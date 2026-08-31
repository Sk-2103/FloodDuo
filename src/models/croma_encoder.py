"""CROMA optical (ViT-L) encoder wrapper for comparative evaluation.

Fully frozen; no adapters. Input: (B, 4, H, W) raw reflectance.
Reuses pangaea-bench's CROMA implementation
(`pangaea.encoders.croma_encoder.CROMA_OPTICAL_Encoder`) and the
`CROMA_large.pt` weights — nothing about CROMA is reimplemented here.

CROMA uses 2D-ALiBi attention whose bias is tied to a *fixed* patch grid
(pretrained at 120px -> 15x15 patches). It therefore cannot run at our
512/1024px tiles on a patch-8 grid (O(N^2) attention + a Python double-loop
bias that is unusable at those token counts). We resize the CROMA input to a
fixed `size` (default 256 -> 32x32 patches), matching the ~32^2 token grid the
patch-16 GFMs (Prithvi/TerraMind) get at the 512px training resolution, and
recompute the ALiBi bias once (vectorised) for that grid.

Output: list of 4 feature maps (B, 1024, size//8, size//8).

Band mapping: PS B/G/R/NIR -> S2 B2/B3/B4/B8, placed into CROMA's 12-band
optical input (the other 8 bands are zero, so their linear-input weight
columns contribute nothing -> equivalent to selecting the 4 pretrained band
columns, mirroring the Prithvi/TerraMind wrappers).
"""

import sys
import math
import logging

import torch
import torch.nn.functional as F
from torch import nn

_PANGAEA = "pretrained/pangaea-bench"
if _PANGAEA not in sys.path:
    sys.path.insert(0, _PANGAEA)

from pangaea.encoders.croma_encoder import CROMA_OPTICAL_Encoder

_logger = logging.getLogger(__name__)

_CKPT = "pretrained/CROMA_large.pt"
# Combined PS train stats (same as the DOFA/Prithvi/TerraMind branches)
_MEAN = [0.08877, 0.11022, 0.12273, 0.25167]
_STD  = [0.08009, 0.07457, 0.09559, 0.13047]
# CROMA 12-band optical order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12.
# PS B(blue)=B2, G=B3, R=B4, NIR=B8  ->  indices into the 12-band tensor.
_PS_TO_S2 = [1, 2, 3, 7]
# pangaea CROMA optical band names (for instantiation only)
_S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
             "B8A", "B9", "B11", "B12"]


def _alibi_vectorised(num_heads: int, num_patches: int) -> torch.Tensor:
    """Vectorised re-implementation of pangaea's get_2dalibi.

    Produces an identical (1, num_heads, N, N) bias: -dist(i, j) * slope[h],
    with patches in row-major (i outer, j inner) order. The upstream version
    is a Python double-loop over all N^2 patch pairs (too slow at N>=1024).
    """
    s = int(math.sqrt(num_patches))
    assert s * s == num_patches
    coords = torch.stack(
        torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij"), dim=-1
    ).reshape(-1, 2).float()                       # (N, 2), row-major
    dist = torch.cdist(coords, coords)             # (N, N) euclidean

    def get_slopes(n):
        def pow2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * start ** i for i in range(n)]
        if math.log2(n).is_integer():
            return pow2(n)
        cp = 2 ** math.floor(math.log2(n))
        return pow2(cp) + get_slopes(2 * cp)[0::2][: n - cp]

    slopes = torch.tensor(get_slopes(num_heads)).view(num_heads, 1, 1)
    return (-dist.unsqueeze(0) * slopes).unsqueeze(0)   # (1, num_heads, N, N)


class CromaEncoder(nn.Module):
    """Frozen CROMA-L optical ViT, 4-band PlanetScope input via a fixed grid."""

    def __init__(self, indices=(5, 11, 17, 23), norm_mode: str = "dataset",
                 size: int = 256):
        super().__init__()
        self.indices = list(indices)
        self.dim = 1024
        self.norm_mode = norm_mode
        self.size = size
        assert size % 8 == 0, "CROMA patch size is 8"

        enc = CROMA_OPTICAL_Encoder(
            encoder_weights=_CKPT,
            input_size=size,
            input_bands={"optical": _S2_BANDS},
            output_layers=list(indices),
            output_dim=1024,
            download_url="",
            size="large",
        )
        enc.load_encoder_weights(_logger)
        enc.requires_grad_(False)
        self.backbone = enc

        n_patches = (size // 8) ** 2
        self.register_buffer(
            "attn_bias", _alibi_vectorised(16, n_patches), persistent=False)

        if norm_mode == "dataset":
            self.register_buffer("mean", torch.tensor(_MEAN).view(1, 4, 1, 1))
            self.register_buffer("std",  torch.tensor(_STD).view(1, 4, 1, 1))

    def forward(self, refl: torch.Tensor) -> list[torch.Tensor]:
        """refl: (B, 4, H, W) raw reflectance -> list of spatial feature maps."""
        if self.norm_mode == "per_image":
            m = refl.mean(dim=(2, 3), keepdim=True)
            s = refl.std(dim=(2, 3), keepdim=True) + 1e-6
            x = (refl - m) / s
        else:
            x = (refl - self.mean) / self.std

        # CROMA runs on a fixed grid; resize the 4-band input to it.
        x = F.interpolate(x, size=(self.size, self.size),
                          mode="bilinear", align_corners=False)
        B = x.shape[0]
        x12 = x.new_zeros(B, 12, self.size, self.size)
        for c_src, c_dst in enumerate(_PS_TO_S2):
            x12[:, c_dst] = x[:, c_src]

        g = self.size // 8
        with torch.no_grad():
            outs = self.backbone.s2_encoder(
                x12, self.attn_bias.to(x.device), self.indices)
            feats = [o.permute(0, 2, 1).reshape(B, self.dim, g, g).contiguous()
                     for o in outs]
        return [f.detach() for f in feats]
