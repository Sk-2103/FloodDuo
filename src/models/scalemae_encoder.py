"""Scale-MAE (ViT-L/16) encoder wrapper for comparative evaluation.

Fully frozen; no adapters. Input: (B, 4, H, W) raw reflectance.
Reuses pangaea-bench's Scale-MAE (`ScaleMAE_Encoder` weights + the
resolution-conditioned sincos pos-embed) — nothing is reimplemented.

Scale-MAE is RGB-only (3 channels) and *scale-aware*: its positional embedding
is conditioned on the input ground sample distance (GSD). Unlike CROMA it uses
standard attention, so it runs at our native resolution (512 -> 32x32,
1024 -> 64x64 at patch-16), matching the Prithvi/TerraMind comparative setup.

We feed R,G,B = PS bands [Red, Green, Blue] (our channel order is B,G,R,NIR, so
indices [2,1,0]) and set the pos-embed resolution to the true 3 m GSD. pangaea's
own forward hardcodes the 224px reshape, so we replicate its forward here but
derive the grid from the actual input size.

Output: list of 4 feature maps (B, 1024, H//16, W//16).
"""

import sys
import logging

import torch
from torch import nn

_PANGAEA = "pretrained/pangaea-bench"
if _PANGAEA not in sys.path:
    sys.path.insert(0, _PANGAEA)

from pangaea.encoders.scalemae_encoder import ScaleMAE_Encoder
from pangaea.encoders.pos_embed import get_2d_sincos_pos_embed_with_resolution

_logger = logging.getLogger(__name__)

_CKPT = "pretrained/scalemae-vitlarge-800.pth"
# Combined PS train stats (B,G,R,NIR) — same as the other comparative branches.
_MEAN = [0.08877, 0.11022, 0.12273, 0.25167]
_STD  = [0.08009, 0.07457, 0.09559, 0.13047]


class ScaleMaeEncoder(nn.Module):
    """Frozen Scale-MAE ViT-L/16, RGB input from 4-band PlanetScope."""

    def __init__(self, indices=(7, 11, 15, 23), norm_mode: str = "dataset",
                 input_res: float = 3.0, rgb_indices=(2, 1, 0)):
        super().__init__()
        self.indices = list(indices)
        self.dim = 1024
        self.patch = 16
        self.norm_mode = norm_mode
        self.input_res = float(input_res)
        self.rgb_indices = list(rgb_indices)   # PS channels feeding R,G,B

        enc = ScaleMAE_Encoder(
            encoder_weights=_CKPT,
            input_size=224, input_bands={"optical": ["B4", "B3", "B2"]},
            output_layers=list(indices), output_dim=1024, download_url="",
            embed_dim=1024, patch_size=16, in_chans=3, depth=24, num_heads=16,
        )
        enc.load_encoder_weights(_logger)
        enc.requires_grad_(False)
        self.backbone = enc

        if norm_mode == "dataset":
            self.register_buffer("mean", torch.tensor(_MEAN).view(1, 4, 1, 1))
            self.register_buffer("std",  torch.tensor(_STD).view(1, 4, 1, 1))

    def forward(self, refl: torch.Tensor) -> list[torch.Tensor]:
        """refl: (B, 4, H, W) raw reflectance -> list of spatial feature maps."""
        x = refl[:, self.rgb_indices]                      # (B, 3, H, W), R,G,B
        if self.norm_mode == "per_image":
            m = x.mean(dim=(2, 3), keepdim=True)
            s = x.std(dim=(2, 3), keepdim=True) + 1e-6
            x = (x - m) / s
        else:
            x = (x - self.mean[:, self.rgb_indices]) / self.std[:, self.rgb_indices]

        B, _, H, W = x.shape
        gh, gw = H // self.patch, W // self.patch
        bb = self.backbone
        res = torch.tensor([self.input_res], device=x.device)

        with torch.no_grad():
            tok = bb.patch_embed(x)                         # (B, gh*gw, D)
            pos = get_2d_sincos_pos_embed_with_resolution(
                self.dim, gh, res, cls_token=True, device=x.device)
            cls = bb.cls_token.expand(B, -1, -1)
            tok = torch.cat([cls, tok], dim=1) + pos.to(tok.dtype)

            feats = []
            for i, blk in enumerate(bb.blocks):
                tok = blk(tok)
                if i in self.indices:
                    f = tok[:, 1:].permute(0, 2, 1).reshape(B, self.dim, gh, gw)
                    feats.append(f.contiguous())
        return [f.detach() for f in feats]
