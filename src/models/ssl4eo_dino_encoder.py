"""SSL4EO-S12 DINO (ViT-S/16) encoder wrapper for comparative evaluation.

Fully frozen; no adapters. Input: (B, 4, H, W) raw reflectance.
Reuses pangaea-bench's `SSL4EO_DINO_Encoder` + the `B13_vits16_dino` weights —
nothing is reimplemented.

This is a 13-band Sentinel-2 model (ViT-Small: dim 384, depth 12). We map our
4 PS bands into the S2 slots — B(blue)->B2, G->B3, R->B4, NIR->B8 — and zero the
other 9 channels (zeros null out those pretrained patch-embed conv columns, i.e.
equivalent to using only the 4 real band weights). So NIR feeds a *pretrained*
projection (B8), unlike Scale-MAE.

CAVEAT: ViT-Small, not ViT-Large like the other comparative FMs — not capacity
matched (smaller model; expect lower scores partly from size).

pangaea's forward hardcodes a 224px reshape; we replicate it but derive the grid
from the actual input, so it runs at native resolution (512 -> 32^2, 1024 -> 64^2).

Output: list of 4 feature maps (B, 384, H//16, W//16).
"""

import sys
import logging

import torch
from torch import nn

_PANGAEA = "pretrained/pangaea-bench"
if _PANGAEA not in sys.path:
    sys.path.insert(0, _PANGAEA)

from pangaea.encoders.ssl4eo_dino_encoder import SSL4EO_DINO_Encoder

_logger = logging.getLogger(__name__)

_CKPT = "pretrained/B13_vits16_dino_0099_ckpt.pth"
# Combined PS train stats (B,G,R,NIR) — same as the other comparative branches.
_MEAN = [0.08877, 0.11022, 0.12273, 0.25167]
_STD  = [0.08009, 0.07457, 0.09559, 0.13047]
# 13-band S2 order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12.
# PS B->B2, G->B3, R->B4, NIR->B8  ->  indices into the 13-band tensor.
_PS_TO_S2 = [1, 2, 3, 7]
_S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
             "B8A", "B9", "B10", "B11", "B12"]


class SSL4EODinoEncoder(nn.Module):
    """Frozen SSL4EO-S12 DINO ViT-S/16, 4-band PlanetScope via 13-band S2 slots."""

    def __init__(self, indices=(3, 5, 7, 11), norm_mode: str = "dataset"):
        super().__init__()
        self.indices = list(indices)
        self.dim = 384
        self.patch = 16
        self.norm_mode = norm_mode

        enc = SSL4EO_DINO_Encoder(
            encoder_weights=_CKPT, input_size=224,
            input_bands={"optical": _S2_BANDS}, output_layers=list(indices),
            output_dim=384, download_url="", in_chans=13, patch_size=16,
            embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        )
        enc.load_encoder_weights(_logger)
        enc.requires_grad_(False)
        self.backbone = enc

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

        B, _, H, W = x.shape
        gh, gw = H // self.patch, W // self.patch
        x13 = x.new_zeros(B, 13, H, W)
        for c_src, c_dst in enumerate(_PS_TO_S2):
            x13[:, c_dst] = x[:, c_src]

        bb = self.backbone
        with torch.no_grad():
            tok = bb.prepare_tokens(x13)          # patch_embed + cls + interp pos
            feats = []
            for i, blk in enumerate(bb.blocks):
                tok = blk(tok)
                if i in self.indices:
                    f = tok[:, 1:].permute(0, 2, 1).reshape(B, self.dim, gh, gw)
                    feats.append(f.contiguous())
        return [f.detach() for f in feats]
