"""Prithvi-EO v2 300M encoder wrapper for comparative evaluation.

Fully frozen; no adapters. Input: (B, 4, H, W) raw reflectance.
Output: list of 4 spatial feature maps (B, 1024, H//16, W//16).

Band mapping: PlanetScope B/G/R/NIR treated as S2 B2/B3/B4/B8
(first 4 of 6 pretrained channels; weight loading truncates automatically).
"""

import sys
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

_PANGAEA = "pretrained/pangaea-bench"
if _PANGAEA not in sys.path:
    sys.path.insert(0, _PANGAEA)

from pangaea.encoders.prithvi_encoder import Prithvi_Encoder

_logger = logging.getLogger(__name__)

_CKPT = "pretrained/Prithvi_EO_V2_300M.pt"
# Combined PS train stats (same as DOFA branch)
_MEAN = [0.08877, 0.11022, 0.12273, 0.25167]
_STD  = [0.08009, 0.07457, 0.09559, 0.13047]


class Prithvi2Encoder(nn.Module):
    """Frozen Prithvi-EO v2 300M, 4-band PlanetScope input."""

    def __init__(self, indices=(5, 11, 17, 23), norm_mode: str = "dataset"):
        super().__init__()
        self.indices = list(indices)
        self.dim = 1024
        self.norm_mode = norm_mode

        # Instantiate with input_size=512 (training crop).
        # pos_embed is sin-cos so it re-interpolates for eval at 1024px.
        enc = Prithvi_Encoder(
            encoder_weights=_CKPT,
            input_bands={"optical": ["B2", "B3", "B4", "B8"]},
            input_size=512,
            output_dim=1024,
            output_layers=list(indices),
            download_url="",
            patch_size=16,
            tubelet_size=1,
            in_chans=4,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            num_frames=1,
        )
        enc.load_encoder_weights(_logger)
        enc.requires_grad_(False)
        self.backbone = enc

        if norm_mode == "dataset":
            self.register_buffer("mean", torch.tensor(_MEAN).view(1, 4, 1, 1))
            self.register_buffer("std",  torch.tensor(_STD).view(1, 4, 1, 1))

    def _pos_embed(self, H: int, W: int) -> torch.Tensor:
        """Return pos_embed interpolated for H×W input (patch_size=16)."""
        pe = self.backbone.pos_embed   # (1, 1+S*S, D), S=32 for init at 512px
        cls_pe, grid_pe = pe[:, :1], pe[:, 1:]
        h0, w0 = H // 16, W // 16
        s0 = int(round(grid_pe.shape[1] ** 0.5))
        if (h0, w0) == (s0, s0):
            return pe
        grid_pe = F.interpolate(
            grid_pe.reshape(1, s0, s0, -1).permute(0, 3, 1, 2),
            size=(h0, w0), mode="bicubic", align_corners=False,
        ).flatten(2).transpose(1, 2)
        return torch.cat([cls_pe, grid_pe], dim=1)

    def forward(self, refl: torch.Tensor) -> list[torch.Tensor]:
        """refl: (B, 4, H, W) raw reflectance → list of spatial feature maps."""
        if self.norm_mode == "per_image":
            m = refl.mean(dim=(2, 3), keepdim=True)
            s = refl.std(dim=(2, 3), keepdim=True) + 1e-6
            x = (refl - m) / s
        else:
            x = (refl - self.mean) / self.std

        B, _, H, W = x.shape
        h, w = H // 16, W // 16

        with torch.no_grad():
            # Prithvi Conv3d patch embed expects (B, C, T, H, W)
            tok = self.backbone.patch_embed(x.unsqueeze(2))  # → (B, h*w, D)
            cls = self.backbone.cls_token.expand(B, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
            tok = tok + self._pos_embed(H, W)

            feats = []
            for i, blk in enumerate(self.backbone.blocks):
                tok = blk(tok)
                if i in self.indices:
                    f = tok[:, 1:].permute(0, 2, 1).reshape(B, self.dim, h, w)
                    feats.append(f.detach())

        return feats
