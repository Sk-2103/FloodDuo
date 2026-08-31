"""TerraMind-L encoder wrapper for comparative evaluation.

Fully frozen; no adapters. Input: (B, 4, H, W) raw reflectance.
Output: list of 4 spatial feature maps (B, 1024, H//16, W//16).

Band mapping: PlanetScope B/G/R/NIR mapped to S2 BLUE/GREEN/RED/NIR_BROAD.
Pretrained weights select the matching columns from the 12-band S2L2A
patch projection via select_modality_patch_embed_weights.
"""

import sys
import torch
from torch import nn

_PANGAEA = "pretrained/pangaea-bench"
if _PANGAEA not in sys.path:
    sys.path.insert(0, _PANGAEA)

from pangaea.encoders.terramind_encoder import terramind_v1_large

_CKPT = "pretrained/TerraMind_v1_large.pt"
# PS bands mapped to S2 band names (for weight column selection)
_PS_BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
# Combined PS train stats (same as DOFA branch)
_MEAN = [0.08877, 0.11022, 0.12273, 0.25167]
_STD  = [0.08009, 0.07457, 0.09559, 0.13047]


class TerramindEncoder(nn.Module):
    """Frozen TerraMind-L (ViT-L/16), 4-band PlanetScope input via S2L2A embed."""

    def __init__(self, indices=(5, 11, 17, 23), norm_mode: str = "dataset"):
        super().__init__()
        self.indices = list(indices)
        self.dim = 1024
        self.norm_mode = norm_mode

        # Build model with full S2L2A embedding (12 channels), then load
        # pretrained weights, then select the 4 PS band columns.
        # Use img_size=224 to match pretrained pos_emb shape; forward
        # interpolates the pos_emb at runtime for any image size.
        # terramind_v1_large already passes pretrained_bands internally.
        model = terramind_v1_large(
            encoder_weights=_CKPT,
            input_size=224,
            input_bands={"optical": _PS_BANDS},
            output_layers=list(indices),
            output_dim=1024,
            download_url="",
            modalities=["S2L2A"],
            img_size=224,
            merge_method="mean",
            patch_size=16,
            bands={"S2L2A": _PS_BANDS},
        )
        model.requires_grad_(False)
        self.backbone = model

        if norm_mode == "dataset":
            self.register_buffer("mean", torch.tensor(_MEAN).view(1, 4, 1, 1))
            self.register_buffer("std",  torch.tensor(_STD).view(1, 4, 1, 1))

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
        bk = self.backbone

        with torch.no_grad():
            # Patch embedding: ImageEncoderEmbedding handles pos_emb
            # interpolation at runtime for any H×W divisible by patch_size.
            mod_key = bk.mod_name_mapping["S2L2A"]
            mod_dict = bk.encoder_embeddings[mod_key]({"tensor": x})
            tok = mod_dict["x"] + mod_dict["emb"]  # (B, N, D)

            taps = []
            for i, blk in enumerate(bk.encoder):
                tok = blk(tok)
                if i in self.indices:
                    taps.append(tok.clone())

            taps[-1] = bk.encoder_norm(tok)

            feats = [
                t.permute(0, 2, 1).reshape(B, self.dim, h, w).detach()
                for t in taps
            ]
        return feats
