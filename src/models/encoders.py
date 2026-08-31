"""Frozen foundation-model encoders with multi-depth feature taps.

Both encoders run under no_grad (fully frozen); trainable ADAC adapters are
applied to each tapped feature map afterwards. Normalization lives inside the
wrappers (buffers), so the data pipeline always provides raw reflectance
(B, 4, H, W) in [0, ~0.5].
"""

import contextlib
import math

import timm
import torch
from torch import nn
import torch.nn.functional as F

from .adapters import build_adapter_stack
from .earth_adapter import inject_earth_adapter_timm, wrap_blocks_ckpt
from .lora import inject_lora
from . import models_dwv

# PlanetScope SuperDove band centers (um): blue, green, red, NIR
PS_WAVELENGTHS = [0.490, 0.565, 0.665, 0.865]
DOFA_CKPT_BASE  = "pretrained/DOFA_ViT_base_e100.pth"
DOFA_CKPT_LARGE = "pretrained/DOFA_ViT_large_e100.pth"
DOFA_CKPT = DOFA_CKPT_BASE  # kept for back-compat


class DinoV3Encoder(nn.Module):
    """DINOv3 on RGB (reflectance -> TCI-like 8-bit range -> model norm)."""

    def __init__(self, name: str = "vit_large_patch16_dinov3.lvd1689m",
                 indices=(5, 11, 17, 23), adapter=True,
                 norm_mode: str = "dataset", lora_rank: int = 0,
                 earth_adapter: bool = False, ea_bottleneck: int = 32,
                 ea_rho: float = 0.25, grad_ckpt: bool = False,
                 rgb_indices=(2, 1, 0)):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.backbone.requires_grad_(False)
        if lora_rank > 0:
            n = inject_lora(self.backbone, rank=lora_rank)
            assert n > 0, "no LoRA targets found in DINOv3"
        if earth_adapter:
            inject_earth_adapter_timm(self.backbone, ea_bottleneck, ea_rho)
        if grad_ckpt:
            self.backbone.blocks = wrap_blocks_ckpt(self.backbone.blocks)
        # in-block trainables -> backbone must run with grad
        self.lora = lora_rank > 0 or earth_adapter
        self.indices = list(indices)
        self.dim = self.backbone.embed_dim
        self.norm_mode = norm_mode
        self.rgb_indices = list(rgb_indices)  # which input bands -> R, G, B
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.adapters = nn.ModuleList(
            [build_adapter_stack(self.dim, adapter) for _ in self.indices])

    def forward(self, inp: torch.Tensor) -> list[torch.Tensor]:
        # select channels; zero-pad if fewer than 3 indices given
        chans = [inp[:, i:i+1] for i in self.rgb_indices]
        while len(chans) < 3:
            chans.append(torch.zeros_like(chans[0]))
        rgb = torch.cat(chans, dim=1)   # [B, 3, H, W]
        if self.norm_mode == "per_image":
            m = rgb.mean(dim=(2, 3), keepdim=True)
            s = rgb.std(dim=(2, 3), keepdim=True) + 1e-6
            x = (rgb - m) / s
        else:
            # reflectance -> [0,1] (TCI: 0.3 refl -> 1.0), then ImageNet norm
            rgb = rgb.mul(1.0 / 0.3).clamp(0, 1)
            x = (rgb - self.mean) / self.std
        ctx = contextlib.nullcontext() if self.lora else torch.no_grad()
        with ctx:
            feats = self.backbone.forward_intermediates(
                x, indices=self.indices, output_fmt="NCHW",
                intermediates_only=True, norm=False)
        if not self.lora:
            feats = [f.detach() for f in feats]
        return [ad(f) for ad, f in zip(self.adapters, feats)]


class DofaEncoder(nn.Module):
    """DOFA on all 4 bands via wavelength-conditioned patch embedding.

    size="base"  → ViT-B/16 (depth 12), default indices (2,5,8,11)
    size="large" → ViT-L/16 (depth 24), default indices (5,11,17,23)
    """

    def __init__(self, ckpt_path: str = None, indices=None,
                 adapter=True, band_mean=None, band_std=None,
                 norm_mode: str = "dataset", upsample: int = 1,
                 lora_rank: int = 0, size: str = "base"):
        super().__init__()
        self.norm_mode = norm_mode
        self.upsample = upsample  # >1: finer patch grid (e.g. 2 -> 1/8 stride)
        if size == "large":
            if ckpt_path is None:
                ckpt_path = DOFA_CKPT_LARGE
            if indices is None:
                indices = (5, 11, 17, 23)
            self.backbone = models_dwv.vit_large_patch16(num_classes=0, global_pool=False)
        else:
            if ckpt_path is None:
                ckpt_path = DOFA_CKPT_BASE
            if indices is None:
                indices = (2, 5, 8, 11)
            self.backbone = models_dwv.vit_base_patch16(num_classes=0, global_pool=False)
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("model", sd)
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        assert not [k for k in missing if "head" not in k and "norm." not in k], \
            f"DOFA missing keys: {missing}"
        self.backbone.requires_grad_(False)
        self.lora = lora_rank > 0
        if self.lora:
            n = inject_lora(self.backbone.blocks, rank=lora_rank)
            assert n > 0, "no LoRA targets found in DOFA"
        self.indices = list(indices)
        self.dim = self.backbone.cls_token.shape[-1]
        # combined FloodPlanet+UFO train stats (configs/band_stats.json)
        mean = band_mean if band_mean is not None else [0.08877, 0.11022, 0.122725, 0.25167]
        std = band_std if band_std is not None else [0.080088, 0.074572, 0.095589, 0.130471]
        self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
        self.adapters = nn.ModuleList(
            [build_adapter_stack(self.dim, adapter) for _ in self.indices])

    def _pos_embed(self, h: int, w: int) -> torch.Tensor:
        pe = self.backbone.pos_embed  # (1, 197, C) for 224
        cls_pe, grid_pe = pe[:, :1], pe[:, 1:]
        s = int(math.sqrt(grid_pe.shape[1]))
        if (h, w) != (s, s):
            grid_pe = F.interpolate(
                grid_pe.reshape(1, s, s, -1).permute(0, 3, 1, 2),
                size=(h, w), mode="bicubic", align_corners=False,
            ).flatten(2).transpose(1, 2)
        return cls_pe, grid_pe

    def forward(self, refl_4band: torch.Tensor) -> list[torch.Tensor]:
        if self.norm_mode == "per_image":
            m = refl_4band.mean(dim=(2, 3), keepdim=True)
            s = refl_4band.std(dim=(2, 3), keepdim=True) + 1e-6
            x = (refl_4band - m) / s
        else:
            x = (refl_4band - self.mean) / self.std
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode="bilinear",
                              align_corners=False)
        b = self.backbone
        B, _, H, W = x.shape
        h, w = H // 16, W // 16
        feats = []
        ctx = contextlib.nullcontext() if self.lora else torch.no_grad()
        with ctx:
            waves = torch.tensor(PS_WAVELENGTHS, device=x.device).float()
            tok, _ = b.patch_embed(x, waves)
            cls_pe, grid_pe = self._pos_embed(h, w)
            tok = tok + grid_pe
            cls_tok = (b.cls_token + cls_pe).expand(B, -1, -1)
            tok = torch.cat([cls_tok, tok], dim=1)
            for i, blk in enumerate(b.blocks):
                tok = blk(tok)
                if i in self.indices:
                    f = tok[:, 1:].transpose(1, 2).reshape(B, self.dim, h, w)
                    feats.append(f)
        if not self.lora:
            feats = [f.detach() for f in feats]
        return [ad(f) for ad, f in zip(self.adapters, feats)]
