"""Clay v1.5 spectral encoder (drop-in alternative to DofaEncoder).

Clay (https://github.com/Clay-foundation/model) uses a wavelength-conditioned
DynamicEmbedding (DOFA-style) + GSD-aware position encoding, ViT-L, patch 8,
trained on much more EO data than DOFA. We bypass Encoder.forward (which masks
75% of patches for MAE) and run patch-embed -> encodings -> transformer layers
manually, tapping multiple depths.

Input policy: the 4-band crop is fed at HALF resolution (512->256), so the
patch-8 grid (32x32) matches DOFA's patch-16 grid at full crop size; gsd=6 m.
"""

import contextlib

import torch
from torch import nn
import torch.nn.functional as F

from .adapters import build_adapter_stack
from .earth_adapter import inject_earth_adapter_clay
from .lora import inject_lora
from .clay.backbone import Transformer
from .clay.factory import DynamicEmbedding
from .clay.utils import posemb_sincos_2d_with_gsd

CLAY_CKPT = "pretrained/clay-v1.5.ckpt"
# PlanetScope SuperDove defaults (µm, reflectance stats)
PS_WAVELENGTHS = [0.490, 0.565, 0.665, 0.865]
PS_BAND_MEAN   = [0.08877, 0.11022, 0.122725, 0.25167]
PS_BAND_STD    = [0.080088, 0.074572, 0.095589, 0.130471]
# Sentinel-2 L2A 13-band wavelengths (µm), B1-B12 + B8A order from dataset
S2_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
                  0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]
# S1 SAR pseudo-wavelengths (Clay pretraining convention, µm proxy)
SAR_WAVELENGTHS = [3.5, 4.0]  # VV, VH


class ClayEncoder(nn.Module):
    def __init__(self, ckpt_path: str = CLAY_CKPT, indices=(5, 11, 17, 23),
                 adapter=True, norm_mode: str = "dataset",
                 patch_size: int = 8, dim: int = 1024, depth: int = 24,
                 heads: int = 16, dim_head: int = 64, mlp_ratio: float = 4.0,
                 downsample: int = 1, gsd: float = 3.0, lora_rank: int = 0,
                 earth_adapter: bool = False, ea_bottleneck: int = 32,
                 ea_rho: float = 0.25, grad_ckpt: bool = False,
                 wavelengths=None, band_mean=None, band_std=None):
        super().__init__()
        self.norm_mode = norm_mode
        self.indices = list(indices)
        self.dim = dim
        self.patch_size = patch_size
        self.downsample = downsample
        self.gsd = gsd

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.patch_embedding = DynamicEmbedding(
            wave_dim=128, num_latent_tokens=128, patch_size=patch_size,
            embed_dim=dim, is_decoder=False)
        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio), fused_attn=True)

        sd = torch.load(ckpt_path, map_location="cpu",
                        weights_only=False)["state_dict"]
        prefix = "model.encoder."
        enc_sd = {k[len(prefix):]: v for k, v in sd.items()
                  if k.startswith(prefix)}
        missing, unexpected = self.load_state_dict(enc_sd, strict=False)
        real_missing = [k for k in missing if not k.startswith("adapters")]
        assert not real_missing, f"Clay missing keys: {real_missing[:5]}"

        for p in self.parameters():
            p.requires_grad_(False)
        if lora_rank > 0:
            n = inject_lora(self.transformer, rank=lora_rank)
            assert n > 0, "no LoRA targets found in Clay"
        if earth_adapter:
            inject_earth_adapter_clay(self.transformer, dim,
                                      ea_bottleneck, ea_rho)
        # in-block trainables -> backbone must run with grad
        self.lora = lora_rank > 0 or earth_adapter
        self.grad_ckpt = grad_ckpt
        # configurable band stats (defaults to PlanetScope 4-band)
        wl      = wavelengths if wavelengths is not None else PS_WAVELENGTHS
        n_bands = len(wl)
        # default mean/std: PS values for 4-band, neutral (0/1) otherwise
        mean = (band_mean if band_mean is not None
                else (PS_BAND_MEAN if n_bands == 4 else [0.0] * n_bands))
        std  = (band_std  if band_std  is not None
                else (PS_BAND_STD  if n_bands == 4 else [1.0] * n_bands))
        self.register_buffer("wavelengths_buf", torch.tensor(wl, dtype=torch.float32))
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, n_bands, 1, 1))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32).view(1, n_bands, 1, 1))
        self.adapters = nn.ModuleList(
            [build_adapter_stack(dim, adapter) for _ in self.indices])

    def forward(self, refl_4band: torch.Tensor) -> list[torch.Tensor]:
        if self.norm_mode == "per_image":
            m = refl_4band.mean(dim=(2, 3), keepdim=True)
            s = refl_4band.std(dim=(2, 3), keepdim=True) + 1e-6
            x = (refl_4band - m) / s
        else:
            x = (refl_4band - self.mean) / self.std
        if self.downsample > 1:
            x = F.interpolate(x, scale_factor=1 / self.downsample,
                              mode="bilinear", antialias=True)
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        feats = []
        ctx = contextlib.nullcontext() if self.lora else torch.no_grad()
        with ctx:
            waves = self.wavelengths_buf.to(device=x.device, dtype=x.dtype)
            patches, _ = self.patch_embedding(x, waves)  # [B, h*w, D]
            pos = posemb_sincos_2d_with_gsd(
                h=h, w=w, dim=self.dim - 8,
                gsd=torch.tensor(self.gsd)).to(device=x.device, dtype=x.dtype)
            # zero time/latlon metadata encodings (last 8 dims), as Clay does
            # for unknown metadata
            meta = torch.zeros(h * w, 8, device=x.device, dtype=x.dtype)
            patches = patches + torch.cat([pos, meta], dim=-1)[None]
            tok = torch.cat(
                [self.cls_token.expand(B, -1, -1).to(x.dtype), patches], dim=1)
            for i, (attn, ff) in enumerate(self.transformer.layers):
                def _layer(t, attn=attn, ff=ff):
                    t = attn(t) + t
                    return ff(t) + t
                if (self.grad_ckpt and self.training
                        and torch.is_grad_enabled()):
                    tok = torch.utils.checkpoint.checkpoint(
                        _layer, tok, use_reentrant=False)
                else:
                    tok = _layer(tok)
                if i in self.indices:
                    f = tok[:, 1:].transpose(1, 2).reshape(B, self.dim, h, w)
                    feats.append(f)
        if not self.lora:
            feats = [f.detach() for f in feats]
        return [ad(f) for ad, f in zip(self.adapters, feats)]
