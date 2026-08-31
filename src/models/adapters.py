"""ADAC: Atrous Depth-wise Convolution Adapter.

Applied to ViT feature maps (B, C, H, W) tapped at multiple depths of a frozen
backbone. Parallel depth-wise 3x3 convolutions with increasing dilation re-inject
multi-scale local structure that patch-16 attention washes out. The residual
branch is zero-gated (gamma init 0) so the adapter starts as identity.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ADAC(nn.Module):
    def __init__(self, dim: int, dilations=(1, 2, 3)):
        super().__init__()
        self.dw = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim, bias=False)
            for d in dilations
        ])
        self.norm = nn.GroupNorm(1, dim)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = sum(conv(x) for conv in self.dw)
        y = self.pw(self.act(self.norm(y)))
        return x + self.gamma * y


class PPAdapter(nn.Module):
    """Pyramid Pooling Adapter: multi-scale pooled context (PSPNet-style)
    re-injected into the feature map through a zero-gated residual. Carries
    region/scene-level context per tap, complementing ADAC's local detail."""

    def __init__(self, dim: int, scales=(1, 2, 3, 6)):
        super().__init__()
        self.scales = scales
        c = dim // 4
        self.reduce = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, c, 1, bias=False), nn.GroupNorm(1, c),
                          nn.GELU())
            for _ in scales
        ])
        self.fuse = nn.Conv2d(dim + c * len(scales), dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        ctx = [x]
        for s, red in zip(self.scales, self.reduce):
            p = red(F.adaptive_avg_pool2d(x, s))
            ctx.append(F.interpolate(p, size=size, mode="bilinear",
                                     align_corners=False))
        return x + self.gamma * self.fuse(torch.cat(ctx, dim=1))


class HFDA(nn.Module):
    """High-Frequency Detail Adapter (proposed): isolates the high-frequency
    residual of the feature map (x minus its local average — an edge/texture
    band the ViT tends to smooth away), sharpens it with depth-wise convs and
    channel gating, and re-injects it zero-gated. Targets thin channels and
    shoreline pixels specifically."""

    def __init__(self, dim: int, pool: int = 3, reduction: int = 8):
        super().__init__()
        self.pool = pool
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(1, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // reduction), nn.GELU(),
            nn.Linear(dim // reduction, dim), nn.Sigmoid(),
        )
        self.pw = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf = x - F.avg_pool2d(x, self.pool, stride=1, padding=self.pool // 2)
        hf = self.dw(hf)
        g = self.gate(hf.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
        return x + self.gamma * self.pw(hf * g)


class StripAdapter(nn.Module):
    """Strip/directional convolution adapter (SegNeXt-MSCA style).

    Paired 1xk / kx1 depth-wise strip kernels integrate evidence ALONG
    elongated structures (rivers, canals, inundation fingers) while staying
    narrow across them; the result gates the input multiplicatively
    (attention) through a zero-init residual."""

    def __init__(self, dim: int, ks=(7, 11)):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.strips = nn.ModuleList()
        for k in ks:
            self.strips.append(nn.ModuleList([
                nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim),
                nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim),
            ]))
        self.pw = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.dw(x)
        a = a + sum(v(h(a)) for h, v in self.strips)
        return x + self.gamma * (self.pw(a) * x)


class WaveletAdapter(nn.Module):
    """Haar-DWT detail adapter: decompose the feature map into LL/LH/HL/HH
    subbands, refine the three ORIENTATION-SELECTIVE detail subbands
    (horizontal/vertical/diagonal edges) with depth-wise convs, inverse-
    transform the details only (LL zeroed), zero-gated residual. Localized
    in space AND frequency — complements Earth-Adapter's global DFT experts."""

    def __init__(self, dim: int):
        super().__init__()
        self.proc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.GroupNorm(1, dim), nn.GELU(), nn.Conv2d(dim, dim, 1))
            for _ in range(3)  # LH, HL, HH
        ])
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "wavelet adapter needs even grid"
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        lh = (x00 - x01 + x10 - x11) / 2
        hl = (x00 + x01 - x10 - x11) / 2
        hh = (x00 - x01 - x10 + x11) / 2
        lh, hl, hh = (p(s) for p, s in zip(self.proc, (lh, hl, hh)))
        # inverse Haar with LL = 0 -> pure detail delta at full grid
        d00 = (lh + hl + hh) / 2
        d01 = (-lh + hl - hh) / 2
        d10 = (lh - hl - hh) / 2
        d11 = (-lh - hl + hh) / 2
        delta = x.new_zeros(B, C, H, W)
        delta[..., 0::2, 0::2] = d00
        delta[..., 0::2, 1::2] = d01
        delta[..., 1::2, 0::2] = d10
        delta[..., 1::2, 1::2] = d11
        return x + self.gamma * delta


ADAPTERS = {"adac": ADAC, "ppad": PPAdapter, "hfda": HFDA,
            "strip": StripAdapter, "wavelet": WaveletAdapter}


def build_adapter_stack(dim: int, kinds) -> nn.Module:
    """kinds: True -> ['adac']; False/None -> identity; list -> sequential."""
    if kinds is True:
        kinds = ["adac"]
    if not kinds:
        return nn.Identity()
    return nn.Sequential(*[ADAPTERS[k](dim) for k in kinds])
