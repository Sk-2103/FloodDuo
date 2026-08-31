"""PPA: Parallelized Patch-Aware Attention (adapted from HCF-Net, arXiv:2403.10778).

Three parallel branches — serial conv (fine detail), local patch attention and
global patch attention — summed, then refined by channel + spatial attention.
Used in the high-resolution stem to keep small water bodies and thin channels
alive before fusion with coarse ViT semantics.
"""

import torch
from torch import nn
import torch.nn.functional as F


class PatchAttention(nn.Module):
    """Gate features with attention computed over p x p average-pooled patches."""

    def __init__(self, dim: int, patches: int):
        super().__init__()
        self.patches = patches
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.GELU(), nn.Conv2d(dim // 4, dim, 1)
        )

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, self.patches)
        g = torch.sigmoid(self.mlp(g))
        g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * g


class ChannelAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction), nn.GELU(), nn.Linear(dim // reduction, dim)
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.mlp(x.mean(dim=(2, 3)))
        mx = self.mlp(x.amax(dim=(2, 3)))
        return x * torch.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(s))


class PPA(nn.Module):
    def __init__(self, in_dim: int, dim: int, local_patches: int = 16,
                 global_patches: int = 4):
        super().__init__()
        self.skip = nn.Conv2d(in_dim, dim, 1)
        self.serial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, dim),
        )
        self.local_attn = PatchAttention(dim, local_patches)
        self.global_attn = PatchAttention(dim, global_patches)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()
        self.norm = nn.GroupNorm(1, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.skip(x)
        y = self.serial(x) + self.local_attn(x) + self.global_attn(x)
        y = self.sa(self.ca(y))
        return self.act(self.norm(x + y))


class DetailStem(nn.Module):
    """High-res stem on the raw 4-band input: PPA-refined skips at 1/2 and 1/4."""

    def __init__(self, in_ch: int = 4, c2: int = 64, c4: int = 128):
        super().__init__()
        self.conv1 = nn.Sequential(  # full res -> 1/2
            nn.Conv2d(in_ch, c2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c2), nn.GELU(),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.GroupNorm(8, c2), nn.GELU(),
        )
        self.ppa2 = PPA(c2, c2, local_patches=32, global_patches=8)
        self.conv2 = nn.Sequential(  # 1/2 -> 1/4
            nn.Conv2d(c2, c4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c4), nn.GELU(),
            nn.Conv2d(c4, c4, 3, padding=1, bias=False),
            nn.GroupNorm(8, c4), nn.GELU(),
        )
        self.ppa4 = PPA(c4, c4, local_patches=16, global_patches=4)

    def forward(self, x):
        f2 = self.ppa2(self.conv1(x))
        f4 = self.ppa4(self.conv2(f2))
        return f2, f4
