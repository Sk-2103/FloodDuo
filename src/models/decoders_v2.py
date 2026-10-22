"""UPerNet and LinkNet decoders over fused multi-depth ViT features.

Both consume the 4 fused maps (same grid, e.g. 1/8 for Clay-full-res fusion)
via a shared Reassemble module that builds a 4-level pyramid (DPT-style:
shallow taps get upsampled, deep taps downsampled), then decode to full res.
No stem skips — these decoders are for the stem-free architecture experiments.
"""

import torch
from torch import nn
import torch.nn.functional as F


def conv_gn(cin, cout, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.GroupNorm(8, cout), nn.GELU(),
    )


class Reassemble(nn.Module):
    """4 same-grid maps -> pyramid at [2x, 1x, 1/2x, 1/4x] of the input grid.
    Depth 0 (shallowest) -> finest level, depth 3 -> coarsest."""

    def __init__(self, dim: int = 256, out_dims=(96, 192, 384, 768)):
        super().__init__()
        self.proc = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, out_dims[0], 2, stride=2),
                          conv_gn(out_dims[0], out_dims[0])),
            conv_gn(dim, out_dims[1]),
            nn.Sequential(nn.Conv2d(dim, out_dims[2], 3, stride=2, padding=1),
                          conv_gn(out_dims[2], out_dims[2])),
            nn.Sequential(nn.Conv2d(dim, out_dims[3], 3, stride=2, padding=1),
                          conv_gn(out_dims[3], out_dims[3]),
                          nn.Conv2d(out_dims[3], out_dims[3], 3, stride=2,
                                    padding=1)),
        ])

    def forward(self, feats: list) -> list:
        return [p(f) for p, f in zip(self.proc, feats)]


class PPM(nn.Module):
    def __init__(self, dim, out, scales=(1, 2, 3, 6)):
        super().__init__()
        c = out // len(scales)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s),
                          nn.Conv2d(dim, c, 1, bias=False),
                          nn.GroupNorm(1, c), nn.GELU())
            for s in scales
        ])
        self.fuse = conv_gn(dim + c * len(scales), out)

    def forward(self, x):
        size = x.shape[-2:]
        outs = [x] + [F.interpolate(st(x), size=size, mode="bilinear",
                                    align_corners=False)
                      for st in self.stages]
        return self.fuse(torch.cat(outs, dim=1))


class UPerNetDecoder(nn.Module):
    def __init__(self, dim: int = 256, channels: int = 192,
                 pyramid_dims=(96, 192, 384, 768)):
        super().__init__()
        self.reassemble = Reassemble(dim, pyramid_dims)
        self.ppm = PPM(pyramid_dims[-1], channels)
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, channels, 1) for c in pyramid_dims[:-1]])
        self.fpn_conv = nn.ModuleList(
            [conv_gn(channels, channels) for _ in pyramid_dims[:-1]])
        self.bottleneck = conv_gn(channels * 4, channels)
        self.head = nn.Sequential(conv_gn(channels, 64), nn.Conv2d(64, 1, 1))

    def forward(self, fused: list, out_size) -> torch.Tensor:
        pyr = self.reassemble(fused)            # fine -> coarse
        laterals = [lat(p) for lat, p in zip(self.lateral, pyr[:-1])]
        laterals.append(self.ppm(pyr[-1]))
        for i in range(len(laterals) - 1, 0, -1):  # top-down
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode="bilinear", align_corners=False)
        outs = [conv(l) for conv, l in zip(self.fpn_conv, laterals[:-1])]
        outs.append(laterals[-1])
        target = outs[0].shape[-2:]
        outs = [F.interpolate(o, size=target, mode="bilinear",
                              align_corners=False) for o in outs]
        x = self.bottleneck(torch.cat(outs, dim=1))
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear",
                             align_corners=False)


class ResidualConvUnit(nn.Module):
    """DPT residual conv unit."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class FeatureFusionBlock(nn.Module):
    """DPT fusion: refine skip, add upsampled deeper path, refine, up x2."""

    def __init__(self, dim):
        super().__init__()
        self.rcu1 = ResidualConvUnit(dim)
        self.rcu2 = ResidualConvUnit(dim)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear",
                             align_corners=False)


class DPTDecoder(nn.Module):
    """DPT head: Reassemble pyramid -> project to common width -> RefineNet
    coarse-to-fine fusion -> conv head."""

    def __init__(self, dim: int = 256, channels: int = 192,
                 pyramid_dims=(96, 192, 384, 768)):
        super().__init__()
        self.reassemble = Reassemble(dim, pyramid_dims)
        self.proj = nn.ModuleList(
            [nn.Conv2d(c, channels, 3, padding=1, bias=False)
             for c in pyramid_dims])
        self.fusion = nn.ModuleList(
            [FeatureFusionBlock(channels) for _ in pyramid_dims])
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels // 2), nn.GELU(),
            nn.Conv2d(channels // 2, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, fused: list, out_size) -> torch.Tensor:
        pyr = [p(f) for p, f in zip(self.proj, self.reassemble(fused))]
        x = self.fusion[-1](pyr[-1])                  # coarsest
        for fuse, skip in zip(self.fusion[-2::-1], pyr[-2::-1]):
            x = fuse(x, skip)
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear",
                             align_corners=False)


class LinkNetBlock(nn.Module):
    """LinkNet decoder block: 1x1 reduce -> deconv x2 -> 1x1 expand."""

    def __init__(self, cin, cout):
        super().__init__()
        c = cin // 4
        self.block = nn.Sequential(
            nn.Conv2d(cin, c, 1, bias=False), nn.GroupNorm(1, c), nn.GELU(),
            nn.ConvTranspose2d(c, c, 2, stride=2),
            nn.GroupNorm(1, c), nn.GELU(),
            nn.Conv2d(c, cout, 1, bias=False), nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class LinkNetDecoder(nn.Module):
    """LinkNet-style: decode coarse->fine with ADDITIVE skips from the
    pyramid (vs UPerNet's concat-all). Lighter, sequential."""

    def __init__(self, dim: int = 256, pyramid_dims=(96, 192, 384, 768)):
        super().__init__()
        self.reassemble = Reassemble(dim, pyramid_dims)
        d = list(pyramid_dims)
        self.dec3 = LinkNetBlock(d[3], d[2])
        self.dec2 = LinkNetBlock(d[2], d[1])
        self.dec1 = LinkNetBlock(d[1], d[0])
        self.final = nn.Sequential(
            LinkNetBlock(d[0], 48), conv_gn(48, 32), nn.Conv2d(32, 1, 1))

    def forward(self, fused: list, out_size) -> torch.Tensor:
        p0, p1, p2, p3 = self.reassemble(fused)  # fine -> coarse
        x = self.dec3(p3)
        x = self.dec2(x + F.interpolate(p2, size=x.shape[-2:], mode="bilinear",
                                        align_corners=False))
        x = self.dec1(x + F.interpolate(p1, size=x.shape[-2:], mode="bilinear",
                                        align_corners=False))
        x = self.final(x + F.interpolate(p0, size=x.shape[-2:],
                                         mode="bilinear", align_corners=False))
        return F.interpolate(x, size=out_size, mode="bilinear",
                             align_corners=False)
