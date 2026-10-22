"""Fine-detail decoder: RefineNet/DPT-style progressive fusion.

Depth maps (all 1/16) are merged residually, then upsampled stage by stage,
fusing the PPA stem skips at 1/4 and 1/2 to recover pixel-level water edges.
"""

import torch
from torch import nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout), nn.GELU(),
    )


class FineDecoder(nn.Module):
    def __init__(self, dim: int = 256, n_depths: int = 4,
                 stem_c2: int = 64, stem_c4: int = 128):
        super().__init__()
        self.depth_merge = nn.ModuleList(
            [conv_block(dim, dim) for _ in range(n_depths)])
        self.up8 = conv_block(dim, 192)            # 1/16 -> 1/8
        self.up4 = conv_block(192, 128)            # 1/8  -> 1/4
        self.skip4 = conv_block(128 + stem_c4, 128)
        self.up2 = conv_block(128, 64)             # 1/4  -> 1/2
        self.skip2 = conv_block(64 + stem_c2, 64)
        self.head = nn.Sequential(
            conv_block(64, 32),
            nn.Conv2d(32, 1, 1),
        )

    @staticmethod
    def _to(x, size):
        if x.shape[-2:] == tuple(size):
            return x
        return F.interpolate(x, size=size, mode="bilinear",
                             align_corners=False)

    def forward(self, depth_feats: list, f2: torch.Tensor,
                f4: torch.Tensor) -> torch.Tensor:
        # target sizes derived from the stem skips (stride-agnostic: depth
        # features may arrive at 1/16 or finer, e.g. Clay patch-8 at 1/8)
        h4, w4 = f4.shape[-2:]
        h2, w2 = f2.shape[-2:]
        # residual merge, deepest first
        x = self.depth_merge[-1](depth_feats[-1])
        for merge, f in zip(self.depth_merge[-2::-1], depth_feats[-2::-1]):
            x = x + merge(f)
        x = self.up8(self._to(x, (h4 // 2, w4 // 2)))   # 1/8
        x = self.up4(self._to(x, (h4, w4)))             # 1/4
        x = self.skip4(torch.cat([x, f4], dim=1))
        x = self.up2(self._to(x, (h2, w2)))             # 1/2
        x = self.skip2(torch.cat([x, f2], dim=1))
        x = self.head(self._to(x, (h2 * 2, w2 * 2)))    # 1/1
        return x  # (B, 1, H, W) logits
