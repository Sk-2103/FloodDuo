"""Classic UNet baseline (from scratch, 4-band input, base width 64).

Input: raw reflectance (B, 4, H, W), z-scored internally with the combined
train statistics — same treatment as the DOFA branch of FloodDualSeg.
"""

import torch
from torch import nn

BAND_MEAN = [0.08877, 0.11022, 0.122725, 0.25167]
BAND_STD = [0.080088, 0.074572, 0.095589, 0.130471]


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 64, norm_mode: str = "dataset"):
        super().__init__()
        self.norm_mode = norm_mode
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.register_buffer("mean", torch.tensor(BAND_MEAN).view(1, 4, 1, 1))
        self.register_buffer("std", torch.tensor(BAND_STD).view(1, 4, 1, 1))
        self.enc = nn.ModuleList([
            double_conv(in_ch, c[0]),
            double_conv(c[0], c[1]),
            double_conv(c[1], c[2]),
            double_conv(c[2], c[3]),
            double_conv(c[3], c[4]),
        ])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(c[4], c[3], 2, stride=2),
            nn.ConvTranspose2d(c[3], c[2], 2, stride=2),
            nn.ConvTranspose2d(c[2], c[1], 2, stride=2),
            nn.ConvTranspose2d(c[1], c[0], 2, stride=2),
        ])
        self.dec = nn.ModuleList([
            double_conv(c[4], c[3]),
            double_conv(c[3], c[2]),
            double_conv(c[2], c[1]),
            double_conv(c[1], c[0]),
        ])
        self.head = nn.Conv2d(c[0], 1, 1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        if self.norm_mode == "per_image":
            m = x.mean(dim=(2, 3), keepdim=True)
            s = x.std(dim=(2, 3), keepdim=True) + 1e-6
            x = (x - m) / s
        else:
            x = (x - self.mean) / self.std
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(self.pool(x)) if i else enc(x)
            skips.append(x)
        x = skips[-1]
        for up, dec, skip in zip(self.up, self.dec, skips[-2::-1]):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)
