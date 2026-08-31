"""LoRA injection for frozen ViT backbones.

LoRA on the attention qkv linears carries the global/semantic adaptation
(low-rank update to how tokens attend), complementing the conv adapters that
carry local spatial detail. Base weights stay frozen; only A/B train.
"""

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        self.scale = alpha / rank

    def forward(self, x):
        return (self.base(x) +
                F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale)


def inject_lora(model: nn.Module, names=("qkv", "to_qkv"),
                rank: int = 8, alpha: int = 16) -> int:
    """Replace every nn.Linear whose attribute name is in `names`."""
    count = 0
    for module in model.modules():
        for attr in names:
            child = getattr(module, attr, None)
            if isinstance(child, nn.Linear):
                setattr(module, attr, LoRALinear(child, rank, alpha))
                count += 1
    return count
