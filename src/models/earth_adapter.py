"""Earth-Adapter (arXiv:2504.06220): Mixture-of-Frequency-Adapters, in-block.

Three low-rank experts per block — spatial (raw features), low-frequency and
high-frequency (DFT decomposition with square cutoff mask at radius rho) —
combined by a softmax router and a zero-init scale, added in parallel to the
FFN. Applied to patch tokens only; prefix (cls/register) tokens pass through.
Replaces LoRA as the in-block adaptation mechanism (arch4 experiment).
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def _expert(dim: int, bottleneck: int) -> nn.Module:
    return nn.Sequential(nn.Linear(dim, bottleneck), nn.ReLU(),
                         nn.Linear(bottleneck, dim))


class EarthAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 32, rho: float = 0.25):
        super().__init__()
        self.rho = rho
        self.spatial = _expert(dim, bottleneck)
        self.low = _expert(dim, bottleneck)
        self.high = _expert(dim, bottleneck)
        self.router = nn.Linear(dim, 3)
        self.alpha = nn.Parameter(torch.zeros(1))

    def _dft_split(self, x: torch.Tensor, h: int, w: int):
        """x: (B, N, D) patch tokens -> low-/high-frequency token tensors."""
        B, N, D = x.shape
        grid = x.transpose(1, 2).reshape(B, D, h, w).float()
        f = torch.fft.fftshift(torch.fft.fft2(grid, norm="ortho"),
                               dim=(-2, -1))
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device), torch.arange(w, device=x.device),
            indexing="ij")
        mask = (torch.maximum((yy - h / 2).abs(), (xx - w / 2).abs())
                <= self.rho * h / 2)
        lf = torch.fft.ifft2(torch.fft.ifftshift(f * mask, dim=(-2, -1)),
                             norm="ortho").real
        hf = torch.fft.ifft2(torch.fft.ifftshift(f * ~mask, dim=(-2, -1)),
                             norm="ortho").real
        to_tok = lambda g: g.reshape(B, D, N).transpose(1, 2).to(x.dtype)
        return to_tok(lf), to_tok(hf)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        lf, hf = self._dft_split(x, h, w)
        experts = torch.stack(
            [self.spatial(x), self.low(lf), self.high(hf)], dim=0)  # (3,B,N,D)
        gate = torch.softmax(self.router(x.mean(dim=1)), dim=-1)    # (B,3)
        delta = torch.einsum("kbnd,bk->bnd", experts, gate)
        return self.alpha * delta


class FFNWithEA(nn.Module):
    """Wraps a block's FFN/MLP: y = ffn(x) + EA(patch_tokens(x))."""

    def __init__(self, ffn: nn.Module, dim: int, num_prefix: int = 0,
                 bottleneck: int = 32, rho: float = 0.25):
        super().__init__()
        self.ffn = ffn
        self.ea = EarthAdapter(dim, bottleneck, rho)
        self.num_prefix = num_prefix

    def forward(self, x):
        y = self.ffn(x)
        p = self.num_prefix
        tok = x[:, p:]
        n = tok.shape[1]
        h = w = int(math.isqrt(n))
        assert h * w == n, f"non-square token grid: {n}"
        delta = self.ea(tok, h, w)
        if p:
            return torch.cat([y[:, :p], y[:, p:] + delta], dim=1)
        return y + delta


class CkptBlock(nn.Module):
    """Gradient-checkpointing wrapper for a transformer block."""

    def __init__(self, blk: nn.Module):
        super().__init__()
        self.blk = blk

    def forward(self, x, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self.blk, x, *args, use_reentrant=False, **kwargs)
        return self.blk(x, *args, **kwargs)


def wrap_blocks_ckpt(blocks: nn.ModuleList) -> nn.ModuleList:
    return nn.ModuleList([CkptBlock(b) for b in blocks])


def inject_earth_adapter_timm(vit, bottleneck: int = 32, rho: float = 0.25):
    """timm ViT: wrap every block's mlp (parallel-to-FFN placement)."""
    p = getattr(vit, "num_prefix_tokens", 1)
    for blk in vit.blocks:
        blk.mlp = FFNWithEA(blk.mlp, vit.embed_dim, num_prefix=p,
                            bottleneck=bottleneck, rho=rho)
    return len(vit.blocks)


def inject_earth_adapter_clay(transformer, dim: int, bottleneck: int = 32,
                              rho: float = 0.25):
    """Clay Transformer: layers are ModuleList([attn, ff]); wrap ff.
    Clay prepends 1 cls token."""
    for layer in transformer.layers:
        layer[1] = FFNWithEA(layer[1], dim, num_prefix=1,
                             bottleneck=bottleneck, rho=rho)
    return len(transformer.layers)
