"""Multi-depth fusion of the two encoder branches.

At each tapped depth, DINOv3 (C_a) and DOFA (C_b) feature maps are projected to
a common width and fused by concat + 1x1 conv, with a learnable per-branch gate
so the model can down-weight a branch per depth.
"""

import torch
from torch import nn
import torch.nn.functional as F


class DepthFusion(nn.Module):
    def __init__(self, dim_a: int, dim_b: int, dim: int = 256):
        super().__init__()
        self.proj_a = nn.Sequential(nn.Conv2d(dim_a, dim, 1), nn.GroupNorm(8, dim))
        self.proj_b = nn.Sequential(nn.Conv2d(dim_b, dim, 1), nn.GroupNorm(8, dim))
        self.gate_a = nn.Parameter(torch.ones(1))
        self.gate_b = nn.Parameter(torch.ones(1))
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
        )

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        if fa.shape[-2:] != fb.shape[-2:]:
            # branches on different grids: upsample the coarser to the finer
            target = (max(fa.shape[-2], fb.shape[-2]),
                      max(fa.shape[-1], fb.shape[-1]))
            if fa.shape[-2:] != target:
                fa = F.interpolate(fa, size=target, mode="bilinear",
                                   align_corners=False)
            if fb.shape[-2:] != target:
                fb = F.interpolate(fb, size=target, mode="bilinear",
                                   align_corners=False)
        a = self.proj_a(fa) * self.gate_a
        b = self.proj_b(fb) * self.gate_b
        return self.fuse(torch.cat([a, b], dim=1))


class MultiDepthFusion(nn.Module):
    def __init__(self, dim_a: int, dim_b: int, n_depths: int = 4, dim: int = 256):
        super().__init__()
        self.levels = nn.ModuleList(
            [DepthFusion(dim_a, dim_b, dim) for _ in range(n_depths)])

    def forward(self, feats_a: list, feats_b: list) -> list:
        return [lvl(fa, fb) for lvl, fa, fb in zip(self.levels, feats_a, feats_b)]


# --------------------------------------------------------------------------
# Disagreement-aware fusion (config: fusion_opts.gated_fusion)
# --------------------------------------------------------------------------

class GatedDepthFusion(nn.Module):
    """D-conditioned, spatially-varying fusion of the two branches.

    Same projections as DepthFusion, but the per-branch combination weights are
    predicted per pixel from concat([proj_a, proj_b, D]) via a small conv stack
    + softmax over the 2 branches. D tells the gate THAT the branches disagree,
    not WHO is right — the policy is learned. An optional small constant bias on
    the spectral (branch-b) gate logit warm-starts trust toward it where D is
    high, but it stays learnable.
    """

    def __init__(self, dim_a: int, dim_b: int, dim: int = 256,
                 spectral_bias: float = 0.5):
        super().__init__()
        self.proj_a = nn.Sequential(nn.Conv2d(dim_a, dim, 1), nn.GroupNorm(8, dim))
        self.proj_b = nn.Sequential(nn.Conv2d(dim_b, dim, 1), nn.GroupNorm(8, dim))
        # NO GroupNorm in the gate: it sees the near-constant D channel early in
        # training -> low-variance group -> GN backward (∝1/std) explodes -> NaN.
        # Final conv zero-init so the gate starts as a clean softmax prior
        # (= the spectral_bias warm-start), then learns the spatial policy.
        self.gate = nn.Sequential(
            nn.Conv2d(2 * dim + 1, dim, 1), nn.GELU(),
            nn.Conv2d(dim, 2, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        # warm-start: bias branch-b (spectral) gate logit upward; learnable.
        self.register_buffer("gate_bias",
                             torch.tensor([0.0, float(spectral_bias)]).view(1, 2, 1, 1))
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
        )

    def forward(self, fa, fb, D):
        target = (max(fa.shape[-2], fb.shape[-2]),
                  max(fa.shape[-1], fb.shape[-1]))
        if fa.shape[-2:] != target:
            fa = F.interpolate(fa, size=target, mode="bilinear", align_corners=False)
        if fb.shape[-2:] != target:
            fb = F.interpolate(fb, size=target, mode="bilinear", align_corners=False)
        if D.shape[-2:] != target:
            D = F.interpolate(D, size=target, mode="bilinear", align_corners=False)
        # GroupNorm + gate softmax + weighted sum are run in fp32: in bf16 the
        # gate path overflows to NaN on a large fraction of batches (GN/softmax
        # dynamic range). The rest of the network stays bf16.
        with torch.autocast(device_type="cuda", enabled=False):
            fa, fb, D = fa.float(), fb.float(), D.float()
            a = self.proj_a(fa)
            b = self.proj_b(fb)
            g = self.gate(torch.cat([a, b, D], dim=1)) + self.gate_bias
            w = torch.softmax(g, dim=1)            # (B,2,h,w)
            fused = w[:, :1] * a + w[:, 1:] * b
            return self.fuse(fused)


class GatedMultiDepthFusion(nn.Module):
    def __init__(self, dim_a, dim_b, n_depths=4, dim=256, spectral_bias=0.5):
        super().__init__()
        self.levels = nn.ModuleList(
            [GatedDepthFusion(dim_a, dim_b, dim, spectral_bias)
             for _ in range(n_depths)])

    def forward(self, feats_a, feats_b, D):
        return [lvl(fa, fb, D) for lvl, fa, fb in zip(self.levels, feats_a, feats_b)]


# --------------------------------------------------------------------------
# Staged cross-branch co-attention (config: fusion_opts.cross_attn) — risky,
# default OFF. Attention is computed at the coarse 32^2 grid only (cheap); the
# native fine grid of branch b is preserved via residual. Zero-init output
# scale -> exact no-op at initialisation.
# --------------------------------------------------------------------------

class _CrossAttn(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.h = heads
        self.nq = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.zeros(1))   # zero-init -> no-op

    def forward(self, q_tok, kv_tok):
        B, Nq, Dm = q_tok.shape
        qn, kvn = self.nq(q_tok), self.nkv(kv_tok)
        q = self.q(qn).view(B, Nq, self.h, Dm // self.h).transpose(1, 2)
        k = self.k(kvn).view(B, kv_tok.shape[1], self.h, Dm // self.h).transpose(1, 2)
        v = self.v(kvn).view(B, kv_tok.shape[1], self.h, Dm // self.h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, Dm)
        return q_tok + self.scale * self.o(out)


class CrossDepthAttn(nn.Module):
    """Per-depth co-attention: a attends to (pooled) b and vice versa.

    a is the coarse branch (32^2); b is the fine branch (e.g. 64^2). b is pooled
    to a's grid for the attention computation only; the result is upsampled back
    and added residually so b keeps its native detail.
    """

    def __init__(self, dim_a, dim_b, heads=4):
        super().__init__()
        assert dim_a == dim_b, "cross-attn assumes matched token dim"
        self.a2b = _CrossAttn(dim_a, heads)
        self.b2a = _CrossAttn(dim_a, heads)

    def forward(self, fa, fb):
        ga = fa.shape[-2:]
        fb_pool = F.adaptive_avg_pool2d(fb, ga)
        B, C, H, W = fa.shape
        a_tok = fa.flatten(2).transpose(1, 2)
        b_tok = fb_pool.flatten(2).transpose(1, 2)
        a_new = self.a2b(a_tok, b_tok)                       # a attends to b
        b_new = self.b2a(b_tok, a_tok)                       # b attends to a
        fa2 = a_new.transpose(1, 2).reshape(B, C, H, W)
        b_upd = (b_new - b_tok).transpose(1, 2).reshape(B, C, *ga)
        fb2 = fb + F.interpolate(b_upd, size=fb.shape[-2:],
                                 mode="bilinear", align_corners=False)
        return fa2, fb2


class CrossAttnStack(nn.Module):
    def __init__(self, dim_a, dim_b, n_depths=4, heads=4):
        super().__init__()
        self.levels = nn.ModuleList(
            [CrossDepthAttn(dim_a, dim_b, heads) for _ in range(n_depths)])

    def forward(self, feats_a, feats_b):
        outs = [lvl(fa, fb) for lvl, fa, fb in zip(self.levels, feats_a, feats_b)]
        return [o[0] for o in outs], [o[1] for o in outs]
