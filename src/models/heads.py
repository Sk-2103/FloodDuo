"""Per-branch auxiliary segmentation heads + disagreement diagnostics.

Used by the disagreement-aware fusion ladder (see CLAUDE.md "Disagreement-aware
fusion"). Each frozen encoder branch gets its own tiny aux head that predicts
water *independently* (deep supervision). The divergence between the two
branches' predicted water probabilities — measured in PREDICTION space, not
feature space — is a label-free uncertainty / OOD signal (D).

Hard constraints honoured here:
- Disagreement is computed from the PRE-FUSION aux predictions only.
- No agreement/consistency coupling between heads (each fits GT on its own).
"""

import torch
from torch import nn
import torch.nn.functional as F


class BranchAuxHead(nn.Module):
    """Lightweight deep-supervision head on a branch's deepest adapted tap.

    1x1 conv -> GN -> GELU -> 1x1 conv -> 1 logit. Returns the logit at the
    tap's native grid (caller upsamples to full res for the loss, and resizes
    to the common coarse grid for the disagreement map).
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def _bern_kl(p, q, eps=1e-4):
    """Bernoulli KL(p || q), elementwise.

    eps=1e-4 (not 1e-6): D is computed in fp32, but the clamp must survive even
    if a caller is in bf16 — 1-1e-6 rounds to exactly 1.0 in bf16, making
    log((1-p)/(1-q)) blow up to inf/NaN. 1-1e-4 is representable.
    """
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


def bernoulli_jsd(p_a, p_b, eps=1e-6):
    """Jensen-Shannon divergence between two Bernoulli prob maps, in [0, ln2]."""
    m = 0.5 * (p_a + p_b)
    return 0.5 * _bern_kl(p_a, m, eps) + 0.5 * _bern_kl(p_b, m, eps)


def bernoulli_entropy(p, eps=1e-6):
    """Binary entropy of a prob map, in [0, ln2]."""
    p = p.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def disagreement_map(logits_a, logits_b, grid):
    """Compute D (JSD) and H (entropy of the mean) at a common coarse grid.

    logits_a, logits_b: (B,1,*,*) pre-fusion aux logits at their native grids.
    grid: (h, w) common low-res grid (e.g. 32x32) — D is low-frequency, cheap.
    Returns D, H each (B,1,h,w), both bounded [0, ln2].

    Computed in fp32 (autocast disabled): the JSD's log/division underflow
    catastrophically in bf16 once the aux predictions saturate near 0/1.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        pa = torch.sigmoid(_resize(logits_a.float(), grid))
        pb = torch.sigmoid(_resize(logits_b.float(), grid))
        D = bernoulli_jsd(pa, pb)
        H = bernoulli_entropy(0.5 * (pa + pb))
    return D, H


def _resize(x, grid):
    if x.shape[-2:] == tuple(grid):
        return x
    return F.interpolate(x, size=grid, mode="bilinear", align_corners=False)
