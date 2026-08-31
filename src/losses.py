"""Segmentation loss: BCE + Dice + boundary-weighted BCE.

Water errors at 3 m concentrate on shoreline pixels; the boundary term
up-weights a band of pixels around GT mask edges (morphological gradient via
max-pooling).
"""

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(logits, target, valid=None, eps=1.0):
    p = torch.sigmoid(logits)
    if valid is not None:
        p = p * valid
        target = target * valid
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def boundary_band(target, width=3):
    """1.0 on pixels within `width` of a GT boundary, else 0."""
    k = 2 * width + 1
    dil = F.max_pool2d(target, k, stride=1, padding=width)
    ero = -F.max_pool2d(-target, k, stride=1, padding=width)
    return (dil - ero).clamp(0, 1)


def aux_seg_loss(logits, target):
    """Deep-supervision loss for a branch aux head: BCE + Dice vs GT.

    No boundary term and — importantly — no coupling between the two branch
    heads: each fits GT independently so their disagreement stays an emergent
    OOD signal (not regularised away).
    """
    target = target.float()
    if target.dim() == 3:
        target = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dl = dice_loss(logits, target)
    return bce + dl


class FloodLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, w_boundary=1.0, band_width=3,
                 ignore_index=None):
        super().__init__()
        self.w_bce, self.w_dice, self.w_boundary = w_bce, w_dice, w_boundary
        self.band_width = band_width
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        target = target.float()
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if self.ignore_index is not None:
            valid = (target != self.ignore_index).float()
            target_clean = target * valid   # zero out ignore pixels
        else:
            valid = None
            target_clean = target

        bce_pix = F.binary_cross_entropy_with_logits(
            logits, target_clean, reduction="none")
        if valid is not None:
            bce = (bce_pix * valid).sum() / (valid.sum() + 1e-6)
        else:
            bce = bce_pix.mean()

        dl = dice_loss(logits, target_clean, valid=valid)

        band = boundary_band(target_clean, self.band_width)
        if valid is not None:
            band = band * valid
        b_bce = (bce_pix * band).sum() / (band.sum() + 1.0)

        loss = self.w_bce * bce + self.w_dice * dl + self.w_boundary * b_bce
        return loss, {"bce": bce.item(), "dice": dl.item(), "boundary": b_bce.item()}
