"""Binary water-segmentation metrics accumulated as confusion counts."""

import torch


class SegMetrics:
    def __init__(self, ignore_index=None):
        self.tp = self.fp = self.fn = self.tn = 0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        pred = (logits >= 0).squeeze(1)
        t = target.float()
        if t.dim() == 4:
            t = t.squeeze(1)
        if self.ignore_index is not None:
            keep = (t != self.ignore_index)
            pred = pred[keep]
            t = t[keep]
        t = t.bool()
        self.tp += int((pred & t).sum())
        self.fp += int((pred & ~t).sum())
        self.fn += int((~pred & t).sum())
        self.tn += int((~pred & ~t).sum())

    def compute(self) -> dict:
        eps = 1e-9
        prec = self.tp / (self.tp + self.fp + eps)
        rec = self.tp / (self.tp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        return {"iou": iou, "f1": f1, "precision": prec, "recall": rec, "acc": acc}
