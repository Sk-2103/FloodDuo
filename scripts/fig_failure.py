#!/usr/bin/env python3
"""Failure-mode figure: three recurring FloodDuo error modes on the lowest-
scoring held-out tiles, selected data-drivenly from the LORO prediction caches.
(a) omission of vegetated/turbid flood water (Ghana), (b) omission of fine-scale
urban water (Houston/HTX), (c) commission over wet/dark land (Nigeria)."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

V = "outputs/viz/manuscript_figs"

def fc(im):  # false color NIR-R-G (4-3-2), 2-98 stretch (matches galleries)
    x = im[[3, 2, 1]].astype(float); o = np.empty_like(x)
    for k in range(3):
        lo, hi = np.percentile(x[k], 2), np.percentile(x[k], 98)
        o[k] = np.clip((x[k] - lo) / (hi - lo + 1e-6), 0, 1)
    return np.transpose(o, (1, 2, 0))

def load(base):
    return np.load(f"{V}/{base}_cache.npz", allow_pickle=True)

def pick(base, mode):
    z = load(base); best = None
    for i in range(len(z["gts"])):
        g = (z["gts"][i] > 0.5); p = (z["pred_FloodDuo"][i] > 0.5)
        if g.mean() <= 0.015: continue
        u = (g | p).sum(); iou = (g & p).sum() / u if u else 1
        fp = ((~g & p).sum() / p.sum()) if p.sum() else 0
        key = fp if mode == "fp" else -iou        # max fp, or min iou
        if best is None or key > best[0]:
            best = (key, i, z["imgs"][i], z["gts"][i], z["pred_FloodDuo"][i], iou, fp)
    return best

rows = [("(a) Omission:\nvegetated / turbid\nwater (Ghana)",
         pick("floodplanet_Ghana", "iou")),
        ("(b) Omission:\nfine-scale urban\nwater (Houston)",
         pick("ufo_HTX", "iou")),
        ("(c) Commission:\nwet / dark land\n(Nigeria)",
         pick("floodplanet_Nigeria", "fp"))]

plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(3, 4, figsize=(13.5, 10.4))
cols = ["False-color (NIR-R-G)", "Ground truth", "FloodDuo", "Error"]
for r, (lab, sel) in enumerate(rows):
    _, i, im, gt, pr, iou, fp = sel
    g = (gt > 0.5); p = (pr > 0.5)
    err = np.full((*g.shape, 3), 0.93)   # TN light grey (matches Fig 5/6)
    err[g & p] = [0.18, 0.70, 0.24]   # TP green
    err[~g & p] = [0.86, 0.14, 0.14]  # FP red (commission)
    err[g & ~p] = [0.13, 0.34, 0.92]  # FN blue (omission)
    ax[r, 0].imshow(fc(im))
    ax[r, 1].imshow(g, cmap="Blues", vmin=0, vmax=1)
    ax[r, 2].imshow(p, cmap="Blues", vmin=0, vmax=1)
    ax[r, 3].imshow(err)
    ax[r, 0].set_ylabel(lab, fontsize=15, labelpad=10, rotation=90,
                        va="center", ha="center", fontweight="bold")
    ax[r, 3].text(0.03, 0.04, f"IoU {iou:.2f}", transform=ax[r, 3].transAxes,
                  fontsize=15, color="w", va="bottom", fontweight="bold",
                  bbox=dict(fc="k", alpha=0.55, pad=2.5))
    if r == 0:
        for c in range(4): ax[r, c].set_title(cols[c], fontsize=18, fontweight="bold")
    for c in range(4): ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
leg = [Patch(fc=[0.18, 0.70, 0.24], label="True positive"),
       Patch(fc=[0.13, 0.34, 0.92], label="False negative (missed water)"),
       Patch(fc=[0.86, 0.14, 0.14], label="False positive (over-prediction)")]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=15,
           frameon=False, bbox_to_anchor=(0.5, -0.015))
fig.tight_layout(rect=[0.02, 0.03, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(f"{V}/Fig_failure.{ext}", dpi=200, bbox_inches="tight")
    print("->", f"{V}/Fig_failure.{ext}")
for lab, sel in rows:
    print(f"{lab[:40]:42} tile_idx={sel[1]} IoU={sel[5]:.2f} FP={sel[6]:.2f}")
