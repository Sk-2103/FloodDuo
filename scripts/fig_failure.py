#!/usr/bin/env python3
"""Figure 16: failure tiles with the disagreement map D. Reads failure_Dmaps.npz."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch, torch.nn.functional as F

from _paths import FIGS_ROOT as V
DM = np.load(f"{V}/failure_Dmaps.npz")

def fc(im):
    x = im[[3, 2, 1]].astype(float); o = np.empty_like(x)
    for k in range(3):
        lo, hi = np.percentile(x[k], 2), np.percentile(x[k], 98)
        o[k] = np.clip((x[k] - lo) / (hi - lo + 1e-6), 0, 1)
    return np.transpose(o, (1, 2, 0))

def pick(base, mode):
    z = np.load(f"{V}/{base}_cache.npz", allow_pickle=True); best = None
    for i in range(len(z["gts"])):
        g = z["gts"][i] > 0.5; p = z["pred_FloodDuo"][i] > 0.5
        if g.mean() <= 0.015: continue
        u = (g | p).sum(); iou = (g & p).sum() / u if u else 1
        fp = ((~g & p).sum() / p.sum()) if p.sum() else 0
        key = fp if mode == "fp" else -iou
        if best is None or key > best[0]: best = (key, i, z["imgs"][i], z["gts"][i], z["pred_FloodDuo"][i], iou)
    return best

rows = [("(a) Omission:\nvegetated / turbid\nwater (Ghana)", "floodplanet_Ghana", "iou"),
        ("(b) Omission:\nfine-scale urban\nwater (Houston)", "ufo_HTX", "iou"),
        ("(c) Commission:\nwet / dark land\n(Nigeria)", "floodplanet_Nigeria", "fp")]

def Dfull(base):
    d = DM[f"{base}_D"]
    return F.interpolate(torch.from_numpy(d)[None, None].float(), size=(1024, 1024), mode="nearest")[0, 0].numpy()

VMAX = 0.12   # shared D scale (ln2 = 0.693 max); failures sit low, so clip for visibility
# This figure's canvas (16.8in) is much larger than the manuscript's other multi-panel
# figures (e.g. boundary figure at 13.2in), but it's placed at the same final printed
# width, so its text was shrinking by more on the page and reading smaller than the
# rest of the manuscript even at a nominally similar point size. Sizes below are bumped
# accordingly (labels ~+1-3pt, legend ~+4pt plus larger handles) to match print scale.
plt.rcParams.update({"font.size": 16})
fig, ax = plt.subplots(3, 5, figsize=(16.8, 10.4))
cols = ["False-color (NIR-R-G)", "Ground truth", "FloodDuo", "Error", "Disagreement D"]
im_d = None
for r, (lab, base, mode) in enumerate(rows):
    _, i, im, gt, pr, iou = pick(base, mode)
    g = gt > 0.5; p = pr > 0.5
    err = np.full((*g.shape, 3), 0.93)
    err[g & p] = [0.18, 0.70, 0.24]; err[~g & p] = [0.86, 0.14, 0.14]; err[g & ~p] = [0.13, 0.34, 0.92]
    D = Dfull(base); mD = float(DM[f"{base}_D"].mean())
    ax[r, 0].imshow(fc(im)); ax[r, 1].imshow(g, cmap="Blues", vmin=0, vmax=1)
    ax[r, 2].imshow(p, cmap="Blues", vmin=0, vmax=1); ax[r, 3].imshow(err)
    im_d = ax[r, 4].imshow(D, cmap="magma", vmin=0, vmax=VMAX)
    ax[r, 0].set_ylabel(lab, fontsize=17, labelpad=10, rotation=90, va="center", ha="center", fontweight="bold")
    ax[r, 3].text(0.03, 0.04, f"IoU {iou:.2f}", transform=ax[r, 3].transAxes, fontsize=16, color="w",
                  va="bottom", fontweight="bold", bbox=dict(fc="k", alpha=0.55, pad=2.5))
    ax[r, 4].text(0.03, 0.04, f"mean D {mD:.3f}", transform=ax[r, 4].transAxes, fontsize=16, color="w",
                  va="bottom", fontweight="bold", bbox=dict(fc="k", alpha=0.55, pad=2.5))
    if r == 0:
        for c in range(5): ax[r, c].set_title(cols[c], fontsize=18, fontweight="bold")
    for c in range(5): ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
leg = [Patch(fc=[0.18, 0.70, 0.24], label="True positive"),
       Patch(fc=[0.13, 0.34, 0.92], label="False negative (missed water)"),
       Patch(fc=[0.86, 0.14, 0.14], label="False positive (over-prediction)")]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=19, frameon=False, bbox_to_anchor=(0.5, -0.02),
           handlelength=3.0, handleheight=1.4, handletextpad=0.8, columnspacing=2.8)
# tight_layout MUST run before the colorbar is carved out of ax[:, 4]: calling it
# after (as before) reflows the 3x5 grid without repositioning the colorbar axes,
# leaving it frozen over row 2 instead of beside column 5.
fig.tight_layout(rect=[0.02, 0.03, 1, 1])
cb = fig.colorbar(im_d, ax=ax[:, 4], fraction=0.046, pad=0.02); cb.set_label("D (JSD, nats)", fontsize=16)
cb.ax.tick_params(labelsize=14)
for ext in ("png", "pdf"):
    fig.savefig(f"{V}/Fig_failure.{ext}", dpi=200, bbox_inches="tight"); print("->", ext)
