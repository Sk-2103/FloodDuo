#!/usr/bin/env python3
"""Boxplot of per-region (LORO) IoU across models. (a) FloodPlanet, (b) UFO.
Each box = distribution of held-out-region IoU for one model; shows how IoU
varies across sites and lets readers compare medians + spread."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "outputs/runs"
OUT = "outputs/viz/manuscript_figs"
# (label, loro subdir) — ordered by FP mean; FloodDuo first/highlighted
M = [("FloodDuo", "loro_arch6/arch6"),
     ("TerraMind", "loro_comparative/terramind"),
     ("DOFA", "loro_comparative/dofa_noada"),
     ("Prithvi", "loro_comparative/prithvi"),
     ("Clay", "loro_comparative/clay_noada"),
     ("CROMA", "loro_comparative/croma"),
     ("SSL4EO-DINO", "loro_comparative/ssl4eo_dino"),
     ("U-Net", "loro_80ep/unet")]


def ious(sub, ds):
    v = []
    for f in glob.glob(f"{RUNS}/{sub}/{ds}/*/result.json"):
        v.append(json.load(open(f))["iou"])
    return v


fig, axes = plt.subplots(2, 1, figsize=(8.5, 9.5), sharex=True)
for ax, ds, ttl in [(axes[0], "floodplanet", "(a) FloodPlanet (19 held-out regions)"),
                    (axes[1], "ufo", "(b) UFO (14 held-out regions)")]:
    data = [ious(sub, ds) for _, sub in M]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                    medianprops=dict(color="black", lw=1.6),
                    meanprops=dict(marker="D", mfc="white", mec="black", ms=6),
                    flierprops=dict(marker="o", ms=4, mfc="0.5", mec="none", alpha=.6))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=("#1b9e77" if i == 0 else "#b0b0b0"),
                alpha=(0.95 if i == 0 else 0.7), edgecolor="black")
    # jittered points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.06, len(d))
        ax.scatter(x, d, s=10, color="black", alpha=0.35, zorder=3)
    ax.set_xticks(range(1, len(M) + 1))
    ax.set_xticklabels([m[0] for m in M], rotation=20, ha="right", fontsize=10)
    ax.set_title(ttl, fontsize=12)
    ax.set_ylabel("Held-out region IoU (water)", fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)
fig.suptitle("Cross-region IoU distribution across models (leave-one-region-out)",
             fontsize=13, y=1.00)
fig.tight_layout()
p = f"{OUT}/Fig_boxplot_LORO_IoU.png"
fig.savefig(p, dpi=200, bbox_inches="tight"); print("->", p)
