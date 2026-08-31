#!/usr/bin/env python3
"""Fig: learning-rate sensitivity (validation IoU, 30-epoch design sweep).
Hard-coded sweep values; single panel, two twin y-axes so each dataset's
near-flat trend is visible on its own scale. Vertical dashed line at the
selected lr=5e-4."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/viz/manuscript_figs"

FP_LR = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
FP_IOU = [0.671, 0.670, 0.670, 0.678, 0.673]
UFO_LR = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
UFO_IOU = [0.872, 0.873, 0.874, 0.873, 0.872]

FP_C = "#1b9e77"
UFO_C = "#7570b3"
SEL = 5e-4

plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(8.5, 6.0))
ax2 = ax.twinx()

l1 = ax.plot(FP_LR, FP_IOU, marker="o", ms=8, lw=2.2, color=FP_C,
             label="FloodPlanet")[0]
l2 = ax2.plot(UFO_LR, UFO_IOU, marker="s", ms=8, lw=2.2, color=UFO_C,
              label="UFO")[0]

# annotate each point so the flat trend is legible on both scales
for x, y in zip(FP_LR, FP_IOU):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=11, color=FP_C)
for x, y in zip(UFO_LR, UFO_IOU):
    ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, -16), ha="center", fontsize=11, color=UFO_C)

ax.axvline(SEL, ls="--", color="0.35", lw=1.6, zorder=0)
ax.annotate("selected", (SEL, 1.0), xycoords=("data", "axes fraction"),
            xytext=(6, -14), textcoords="offset points", fontsize=12.5,
            color="0.25", fontweight="bold", ha="left")

ax.set_xscale("log")
ax.set_xticks(FP_LR)
ax.set_xticklabels(["1e-4", "2e-4", "3e-4", "5e-4", "1e-3"], fontsize=12.5)
ax.set_xlabel("Learning rate (log scale)", fontsize=16)
ax.set_ylabel("Validation IoU — FloodPlanet", fontsize=15, color=FP_C)
ax2.set_ylabel("Validation IoU — UFO", fontsize=15, color=UFO_C)
ax.tick_params(axis="y", labelcolor=FP_C, labelsize=12.5)
ax2.tick_params(axis="y", labelcolor=UFO_C, labelsize=12.5)

# pad each scale a little so annotations breathe and flatness is visible
ax.set_ylim(min(FP_IOU) - 0.010, max(FP_IOU) + 0.010)
ax2.set_ylim(min(UFO_IOU) - 0.008, max(UFO_IOU) + 0.008)

ax.set_title("Learning-rate sensitivity (validation)", fontsize=16,
             fontweight="bold")
ax.grid(ls=":", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(handles=[l1, l2], loc="center right", fontsize=13.5,
          framealpha=0.9)

fig.tight_layout()
for ext in ("png", "pdf"):
    p = f"{OUT}/Fig_lr_sensitivity.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("->", p)
