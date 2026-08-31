#!/usr/bin/env python3
"""F2: Per-region precision-recall scatter (failure modes).
Each point = one held-out region. Baselines drift to low recall (missed water)
in unseen regions; FloodDuo stays near the P=R diagonal at higher F1."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "outputs/runs"
VIZ = "outputs/viz/manuscript_figs"
MODELS = [("FloodDuo", "loro_arch6/arch6", "#1b9e77", "o"),
          ("TerraMind", "loro_comparative/terramind", "#7570b3", "s"),
          ("U-Net", "loro_80ep/unet", "#d95f02", "^")]


def load(sub, ds):
    out = []
    for f in glob.glob(f"{RUNS}/{sub}/{ds}/*/result.json"):
        j = json.load(open(f)); out.append((j["precision"], j["recall"]))
    return out


plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9))
for ax, ds, ttl in [(axes[0], "floodplanet", "(a) FloodPlanet"),
                    (axes[1], "ufo", "(b) UFO")]:
    # iso-F1 curves
    P = np.linspace(0.3, 1, 200)
    for f1 in [0.6, 0.7, 0.8, 0.9]:
        R = f1 * P / (2 * P - f1)
        m = (R > 0) & (R <= 1.001)
        ax.plot(P[m], R[m], color="0.8", lw=0.8, zorder=1)
        # label near right
        xi = P[m][-1]; ax.text(xi, R[m][-1], f"F1={f1}", fontsize=7, color="0.6",
                               va="bottom", ha="right")
    ax.plot([0.3, 1], [0.3, 1], ls="--", color="0.5", lw=1, zorder=1)
    for name, sub, c, mk in MODELS:
        pts = load(sub, ds)
        if not pts: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, c=c, marker=mk, s=55, alpha=0.8, edgecolor="k",
                   lw=0.5, label=name, zorder=5)
    ax.set_xlim(0.45, 1.0); ax.set_ylim(0.35, 1.0)
    ax.set_xlabel("Precision (water)", fontsize=14)
    if ax is axes[0]:
        ax.set_ylabel("Recall (water)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.text(0.5, 0.04, ttl, transform=ax.transAxes, ha="center", fontsize=15,
            fontweight="bold", bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=3))
    ax.grid(ls=":", alpha=0.4); ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=11, title="per region")
fig.tight_layout()
for ext in ("png", "pdf"):
    p = f"{VIZ}/Fig_pr_scatter.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight"); print("->", p)
