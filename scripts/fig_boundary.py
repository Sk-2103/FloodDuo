#!/usr/bin/env python3
"""F3: Error rate vs distance-to-water-boundary (quantifies fine-detail gain).
Reads diag_stats.json. Lower = better; FloodDuo should win most near boundaries."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZ = "outputs/viz/manuscript_figs"
s = json.load(open(f"{VIZ}/diag_stats.json"))
edges = np.array(s["dist_edges"]) * 3.0   # px -> meters (3 m GSD)
centers = 0.5 * (edges[:-1] + edges[1:])
COL = {"FloodDuo": "#1b9e77", "TerraMind": "#7570b3", "U-Net": "#d95f02"}

plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
for ax, ds, ttl in [(axes[0], "floodplanet", "(a) FloodPlanet"), (axes[1], "ufo", "(b) UFO")]:
    for m in ["FloodDuo", "TerraMind", "U-Net"]:
        h = np.array(s["boundary"][ds][m])
        rate = h[:, 0] / np.clip(h[:, 1], 1, None)
        ax.plot(centers, rate, "-o", color=COL[m], lw=2.2, ms=6, label=m)
    ax.set_xlabel("Distance to water boundary (m)", fontsize=14)
    ax.tick_params(labelsize=12); ax.grid(ls=":", alpha=0.4); ax.set_axisbelow(True)
    ax.set_xscale("log"); ax.legend(fontsize=12)
    ax.text(0.5, 0.05, ttl, transform=ax.transAxes, ha="center", fontsize=15,
            fontweight="bold", bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=3))
axes[0].set_ylabel("Pixel error rate", fontsize=14)
fig.tight_layout()
for ext in ("png", "pdf"):
    p = f"{VIZ}/Fig_boundary.{ext}"; fig.savefig(p, dpi=200, bbox_inches="tight"); print("->", p)
