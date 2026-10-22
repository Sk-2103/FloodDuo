#!/usr/bin/env python3
"""Figure 13: disagreement map D as a reliability signal.
Reads disagreement_stats.json + disagreement_cells.npz."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from _paths import FIGS_ROOT as V
S = json.load(open(f"{V}/disagreement_stats.json"))
Z = np.load(f"{V}/disagreement_cells.npz", allow_pickle=True)
DS = [("floodplanet", "FloodPlanet"), ("ufo", "UFO")]
COL = {"floodplanet": "#0072B2", "ufo": "#D55E00"}
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif", "axes.titlesize": 12,
                     "axes.titleweight": "bold", "axes.linewidth": 0.8})
fig, AX = plt.subplots(2, 3, figsize=(12.5, 7.6))
for r, (ds, nice) in enumerate(DS):
    c = COL[ds]; o = S[ds]
    # (col 0) error vs D decile
    a = AX[r, 0]; cur = o["curve"]; x = np.arange(1, len(cur) + 1)
    y = [u["err"] for u in cur]; lo = [u["err"] - u["ci"][0] for u in cur]; hi = [u["ci"][1] - u["err"] for u in cur]
    a.errorbar(x, y, yerr=[lo, hi], marker="o", color=c, lw=2, capsize=3)
    a.set_xlabel("Disagreement D decile (low → high)"); a.set_ylabel("Cell error rate")
    a.set_title(f"({chr(97+3*r)}) {nice}: error vs D", loc="left")
    a.text(0.04, 0.92, f"Spearman $\\rho$ = {o['spearman']:.2f}\n[{o['spearman_ci'][0]:.2f}, {o['spearman_ci'][1]:.2f}]",
           transform=a.transAxes, va="top", fontsize=10)
    # (col 1) region mean-D vs IoU
    a = AX[r, 1]; md = Z[f"{ds}_rmeanD"]; io = Z[f"{ds}_rIoU"]
    a.scatter(md, io, color=c, s=36, edgecolor="k", lw=0.5)
    b1, b0 = np.polyfit(md, io, 1); xs = np.linspace(md.min(), md.max(), 20)
    a.plot(xs, b1 * xs + b0, color=c, lw=1.3, ls="--")
    a.set_xlabel("Region mean D"); a.set_ylabel("Region held-out IoU")
    a.set_title(f"({chr(98+3*r)}) {nice}: region D vs IoU", loc="left")
    a.text(0.96, 0.92, f"r = {o['region_r']:.2f}\n[{o['region_r_ci'][0]:.2f}, {o['region_r_ci'][1]:.2f}]",
           transform=a.transAxes, va="top", ha="right", fontsize=10)
    # (col 2) reliability of fitted g(D)
    a = AX[r, 2]; eb = o["ece_bins"]; conf = [u[0] for u in eb]; acc = [u[1] for u in eb]
    mx = max(max(conf), max(acc)) * 1.05
    a.plot([0, mx], [0, mx], color="0.6", ls="--", lw=1)
    a.plot(conf, acc, marker="s", color=c, lw=2)
    a.set_xlabel("Fitted P(error) from D"); a.set_ylabel("Observed error rate")
    a.set_title(f"({chr(99+3*r)}) {nice}: fitted-mapping calibration", loc="left")
    a.text(0.04, 0.92, f"ECE = {o['ece_fitted']:.3f}\n(LORO-fit, 10 eq-count bins)",
           transform=a.transAxes, va="top", fontsize=10)
for a in AX.ravel():
    a.grid(lw=0.4, alpha=0.3); a.set_axisbelow(True)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{V}/Fig_disagreement.{ext}", dpi=240, bbox_inches="tight"); print("->", ext)
