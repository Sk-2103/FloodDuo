#!/usr/bin/env python3
"""Figure 9: boundary-resolved error analysis. Reads boundary_stats.json."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import FIGS_ROOT as V
S = json.load(open(f"{V}/boundary_stats.json"))
EDGES = np.array(S["_edges"]); TAUS = S["_taus"]
CEN = (EDGES[:-1] + EDGES[1:]) / 2.0
DS = [("floodplanet", "FloodPlanet"), ("ufo", "UFO")]
MODELS = ["FloodDuo", "U-Net", "TerraMind"]
MCOL = {"FloodDuo": "#0072B2", "U-Net": "#000000", "TerraMind": "#D55E00"}
MLS  = {"FloodDuo": "-", "U-Net": "--", "TerraMind": ":"}
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif", "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "axes.linewidth": 0.8})
# axes.titlesize trimmed from 12 to 10.5: at loc="left", the long "(x) FloodPlanet: ..."
# titles are wider than their own axis and bleed into the neighboring panel; this small
# reduction (plus the -0.02 x-nudge on column b below) clears both neighbors with margin
# to spare, without touching wspace/tight_layout (which fought each other on this figure).
fig, AX = plt.subplots(2, 3, figsize=(13.2, 7.6))
for r, (ds, nice) in enumerate(DS):
    # (col 0) signed-distance error: FN inside (x<0), FP outside (x>0)
    a = AX[r, 0]
    a.axvspan(-3, 3, color="0.85", alpha=0.6, lw=0, zorder=0)
    a.axvline(0, color="0.4", lw=0.8, zorder=1)
    for m in MODELS:
        A = S[m][ds]
        fn = np.array(A["fn_rate"]); fp = np.array(A["fp_rate"])
        err = np.where(CEN < 0, fn, fp)
        a.plot(CEN, err, MLS[m], color=MCOL[m], lw=2, label=m, zorder=3)
    a.set_xlabel("Signed distance to water boundary (m)\n(negative = inside water)", fontsize=12.5)
    a.set_ylabel("Pixel error rate", fontsize=12.5)
    a.set_title(f"({chr(97+3*r)}) {nice}: omission vs commission", loc="left")
    a.set_xlim(-30, 30)
    # (col 1) Boundary-F1 vs tolerance
    a = AX[r, 1]
    for m in MODELS:
        bf = S[m][ds]["bf"]; y = [bf[str(t)][0] for t in TAUS]
        lo = [bf[str(t)][0] - bf[str(t)][1] for t in TAUS]
        hi = [bf[str(t)][2] - bf[str(t)][0] for t in TAUS]
        a.errorbar(TAUS, y, yerr=[lo, hi], marker="o", ls=MLS[m], color=MCOL[m], lw=2,
                   capsize=3, label=m)
    a.set_xlabel("Boundary tolerance $\\tau$ (m)", fontsize=12.5); a.set_ylabel("Boundary-F1", fontsize=12.5)
    ttl = a.set_title(f"({chr(98+3*r)}) {nice}: boundary-F1 vs tolerance", loc="left")
    if r == 0:  # "FloodPlanet" is long enough to run into panel c's y-label; "UFO" (row 1) is not
        ttl.set_x(-0.02)
    a.set_xticks(TAUS)
    # (col 2) narrow-feature recall
    a = AX[r, 2]
    xs = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        v, lo, hi = S[m][ds]["narrow_recall"]
        a.bar(i, v, color=MCOL[m], alpha=0.85, width=0.62,
              yerr=[[v - lo], [hi - v]], capsize=4, ecolor="0.3")
        a.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=11)
    a.set_xticks(xs); a.set_xticklabels(MODELS, fontsize=9)
    a.set_ylabel("Recall, narrow features ($\\leq$2 px)", fontsize=12.5)
    a.set_title(f"({chr(99+3*r)}) {nice}: narrow-feature recall", loc="left")
    a.set_ylim(0, max(S[m][ds]["narrow_recall"][2] for m in MODELS) * 1.18)
for a in AX.ravel():
    a.grid(lw=0.4, alpha=0.3); a.set_axisbelow(True)
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=MCOL[m], ls=MLS[m], lw=3, marker="o", ms=9) for m in MODELS]
fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.legend(handles, MODELS, loc="lower center", ncol=len(MODELS), fontsize=14,
           frameon=False, handlelength=2.6, columnspacing=2.6, bbox_to_anchor=(0.5, 0.005))
for ext in ("png", "pdf"):
    fig.savefig(f"{V}/Fig_boundary.{ext}", dpi=240, bbox_inches="tight"); print("->", ext)
