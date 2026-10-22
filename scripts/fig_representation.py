#!/usr/bin/env python3
"""Figure 11: encoder representation analysis.
Reads representation_stats.json + representation_rerun.json."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _paths import FIGS_ROOT as V
S = json.load(open(f"{V}/representation_stats.json"))
R = json.load(open(f"{V}/representation_rerun.json"))
DIN, CLA, NEU = "#D55E00", "#009E73", "#4C72B0"
bands = ["B", "G", "R", "NIR", "ndwi"]; labels = ["B", "G", "R", "NIR", "NDWI"]; xb = np.arange(5)
plt.rcParams.update({"font.size": 10})

cka = R["cka_by_depth"]
probe = R["probe_R2_grouped"]
hf_dino, hf_clay = R["hf_common"]["dino"], R["hf_common"]["clay"]
sd_ = np.array(S["spectrum"]["dino"]); sc_ = np.array(S["spectrum"]["clay"])

fig, AX = plt.subplots(2, 2, figsize=(7.2, 5.6)); (a0, a1), (a2, a3) = AX
# (a) CKA
a0.bar(np.arange(1, 5), cka, color=NEU, width=0.62, edgecolor="none")
a0.set_ylim(0, 1); a0.set_yticks(np.arange(0, 1.01, 0.25)); a0.set_xticks(np.arange(1, 5))
a0.set_xlabel("Encoder block depth"); a0.set_ylabel("Linear CKA"); a0.set_title("(a)", loc="left")
# (b) region-separated linear-probe decodability
_dv = [max(0.0, probe[b]["dino"]) for b in bands]; _cv = [probe[b]["clay"] for b in bands]
a1.axhline(0, color="0.75", lw=0.8, zorder=1)
a1.vlines(xb - 0.12, 0, _dv, color=DIN, lw=1.6, zorder=2); a1.vlines(xb + 0.12, 0, _cv, color=CLA, lw=1.6, zorder=2)
a1.scatter(xb - 0.12, _dv, color=DIN, s=44, marker="o", edgecolor="black", lw=0.5, zorder=3)
a1.scatter(xb + 0.12, _cv, color=CLA, s=44, marker="D", edgecolor="black", lw=0.5, zorder=3)
a1.set_xticks(xb); a1.set_xticklabels(labels); a1.set_xlim(-0.5, 4.5)
a1.set_ylim(-0.06, 1.0); a1.set_yticks(np.arange(0, 1.01, 0.25))
a1.set_xlabel("Spectral target"); a1.set_ylabel("Linear-probe $R^2$ (region-held-out)"); a1.set_title("(b)", loc="left")
# (c) radial power spectrum (native)
a2.plot(np.arange(len(sd_)) / len(sd_), sd_, color=DIN, lw=1.7)
a2.plot(np.arange(len(sc_)) / len(sc_), sc_, color=CLA, lw=1.7)
a2.set_yscale("log"); a2.set_xlim(0, 1); a2.set_xlabel("Normalized spatial frequency")
a2.set_ylabel("Normalized power"); a2.set_title("(c)", loc="left")
# (d) common-scale HF energy fraction vs spectral probe R2
spec_d = float(np.mean([max(0.0, probe[b]["dino"]) for b in bands])); spec_c = float(np.mean([probe[b]["clay"] for b in bands]))
a3.scatter([spec_d], [hf_dino], s=130, color=DIN, edgecolor="black", lw=0.7, marker="o", zorder=3)
a3.scatter([spec_c], [hf_clay], s=130, color=CLA, edgecolor="black", lw=0.7, marker="D", zorder=3)
a3.set_xlim(-0.08, 1.0); a3.set_ylim(0, 0.5); a3.set_yticks(np.arange(0, 0.51, 0.1))
a3.set_xlabel("Mean spectral probe $R^2$"); a3.set_ylabel("HF energy fraction ($<$300 m)"); a3.set_title("(d)", loc="left")
for a in (a0, a1, a2, a3):
    a.grid(axis="y", lw=0.5, alpha=0.25); a.set_axisbelow(True)
handles = [Line2D([0], [0], color=DIN, lw=2.2, marker="o", ms=6, mec="black", mew=0.6, label="DINOv3-L"),
           Line2D([0], [0], color=CLA, lw=2.2, marker="D", ms=6, mec="black", mew=0.6, label="Clay")]
fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), handlelength=1.8, columnspacing=1.6)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(f"{V}/Fig_representation.{ext}", dpi=300, bbox_inches="tight"); print("->", ext)
print(f"HF common-scale: dino {hf_dino:.3f} clay {hf_clay:.3f}; mean probe R2 dino {spec_d:.3f} clay {spec_c:.3f}")
