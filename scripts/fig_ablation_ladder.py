#!/usr/bin/env python3
"""Fig: component ablation "waterfall"/step chart — how FloodDuo's region-mean
LORO IoU builds up across the design. (a) FloodPlanet, (b) UFO.
Region-mean IoU = mean of per-fold result.json "iou" over a run's
{floodplanet,ufo}/ subfolders; error bars = std across regions.
Standalone: reads only result.json under RUNS; writes PNG+PDF."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "outputs/runs"
OUT = "outputs/viz/manuscript_figs"

# (label, loro subdir) in build-up order; last step = FloodDuo (highlighted)
STEPS = [
    ("Spectral only\n(DOFA-L)", "loro_dofal_only"),
    ("+DINOv3\n(dual)", "loro_dofa"),
    ("+Clay,\nscalar gate", "loro_arch6/arch6_v0"),
    ("+Disagreement gate\n(FloodDuo)", "loro_arch6/arch6"),
]

FLOODDUO = "#1b9e77"
GREY = "#b0b0b0"


def region_stats(sub, ds):
    v = [json.load(open(f))["iou"]
         for f in glob.glob(f"{RUNS}/{sub}/{ds}/*/result.json")]
    v = np.asarray(v, float)
    return v.mean(), v.std(), len(v)


plt.rcParams.update({"font.size": 14})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))

reported = {}
for ax, ds, ttl in [(axes[0], "floodplanet", "(a) FloodPlanet"),
                    (axes[1], "ufo", "(b) UFO")]:
    means, stds, ns = [], [], []
    for _, sub in STEPS:
        m, s, n = region_stats(sub, ds)
        means.append(m); stds.append(s); ns.append(n)
    reported[ds] = (means, stds, ns)
    x = np.arange(len(STEPS))
    colors = [GREY] * (len(STEPS) - 1) + [FLOODDUO]
    alphas = [0.75] * (len(STEPS) - 1) + [0.95]
    bars = ax.bar(x, means, yerr=stds, width=0.62, color=colors,
                  edgecolor="black", lw=0.9, capsize=4,
                  error_kw=dict(lw=1.1, ecolor="0.3"))
    for b, a in zip(bars, alphas):
        b.set_alpha(a)
    # annotate value + delta vs previous step
    ymax = max(m + s for m, s in zip(means, stds))
    for i, (m, s) in enumerate(zip(means, stds)):
        lbl = f"{m:.3f}"
        if i > 0:
            d = m - means[i - 1]
            lbl += f"\n({'+' if d >= 0 else ''}{d:.3f})"
        ax.text(i, m + s + 0.012 * ymax, lbl, ha="center", va="bottom",
                fontsize=12.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in STEPS], rotation=20, ha="right",
                       fontsize=11.5)
    ax.set_ylabel("LORO region-mean IoU (water)", fontsize=14)
    ax.set_title(ttl, fontsize=15, fontweight="bold")
    lo = min(means) - max(stds)
    ax.set_ylim(max(0, lo - 0.03), ymax + 0.10 * ymax)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)

fig.suptitle("Component ablation: cross-region IoU across FloodDuo's design "
             "(leave-one-region-out)", fontsize=14, y=1.00)
fig.tight_layout()
for ext in ("png", "pdf"):
    p = f"{OUT}/Fig_ablation_ladder.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("->", p)

# print reported numbers for the caller
for ds in ("floodplanet", "ufo"):
    means, stds, ns = reported[ds]
    print(f"\n{ds}:")
    for i, (lbl, _) in enumerate(STEPS):
        d = "" if i == 0 else f"  d={means[i]-means[i-1]:+.4f}"
        clean = lbl.replace("\n", " ")
        print(f"  {clean:28s} n={ns[i]:2d}  IoU={means[i]:.4f} +/- {stds[i]:.4f}{d}")
