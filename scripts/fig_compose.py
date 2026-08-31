#!/usr/bin/env python3
"""Compose the two manuscript qualitative figures from cached predictions.

Fig 1 (generalization): FP Ghana — U-Net misses water in an unseen region
       (global-context failure) vs FloodDuo. cols: RGB | GT | U-Net | FloodDuo.
Fig 2 (fine detail): UFO HTX — a single ViT FM gives coarse/blocky boundaries
       vs FloodDuo's sharp ones. cols: RGB | GT | TerraMind | FloodDuo, + zoom.
Error maps: TP green, FP red, FN blue, TN light grey.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

V = "outputs/viz/manuscript_figs"
ECMAP = ListedColormap([(0.93, 0.93, 0.93), (0.20, 0.70, 0.25),
                        (0.85, 0.15, 0.15), (0.15, 0.35, 0.90)])
BLUE = ListedColormap([(0.12, 0.45, 1.0)])


def rgb(img):
    # per-channel 2-98% percentile stretch for display (urban scenes are dark)
    x = img[[2, 1, 0]].astype(np.float32)
    out = np.empty_like(x)
    for k in range(3):
        lo, hi = np.percentile(x[k], 2), np.percentile(x[k], 98)
        out[k] = np.clip((x[k] - lo) / (hi - lo + 1e-6), 0, 1)
    return np.transpose(out, (1, 2, 0))


def err(pred, gt):
    p = (pred >= 0.5).astype(np.uint8); g = (gt >= 0.5).astype(np.uint8)
    e = np.zeros_like(p); e[(p == 1) & (g == 1)] = 1
    e[(p == 1) & (g == 0)] = 2; e[(p == 0) & (g == 1)] = 3
    return e


def iou(p, g):
    p = p >= 0.5; g = g >= 0.5; u = (p | g).sum()
    return (p & g).sum() / u if u else 1.0


def best_zoom(gt, errA, errB, win=300):
    """window (r,c) maximising boundary activity + model error contrast."""
    H, W = gt.shape
    bnd = (np.abs(np.gradient(gt.astype(float))[0]) +
           np.abs(np.gradient(gt.astype(float))[1])) > 0
    score_map = bnd.astype(float) + (errA != errB).astype(float)
    best, br, bc = -1, 0, 0
    for r in range(0, H - win + 1, 60):
        for c in range(0, W - win + 1, 60):
            s = score_map[r:r+win, c:c+win].sum()
            if s > best: best, br, bc = s, r, c
    return br, bc, win


def legend(fig):
    handles = [Line2D([0], [0], marker='s', ls='', mfc=ECMAP(i), mec='none',
               ms=10, label=l) for i, l in
               [(1, 'True positive'), (2, 'False positive (over-pred)'),
                (3, 'False negative (missed water)')]]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))


# ---------------- Figure 1: generalization (Ghana) ----------------
def fig_generalization(idxs):
    z = np.load(f"{V}/floodplanet_Ghana_cache.npz")
    cols = ["RGB", "Ground truth", "U-Net (CNN)", "FloodDuo (ours)"]
    n = len(idxs)
    fig, ax = plt.subplots(n, 4, figsize=(4 * 2.7, n * 2.7))
    if n == 1: ax = ax[None, :]
    for r, i in enumerate(idxs):
        img, gt = z["imgs"][i], z["gts"][i]
        ax[r, 0].imshow(rgb(img))
        ax[r, 1].imshow(rgb(img)); ax[r, 1].imshow(np.ma.masked_where(gt < .5, gt), cmap=BLUE, alpha=.55)
        for c, mn in [(2, "UNet"), (3, "arch6")]:
            ax[r, c].imshow(err(z[f"pred_{mn}"][i], gt), cmap=ECMAP, vmin=0, vmax=3)
            ax[r, c].text(0.03, 0.95, f"IoU {iou(z[f'pred_{mn}'][i], gt):.2f}",
                          transform=ax[r, c].transAxes, va='top', ha='left',
                          fontsize=11, color='black', weight='bold',
                          bbox=dict(fc='white', ec='none', alpha=.8, pad=1.5))
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(cols[c], fontsize=13)
    legend(fig)
    fig.suptitle("Cross-region generalization (held-out region: Ghana). "
                 "The CNN misses water it has not seen; FloodDuo recovers it.",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    p = f"{V}/Fig_generalization_Ghana.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); print("->", p)


# ---------------- Figure 2: fine detail (HTX) ----------------
def fig_detail(idxs, fm="TerraMind"):
    z = np.load(f"{V}/ufo_HTX_cache.npz")
    cols = ["RGB", "Ground truth", f"{fm} (single ViT)", "FloodDuo (ours)"]
    n = len(idxs)
    fig, ax = plt.subplots(n, 4, figsize=(4 * 2.7, n * 2.7))
    if n == 1: ax = ax[None, :]
    for r, i in enumerate(idxs):
        img, gt = z["imgs"][i], z["gts"][i]
        eA, eB = err(z[f"pred_{fm}"][i], gt), err(z["pred_arch6"][i], gt)
        zr, zc, w = best_zoom(gt, eA, eB)
        ax[r, 0].imshow(rgb(img))
        ax[r, 1].imshow(rgb(img)); ax[r, 1].imshow(np.ma.masked_where(gt < .5, gt), cmap=BLUE, alpha=.55)
        for c, (mn, e) in [(2, (fm, eA)), (3, ("arch6", eB))]:
            ax[r, c].imshow(e, cmap=ECMAP, vmin=0, vmax=3)
            ax[r, c].text(0.03, 0.95, f"IoU {iou(z[f'pred_{mn}'][i], gt):.2f}",
                          transform=ax[r, c].transAxes, va='top', ha='left',
                          fontsize=11, weight='bold',
                          bbox=dict(fc='white', ec='none', alpha=.8, pad=1.5))
            # zoom rectangle + inset
            ax[r, c].add_patch(Rectangle((zc, zr), w, w, ec='black', fc='none', lw=1.3))
            axins = ax[r, c].inset_axes([0.62, 0.0, 0.38, 0.38])
            axins.imshow(e[zr:zr+w, zc:zc+w], cmap=ECMAP, vmin=0, vmax=3)
            axins.set_xticks([]); axins.set_yticks([])
            for s in axins.spines.values(): s.set_edgecolor('black'); s.set_linewidth(1.2)
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(cols[c], fontsize=13)
    legend(fig)
    fig.suptitle("Fine spatial detail (held-out urban region: Houston). "
                 "The single ViT encoder yields coarse boundaries; FloodDuo resolves "
                 "thin channels (inset).", fontsize=13, y=1.005)
    fig.tight_layout()
    p = f"{V}/Fig_detail_HTX.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); print("->", p)


if __name__ == "__main__":
    fig_generalization([6, 8, 1])
    fig_detail([13, 8, 3], fm="TerraMind")
