#!/usr/bin/env python3
"""Manuscript figure generator: inference cache + gallery/detail figures.

Columns: RGB(4-3-2 false colour) | GT | U-Net | DOFA | TerraMind | FloodDuo.
Error maps: TP green, FP red, FN blue, TN light grey. IoU printed per panel.
Caches predictions per region so re-rendering is instant.

  # cache + render overall gallery (multiple FP regions pooled)
  python scripts/fig_make.py --dataset floodplanet --regions Ghana Nigeria US-Oklahoma --mode overall
  # detail gallery with zoom insets
  python scripts/fig_make.py --dataset ufo --regions HTX MID --mode detail
"""
import argparse, os, yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from src.train import build_model
from src.data.flood_dataset import FloodDataset

RUNS = "outputs/runs"
V = "outputs/viz/manuscript_figs"
REG = {
    "UNet":      ("configs/unet_fp.yaml",       "loro_80ep/unet"),
    "DOFA":      ("configs/dofa_noada_fp.yaml", "loro_comparative/dofa_noada"),
    "TerraMind": ("configs/terramind_fp.yaml",  "loro_comparative/terramind"),
    "CROMA":     ("configs/croma_fp.yaml",      "loro_comparative/croma"),
    "FloodDuo":  ("configs/arch6_fp.yaml",      "loro_arch6/arch6"),
}
LABEL = {"UNet": "U-Net", "DOFA": "DOFA", "TerraMind": "TerraMind",
         "CROMA": "CROMA", "FloodDuo": "FloodDuo (ours)"}
ECMAP = ListedColormap([(0.93, 0.93, 0.93), (0.18, 0.70, 0.24),
                        (0.86, 0.14, 0.14), (0.13, 0.34, 0.92)])
WCMAP = ListedColormap([(0.10, 0.45, 1.0)])


def cir(img):
    """4-3-2 false colour (NIR,R,G = idx 3,2,1) with 2-98% per-channel stretch."""
    x = img[[3, 2, 1]].astype(np.float32); o = np.empty_like(x)
    for k in range(3):
        lo, hi = np.percentile(x[k], 2), np.percentile(x[k], 98)
        o[k] = np.clip((x[k] - lo) / (hi - lo + 1e-6), 0, 1)
    return np.transpose(o, (1, 2, 0))


def err(p, g):
    p = (p >= .5).astype(np.uint8); g = (g >= .5).astype(np.uint8)
    e = np.zeros_like(p); e[(p == 1) & (g == 1)] = 1
    e[(p == 1) & (g == 0)] = 2; e[(p == 0) & (g == 1)] = 3
    return e


def iou(p, g):
    p = p >= .5; g = g >= .5; u = (p | g).sum()
    return (p & g).sum() / u if u else 1.0


def cache_path(ds, reg):
    return f"{V}/{ds}_{reg}_cache.npz"


@torch.no_grad()
def ensure_cache(ds, reg, models):
    cp = cache_path(ds, reg)
    have = set()
    if os.path.exists(cp):
        z = np.load(cp); have = {k[5:] for k in z.files if k.startswith("pred_")}
        if set(models) <= have:
            return
    data = FloodDataset(datasets=(ds,), split=("train", "val", "test"),
                        train=False, include_regions={reg})
    tiles = [(b["name"][0], b["image"][0].numpy(),
              (b["mask"][0, 0] if b["mask"].dim() == 4 else b["mask"][0]).numpy())
             for b in DataLoader(data, batch_size=1, num_workers=4)]
    out = {"names": [t[0] for t in tiles],
           "imgs": np.stack([t[1] for t in tiles]),
           "gts": np.stack([t[2] for t in tiles])}
    if os.path.exists(cp):
        z = np.load(cp)
        for k in z.files:
            if k.startswith("pred_"):
                out[k] = z[k]
    for mn in models:
        if mn in have:
            continue
        cfg, sub = REG[mn]
        mcfg = yaml.safe_load(open(cfg))["model"]
        mcfg["norm_mode"] = "dataset" if ds == "floodplanet" else "per_image"
        m = build_model(mcfg).cuda().eval()
        m.load_state_dict(torch.load(f"{RUNS}/{sub}/{ds}/{reg}/last.pt",
                          map_location="cpu", weights_only=False)["model"])
        ps = []
        for _, img, _ in tiles:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(torch.from_numpy(img)[None].cuda())
            ps.append(torch.sigmoid(lo.float())[0, 0].cpu().numpy())
        out[f"pred_{mn}"] = np.stack(ps)
        del m; torch.cuda.empty_cache()
        print(f"  cached {ds}/{reg} {mn}")
    np.savez_compressed(cp, **out)


def best_zoom(gt, eA, eB, win=300):
    H, W = gt.shape
    gy, gx = np.gradient(gt.astype(float))
    score = ((np.abs(gy) + np.abs(gx)) > 0).astype(float) + (eA != eB)
    best = (-1, 0, 0)
    for r in range(0, H - win + 1, 60):
        for c in range(0, W - win + 1, 60):
            s = score[r:r+win, c:c+win].sum()
            if s > best[0]: best = (s, r, c)
    return best[1], best[2], win


def gather(ds, regions, models):
    rows = []
    for reg in regions:
        z = np.load(cache_path(ds, reg))
        for i, n in enumerate(z["names"]):
            rows.append(dict(reg=reg, name=str(n), img=z["imgs"][i], gt=z["gts"][i],
                             preds={m: z[f"pred_{m}"][i] for m in models}))
    return rows


def render(ds, rows, models, mode, tag):
    ncol = 2 + len(models); nrow = len(rows)
    cell = 2.5
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * cell, nrow * cell))
    if nrow == 1: ax = ax[None, :]
    heads = ["RGB (4-3-2)", "Ground truth"] + [LABEL[m] for m in models]
    for r, row in enumerate(rows):
        base = cir(row["img"]); gt = row["gt"]
        ax[r, 0].imshow(base)
        ax[r, 0].set_ylabel(f"{row['reg']}", fontsize=10)
        ax[r, 1].imshow(base)
        ax[r, 1].imshow(np.ma.masked_where(gt < .5, gt), cmap=WCMAP, alpha=.55)
        eF = err(row["preds"]["FloodDuo"], gt)
        for c, mn in enumerate(models):
            a = ax[r, 2 + c]; e = err(row["preds"][mn], gt)
            a.imshow(e, cmap=ECMAP, vmin=0, vmax=3)
            a.text(0.035, 0.965, f"{iou(row['preds'][mn], gt):.2f}",
                   transform=a.transAxes, va='top', ha='left', fontsize=11,
                   weight='bold', bbox=dict(fc='white', ec='none', alpha=.85, pad=1.4))
            if mode == "detail":
                zr, zc, w = best_zoom(gt, err(row["preds"][models[-2]], gt)
                                      if len(models) > 1 else eF, eF)
                a.add_patch(Rectangle((zc, zr), w, w, ec='k', fc='none', lw=1.2))
                ins = a.inset_axes([0.6, 0.0, 0.4, 0.4])
                ins.imshow(e[zr:zr+w, zc:zc+w], cmap=ECMAP, vmin=0, vmax=3)
                ins.set_xticks([]); ins.set_yticks([])
                for s in ins.spines.values(): s.set_edgecolor('k'); s.set_linewidth(1.1)
        for c in range(ncol):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(heads[c], fontsize=12)
    fig.subplots_adjust(wspace=0.02, hspace=0.04, bottom=0.06)
    handles = [Line2D([0], [0], marker='s', ls='', mfc=ECMAP(i), mec='none', ms=11,
               label=l) for i, l in [(1, 'True positive'),
               (2, 'False positive (over-prediction)'), (3, 'False negative (missed water)')]]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.0))
    p = f"{V}/Gallery_{tag}.png"
    fig.savefig(p, dpi=190, bbox_inches="tight"); plt.close(fig); print("->", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["floodplanet", "ufo"])
    ap.add_argument("--regions", nargs="+", required=True)
    ap.add_argument("--models", nargs="+",
                    default=["UNet", "DOFA", "TerraMind", "FloodDuo"])
    ap.add_argument("--mode", choices=["overall", "detail"], default="overall")
    ap.add_argument("--ntiles", type=int, default=8)
    ap.add_argument("--exclude-tiles", default=None,
                    help="json list of tile names to exclude (avoid scene reuse)")
    args = ap.parse_args()
    import json
    exclude = set()
    if args.exclude_tiles and os.path.exists(args.exclude_tiles):
        exclude = set(json.load(open(args.exclude_tiles)))
    os.makedirs(V, exist_ok=True)
    for reg in args.regions:
        ensure_cache(args.dataset, reg, args.models)
    rows = gather(args.dataset, args.regions, args.models)
    # select tiles
    for row in rows:
        row["wf"] = float((row["gt"] >= .5).mean())
        row["adv"] = iou(row["preds"]["FloodDuo"], row["gt"]) - np.mean(
            [iou(row["preds"][m], row["gt"]) for m in args.models if m != "FloodDuo"])
    if args.mode == "detail":
        cand = [r for r in rows if 0.02 < r["wf"] < 0.35]
    else:
        cand = [r for r in rows if r["wf"] > 0.05]
    cand = [r for r in cand if r["name"] not in exclude] or \
           [r for r in (cand or rows) if r["name"] not in exclude]
    # round-robin across regions so the gallery spans regions, best tile first
    from collections import defaultdict
    groups = defaultdict(list)
    for r in cand:
        groups[r["reg"]].append(r)
    for g in groups.values():
        g.sort(key=lambda r: -r["adv"])
    sel = []
    while len(sel) < args.ntiles and any(groups.values()):
        for reg in args.regions:
            if groups.get(reg):
                sel.append(groups[reg].pop(0))
                if len(sel) >= args.ntiles:
                    break
    tag = f"{args.dataset}_{args.mode}_" + "-".join(args.regions)[:40]
    json.dump([r["name"] for r in sel], open(f"{V}/Gallery_{tag}.tiles.json", "w"))
    render(args.dataset, sel, args.models, args.mode, tag)


if __name__ == "__main__":
    main()
