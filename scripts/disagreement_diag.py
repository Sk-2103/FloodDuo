#!/usr/bin/env python3
"""Diagnostics for the disagreement map D as a label-free OOD / uncertainty signal.

Consumes the per-tile dumps written by `src.eval` for disagreement-enabled runs:
each tile contributes  <name>_D.npy, <name>_pred.npy, <name>_gt.npy  in a run's
`dmaps_<split>/` dir (D in [0, ln2]; pred = water probability; gt in {0,1}).
Per-region mean-D vs IoU comes from the run's `eval_<ds>_<split>.json` per_tile
records (each has `mean_D` and `iou`), optionally pooled across LORO folds.

Produces:
  (a) error-rate vs D-bin curve + a reliability/ECE summary (does higher D
      actually predict higher pixel error?), and
  (b) per-region scatter: mean-D (x) vs IoU-drop from the best region (y).

Usage:
  # pixel-level error-vs-D + reliability over dumped tiles in one or more runs
  python scripts/disagreement_diag.py pixel  --dmaps RUN/dmaps_test [more ...] \
         --out RUN/viz/disagreement
  # region-level scatter from eval jsons (e.g. LORO folds)
  python scripts/disagreement_diag.py region --evals 'runs/loro_*/.../eval_*_test.json' \
         --out RUN/viz/disagreement
"""
import argparse
import glob
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except Exception:
    _MPL = False


def _load_tiles(dmap_dirs):
    Ds, Es = [], []          # per-pixel D and per-pixel error, flattened
    for d in dmap_dirs:
        for dp in sorted(glob.glob(os.path.join(d, "*_D.npy"))):
            base = dp[:-6]
            pred = np.load(base + "_pred.npy")
            gt = np.load(base + "_gt.npy")
            D = np.load(dp)
            # D is low-res; upsample by nearest repeat to pred grid for binning
            if D.shape != pred.shape:
                fy, fx = pred.shape[0] // D.shape[0], pred.shape[1] // D.shape[1]
                D = np.kron(D, np.ones((max(fy, 1), max(fx, 1))))[:pred.shape[0], :pred.shape[1]]
            err = (np.abs(pred - gt) > 0.5).astype(np.float32)
            Ds.append(D.ravel())
            Es.append(err.ravel())
    if not Ds:
        raise SystemExit("no *_D.npy dumps found; run src.eval on a disagreement run")
    return np.concatenate(Ds), np.concatenate(Es)


def pixel_cmd(args):
    D, E = _load_tiles(args.dmaps)
    nbins = args.bins
    edges = np.linspace(0, np.log(2) + 1e-9, nbins + 1)
    idx = np.clip(np.digitize(D, edges) - 1, 0, nbins - 1)
    err_rate = np.array([E[idx == b].mean() if (idx == b).any() else np.nan
                         for b in range(nbins)])
    frac = np.array([(idx == b).mean() for b in range(nbins)])
    # ECE-style: |error-rate - normalized-D| weighted by bin mass
    centers = 0.5 * (edges[:-1] + edges[1:])
    dnorm = centers / np.log(2)
    valid = ~np.isnan(err_rate)
    ece = float(np.sum(frac[valid] * np.abs(err_rate[valid] - dnorm[valid])))
    # rank correlation between D and error (monotonic calibration)
    overall = float(E.mean())
    print(f"pixels={D.size:,} overall error-rate={overall:.4f} ECE-vs-D={ece:.4f}")
    for b in range(nbins):
        print(f"  D[{edges[b]:.3f},{edges[b+1]:.3f}) mass={frac[b]:.3f} "
              f"err={err_rate[b]:.4f}")
    if _MPL:
        os.makedirs(args.out, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(centers[valid], err_rate[valid], "o-")
        ax[0].axhline(overall, ls="--", c="grey", label=f"mean {overall:.3f}")
        ax[0].set_xlabel("disagreement D"); ax[0].set_ylabel("pixel error rate")
        ax[0].set_title(f"error vs D (ECE {ece:.3f})"); ax[0].legend()
        ax[1].plot(dnorm[valid], err_rate[valid], "o-")
        ax[1].plot([0, 1], [0, 1], ls="--", c="grey")
        ax[1].set_xlabel("normalized D"); ax[1].set_ylabel("error rate")
        ax[1].set_title("reliability")
        fig.tight_layout(); p = os.path.join(args.out, "error_vs_D.png")
        fig.savefig(p, dpi=120); print(f"-> {p}")


def region_cmd(args):
    files = []
    for pat in args.evals:
        files += glob.glob(pat)
    rows = []
    for f in files:
        j = json.load(open(f))
        region = j.get("region") or os.path.basename(os.path.dirname(f))
        tiles = j.get("per_tile", [])
        mds = [t["mean_D"] for t in tiles if "mean_D" in t]
        if not mds:
            continue
        rows.append((region, float(np.mean(mds)),
                     float(j["overall"]["iou"])))
    if not rows:
        raise SystemExit("no eval jsons with per-tile mean_D found")
    rows.sort(key=lambda r: -r[2])
    best_iou = rows[0][2]
    print(f"{'region':16s} {'meanD':>8s} {'IoU':>7s} {'IoU-drop':>9s}")
    xs, ys = [], []
    for region, md, iou in rows:
        drop = best_iou - iou
        xs.append(md); ys.append(drop)
        print(f"{region:16s} {md:8.4f} {iou:7.4f} {drop:9.4f}")
    if len(xs) > 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        print(f"Pearson(mean-D, IoU-drop) = {r:.3f}")
    if _MPL:
        os.makedirs(args.out, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(xs, ys)
        for (region, md, iou) in rows:
            ax.annotate(region, (md, best_iou - iou), fontsize=7)
        ax.set_xlabel("mean disagreement D"); ax.set_ylabel("IoU drop vs best region")
        ax.set_title("region OOD: D vs generalization gap")
        fig.tight_layout(); p = os.path.join(args.out, "region_D_vs_iou.png")
        fig.savefig(p, dpi=120); print(f"-> {p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pixel"); pp.add_argument("--dmaps", nargs="+", required=True)
    pp.add_argument("--bins", type=int, default=10)
    pp.add_argument("--out", default="viz/disagreement"); pp.set_defaults(fn=pixel_cmd)
    rp = sub.add_parser("region"); rp.add_argument("--evals", nargs="+", required=True)
    rp.add_argument("--out", default="viz/disagreement"); rp.set_defaults(fn=region_cmd)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
