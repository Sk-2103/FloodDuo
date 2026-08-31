#!/usr/bin/env python3
"""Qualitative comparison figures for the manuscript.

Loads LORO checkpoints (last.pt) for UNet / single-encoder FMs / arch6, predicts
the held-out tiles of a region, and renders RGB | GT | per-model error maps
(TP green / FP red / FN blue) with per-tile IoU. Caches predictions to .npz so
the final composed figure doesn't re-run inference.

Usage:
  python scripts/fig_qualitative.py --dataset floodplanet --region Ghana \
      --models UNet TerraMind arch6 --out viz/manuscript_figs
"""
import argparse, os, yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from src.train import build_model
from src.data.flood_dataset import FloodDataset
from src.metrics import SegMetrics

RUNS = "outputs/runs"
# name -> (model-config, loro run subdir)
REG = {
    "UNet":      ("configs/unet_fp.yaml",      "loro_80ep/unet"),
    "TerraMind": ("configs/terramind_fp.yaml", "loro_comparative/terramind"),
    "CROMA":     ("configs/croma_fp.yaml",     "loro_comparative/croma"),
    "Clay":      ("configs/clay_noada_fp.yaml","loro_comparative/clay_noada"),
    "Prithvi":   ("configs/prithvi_fp.yaml",   "loro_comparative/prithvi"),
    "arch6":     ("configs/arch6_fp.yaml",     "loro_arch6/arch6"),
}


def load_model(cfg_path, ckpt, dataset):
    mcfg = yaml.safe_load(open(cfg_path))["model"]
    mcfg["norm_mode"] = "dataset" if dataset == "floodplanet" else "per_image"
    m = build_model(mcfg).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def predict(m, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(x.cuda())
    return torch.sigmoid(logits.float())[:, 0].cpu().numpy()


def rgb(img):  # img (4,H,W) reflectance, order B,G,R,NIR -> display R,G,B
    r = np.clip(img[[2, 1, 0]] / 0.3, 0, 1)
    return np.transpose(r, (1, 2, 0))


def errmap(pred, gt):  # 0 TN,1 TP,2 FP,3 FN
    p = (pred >= 0.5).astype(np.uint8); g = (gt >= 0.5).astype(np.uint8)
    e = np.zeros_like(p)
    e[(p == 1) & (g == 1)] = 1
    e[(p == 1) & (g == 0)] = 2
    e[(p == 0) & (g == 1)] = 3
    return e

ECMAP = ListedColormap([(0.92, 0.92, 0.92), (0.20, 0.70, 0.25),
                        (0.85, 0.15, 0.15), (0.15, 0.35, 0.90)])  # TN,TP,FP,FN


def iou(pred, gt):
    p = (pred >= 0.5); g = (gt >= 0.5)
    i = (p & g).sum(); u = (p | g).sum()
    return float(i / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["floodplanet", "ufo"])
    ap.add_argument("--region", required=True)
    ap.add_argument("--models", nargs="+", default=["UNet", "TerraMind", "arch6"])
    ap.add_argument("--out", default="viz/manuscript_figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = FloodDataset(datasets=(args.dataset,), split=("train", "val", "test"),
                      train=False, include_regions={args.region})
    loader = DataLoader(ds, batch_size=1, num_workers=4)
    tiles = []
    for b in loader:
        tiles.append((b["name"][0], b["image"][0].numpy(), b["mask"][0, 0].numpy()
                      if b["mask"].dim() == 4 else b["mask"][0].numpy()))
    print(f"{args.region}: {len(tiles)} held-out tiles")

    preds = {}  # model -> {name: prob}
    for mn in args.models:
        cfg, sub = REG[mn]
        ckpt = f"{RUNS}/{sub}/{args.dataset}/{args.region}/last.pt"
        m = load_model(cfg, ckpt, args.dataset)
        preds[mn] = {}
        for name, img, gt in tiles:
            x = torch.from_numpy(img)[None]
            preds[mn][name] = predict(m, x)[0]
        del m; torch.cuda.empty_cache()
        ious = [iou(preds[mn][n], gt) for n, _, gt in tiles]
        print(f"  {mn}: per-tile IoU mean {np.mean(ious):.3f}")

    # cache
    np.savez_compressed(f"{args.out}/{args.dataset}_{args.region}_cache.npz",
                        names=[t[0] for t in tiles],
                        imgs=np.stack([t[1] for t in tiles]),
                        gts=np.stack([t[2] for t in tiles]),
                        **{f"pred_{mn}": np.stack([preds[mn][t[0]] for t in tiles])
                           for mn in args.models})

    # contact sheet
    ncol = 2 + len(args.models)
    nrow = len(tiles)
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow))
    if nrow == 1: ax = ax[None, :]
    for r, (name, img, gt) in enumerate(tiles):
        ax[r, 0].imshow(rgb(img)); ax[r, 0].set_ylabel(name[:18], fontsize=6)
        if r == 0: ax[r, 0].set_title("RGB", fontsize=9)
        ax[r, 1].imshow(rgb(img)); ax[r, 1].imshow(np.ma.masked_where(gt < 0.5, gt),
                        cmap=ListedColormap([(0.15, 0.4, 1.0)]), alpha=0.55)
        if r == 0: ax[r, 1].set_title("Ground truth", fontsize=9)
        for c, mn in enumerate(args.models):
            a = ax[r, 2 + c]
            a.imshow(errmap(preds[mn][name], gt), cmap=ECMAP, vmin=0, vmax=3)
            a.set_title(f"{mn}\nIoU {iou(preds[mn][name], gt):.3f}" if r == 0
                        else f"IoU {iou(preds[mn][name], gt):.3f}", fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{args.dataset} / {args.region}  (TP green · FP red · FN blue)",
                 fontsize=11)
    fig.tight_layout()
    p = f"{args.out}/contact_{args.dataset}_{args.region}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); print("->", p)


if __name__ == "__main__":
    main()
