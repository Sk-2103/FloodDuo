#!/usr/bin/env python3
"""Figure 12: encoder complementarity on 4 fixed example tiles
(DINOv3-only / Clay-only / FloodDuo error maps)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.train import build_model
from src.data.flood_dataset import FloodDataset

from _paths import RUNS_ROOT as RUNS
from _paths import FIGS_ROOT as V
DEV = "cuda:0"; NORM = {"floodplanet": "dataset", "ufo": "per_image"}
MODELS = {"DINOv3-only": ("loro_single_enc/dinoonly", "configs/dinoonly_{s}.yaml"),
          "Clay-only":   ("loro_single_enc/clayonly", "configs/clayonly_{s}.yaml"),
          "FloodDuo":    ("loro_arch6/arch6",         "configs/arch6_{s}.yaml")}
# (short row label, dataset, region, tile stem)
PICKS = [
    ("(a) Spectrally mixed water",     "ufo",         "HTX",     "UFO_HTX_HTX_20170831_172559_100d_5h5l"),
    ("(b) Turbid water",               "floodplanet", "Bolivia", "FP_Bolivia_BOL_1660"),
    ("(c) Vegetated / turbid water",   "floodplanet", "Somalia", "FP_Somalia_SOM_26_62"),
    ("(d) Urban fine-scale",           "ufo",         "HTX",     "UFO_HTX_HTX_20170831_172612_100d_mhby"),
]
ECMAP = ListedColormap([(1, 1, 1), (0.15, 0.6, 0.2), (0.85, 0.15, 0.15), (0.15, 0.35, 0.9)])

def cir(img):  # false color 4-3-2 = NIR, R, G, per-band 2-98 percentile stretch
    x = img[[3, 2, 1]].astype(np.float32); out = np.empty_like(x)
    for i in range(3):
        lo, hi = np.percentile(x[i], (2, 98)); out[i] = np.clip((x[i] - lo) / (hi - lo + 1e-6), 0, 1)
    return out.transpose(1, 2, 0)
def errmap(prob, gt):
    p = prob >= 0.5; g = gt >= 0.5; m = np.zeros_like(gt, np.uint8)
    m[p & g] = 1; m[p & ~g] = 2; m[~p & g] = 3; return m
def iou(prob, gt):
    p = prob >= 0.5; g = gt >= 0.5; u = (p | g).sum(); return float((p & g).sum() / u) if u else 1.0

@torch.no_grad()
def get_tile(ds, region, stem):
    short = "fp" if ds == "floodplanet" else "ufo"
    tiles = FloodDataset(datasets=(ds,), split=("train", "val", "test"), train=False, include_regions={region})
    rec = None
    for i in range(len(tiles)):
        if tiles[i]["name"] == stem: rec = tiles[i]; break
    assert rec is not None, f"{stem} not found in {region}"
    img = rec["image"].numpy(); gt = rec["mask"].numpy(); x = rec["image"][None].to(DEV); pr = {}
    for mn, (rundir, cfgt) in MODELS.items():
        mcfg = yaml.safe_load(open(cfgt.format(s=short)))["model"]; mcfg["norm_mode"] = NORM[ds]
        net = build_model(mcfg).to(DEV).eval()
        sd = torch.load(f"{RUNS}/{rundir}/{ds}/{region}/last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
        with torch.autocast("cuda", dtype=torch.bfloat16): pr[mn] = torch.sigmoid(net(x).float())[0, 0].cpu().numpy()
        del net; torch.cuda.empty_cache()
    return img, gt, pr

cols = ["False color (4-3-2)", "Ground truth", "DINOv3-only", "Clay-only", "FloodDuo"]
fig, ax = plt.subplots(4, 5, figsize=(14, 11.6))
for r, (label, ds, region, stem) in enumerate(PICKS):
    img, gt, pr = get_tile(ds, region, stem); fc = cir(img)
    ax[r, 0].imshow(fc)
    ax[r, 0].set_ylabel(label, fontsize=14, labelpad=8)
    ax[r, 1].imshow(fc); ax[r, 1].imshow(np.ma.masked_where(gt < 0.5, gt),
                                         cmap=ListedColormap([(0.15, 0.4, 1.0)]), alpha=0.55)
    for c, mn in enumerate(["DINOv3-only", "Clay-only", "FloodDuo"], start=2):
        ax[r, c].imshow(errmap(pr[mn], gt), cmap=ECMAP, vmin=0, vmax=3)
        ax[r, c].text(0.04, 0.96, f"IoU {iou(pr[mn], gt):.2f}", transform=ax[r, c].transAxes,
                      ha="left", va="top", fontsize=13,
                      bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for c in range(5):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    print(f"  rendered {label}: dino {iou(pr['DINOv3-only'],gt):.2f} clay {iou(pr['Clay-only'],gt):.2f} full {iou(pr['FloodDuo'],gt):.2f}", flush=True)
for c, name in enumerate(cols):
    ax[0, c].set_title(name, fontsize=15, fontweight="bold", pad=8)
from matplotlib.patches import Patch
handles = [Patch(fc=(0.15, 0.6, 0.2), label="True positive"),
           Patch(fc=(0.85, 0.15, 0.15), label="False water (commission)"),
           Patch(fc=(0.15, 0.35, 0.9), label="Missed water (omission)")]
fig.tight_layout(rect=[0, 0.035, 1, 0.985])
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=17, frameon=False,
           bbox_to_anchor=(0.5, 0.004), handlelength=1.6, columnspacing=2.4, handleheight=1.3)
for ext in ("png", "pdf"):
    fig.savefig(f"{V}/Fig_complementarity_maps.{ext}", dpi=240, bbox_inches="tight"); print("->", ext)
