#!/usr/bin/env python3
"""F1: Global map of held-out-region performance.
(a) FloodDuo region-mean IoU; (b) Delta(FloodDuo - U-Net). Region centroids from
real tile geocoordinates (rasterio); Natural Earth basemap."""
import json, glob, collections
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from src.data.flood_dataset import FloodDataset, tile_region

RUNS = "outputs/runs"
VIZ = "outputs/viz/manuscript_figs"
WORLD = "outputs/viz/assets/world_110m.geojson"
COORDS = "outputs/viz/assets/region_coords.json"


def region_centroids():
    import os
    if os.path.exists(COORDS):
        return json.load(open(COORDS))
    out = {}
    for ds in ["floodplanet", "ufo"]:
        d = FloodDataset(datasets=(ds,), split=("train", "val", "test"), train=False)
        acc = collections.defaultdict(list)
        for img, msk, _ in d.items:
            r = tile_region(img)
            with rasterio.open(img) as s:
                b = s.bounds
                acc[r].append(((b.left + b.right) / 2, (b.top + b.bottom) / 2))
        out[ds] = {r: [float(np.mean([p[0] for p in v])),
                       float(np.mean([p[1] for p in v])), len(v)] for r, v in acc.items()}
    json.dump(out, open(COORDS, "w")); return out


def iou(sub, ds):
    return {json.load(open(f))["region"]: json.load(open(f))["iou"]
            for f in glob.glob(f"{RUNS}/{sub}/{ds}/*/result.json")}


def main():
    coords = region_centroids()
    world = gpd.read_file(WORLD)
    fd = {ds: iou(f"loro_arch6/arch6", ds) for ds in ["floodplanet", "ufo"]}
    un = {ds: iou(f"loro_80ep/unet", ds) for ds in ["floodplanet", "ufo"]}

    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 1, figsize=(10, 9.6))
    for ax, mode, lab in zip(axes, ["iou", "delta"],
                             ["(a) FloodDuo cross-region IoU", "(b) Improvement over U-Net (Δ IoU)"]):
        world.plot(ax=ax, color="#eef0f2", edgecolor="#c8ccd0", lw=0.4)
        ax.set_xlim(-130, 155); ax.set_ylim(-45, 62); ax.set_facecolor("#f7fbff")
        for ds, mk in [("floodplanet", "o"), ("ufo", "^")]:
            xs, ys, vs = [], [], []
            for r, (lon, lat, n) in coords[ds].items():
                if r not in fd[ds]: continue
                xs.append(lon); ys.append(lat)
                vs.append(fd[ds][r] if mode == "iou" else fd[ds][r] - un[ds].get(r, np.nan))
            kw = dict(cmap="viridis", vmin=0.45, vmax=0.95) if mode == "iou" \
                 else dict(cmap="RdBu_r", vmin=-0.2, vmax=0.2)
            sc = ax.scatter(xs, ys, c=vs, s=130, marker=mk, edgecolor="k", lw=0.7, zorder=5, **kw)
        ax.set_xticks([]); ax.set_yticks([])
        # horizontal colorbar beneath the map
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03,
                          aspect=40, shrink=0.65)
        cb.set_label("Held-out region IoU" if mode == "iou" else "Δ IoU (FloodDuo − U-Net)",
                     fontsize=14); cb.ax.tick_params(labelsize=12)
        # panel label inside, lower-center
        ax.text(0.5, 0.05, lab, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=16, fontweight="bold",
                bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=3))
        ax.legend(handles=[Line2D([0],[0],marker="o",ls="",mfc="grey",mec="k",ms=10,label="FloodPlanet"),
                           Line2D([0],[0],marker="^",ls="",mfc="grey",mec="k",ms=10,label="UFO")],
                  loc="lower left", fontsize=12, frameon=True)
    fig.tight_layout(h_pad=0.2)
    fig.subplots_adjust(hspace=0.04)
    for ext in ("png", "pdf"):
        p = f"{VIZ}/Fig_geomap.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight"); print("->", p)


if __name__ == "__main__":
    main()
