"""Compute per-band mean/std and class balance over the train splits."""

import json
from pathlib import Path

import numpy as np
import rasterio

DATA_ROOT = Path("data/floodbench")
OUT = Path(__file__).resolve().parent.parent / "configs" / "band_stats.json"

stats = {}
for ds in ["floodplanet_planetScope", "UFO"]:
    s = np.zeros(4)
    s2 = np.zeros(4)
    n = 0
    water = 0
    total = 0
    for f in sorted((DATA_ROOT / ds / "train" / "image").glob("*.tif")):
        with rasterio.open(f) as src:
            img = src.read().astype(np.float64)  # (4,H,W)
        s += img.sum(axis=(1, 2))
        s2 += (img ** 2).sum(axis=(1, 2))
        n += img.shape[1] * img.shape[2]
        with rasterio.open(str(f).replace("/image/", "/mask/")) as src:
            m = src.read(1)
        water += int((m == 1).sum())
        total += m.size
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    stats[ds] = {
        "bands": ["blue", "green", "red", "nir"],
        "mean": mean.round(6).tolist(),
        "std": std.round(6).tolist(),
        "water_fraction": round(water / total, 6),
        "n_train_tiles": int(n / (1024 * 1024)),
    }
    print(ds, stats[ds])

# combined (pixel-weighted over both train sets)
ws = [stats[d]["n_train_tiles"] for d in stats]
means = np.array([stats[d]["mean"] for d in stats])
stds = np.array([stats[d]["std"] for d in stats])
w = np.array(ws)[:, None] / sum(ws)
comb_mean = (means * w).sum(0)
comb_var = ((stds ** 2 + means ** 2) * w).sum(0) - comb_mean ** 2
stats["combined"] = {
    "bands": ["blue", "green", "red", "nir"],
    "mean": comb_mean.round(6).tolist(),
    "std": np.sqrt(comb_var).round(6).tolist(),
}
print("combined", stats["combined"])
OUT.write_text(json.dumps(stats, indent=2))
print(f"saved -> {OUT}")
