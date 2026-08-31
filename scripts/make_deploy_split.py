#!/usr/bin/env python3
"""Combined FP+UFO deployment split: pool ALL tiles (train/val/test of both
datasets), hold out ~15% for validation (checkpoint selection), stratified by
dataset+region so every region is represented in both train and val.
Writes configs/deploy_split.json.
"""
import glob, json, os, random
from collections import defaultdict

ROOT = "data/floodbench"
DS = {"floodplanet": "floodplanet_planetScope", "ufo": "UFO"}
VAL_FRAC = 0.15
SEED = 42
rng = random.Random(SEED)

def region(stem):  # filename token 2 (matches src/data tile_region convention)
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else stem

# gather all tiles grouped by (dataset, region)
groups = defaultdict(list)
allrows = []
for ds, d in DS.items():
    for sp in ("train", "val", "test"):
        for f in sorted(glob.glob(f"{ROOT}/{d}/{sp}/image/*.tif")):
            stem = os.path.splitext(os.path.basename(f))[0]
            groups[(ds, region(stem))].append(stem)
            allrows.append((ds, region(stem), stem))

train, val = [], []
for key, stems in groups.items():
    stems = sorted(stems); rng.shuffle(stems)
    k = max(1, round(len(stems) * VAL_FRAC)) if len(stems) >= 3 else 0  # tiny groups stay in train
    val += stems[:k]; train += stems[k:]

train, val = sorted(train), sorted(val)
out = {
    "seed": SEED, "val_frac": VAL_FRAC,
    "n_total": len(allrows), "n_train": len(train), "n_val": len(val),
    "train": train, "val": val,
}
os.makedirs("configs", exist_ok=True)
json.dump(out, open("configs/deploy_split.json", "w"), indent=2)

# report
by_ds = defaultdict(lambda: [0, 0])
vs = set(val)
for ds, reg, stem in allrows:
    by_ds[ds][1 if stem in vs else 0] += 1
print(f"total {len(allrows)} | train {len(train)} | val {len(val)} "
      f"({100*len(val)/len(allrows):.1f}%)")
for ds in DS:
    print(f"  {ds}: train {by_ds[ds][0]} / val {by_ds[ds][1]}")
print(f"regions covered: {len(groups)} | saved configs/deploy_split.json")
