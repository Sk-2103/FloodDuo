#!/usr/bin/env python3
"""Print LORO region-wise IoU tables: FloodDuo | comparative FM baselines | U-Net.

Usage:  python scripts/show_loro.py [--budget 40|80]
"""
import argparse
import glob
import json
import os
import statistics as st

from _paths import RUNS_ROOT as RUNS

# (column header, base dir relative to RUNS) for the 80-ep comparative set.
# arch5 (ours, dual+adapters) | frozen FM + UPerNet, NO adapters | UNet (scratch)
SPECS_80 = [
    ("arch5 (ours)", "loro_80ep/arch5"),
    ("clay",         "loro_comparative/clay_noada"),
    ("dofa",         "loro_comparative/dofa_noada"),
    ("prithvi",      "loro_comparative/prithvi"),
    ("terramind",    "loro_comparative/terramind"),
    ("croma",        "loro_comparative/croma"),
    ("ssl4eo_dino*", "loro_comparative/ssl4eo_dino"),  # *ViT-S (not ViT-L)
    ("UNet",         "loro_80ep/unet"),
]
# 40-ep budget (primary for paper): only arch5 + UNet have matched 40ep runs.
SPECS_40 = [
    ("arch5 (ours)", "loro_arch5"),
    ("UNet",         "loro_unet"),
]


def load(base):
    d = {"floodplanet": {}, "ufo": {}}
    for ds in d:
        for f in glob.glob(os.path.join(RUNS, base, ds, "*", "result.json")):
            j = json.load(open(f))
            d[ds][j["region"]] = j["iou"]
    return d


def table(ds, specs, data):
    hdr = [h for h, _ in specs]
    regs = sorted(data[hdr[0]][ds])
    print(f"### {ds.upper()}")
    print("| region | " + " | ".join(hdr) + " |")
    print("|" + "---|" * (len(hdr) + 1))
    for r in regs:
        cells = [f"{data[h][ds].get(r, float('nan')):.3f}" for h in hdr]
        print(f"| {r} | " + " | ".join(cells) + " |")
    mean = [f"**{st.mean(list(data[h][ds].values())):.3f}**" for h in hdr]
    sd = [f"{st.pstdev(list(data[h][ds].values())):.3f}" for h in hdr]
    print("| **MEAN** | " + " | ".join(mean) + " |")
    print("| ±std | " + " | ".join(sd) + " |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=80, choices=[40, 80])
    args = ap.parse_args()
    specs = SPECS_80 if args.budget == 80 else SPECS_40
    data = {h: load(b) for h, b in specs}
    print(f"# LORO held-out-region IoU ({args.budget} ep)\n")
    for ds in ("floodplanet", "ufo"):
        table(ds, specs, data)


if __name__ == "__main__":
    main()
