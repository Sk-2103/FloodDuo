#!/usr/bin/env python3
"""Per-tile water-fraction (predicted vs. reference) across the LORO test set.
Writes waterfrac_{ds}.json, used by area_full_sample.py.
"""
import json, os, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.train import build_model
from src.data.flood_dataset import FloodDataset

from _paths import RUNS_ROOT as RUNS
from _paths import FIGS_ROOT as OUT
REG = {
    "FloodDuo":  ("configs/arch6_fp.yaml",      "loro_arch6/arch6"),
    "TerraMind": ("configs/terramind_fp.yaml",  "loro_comparative/terramind"),
    "DOFA":      ("configs/dofa_noada_fp.yaml", "loro_comparative/dofa_noada"),
    "Prithvi":   ("configs/prithvi_fp.yaml",    "loro_comparative/prithvi"),
    "Clay":      ("configs/clay_noada_fp.yaml", "loro_comparative/clay_noada"),
    "CROMA":     ("configs/croma_fp.yaml",      "loro_comparative/croma"),
    "SSL4EO-DINO": ("configs/ssl4eo_dino_fp.yaml", "loro_comparative/ssl4eo_dino"),
    "U-Net":     ("configs/unet_fp.yaml",       "loro_80ep/unet"),
}
REGIONS = {
  "floodplanet": ["Bolivia","US-Kansas","Spain","Somalia","Ghana","Cambodia",
    "Paraguay","US-Nebraska","US-Alabama","US-Carolina","Nigeria","Bangladesh",
    "US-Dakota","Uzbekistan","Colombia","US-Oklahoma","US-Texas","Nepal","US-Arkansas"],
  "ufo": ["HTX","SPS","NSW","KTM","GIL","CMO","PNE","QUE","DKA","CTO","SLC","BNA","MID","BEI"],
}
MODELS = ["FloodDuo","TerraMind","DOFA","Prithvi","Clay","CROMA","SSL4EO-DINO","U-Net"]


@torch.no_grad()
def run(ds):
    norm = "dataset" if ds == "floodplanet" else "per_image"
    out_path = f"{OUT}/waterfrac_{ds}.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for mn in MODELS:
        if mn in res:
            continue
        cfg, sub = REG[mn]
        rows = []
        for reg in REGIONS[ds]:
            ck = f"{RUNS}/{sub}/{ds}/{reg}/last.pt"
            if not os.path.exists(ck):
                print(f"  MISSING {mn} {ds}/{reg}"); continue
            mcfg = yaml.safe_load(open(cfg))["model"]; mcfg["norm_mode"] = norm
            m = build_model(mcfg).cuda().eval()
            m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
            data = FloodDataset(datasets=(ds,), split=("train","val","test"),
                                train=False, include_regions={reg})
            for b in DataLoader(data, batch_size=1, num_workers=4):
                x = b["image"].cuda()
                gt = (b["mask"][0,0] if b["mask"].dim()==4 else b["mask"][0]).numpy()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = torch.sigmoid(m(x).float())[0,0].cpu().numpy()
                pb = p >= .5; gb = gt >= .5; u = (pb|gb).sum()
                rows.append(dict(region=reg, name=str(b["name"][0]),
                                 gt=float(gb.mean()), pred=float(pb.mean()),
                                 iou=float((pb&gb).sum()/u) if u else 1.0))
            del m; torch.cuda.empty_cache()
        res[mn] = rows
        json.dump(res, open(out_path, "w"))
        print(f"  done {ds} {mn}: {len(rows)} tiles")
    print("WROTE", out_path)


if __name__ == "__main__":
    import sys
    for ds in (sys.argv[1:] or ["floodplanet", "ufo"]):
        run(ds)
    print("STAT_WATERFRAC_DONE")
