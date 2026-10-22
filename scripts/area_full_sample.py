#!/usr/bin/env python3
"""Flood-area agreement (R2, MAE, RMSE) per model, full sample + |z|>3-filtered.
Reads waterfrac_{ds}.json."""
import json, numpy as np
from collections import Counter

from _paths import FIGS_ROOT as OUT
TILE_KM2 = (1024 * 3.0) ** 2 / 1e6                       # 9.437 km^2
MODELS = ["FloodDuo", "TerraMind", "Prithvi", "DOFA", "Clay", "CROMA",
          "SSL4EO-DINO", "U-Net"]
Z = 3.0

def metrics(o, p):
    o, p = np.asarray(o), np.asarray(p)
    ss_res = np.sum((p - o) ** 2); ss_tot = np.sum((o - o.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = np.corrcoef(o, p)[0, 1] if len(o) > 1 else float("nan")
    return dict(R2=float(r2), r=float(r),
                MAE=float(np.mean(np.abs(p - o))), RMSE=float(np.sqrt(np.mean((p - o) ** 2))),
                MAE_pct=float(np.mean(np.abs(p - o)) / TILE_KM2 * 100),
                RMSE_pct=float(np.sqrt(np.mean((p - o) ** 2)) / TILE_KM2 * 100), n=int(len(o)))

res = {}
for ds in ["floodplanet", "ufo"]:
    d = json.load(open(f"{OUT}/waterfrac_{ds}.json")); res[ds] = {}
    for m in MODELS:
        if m not in d: continue
        obs = np.array([t["gt"] for t in d[m]]) * TILE_KM2
        pred = np.array([t["pred"] for t in d[m]]) * TILE_KM2
        regs = [t["region"] for t in d[m]]; names = [t["name"] for t in d[m]]
        full = metrics(obs, pred)
        resid = pred - obs; s = resid.std()
        z = (resid - resid.mean()) / s if s > 0 else np.zeros_like(resid)
        keep = np.abs(z) <= Z
        filt = metrics(obs[keep], pred[keep])
        exreg = Counter(r for r, k in zip(regs, keep) if not k)
        res[ds][m] = dict(full=full, filtered=filt, n_excl=int((~keep).sum()),
                          excl_regions=dict(exreg),
                          excl_names=[n for n, k in zip(names, keep) if not k])
json.dump(res, open(f"{OUT}/area_full_sample.json", "w"), indent=2)

for ds in res:
    print(f"\n================= {ds.upper()} =================")
    print(f"{'Model':14s} | {'FULL R2':>7s} {'MAE%':>6s} {'RMSE%':>6s} (km2 MAE/RMSE) | "
          f"{'FILT R2':>7s} {'MAE%':>6s} {'RMSE%':>6s} | n_excl")
    for m in MODELS:
        if m not in res[ds]: continue
        a = res[ds][m]; f = a["full"]; q = a["filtered"]
        print(f"{m:14s} | {f['R2']:7.3f} {f['MAE_pct']:6.2f} {f['RMSE_pct']:6.2f} "
              f"({f['MAE']:.2f}/{f['RMSE']:.2f}) | "
              f"{q['R2']:7.3f} {q['MAE_pct']:6.2f} {q['RMSE_pct']:6.2f} | "
              f"{a['n_excl']:2d} {dict(a['excl_regions'])}")
