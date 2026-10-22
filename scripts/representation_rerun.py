#!/usr/bin/env python3
"""Region-grouped linear probe + common-scale high-frequency fraction, for
Figure 11. Writes representation_rerun.json.
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml
import torch.nn.functional as F
from scipy import ndimage as nd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GroupKFold
from src.train import build_model

from _paths import FIGS_ROOT as V
FOLD = os.path.join(RUNS_ROOT, "loro_arch6", "arch6", "floodplanet")
DEV = "cuda:0"
TILE_M = 1024 * 3.0          # 3072 m
LAMBDA_C = 300.0             # common ground-distance HF cutoff (m); both grids resolve it

def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).norm() ** 2
    return (xy / ((X.T @ X).norm() * (Y.T @ Y).norm() + 1e-8)).item()

def radial_power(fmap):
    m = fmap.mean(0).float(); m = m - m.mean()
    P = (torch.fft.fftshift(torch.fft.fft2(m)).abs() ** 2).cpu().numpy()
    H, W = P.shape; y, x = np.indices((H, W)); r = np.hypot(y - H // 2, x - W // 2).astype(int)
    prof = np.bincount(r.ravel(), P.ravel())      # power per radial bin r (cycles/tile)
    return prof

def hf_common(prof):
    r = np.arange(len(prof))
    r_cut = TILE_M / LAMBDA_C                      # bins above this = wavelength < LAMBDA_C
    return float(prof[r > r_cut].sum() / (prof.sum() + 1e-8))

def hf_native(prof):                              # original normalized>0.5 definition
    f = np.arange(len(prof)) / len(prof)
    return float(prof[f > 0.5].sum() / (prof.sum() + 1e-8))

mcfg = yaml.safe_load(open("configs/arch6_fp.yaml"))["model"]; mcfg["norm_mode"] = "dataset"
net = build_model(mcfg).to(DEV).eval()
def load_fold(region):
    ck = f"{FOLD}/{region}/last.pt"
    if not os.path.exists(ck): return False
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    net.load_state_dict(sd, strict=False); return True

cka_d = [[] for _ in range(4)]
hf_dino_c, hf_clay_c, hf_dino_n, hf_clay_n = [], [], [], []
PX = {k: [] for k in ["Xd", "Xc", "ndwi", "dist", "B", "G", "R", "NIR", "grp"]}
rng = np.random.default_rng(0); ntiles = 0
for ridx, cp in enumerate(sorted(glob.glob(f"{V}/floodplanet_*_cache.npz"))):
    region = os.path.basename(cp).replace("floodplanet_", "").replace("_cache.npz", "")
    if not load_fold(region): print("no fold:", region, flush=True); continue
    z = np.load(cp, allow_pickle=True); used = 0
    for i in range(len(z["gts"])):
        if z["gts"][i].mean() <= 0.02 or used >= 8: continue
        im = z["imgs"][i]; gt = z["gts"][i]; used += 1; ntiles += 1
        x = torch.from_numpy(im)[None].to(DEV)
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fa = net.dino(x); fb = net.dofa(x)
        for k in range(4):
            A = fa[k][0].float(); B = fb[k][0].float()
            A = F.interpolate(A[None], size=B.shape[-2:], mode="bilinear")[0]
            cka_d[k].append(linear_cka(A.reshape(A.shape[0], -1).T, B.reshape(B.shape[0], -1).T))
        pa = radial_power(fa[-1][0]); pb = radial_power(fb[-1][0])
        hf_dino_c.append(hf_common(pa)); hf_clay_c.append(hf_common(pb))
        hf_dino_n.append(hf_native(pa)); hf_clay_n.append(hf_native(pb))
        A = F.interpolate(fa[-1].float(), size=(64, 64), mode="bilinear")[0]
        Bf = fb[-1][0].float()
        Xd = A.reshape(1024, -1).T.cpu().numpy(); Xc = Bf.reshape(1024, -1).T.cpu().numpy()
        img64 = F.interpolate(torch.from_numpy(im)[None].float(), size=(64, 64), mode="bilinear")[0].numpy()
        Bb, Gg, Rr, NN = img64[0], img64[1], img64[2], img64[3]
        ndwi = ((Gg - NN) / (Gg + NN + 1e-6)).ravel()
        g64 = F.interpolate(torch.from_numpy((gt > 0.5).astype(np.float32))[None, None], size=(64, 64))[0, 0].numpy()
        dist = (nd.distance_transform_edt(g64 < 0.5) - nd.distance_transform_edt(g64 >= 0.5)).ravel()
        idx = rng.choice(Xd.shape[0], size=min(300, Xd.shape[0]), replace=False)
        PX["Xd"].append(Xd[idx]); PX["Xc"].append(Xc[idx])
        PX["ndwi"].append(ndwi[idx]); PX["dist"].append(dist[idx])
        PX["B"].append(Bb.ravel()[idx]); PX["G"].append(Gg.ravel()[idx])
        PX["R"].append(Rr.ravel()[idx]); PX["NIR"].append(NN.ravel()[idx])
        PX["grp"].append(np.full(len(idx), ridx))
    print(f"  {region}: {used} tiles", flush=True)

Xd = np.concatenate(PX["Xd"]); Xc = np.concatenate(PX["Xc"]); grp = np.concatenate(PX["grp"])
Xd = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-6); Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
ngroups = len(np.unique(grp)); nsplit = min(5, ngroups)
def probeR2_grouped(X, y):
    return float(cross_val_score(Ridge(alpha=10.0), X, y, groups=grp,
                                 cv=GroupKFold(n_splits=nsplit), scoring="r2").mean())
targets = {k: np.concatenate(PX[k]) for k in ["ndwi", "dist", "B", "G", "R", "NIR"]}
probe = {t: {"dino": probeR2_grouped(Xd, targets[t]), "clay": probeR2_grouped(Xc, targets[t])} for t in targets}

out = dict(n_tiles=ntiles, n_regions=ngroups, group_splits=nsplit,
           cka_by_depth=[float(np.mean(c)) for c in cka_d],
           probe_R2_grouped=probe,
           lambda_c_m=LAMBDA_C,
           hf_common=dict(dino=float(np.mean(hf_dino_c)), clay=float(np.mean(hf_clay_c))),
           hf_native=dict(dino=float(np.mean(hf_dino_n)), clay=float(np.mean(hf_clay_n))))
json.dump(out, open(f"{V}/representation_rerun.json", "w"), indent=2)
print("\n=== REPRESENTATION RERUN (region-grouped probe, common-scale HF) ===")
print("CKA by depth:", [round(x, 3) for x in out["cka_by_depth"]])
print(f"HF fraction (<{LAMBDA_C:.0f} m, common scale): DINOv3 {out['hf_common']['dino']:.3f}  Clay {out['hf_common']['clay']:.3f}")
print(f"HF fraction (native norm>0.5, old):          DINOv3 {out['hf_native']['dino']:.3f}  Clay {out['hf_native']['clay']:.3f}")
print("Region-grouped probe R2 (GroupKFold):")
for t in targets:
    print(f"   {t:5s}  dino {probe[t]['dino']:+.3f}   clay {probe[t]['clay']:+.3f}")
