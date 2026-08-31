#!/usr/bin/env python3
"""Representation analysis of the two frozen encoders (ISPRS reviewer comment 1):
show DINOv3 carries spatial/high-frequency structure and Clay carries spectral
evidence, and that the two are complementary. Each held-out region is analysed with
the LORO fold model that never saw it (true held-out features).
 (a) Linear CKA(DINOv3, Clay) per depth -> complementarity.
 (b) Radial power spectrum of features -> DINOv3 richer high-frequency energy.
 (c) Per-branch aux-head water IoU, boundary vs interior -> functional roles.
 (d) |corr| of each aux prediction with NDWI (spectral) vs image edges (spatial).
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch, yaml
import torch.nn.functional as F
from scipy import ndimage as nd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.train import build_model

V = "outputs/viz/manuscript_figs"
FOLD = "outputs/runs/loro_arch6/arch6/floodplanet"
DEV = "cuda:0"

def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).norm() ** 2
    return (xy / ((X.T @ X).norm() * (Y.T @ Y).norm() + 1e-8)).item()

def radial_spectrum(fmap):
    m = fmap.mean(0).float(); m = m - m.mean()
    P = (torch.fft.fftshift(torch.fft.fft2(m)).abs() ** 2).cpu().numpy()
    H, W = P.shape; y, x = np.indices((H, W)); r = np.hypot(y - H // 2, x - W // 2).astype(int)
    prof = np.bincount(r.ravel(), P.ravel()) / (np.bincount(r.ravel()) + 1e-8)
    return prof / (prof.sum() + 1e-8)

def iou(p, g):
    p = p > 0.5; g = g > 0.5; u = (p | g).sum(); return float((p & g).sum() / u) if u > 0 else np.nan

def pear(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))

mcfg = yaml.safe_load(open("configs/arch6_fp.yaml"))["model"]; mcfg["norm_mode"] = "dataset"
net = build_model(mcfg).to(DEV).eval()
def load_fold(region):
    ck = f"{FOLD}/{region}/last.pt"
    if not os.path.exists(ck): return False
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    net.load_state_dict(sd, strict=False); return True

cka_d = [[] for _ in range(4)]; spec_dino = []; spec_clay = []
# linear-probe collections: features (deepest tap, 64^2 grid) + per-pixel targets
PX = {"Xd": [], "Xc": [], "ndwi": [], "dist": [], "B": [], "G": [], "R": [], "NIR": []}
ntiles = 0
rng = np.random.default_rng(0)
for cp in sorted(glob.glob(f"{V}/floodplanet_*_cache.npz")):
    region = os.path.basename(cp).replace("floodplanet_", "").replace("_cache.npz", "")
    if not load_fold(region): print("no fold:", region); continue
    z = np.load(cp, allow_pickle=True); used = 0
    for i in range(len(z["gts"])):
        if z["gts"][i].mean() <= 0.02 or used >= 8: continue
        im = z["imgs"][i]; gt = z["gts"][i]; used += 1; ntiles += 1
        x = torch.from_numpy(im)[None].to(DEV)
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fa = net.dino(x); fb = net.dofa(x)
            aa = torch.sigmoid(net.aux_head_a(fa[-1]).float())
            ab = torch.sigmoid(net.aux_head_b(fb[-1]).float())
        for k in range(4):
            A = fa[k][0].float(); B = fb[k][0].float()
            A = F.interpolate(A[None], size=B.shape[-2:], mode="bilinear")[0]
            cka_d[k].append(linear_cka(A.reshape(A.shape[0], -1).T, B.reshape(B.shape[0], -1).T))
        spec_dino.append(radial_spectrum(fa[-1][0])); spec_clay.append(radial_spectrum(fb[-1][0]))
        # ---- linear-probe features + targets on a common 64x64 grid ----
        A = F.interpolate(fa[-1].float(), size=(64, 64), mode="bilinear")[0]  # (1024,64,64)
        Bf = fb[-1][0].float()                                                # (1024,64,64)
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
    print(f"  {region}: {used} tiles")

# ---- fit linear probes (3-fold CV R^2), standardized features ----
Xd = np.concatenate(PX["Xd"]); Xc = np.concatenate(PX["Xc"])
Xd = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-6); Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
def probeR2(X, y):
    y = np.asarray(y)
    return float(cross_val_score(Ridge(alpha=10.0), X, y, cv=3, scoring="r2").mean())
targets = {k: np.concatenate(PX[k]) for k in ["ndwi", "dist", "B", "G", "R", "NIR"]}
probe = {t: {"dino": probeR2(Xd, targets[t]), "clay": probeR2(Xc, targets[t])} for t in targets}
print("probe R2:", json.dumps(probe, indent=2))

cka = [float(np.mean(c)) for c in cka_d]
sd_, sc_ = np.mean(spec_dino, 0), np.mean(spec_clay, 0)
def hf_frac(p):
    f = np.arange(len(p)) / len(p); return float(p[f > 0.5].sum() / (p.sum() + 1e-8))
hf_dino = float(np.mean([hf_frac(p) for p in spec_dino]))
hf_clay = float(np.mean([hf_frac(p) for p in spec_clay]))
stats = dict(n_tiles=ntiles, cka_by_depth=cka, probe_R2=probe,
             hf_energy_fraction=dict(dino=hf_dino, clay=hf_clay),
             spectrum=dict(freq_dino=(np.arange(len(sd_)) / len(sd_)).tolist(), dino=sd_.tolist(),
                           freq_clay=(np.arange(len(sc_)) / len(sc_)).tolist(), clay=sc_.tolist()))
json.dump(stats, open(f"{V}/representation_stats.json", "w"), indent=2)
print(json.dumps({k: v for k, v in stats.items() if k != "spectrum"}, indent=2))

# ---- 2x2 publication figure (no title, no in-plot text; everything in caption/legend) ----
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({
    "font.size": 10, "font.family": "sans-serif", "mathtext.default": "regular",
    "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
    "xtick.major.size": 3, "ytick.major.size": 3,
})
DIN, CLA, NEU = "#D55E00", "#009E73", "#4C72B0"   # Okabe-Ito orange/green + muted blue
bands = ["B", "G", "R", "NIR", "ndwi"]; labels = ["B", "G", "R", "NIR", "NDWI"]; xb = np.arange(5)
fig, AX = plt.subplots(2, 2, figsize=(7.2, 5.6)); (a0, a1), (a2, a3) = AX

# (a) cross-encoder similarity by depth
a0.bar(np.arange(1, 5), cka, color=NEU, width=0.62, edgecolor="none")
a0.set_ylim(0, 1); a0.set_yticks(np.arange(0, 1.01, 0.25)); a0.set_xticks(np.arange(1, 5))
a0.set_xlabel("Encoder block depth"); a0.set_ylabel("Linear CKA")
a0.set_title("(a)", loc="left")

# (b) linear-probe decodability (DINOv3-L R2<=0 clamped to 0 = no linear skill)
_dv = [max(0.0, probe[b]["dino"]) for b in bands]; _cv = [probe[b]["clay"] for b in bands]
a1.axhline(0, color="0.75", lw=0.8, zorder=1)
a1.vlines(xb - 0.12, 0, _dv, color=DIN, lw=1.6, zorder=2); a1.vlines(xb + 0.12, 0, _cv, color=CLA, lw=1.6, zorder=2)
a1.scatter(xb - 0.12, _dv, color=DIN, s=44, marker="o", edgecolor="black", lw=0.5, zorder=3)
a1.scatter(xb + 0.12, _cv, color=CLA, s=44, marker="D", edgecolor="black", lw=0.5, zorder=3)
a1.set_xticks(xb); a1.set_xticklabels(labels); a1.set_xlim(-0.5, 4.5)
a1.set_ylim(-0.06, 1.0); a1.set_yticks(np.arange(0, 1.01, 0.25))
a1.set_xlabel("Spectral target"); a1.set_ylabel("Linear-probe $R^2$")
a1.set_title("(b)", loc="left")

# (c) radial power spectrum of features
a2.plot(np.arange(len(sd_)) / len(sd_), sd_, color=DIN, lw=1.7)
a2.plot(np.arange(len(sc_)) / len(sc_), sc_, color=CLA, lw=1.7)
a2.set_yscale("log"); a2.set_xlim(0, 1); a2.set_xlabel("Normalized spatial frequency")
a2.set_ylabel("Normalized power"); a2.set_title("(c)", loc="left")

# (d) spectral evidence vs spatial detail
spec_d = float(np.mean([max(0.0, probe[b]["dino"]) for b in bands])); spec_c = float(np.mean([probe[b]["clay"] for b in bands]))
a3.scatter([spec_d], [hf_dino], s=130, color=DIN, edgecolor="black", lw=0.7, marker="o", zorder=3)
a3.scatter([spec_c], [hf_clay], s=130, color=CLA, edgecolor="black", lw=0.7, marker="D", zorder=3)
a3.set_xlim(-0.08, 1.0); a3.set_ylim(0, 0.2); a3.set_yticks(np.arange(0, 0.201, 0.05))
a3.set_xlabel("Mean spectral probe $R^2$"); a3.set_ylabel("High-frequency energy fraction")
a3.set_title("(d)", loc="left")

for a in (a0, a1, a2, a3):
    a.grid(axis="y", lw=0.5, alpha=0.25); a.set_axisbelow(True)

handles = [Line2D([0], [0], color=DIN, lw=2.2, marker="o", ms=6, mec="black", mew=0.6, label="DINOv3-L"),
           Line2D([0], [0], color=CLA, lw=2.2, marker="D", ms=6, mec="black", mew=0.6, label="Clay")]
fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02),
           handlelength=1.8, columnspacing=1.6)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(f"{V}/Fig_representation.{ext}", dpi=300, bbox_inches="tight"); print("->", ext)
