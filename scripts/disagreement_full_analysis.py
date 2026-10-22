#!/usr/bin/env python3
"""Disagreement map D: error association and LORO-fit calibration, at the
32x32 cell grid. Writes disagreement_stats.json / disagreement_cells.npz.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, yaml
from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression
from src.train import build_model
from src.data.flood_dataset import FloodDataset, DATASETS

from _paths import RUNS_ROOT as RUNS
from _paths import FIGS_ROOT as V
DEV = "cuda:0"; NORM = {"floodplanet": "dataset", "ufo": "per_image"}
G = 32  # D grid
rng = np.random.default_rng(0)

@torch.no_grad()
def collect():
    data = {}
    for ds in DATASETS:
        short = "fp" if ds == "floodplanet" else "ufo"
        mcfg = yaml.safe_load(open(f"configs/arch6_{short}.yaml"))["model"]; mcfg["norm_mode"] = NORM[ds]
        Dc, Ec, Rc = [], [], []           # cell-level D, error-rate, region-id
        rmeanD, rIoU, rname = [], [], []  # region-level
        for ridx, rdir in enumerate(sorted(glob.glob(f"{RUNS}/loro_arch6/arch6/{ds}/*"))):
            region = os.path.basename(rdir)
            ck = f"{rdir}/last.pt"
            if not os.path.exists(ck): continue
            net = build_model(mcfg).to(DEV).eval()
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            net.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
            tiles = FloodDataset(datasets=(ds,), split=("train", "val", "test"),
                                 train=False, include_regions={region})
            inter = union = 0; mDs = []
            for i in range(len(tiles)):
                b = tiles[i]; x = b["image"][None].to(DEV); gt = b["mask"].numpy()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit, ex = net(x, return_aux=True)
                prob = torch.sigmoid(logit.float())[0, 0].cpu().numpy()
                D = ex["D"].float()[0, 0].cpu().numpy()                 # (32,32), in [0, ln2]
                err = (prob >= 0.5) != (gt >= 0.5)                      # full-res binary error
                errcell = F.avg_pool2d(torch.from_numpy(err.astype(np.float32))[None, None],
                                       kernel_size=err.shape[0] // G)[0, 0].numpy()  # (32,32) local error rate
                Dc.append(D.ravel()); Ec.append(errcell.ravel()); Rc.append(np.full(G * G, ridx))
                p = prob >= 0.5; g = gt >= 0.5; inter += (p & g).sum(); union += (p | g).sum()
                mDs.append(float(D.mean()))
            rmeanD.append(float(np.mean(mDs))); rIoU.append(float(inter / union) if union else 1.0); rname.append(region)
            del net; torch.cuda.empty_cache()
        data[ds] = dict(D=np.concatenate(Dc), E=np.concatenate(Ec), R=np.concatenate(Rc),
                        rmeanD=np.array(rmeanD), rIoU=np.array(rIoU), rname=rname)
        print(f"  collected {ds}: {len(data[ds]['D'])} cells, {len(rmeanD)} regions", flush=True)
    np.savez(f"{V}/disagreement_cells.npz", **{f"{ds}_{k}": v for ds in data for k, v in data[ds].items()})
    return data

def boot_ci(fn, *arrs, groups=None, n=1000):
    """Bootstrap 95% CI; if groups given, resample whole regions (cluster boot)."""
    vals = []
    if groups is not None:
        uq = np.unique(groups)
        for _ in range(n):
            pick = rng.choice(uq, size=len(uq), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in pick])
            vals.append(fn(*[a[idx] for a in arrs]))
    else:
        m = len(arrs[0])
        for _ in range(n):
            idx = rng.integers(0, m, m); vals.append(fn(*[a[idx] for a in arrs]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def ece_equal_count(p, y, K=10):
    """ECE with K equal-COUNT bins, population-weighted. Returns ece, bins list."""
    order = np.argsort(p); p, y = p[order], y[order]; N = len(p)
    ece = 0.0; bins = []
    for b in np.array_split(np.arange(N), K):
        if len(b) == 0: continue
        conf = p[b].mean(); acc = y[b].mean(); w = len(b) / N
        ece += w * abs(conf - acc); bins.append((conf, acc, len(b)))
    return ece, bins

def analyze(data):
    out = {}
    for ds in DATASETS:
        d = data[ds]; D, E, R = d["D"], d["E"], d["R"]
        # (1) error association: Spearman(D, cell error rate), region-cluster bootstrap
        rho = spearmanr(D, E).statistic
        rho_ci = boot_ci(lambda a, b: spearmanr(a, b).statistic, D, E, groups=R)
        # (2) binned error-vs-D (deciles, equal count): mean err + count + CI
        order = np.argsort(D); curve = []
        for b in np.array_split(order, 10):
            lo = boot_ci(lambda a: a.mean(), E[b]); curve.append(
                dict(D=float(D[b].mean()), err=float(E[b].mean()), n=int(len(b)), ci=lo))
        # (3) region-level mean-D vs IoU, region bootstrap
        rr = pearsonr(d["rmeanD"], d["rIoU"]).statistic
        idxg = np.arange(len(d["rmeanD"]))
        rr_ci = boot_ci(lambda i: pearsonr(d["rmeanD"][i.astype(int)], d["rIoU"][i.astype(int)]).statistic,
                        idxg.astype(float))
        # (4) LORO-fitted isotonic g(D)->error rate; ECE on held-out-region predictions
        phat = np.zeros_like(E)
        for g in np.unique(R):
            tr = R != g
            iso = IsotonicRegression(out_of_bounds="clip").fit(D[tr], E[tr])
            phat[R == g] = iso.predict(D[R == g])
        ece, ebins = ece_equal_count(phat, E, K=10)
        out[ds] = dict(spearman=rho, spearman_ci=rho_ci,
                       region_r=rr, region_r_ci=rr_ci, n_regions=len(d["rmeanD"]),
                       curve=curve, ece_fitted=ece, ece_bins=ebins)
    json.dump(out, open(f"{V}/disagreement_stats.json", "w"), indent=2, default=float)
    for ds in DATASETS:
        o = out[ds]
        print(f"\n=== {ds.upper()} ===")
        print(f"  Spearman(D, cell error) = {o['spearman']:.3f}  CI {tuple(round(x,3) for x in o['spearman_ci'])}")
        print(f"  region mean-D vs IoU r  = {o['region_r']:.3f}  CI {tuple(round(x,3) for x in o['region_r_ci'])}  (n={o['n_regions']})")
        print(f"  ECE of LORO-fitted g(D) = {o['ece_fitted']:.3f}  (10 equal-count bins, pop-weighted)")
        print("  error-vs-D (decile: D, err, n):")
        for c in o["curve"]: print(f"    D={c['D']:.3f}  err={c['err']:.3f}  n={c['n']}  CI[{c['ci'][0]:.3f},{c['ci'][1]:.3f}]")
    return out

if __name__ == "__main__":
    analyze(collect())
