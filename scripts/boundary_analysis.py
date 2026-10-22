#!/usr/bin/env python3
"""Boundary-resolved error analysis: signed-distance omission/commission,
boundary-F1 at tau=3/6/9/12m, narrow-feature recall. Writes boundary_stats.json.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml
from scipy import ndimage as nd
from src.train import build_model
from src.data.flood_dataset import FloodDataset, DATASETS

from _paths import RUNS_ROOT as RUNS
from _paths import FIGS_ROOT as V
DEV = "cuda:0"; NORM = {"floodplanet": "dataset", "ufo": "per_image"}; GSD = 3.0
MODELS = {"FloodDuo": ("loro_arch6/arch6", "configs/arch6_{s}.yaml"),
          "U-Net":    ("loro_80ep/unet", "configs/unet_{s}.yaml"),
          "TerraMind":("loro_comparative/terramind", "configs/terramind_{s}.yaml")}
TAUS = [3, 6, 9, 12]                      # metres
EDGES = np.arange(-30, 30.1, 3)           # signed-distance bins (m); neg=inside water

def boundary(mask):  # 1-px inner boundary
    m = mask > 0.5
    return m ^ nd.binary_erosion(m)

def bf_score(pred, gt, tau_px):
    gb, pb = boundary(gt), boundary(pred)
    if gb.sum() == 0 and pb.sum() == 0: return 1.0
    if gb.sum() == 0 or pb.sum() == 0: return 0.0
    dgt = nd.distance_transform_edt(~gb); dpr = nd.distance_transform_edt(~pb)
    prec = (dgt[pb] <= tau_px).mean(); rec = (dpr[gb] <= tau_px).mean()
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

@torch.no_grad()
def run():
    acc = {m: {ds: dict(fn_c=np.zeros(len(EDGES)-1), fn_t=np.zeros(len(EDGES)-1),
                        fp_c=np.zeros(len(EDGES)-1), fp_t=np.zeros(len(EDGES)-1),
                        bf={t: [] for t in TAUS}, nar=[], regs=[])
               for ds in DATASETS} for m in MODELS}
    for ds in DATASETS:
        short = "fp" if ds == "floodplanet" else "ufo"
        for ri, rdir in enumerate(sorted(glob.glob(f"{RUNS}/loro_arch6/arch6/{ds}/*"))):
            region = os.path.basename(rdir)
            tiles = FloodDataset(datasets=(ds,), split=("train", "val", "test"), train=False, include_regions={region})
            for mname, (rundir, cfgt) in MODELS.items():
                ck = f"{RUNS}/{rundir}/{ds}/{region}/last.pt"
                if not os.path.exists(ck): continue
                mcfg = yaml.safe_load(open(cfgt.format(s=short)))["model"]; mcfg["norm_mode"] = NORM[ds]
                net = build_model(mcfg).to(DEV).eval()
                sd = torch.load(ck, map_location="cpu", weights_only=False)
                net.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
                bf_tile = {t: [] for t in TAUS}; nar_tp = nar_tot = 0
                for i in range(len(tiles)):
                    b = tiles[i]; x = b["image"][None].to(DEV); gt = b["mask"].numpy() > 0.5
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = torch.sigmoid(net(x).float())[0, 0].cpu().numpy() >= 0.5
                    A = acc[mname][ds]
                    din = nd.distance_transform_edt(gt); dout = nd.distance_transform_edt(~gt)
                    signed = (dout - din) * GSD                   # metres, neg inside
                    err = pred != gt
                    # FN inside water (gt true), FP outside (gt false), binned by signed dist
                    idx = np.digitize(signed, EDGES) - 1
                    for bb in range(len(EDGES) - 1):
                        cell = idx == bb
                        gin = cell & gt; gout = cell & (~gt)
                        A["fn_t"][bb] += gin.sum(); A["fn_c"][bb] += (gin & ~pred).sum()
                        A["fp_t"][bb] += gout.sum(); A["fp_c"][bb] += (gout & pred).sum()
                    for t in TAUS: bf_tile[t].append(bf_score(pred, gt, t / GSD))
                    narrow = gt & (din <= 1)                      # width <= 2 px (6 m)
                    nar_tp += (narrow & pred).sum(); nar_tot += narrow.sum()
                    del err
                for t in TAUS: acc[mname][ds]["bf"][t].append(float(np.mean(bf_tile[t])))
                acc[mname][ds]["nar"].append(float(nar_tp / nar_tot) if nar_tot else np.nan)
                acc[mname][ds]["regs"].append(region)
                del net; torch.cuda.empty_cache()
            print(f"  {ds}/{region} done", flush=True)
    # summarize
    rng = np.random.default_rng(0)
    def ci(v):
        v = np.array([x for x in v if np.isfinite(x)])
        bs = [rng.choice(v, len(v), replace=True).mean() for _ in range(1000)]
        return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    out = {}
    for m in MODELS:
        out[m] = {}
        for ds in DATASETS:
            A = acc[m][ds]
            out[m][ds] = dict(
                bf={t: ci(A["bf"][t]) for t in TAUS},
                narrow_recall=ci(A["nar"]),
                fn_rate=(A["fn_c"] / np.maximum(A["fn_t"], 1)).tolist(),
                fp_rate=(A["fp_c"] / np.maximum(A["fp_t"], 1)).tolist(),
                fn_t=A["fn_t"].tolist(), fp_t=A["fp_t"].tolist())
    out["_edges"] = EDGES.tolist(); out["_taus"] = TAUS
    json.dump(out, open(f"{V}/boundary_stats.json", "w"), indent=2, default=float)
    for ds in DATASETS:
        print(f"\n=== {ds.upper()} ===")
        print("  Boundary-F1 (BF) at tau (m):   " + "  ".join(f"{t}m" for t in TAUS))
        for m in MODELS:
            print(f"    {m:10s} " + "  ".join(f"{out[m][ds]['bf'][t][0]:.3f}" for t in TAUS))
        print("  Narrow-feature recall (width<=2px):")
        for m in MODELS:
            r = out[m][ds]["narrow_recall"]; print(f"    {m:10s} {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]")
    return out

if __name__ == "__main__":
    run()
