"""D's blind spot: error mass D does/doesn't flag, plus D maps for the Figure 16
failure tiles. Writes failure_disagreement.json and failure_Dmaps.npz.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, yaml
from src.train import build_model
from src.data.flood_dataset import FloodDataset, DATASETS

from _paths import RUNS_ROOT as RUNS
from _paths import FIGS_ROOT as V
DEV = "cuda:0"; NORM = {"floodplanet": "dataset", "ufo": "per_image"}; G = 32

@torch.no_grad()
def forward(net, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit, ex = net(x, return_aux=True)
    prob = torch.sigmoid(logit.float())[0, 0].cpu().numpy()
    D = ex["D"].float()[0, 0].cpu().numpy()                     # (32,32), [0,ln2]
    ad = torch.sigmoid(ex["aux_dino"].float())[0, 0].cpu().numpy()
    ac = torch.sigmoid(ex["aux_clay"].float())[0, 0].cpu().numpy()
    # resize aux to 32x32 (D grid) if needed
    def to32(a):
        if a.shape == (G, G): return a
        return F.interpolate(torch.from_numpy(a)[None, None], size=(G, G), mode="bilinear")[0, 0].numpy()
    return prob, D, to32(ad), to32(ac)

def cells(err_full):
    t = torch.from_numpy(err_full.astype(np.float32))[None, None]
    return F.avg_pool2d(t, kernel_size=err_full.shape[0] // G)[0, 0].numpy()   # (32,32) local error rate

@torch.no_grad()
def run():
    Dall, Eall, agree = [], [], []            # cell D, cell error rate, branch-agree(bool)
    tmeanD, tIoU = [], []
    for ds in DATASETS:
        short = "fp" if ds == "floodplanet" else "ufo"
        mcfg = yaml.safe_load(open(f"configs/arch6_{short}.yaml"))["model"]; mcfg["norm_mode"] = NORM[ds]
        for rdir in sorted(glob.glob(f"{RUNS}/loro_arch6/arch6/{ds}/*")):
            region = os.path.basename(rdir); ck = f"{rdir}/last.pt"
            if not os.path.exists(ck): continue
            net = build_model(mcfg).to(DEV).eval()
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            net.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
            tiles = FloodDataset(datasets=(ds,), split=("train", "val", "test"), train=False, include_regions={region})
            for i in range(len(tiles)):
                b = tiles[i]; x = b["image"][None].to(DEV); gt = b["mask"].numpy() > 0.5
                prob, D, ad, ac = forward(net, x)
                pred = prob >= 0.5
                err = cells((pred != gt))                       # (32,32) local error rate
                ag = (ad >= 0.5) == (ac >= 0.5)                 # branches agree on class
                Dall.append(D.ravel()); Eall.append(err.ravel()); agree.append(ag.ravel())
                inter = (pred & gt).sum(); union = (pred | gt).sum()
                tmeanD.append(float(D.mean())); tIoU.append(float(inter / union) if union else 1.0)
            del net; torch.cuda.empty_cache()
        print(f"  {ds} done", flush=True)
    D = np.concatenate(Dall); E = np.concatenate(Eall); A = np.concatenate(agree)
    tmeanD = np.array(tmeanD); tIoU = np.array(tIoU)
    med = np.median(D)
    errmass = E.sum()
    out = dict(
        n_cells=int(len(D)), n_tiles=int(len(tIoU)), D_median=float(med),
        # error-rate in low vs high D halves
        err_rate_lowD=float(E[D <= med].mean()), err_rate_highD=float(E[D > med].mean()),
        # fraction of total error MASS that falls in the low-D (agreement) half -> D's blind spot
        errmass_frac_lowD=float(E[D <= med].sum() / errmass),
        # fraction of error mass in cells where the two branches AGREE on the class
        errmass_frac_branches_agree=float(E[A].sum() / errmass),
        frac_cells_branches_agree=float(A.mean()),
        # worst-tile check: mean D on worst-IoU vs best-IoU tertiles
        meanD_worst_tertile=float(tmeanD[tIoU <= np.quantile(tIoU, 1/3)].mean()),
        meanD_best_tertile=float(tmeanD[tIoU >= np.quantile(tIoU, 2/3)].mean()),
        tile_D_IoU_pearson=float(np.corrcoef(tmeanD, tIoU)[0, 1]),
    )
    json.dump(out, open(f"{V}/failure_disagreement.json", "w"), indent=2)
    print("\n=== FAILURE / DISAGREEMENT (comment 10) ===")
    for k, v in out.items(): print(f"  {k}: {v}")
    return out

# ---- Part B: D maps for the three Fig 16 failure tiles ----
def pick(base, mode):
    z = np.load(f"{V}/{base}_cache.npz", allow_pickle=True); best = None
    for i in range(len(z["gts"])):
        g = z["gts"][i] > 0.5; p = z["pred_FloodDuo"][i] > 0.5
        if g.mean() <= 0.015: continue
        u = (g | p).sum(); iou = (g & p).sum() / u if u else 1
        fp = ((~g & p).sum() / p.sum()) if p.sum() else 0
        key = fp if mode == "fp" else -iou
        if best is None or key > best[0]: best = (key, i, z["imgs"][i], z["gts"][i])
    return best

@torch.no_grad()
def figtiles():
    specs = [("floodplanet", "Ghana", "floodplanet_Ghana", "iou"),
             ("ufo", "HTX", "ufo_HTX", "iou"),
             ("floodplanet", "Nigeria", "floodplanet_Nigeria", "fp")]
    saved = {}
    for ds, region, base, mode in specs:
        short = "fp" if ds == "floodplanet" else "ufo"
        mcfg = yaml.safe_load(open(f"configs/arch6_{short}.yaml"))["model"]; mcfg["norm_mode"] = NORM[ds]
        net = build_model(mcfg).to(DEV).eval()
        sd = torch.load(f"{RUNS}/loro_arch6/arch6/{ds}/{region}/last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
        _, idx, im, gt = pick(base, mode)
        # normalize/prepare exactly as dataset: the cache imgs are raw reflectance (as model expects)
        x = torch.from_numpy(im)[None].float().to(DEV)
        prob, D, ad, ac = forward(net, x)
        saved[f"{base}_D"] = D; saved[f"{base}_idx"] = idx
        print(f"{base} idx={idx} meanD={D.mean():.3f}", flush=True)
        del net; torch.cuda.empty_cache()
    np.savez(f"{V}/failure_Dmaps.npz", **saved)

if __name__ == "__main__":
    run()
    figtiles()
