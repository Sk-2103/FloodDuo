#!/usr/bin/env python3
"""Diagnostic inference for F3 (boundary-distance error), F4 (disagreement D
calibration + region mean-D + example D maps), F5 (gate-weight maps).
FloodDuo hooked for D and per-pixel gate weights; TerraMind/U-Net for boundary."""
import json, os, glob, yaml, copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
from torch.utils.data import DataLoader
from src.train import build_model
from src.data.flood_dataset import FloodDataset
from src.models import fusion as fusion_mod

RUNS = "outputs/runs"
VIZ = "outputs/viz/manuscript_figs"
REGIONS = {
 "floodplanet": ["Bolivia","US-Kansas","Spain","Somalia","Ghana","Cambodia","Paraguay",
   "US-Nebraska","US-Alabama","US-Carolina","Nigeria","Bangladesh","US-Dakota","Uzbekistan",
   "Colombia","US-Oklahoma","US-Texas","Nepal","US-Arkansas"],
 "ufo": ["HTX","SPS","NSW","KTM","GIL","CMO","PNE","QUE","DKA","CTO","SLC","BNA","MID","BEI"]}
SPEC = {"FloodDuo": ("configs/arch6_fp.yaml", "loro_arch6/arch6"),
        "TerraMind": ("configs/terramind_fp.yaml", "loro_comparative/terramind"),
        "U-Net": ("configs/unet_fp.yaml", "loro_80ep/unet")}
DBINS = np.linspace(0, np.log(2), 11)          # disagreement bins
DIST_EDGES = np.array([0,1,2,3,5,8,12,18,28,45,70])  # px distance-to-boundary bins

# ---- gate-weight capture: monkeypatch GatedDepthFusion.forward to stash w ----
_GATE = {}
def patched_forward(self, fa, fb, D):
    target = (max(fa.shape[-2], fb.shape[-2]), max(fa.shape[-1], fb.shape[-1]))
    if fa.shape[-2:] != target: fa = F.interpolate(fa, target, mode="bilinear", align_corners=False)
    if fb.shape[-2:] != target: fb = F.interpolate(fb, target, mode="bilinear", align_corners=False)
    if D.shape[-2:] != target: D = F.interpolate(D, target, mode="bilinear", align_corners=False)
    with torch.autocast(device_type="cuda", enabled=False):
        fa, fb, D = fa.float(), fb.float(), D.float()
        a = self.proj_a(fa); b = self.proj_b(fb)
        g = self.gate(torch.cat([a, b, D], 1)) + self.gate_bias
        w = torch.softmax(g, 1)
        _GATE["w"] = w.detach()           # (B,2,h,w): [:,0]=spatial(DINO), [:,1]=spectral(Clay)
        fused = w[:, :1] * a + w[:, 1:] * b
        return self.fuse(fused)
fusion_mod.GatedDepthFusion.forward = patched_forward


def load(name, ds, reg):
    cfg, sub = SPEC[name]
    mcfg = yaml.safe_load(open(cfg))["model"]
    mcfg["norm_mode"] = "dataset" if ds == "floodplanet" else "per_image"
    m = build_model(mcfg).cuda().eval()
    m.load_state_dict(torch.load(f"{RUNS}/{sub}/{ds}/{reg}/last.pt", map_location="cpu",
                      weights_only=False)["model"])
    return m


def boundary_dist(gt):
    g = gt >= 0.5
    er = binary_erosion(g, iterations=1, border_value=1)
    bnd = g ^ er          # 1-px boundary of water regions
    bnd |= (g ^ binary_erosion(~g, iterations=1, border_value=1)) & ~g
    return distance_transform_edt(~bnd)


@torch.no_grad()
def main():
    # accumulators
    bnd_hist = {ds: {m: np.zeros((len(DIST_EDGES)-1, 2)) for m in SPEC} for ds in REGIONS}  # err,count
    dcal = {ds: np.zeros((len(DBINS)-1, 2)) for ds in REGIONS}    # err,count over D bins
    regD = {ds: {} for ds in REGIONS}                             # region -> mean D
    examples = {ds: [] for ds in REGIONS}

    for ds in REGIONS:
        for name in SPEC:
            need_fd = (name == "FloodDuo")
            for reg in REGIONS[ds]:
                m = load(name, ds, reg)
                data = FloodDataset(datasets=(ds,), split=("train","val","test"),
                                    train=False, include_regions={reg})
                Dvals = []
                for b in DataLoader(data, batch_size=1, num_workers=4):
                    x = b["image"].cuda(); gt = (b["mask"][0,0] if b["mask"].dim()==4 else b["mask"][0]).numpy()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if need_fd:
                            logits, ex = m(x, return_aux=True)
                        else:
                            logits = m(x)
                    pred = (torch.sigmoid(logits.float())[0,0].cpu().numpy() >= 0.5)
                    err = (pred != (gt >= 0.5))
                    # boundary hist
                    dist = boundary_dist(gt)
                    idx = np.clip(np.digitize(dist, DIST_EDGES) - 1, 0, len(DIST_EDGES)-2)
                    for bX in range(len(DIST_EDGES)-1):
                        msk = idx == bX
                        bnd_hist[ds][name][bX,0] += err[msk].sum(); bnd_hist[ds][name][bX,1] += msk.sum()
                    if need_fd:
                        D = ex["D"][0,0].float().cpu().numpy()          # 32x32 in [0,ln2]
                        Dup = np.kron(D, np.ones((gt.shape[0]//D.shape[0], gt.shape[1]//D.shape[1])))
                        Dup = Dup[:gt.shape[0], :gt.shape[1]]
                        di = np.clip(np.digitize(Dup, DBINS)-1, 0, len(DBINS)-2)
                        for bX in range(len(DBINS)-1):
                            msk = di == bX
                            dcal[ds][bX,0] += err[msk].sum(); dcal[ds][bX,1] += msk.sum()
                        Dvals.append(float(D.mean()))
                        # gate map (mean over depths captured: last level stashed -> recompute mean)
                        w = _GATE.get("w")  # last depth; use as representative
                        wsp = w[0,0].cpu().numpy() if w is not None else None
                        if len(examples[ds]) < 4 and float((gt>=0.5).mean()) > 0.12:
                            examples[ds].append(dict(region=reg, name=str(b["name"][0]),
                                img=b["image"][0].numpy(), gt=gt, pred=pred.astype(np.uint8),
                                D=D, wsp=wsp))
                    del x, logits
                if need_fd and Dvals:
                    regD[ds][reg] = float(np.mean(Dvals))
                del m; torch.cuda.empty_cache()
            print(f"done {ds} {name}", flush=True)

    # save (json-friendly)
    out = dict(
      boundary={ds: {m: bnd_hist[ds][m].tolist() for m in SPEC} for ds in REGIONS},
      dist_edges=DIST_EDGES.tolist(),
      dcal={ds: dcal[ds].tolist() for ds in REGIONS}, dbins=DBINS.tolist(),
      regD=regD,
      regIoU={ds: {json.load(open(f))["region"]: json.load(open(f))["iou"]
                   for f in glob.glob(f"{RUNS}/loro_arch6/arch6/{ds}/*/result.json")} for ds in REGIONS})
    json.dump(out, open(f"{VIZ}/diag_stats.json", "w"))
    for ds in REGIONS:
        np.savez_compressed(f"{VIZ}/diag_examples_{ds}.npz",
            **{f"{i}_{k}": v for i, e in enumerate(examples[ds]) for k, v in e.items()
               if k not in ("region","name")},
            meta=json.dumps([{"region": e["region"], "name": e["name"]} for e in examples[ds]]))
    print("DIAG_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
