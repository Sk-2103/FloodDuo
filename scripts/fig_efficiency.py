#!/usr/bin/env python3
"""F6: Accuracy-efficiency. Measures trainable/total params + inference latency
(per 1024x1024 tile, bf16) for each model, plots vs LORO mean IoU.
Caches measurements so re-plotting is instant."""
import json, os, time, glob, yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.train import build_model

RUNS = "outputs/runs"
VIZ = "outputs/viz/manuscript_figs"
CACHE = f"{VIZ}/efficiency_train.json"
SPEC = [("FloodDuo", "configs/arch6_fp.yaml", "loro_arch6/arch6", "#1b9e77"),
        ("TerraMind", "configs/terramind_fp.yaml", "loro_comparative/terramind", "#7570b3"),
        ("Prithvi", "configs/prithvi_fp.yaml", "loro_comparative/prithvi", "#7570b3"),
        ("DOFA", "configs/dofa_noada_fp.yaml", "loro_comparative/dofa_noada", "#7570b3"),
        ("Clay", "configs/clay_noada_fp.yaml", "loro_comparative/clay_noada", "#7570b3"),
        ("CROMA", "configs/croma_fp.yaml", "loro_comparative/croma", "#7570b3"),
        ("SSL4EO-DINO", "configs/ssl4eo_dino_fp.yaml", "loro_comparative/ssl4eo_dino", "#7570b3"),
        ("U-Net", "configs/unet_fp.yaml", "loro_80ep/unet", "#d95f02")]


def loro_mean(sub):
    v = {}
    for ds in ["floodplanet", "ufo"]:
        xs = [json.load(open(f))["iou"] for f in glob.glob(f"{RUNS}/{sub}/{ds}/*/result.json")]
        v[ds] = float(np.mean(xs)) if xs else None
    return v


def measure():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    from torch.utils.flop_counter import FlopCounterMode
    res = {}
    for name, cfg, sub, _ in SPEC:
        mcfg = yaml.safe_load(open(cfg))["model"]; mcfg["norm_mode"] = "dataset"
        mcfg["grad_ckpt"] = False  # one clean fwd+bwd, no checkpoint recompute
        try:
            m = build_model(mcfg).cuda()
        except TypeError:  # e.g. U-Net does not accept grad_ckpt
            mcfg.pop("grad_ckpt", None); m = build_model(mcfg).cuda()
        ntr = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
        ntot = sum(p.numel() for p in m.parameters()) / 1e6
        # inference latency (per 1024^2 tile, bf16, no grad) — kept for reference
        m.eval()
        with torch.no_grad():
            xi = torch.rand(1, 4, 1024, 1024).cuda() * 0.4
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.bfloat16): m(xi)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(6):
                with torch.autocast("cuda", dtype=torch.bfloat16): m(xi)
            torch.cuda.synchronize(); ms = (time.time() - t0) / 6 * 1000
            del xi
        torch.cuda.empty_cache()
        # TRAINING-step GFLOPs: forward + backward at the 512^2 training res.
        # Backbones are frozen (no weight grads), so this is the real training
        # cost the user pays per step — fairer than full forward at inference.
        m.train(); gflops = float("nan")
        for res_px in (512, 384, 256):
            try:
                m.zero_grad(set_to_none=True)
                xt = torch.rand(1, 4, res_px, res_px).cuda() * 0.4
                fcm = FlopCounterMode(display=False)
                with fcm:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        try: out = m(xt, return_aux=True)
                        except TypeError: out = m(xt)
                    main = out[0] if isinstance(out, tuple) else out
                    extras = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
                    ts = [main] + ([v for v in extras.values()
                                    if torch.is_tensor(v) and v.requires_grad]
                                   if isinstance(extras, dict) else [])
                    loss = sum(t.float().pow(2).mean() for t in ts)
                    loss.backward()
                gflops = fcm.get_total_flops() / 1e9 * (512.0 / res_px) ** 2
                del xt, out, loss
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower(): raise
                print(f"  {name}: OOM @ {res_px}^2, retrying smaller"); torch.cuda.empty_cache()
        res[name] = dict(trainable_M=ntr, total_M=ntot, ms=ms, gflops=gflops, loro=loro_mean(sub))
        print(f"{name}: train {ntr:.1f}M total {ntot:.0f}M {ms:.0f}ms "
              f"{gflops:.0f}GFLOP/step FP {res[name]['loro']['floodplanet']:.3f}")
        del m; torch.cuda.empty_cache()
    json.dump(res, open(CACHE, "w"), indent=2); return res


def main():
    r = measure()
    plt.rcParams.update({"font.size": 15})
    # per-model label offset (points) to de-clutter; ha picked from x-sign
    OFF = {"FloodDuo": (10, 4), "Clay": (-10, -16), "TerraMind": (10, 6),
           "DOFA": (10, -14), "Prithvi": (10, 8), "CROMA": (10, 4),
           "SSL4EO-DINO": (10, -14), "U-Net": (10, 8)}
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.3))
    for ax, ds, ttl in [(axes[0], "floodplanet", "(a) FloodPlanet"), (axes[1], "ufo", "(b) UFO")]:
        for name, cfg, sub, c in SPEC:
            d = r[name]; y = d["loro"][ds]
            ax.scatter(d["gflops"], y, s=60 + d["total_M"]/3, c=c, edgecolor="k", lw=0.8,
                       alpha=0.85, zorder=5)
            ox, oy = OFF.get(name, (10, 5))
            ax.annotate(name, (d["gflops"], y), fontsize=13.5, xytext=(ox, oy),
                        textcoords="offset points", ha=("right" if ox < 0 else "left"),
                        zorder=6)
        ax.set_xscale("log"); ax.set_xlim(100, 1e5)
        ax.set_xlabel("Training compute (GFLOPs / step, 512² tile, log)", fontsize=16)
        if ax is axes[0]:
            ax.set_ylabel("LORO region-mean IoU", fontsize=16)
        ax.tick_params(labelsize=13.5)
        ax.grid(ls=":", alpha=0.4); ax.set_axisbelow(True)
        ax.text(0.975, 0.05, ttl, transform=ax.transAxes, ha="right", fontsize=17,
                fontweight="bold", bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=3))
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([0],[0],marker='o',ls='',mfc="#1b9e77",mec='k',ms=11,label="FloodDuo (dual, ours)"),
                            Line2D([0],[0],marker='o',ls='',mfc="#7570b3",mec='k',ms=11,label="single-encoder FM"),
                            Line2D([0],[0],marker='o',ls='',mfc="#d95f02",mec='k',ms=11,label="U-Net (scratch)")],
                   loc="upper left", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{VIZ}/Fig_efficiency.{ext}"; fig.savefig(p, dpi=200, bbox_inches="tight"); print("->", p)


if __name__ == "__main__":
    main()
