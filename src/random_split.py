"""Random-split (region-sharing) counterpart to the LORO folds, for the direct
random-split vs LORO comparison requested in review.

Identical protocol to src.loro_fold (seed, 80-epoch budget, no val selection,
last-epoch eval, norm rule, arch6 model config, loss, aux wiring) EXCEPT the
split: train on the FloodBench standard `train` split and evaluate on the
standard `test` split, so regions are shared between train and test. We report
the SAME region-mean IoU as LORO (per-region confusion counts -> unweighted mean
across regions) plus the micro (overall) IoU.

Usage:
    python -m src.random_split --dataset floodplanet \
        --model-config configs/arch6_fp.yaml --epochs 80 --lr 5e-4 --device cuda:0
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.flood_dataset import FloodDataset, tile_region
from .losses import FloodLoss, aux_seg_loss
from .metrics import SegMetrics
from .train import build_model

from .paths import RUNS_ROOT as _RUNS_ROOT
RUNS_ROOT = _RUNS_ROOT / "randsplit"
NORM = {"floodplanet": "dataset", "ufo": "per_image"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["floodplanet", "ufo"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--out-root", default=str(RUNS_ROOT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--norm-mode", default=None, choices=["dataset", "per_image"])
    args = ap.parse_args()

    run_dir = Path(args.out_root) / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ONLY difference from LORO: standard FloodBench splits (regions shared)
    train_ds = FloodDataset(datasets=(args.dataset,), split="train",
                            crop=512, train=True)
    test_ds = FloodDataset(datasets=(args.dataset,), split="test", train=False)
    norm_eff = args.norm_mode or NORM[args.dataset]
    print(f"[randsplit/{args.dataset}] train {len(train_ds)} tiles, "
          f"test {len(test_ds)} tiles, norm={norm_eff}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=12, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=2, num_workers=4, pin_memory=True)

    mcfg = yaml.safe_load(Path(args.model_config).read_text())["model"]
    mcfg["norm_mode"] = norm_eff
    model = build_model(mcfg).to(device)
    loss_fn = FloodLoss()
    aux_on = getattr(model, "aux_on", False)
    aux_w = getattr(model, "aux_loss_weight", 0.0)
    if aux_on:
        print(f"[randsplit/{args.dataset}] aux ON (w {aux_w}); "
              f"disag={getattr(model,'disagreement_on',False)} "
              f"gated={getattr(model,'gated_fusion_on',False)}", flush=True)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=0.01)
    total = args.epochs * len(train_loader)
    warmup = min(300, total // 20)
    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ep_loss = 0.0
        for batch in train_loader:
            lr = (args.lr * (step + 1) / warmup if step < warmup else
                  args.lr * 0.5 * (1 + math.cos(
                      math.pi * (step - warmup) / max(1, total - warmup))))
            for g in opt.param_groups:
                g["lr"] = lr
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if aux_on:
                    logits, extras = model(x, return_aux=True)
                    loss, _ = loss_fn(logits.float(), y)
                    loss = loss + aux_w * (
                        aux_seg_loss(extras["aux_dino"].float(), y)
                        + aux_seg_loss(extras["aux_clay"].float(), y))
                else:
                    logits = model(x)
                    loss, _ = loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            step += 1
            ep_loss += loss.item()
        print(f"ep {epoch:03d} loss {ep_loss/len(train_loader):.4f} "
              f"{time.time()-t0:.0f}s", flush=True)

    # eval: micro overall + per-region (region-mean, same as LORO)
    model.eval()
    overall = SegMetrics()
    region_m = defaultdict(SegMetrics)
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            names = batch["name"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).float()
            overall.update(logits, y)
            for i in range(x.size(0)):
                r = names[i].split("_")[1]
                region_m[r].update(logits[i:i+1], y[i:i+1])
    region_iou = {r: round(m.compute()["iou"], 5) for r, m in sorted(region_m.items())}
    region_mean = float(np.mean(list(region_iou.values())))
    ov = {k: round(v, 5) for k, v in overall.compute().items()}
    out = {"dataset": args.dataset, "protocol": "random_split",
           "n_train": len(train_ds), "n_test": len(test_ds),
           "epochs": args.epochs, "lr": args.lr, "norm": norm_eff,
           "model_config": args.model_config, "n_regions": len(region_iou),
           "region_mean_iou": round(region_mean, 5),
           "overall_micro": ov, "region_iou": region_iou}
    (run_dir / "result.json").write_text(json.dumps(out, indent=2))
    torch.save({"model": model.state_dict(), "result": out}, run_dir / "last.pt")
    print("RESULT", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
