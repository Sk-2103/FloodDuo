"""One leave-one-region-out fold of the Clay+DINOv3 model.

Pool all splits of one dataset; train on every region except --region with a
fixed epoch budget (no val-based selection -> no leakage, comparable folds);
evaluate the final model on the held-out region. Norm follows the standing
rule (FP=dataset stats, UFO=per_image).

Usage:
    python -m src.loro_fold --dataset floodplanet --region Bolivia \
        --epochs 40 --device cuda:0
"""

import os
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.flood_dataset import FloodDataset
from .losses import FloodLoss, aux_seg_loss
from .metrics import SegMetrics
from .models.flood_model import FloodDualSeg
from .train import build_model

RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "runs/loro"))
ALL_SPLITS = ("train", "val", "test")
NORM = {"floodplanet": "dataset", "ufo": "per_image"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["floodplanet", "ufo"])
    ap.add_argument("--region", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-config", default=None,
                    help="yaml whose model: block defines the architecture "
                         "(norm_mode overridden by dataset rule)")
    ap.add_argument("--spectral-aug", action="store_true",
                    help="train-time radiometric jitter (defaults: p=0.8, "
                         "gain 0.2, offset 0.02, gamma 0.25)")
    ap.add_argument("--swir-aux", action="store_true",
                    help="MNDWI distillation from paired S2 (FP only; needs "
                         "swir_aux_head in model config)")
    ap.add_argument("--w-swir", type=float, default=0.5)
    ap.add_argument("--out-root", default=str(RUNS_ROOT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patchify", action="store_true",
                    help="deterministic non-overlapping 512 grid crops (full "
                         "per-epoch coverage) instead of one random crop per tile")
    ap.add_argument("--norm-mode", default=None, choices=["dataset", "per_image"],
                    help="override the FP=dataset/UFO=per_image rule (applies to both encoders)")
    args = ap.parse_args()

    run_dir = Path(args.out_root) / args.dataset / args.region
    run_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = FloodDataset(datasets=(args.dataset,), split=ALL_SPLITS,
                            crop=512, train=True,
                            exclude_regions={args.region},
                            spectral_aug=True if args.spectral_aug else None,
                            swir_aux=args.swir_aux, patchify=args.patchify)
    test_ds = FloodDataset(datasets=(args.dataset,), split=ALL_SPLITS,
                           train=False, include_regions={args.region})
    norm_eff = args.norm_mode or NORM[args.dataset]   # --norm-mode overrides the dataset rule
    print(f"[{args.dataset}/{args.region}] train {len(train_ds)} tiles, "
          f"held-out {len(test_ds)} tiles, norm={norm_eff}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=12, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=2, num_workers=4,
                             pin_memory=True)

    if args.model_config:
        mcfg = yaml.safe_load(Path(args.model_config).read_text())["model"]
        mcfg["norm_mode"] = norm_eff
        model = build_model(mcfg).to(device)
    else:
        model = FloodDualSeg(spectral="clay", clay_indices=(5, 11, 17, 23),
                             norm_mode=norm_eff).to(device)
    loss_fn = FloodLoss()
    # disagreement-aware fusion: per-branch deep-supervision aux loss (must be
    # wired here too, else aux heads get no gradient and D is meaningless)
    aux_on = getattr(model, "aux_on", False)
    aux_w = getattr(model, "aux_loss_weight", 0.0)
    need_extras = args.swir_aux or aux_on
    if aux_on:
        print(f"[{args.region}] aux deep-supervision ON (w {aux_w}); "
              f"disag={getattr(model,'disagreement_on',False)} "
              f"gated={getattr(model,'gated_fusion_on',False)} "
              f"xattn={getattr(model,'cross_attn_on',False)}", flush=True)
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
                if need_extras:
                    logits, extras = model(x, return_aux=True)
                    loss, _ = loss_fn(logits.float(), y)
                    if aux_on:
                        loss = loss + aux_w * (
                            aux_seg_loss(extras["aux_dino"].float(), y)
                            + aux_seg_loss(extras["aux_clay"].float(), y))
                    if args.swir_aux:
                        aux = extras.get("swir")
                        valid = batch["aux_valid"].to(device).view(-1, 1, 1, 1)
                        if valid.sum() > 0 and aux is not None:
                            tgt = torch.nn.functional.interpolate(
                                batch["mndwi"].to(device)[:, None],
                                size=aux.shape[-2:], mode="bilinear",
                                align_corners=False)
                            l_swir = ((aux.float() - tgt).abs() * valid).sum() / (
                                valid.sum() * aux[0, 0].numel())
                            loss = loss + args.w_swir * l_swir
                else:
                    logits = model(x)
                    loss, _ = loss_fn(logits.float(), y)
            if not torch.isfinite(loss):       # skip a bad batch, keep lr sched
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

    model.eval()
    m = SegMetrics()
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            m.update(logits.float(), y)
    res = {k: round(v, 5) for k, v in m.compute().items()}
    out = {"dataset": args.dataset, "region": args.region,
           "n_train": len(train_ds), "n_test": len(test_ds),
           "epochs": args.epochs, "norm": norm_eff,
           "model_config": args.model_config or "default-clay-dual",
           "spectral_aug": args.spectral_aug, "swir_aux": args.swir_aux,
           "patchify": args.patchify, **res}
    (run_dir / "result.json").write_text(json.dumps(out, indent=2))
    torch.save({"model": model.state_dict(), "fold": out},
               run_dir / "last.pt")
    print("RESULT", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
