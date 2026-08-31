"""Deployment training: pool ALL FP+UFO data (combined stratified split),
train the champion architecture (arch6: DINOv3-L + Clay + EA/ADAC/PPAdapter +
disagreement-gated fusion) with weight EMA and best-on-val checkpoint
selection. This is the RELEASE model, distinct from the LORO benchmark model.

Usage:
    python -m src.train_deploy --config configs/arch6_fp.yaml \
        --split configs/deploy_split.json --epochs 120 --lr 5e-4 \
        --norm-mode per_image --spectral-aug --device cuda:0
"""
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .data.flood_dataset import FloodDataset
from .losses import FloodLoss, aux_seg_loss
from .metrics import SegMetrics
from .train import build_model


def evaluate(model, loader, device):
    model.eval(); m = SegMetrics()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            m.update(logits.float(), y)
    model.train()
    return m.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arch6_fp.yaml")
    ap.add_argument("--split", default="configs/deploy_split.json")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--norm-mode", default="per_image", choices=["dataset", "per_image"])
    ap.add_argument("--spectral-aug", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/deploy/floodduo_ema")
    args = ap.parse_args()

    run = Path(args.out); run.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sp = json.load(open(args.split))
    train_names, val_names = set(sp["train"]), set(sp["val"])
    DSS = ("floodplanet", "ufo"); SPLITS = ("train", "val", "test")
    train_ds = FloodDataset(datasets=DSS, split=SPLITS, crop=512, train=True,
                            include_names=train_names,
                            spectral_aug=True if args.spectral_aug else None)
    val_ds = FloodDataset(datasets=DSS, split=SPLITS, train=False,
                          include_names=val_names)
    print(f"deploy: train {len(train_ds)} tiles, val {len(val_ds)} tiles, "
          f"norm={args.norm_mode}, spectral_aug={args.spectral_aug}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=12, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=True)

    mcfg = yaml.safe_load(Path(args.config).read_text())["model"]
    mcfg["norm_mode"] = args.norm_mode
    model = build_model(mcfg).to(dev)
    loss_fn = FloodLoss()
    aux_on = getattr(model, "aux_on", False)
    aux_w = getattr(model, "aux_loss_weight", 0.0)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.01)

    # EMA over trainable params only (frozen backbone excluded -> cheap)
    ema = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}
    def ema_update():
        d = args.ema_decay
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    ema[n].mul_(d).add_(p.detach().float(), alpha=1 - d)
    def swap_in_ema():
        backup = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    backup[n] = p.detach().clone(); p.copy_(ema[n].to(p.dtype))
        return backup
    def restore(backup):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in backup: p.copy_(backup[n])

    total = args.epochs * len(train_loader); warmup = min(300, total // 20); step = 0
    best_iou, best_epoch, history = -1.0, 0, []
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time(); ep_loss = 0.0
        for batch in train_loader:
            lr = (args.lr * (step + 1) / warmup if step < warmup else
                  args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
            for g in opt.param_groups: g["lr"] = lr
            x = batch["image"].to(dev, non_blocking=True)
            y = batch["mask"].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if aux_on:
                    logits, extras = model(x, return_aux=True)
                    loss, _ = loss_fn(logits.float(), y)
                    loss = loss + aux_w * (aux_seg_loss(extras["aux_dino"].float(), y)
                                           + aux_seg_loss(extras["aux_clay"].float(), y))
                else:
                    logits = model(x); loss, _ = loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                step += 1; continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step(); ema_update(); step += 1; ep_loss += loss.item()
        msg = f"ep {epoch:03d} loss {ep_loss/len(train_loader):.4f} {time.time()-t0:.0f}s"

        if epoch % args.eval_every == 0 or epoch > args.epochs - 10:
            backup = swap_in_ema()
            vm = evaluate(model, val_loader, dev)
            restore(backup)
            iou = vm["iou"]; history.append({"epoch": epoch, "val_iou": round(iou, 5)})
            msg += f" | EMA val IoU {iou:.4f}"
            if iou > best_iou:
                best_iou, best_epoch = iou, epoch
                torch.save({"model_ema": {n: ema[n] for n in ema},
                            "trainable_state": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                            "config": mcfg, "norm_mode": args.norm_mode,
                            "val_iou": iou, "epoch": epoch}, run / "best_ema.pt")
                msg += "  <= best (saved)"
        print(msg, flush=True)

    torch.save({"model_state": model.state_dict(), "ema": ema, "config": mcfg,
                "norm_mode": args.norm_mode, "epoch": args.epochs}, run / "last.pt")
    json.dump({"best_val_iou": best_iou, "best_epoch": best_epoch,
               "n_train": len(train_ds), "n_val": len(val_ds),
               "epochs": args.epochs, "norm_mode": args.norm_mode,
               "spectral_aug": bool(args.spectral_aug), "history": history},
              open(run / "deploy_result.json", "w"), indent=2)
    print(f"DONE best EMA val IoU {best_iou:.4f} @ epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
