"""Train FloodDualSeg (or single-branch ablations) on FloodPlanet + UFO.

Usage (base conda env):
    python -m src.train --config configs/dual_base.yaml [--name run_name]
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data.flood_dataset import FloodDataset
from .data.ext_datasets import Sen1Floods11Dataset, KuroSiwoDataset
from .losses import FloodLoss, aux_seg_loss
from .metrics import SegMetrics
from .models.flood_model import FloodDualSeg
from .models.unet import UNet

RUNS_ROOT = Path("outputs/runs")
ARCHS = {"dual": FloodDualSeg, "unet": UNet}


def build_model(mcfg: dict):
    mcfg = dict(mcfg)
    arch = mcfg.pop("arch", "dual")
    return ARCHS[arch](**mcfg)


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))


EXT_DS_CLS = {"sen1floods11": Sen1Floods11Dataset, "kurosiwo": KuroSiwoDataset}


def build_ext_loaders(ds_name, tcfg, crop, workers):
    """Build train + val loaders for external (non FP/UFO) datasets."""
    cls = EXT_DS_CLS[ds_name]
    kw = dict(spectral_aug=tcfg.get("spectral_aug"),
              max_tiles=tcfg.get("max_tiles"))
    # kurosiwo has max_tiles; s11 ignores it
    if ds_name == "sen1floods11":
        kw.pop("max_tiles", None)
        kw["crop"] = crop
    tr = cls(split="train", train=True, **kw)
    va = cls(split="val",   train=False)
    tr_l = DataLoader(tr, batch_size=tcfg.get("batch_size", 8), shuffle=True,
                      num_workers=workers, pin_memory=True,
                      drop_last=True, persistent_workers=True)
    va_l = DataLoader(va, batch_size=4, num_workers=4, pin_memory=True,
                      persistent_workers=True)
    return tr_l, {ds_name: va_l}, cls.IGNORE_INDEX


@torch.no_grad()
def validate(model, loaders, device, ignore_index=None):
    model.eval()
    out = {}
    for ds, loader in loaders.items():
        m = SegMetrics(ignore_index=ignore_index)
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            m.update(logits.float(), y)
        out[ds] = m.compute()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    name = args.name or cfg.get("name", Path(args.config).stem)
    run_dir = RUNS_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg))
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True

    model = build_model(cfg.get("model", {})).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{name}] trainable {n_train/1e6:.2f}M / total {n_total/1e6:.1f}M params")
    # disagreement-aware fusion: per-branch deep-supervision aux loss
    aux_on = getattr(model, "aux_on", False)
    aux_w = getattr(model, "aux_loss_weight", 0.0)
    if aux_on:
        print(f"[{name}] aux deep-supervision ON (weight {aux_w}); "
              f"disagreement={getattr(model,'disagreement_on',False)} "
              f"cross_attn={getattr(model,'cross_attn_on',False)} "
              f"gated_fusion={getattr(model,'gated_fusion_on',False)}")

    tcfg = cfg.get("train", {})
    crop = tcfg.get("crop", 512)
    bs = tcfg.get("batch_size", 8)
    epochs = tcfg.get("epochs", 60)
    base_lr = tcfg.get("lr", 3e-4)
    wd = tcfg.get("weight_decay", 0.01)

    workers = tcfg.get("num_workers", 16)
    ext_ds = tcfg.get("ext_dataset")  # "sen1floods11" | "kurosiwo" | None
    ignore_index = None

    swir_aux = False   # overridden below for FP/UFO path
    w_swir = 0.5
    if ext_ds:
        # external dataset path: single dataset, different loader factory
        train_loader, val_loaders, ignore_index = build_ext_loaders(
            ext_ds, tcfg, crop, workers)
        datasets = (ext_ds,)
        print(f"ext dataset={ext_ds} tiles={len(train_loader.dataset)} "
              f"ignore_index={ignore_index} workers={workers}")
    else:
        # STRICT dataset independence: a run trains AND validates only on its own
        # dataset(s); never evaluate on another dataset's val split.
        datasets = tuple(tcfg.get("datasets", ("floodplanet", "ufo")))
        swir_aux = bool(tcfg.get("swir_aux", False))
        w_swir = float(tcfg.get("w_swir", 0.5))
        train_ds = FloodDataset(datasets=datasets, split="train", crop=crop,
                                spectral_aug=tcfg.get("spectral_aug"),
                                swir_aux=swir_aux)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=workers, pin_memory=True,
                                  drop_last=True, persistent_workers=True)
        val_loaders = {
            ds: DataLoader(FloodDataset(datasets=(ds,), split="val"),
                           batch_size=2, num_workers=4, pin_memory=True,
                           persistent_workers=True)
            for ds in datasets
        }
        print(f"train datasets={datasets} tiles={len(train_ds)} workers={workers}")

    loss_cfg = dict(cfg.get("loss", {}))
    if ignore_index is not None:
        loss_cfg["ignore_index"] = ignore_index
    loss_fn = FloodLoss(**loss_cfg)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=base_lr,
                            weight_decay=wd)
    total_steps = epochs * len(train_loader)
    warmup = tcfg.get("warmup_steps", min(500, total_steps // 20))

    log_path = run_dir / "log.csv"
    metric_cols = [f"{ds}_{m}" for ds in datasets
                   for m in ("iou", "f1", "prec", "rec")]
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "step", "loss", "bce", "dice", "boundary", "aux", "meanD",
             "lr", *metric_cols, "mean_iou", "sec"])

    best = {ds: 0.0 for ds in datasets}
    if len(datasets) > 1:
        best["mean"] = 0.0
    step = 0
    need_extras = swir_aux or aux_on
    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        ep_loss = 0.0
        ep_parts = {"bce": 0, "dice": 0, "boundary": 0, "aux": 0, "meanD": 0}
        for batch in train_loader:
            lr = cosine_lr(step, total_steps, warmup, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if need_extras:
                    logits, extras = model(x, return_aux=True)
                    loss, parts = loss_fn(logits.float(), y)
                    if aux_on:
                        # independent deep supervision on each branch head (no
                        # coupling between heads -> disagreement stays emergent)
                        l_aux = (aux_seg_loss(extras["aux_dino"].float(), y)
                                 + aux_seg_loss(extras["aux_clay"].float(), y))
                        loss = loss + aux_w * l_aux
                        parts["aux"] = l_aux.item()
                        if extras.get("D") is not None:
                            parts["meanD"] = extras["D"].float().mean().item()
                    if swir_aux:
                        aux = extras.get("swir")
                        valid = batch["aux_valid"].to(device).view(-1, 1, 1, 1)
                        if valid.sum() > 0 and aux is not None:
                            tgt = torch.nn.functional.interpolate(
                                batch["mndwi"].to(device)[:, None],
                                size=aux.shape[-2:], mode="bilinear",
                                align_corners=False)
                            l_swir = ((aux.float() - tgt).abs() * valid).sum() / (
                                valid.sum() * aux[0, 0].numel())
                            loss = loss + w_swir * l_swir
                            parts["swir"] = l_swir.item()
                else:
                    logits = model(x)
                    loss, parts = loss_fn(logits.float(), y)
            if not torch.isfinite(loss):       # skip a bad batch, keep lr sched
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            step += 1
            ep_loss += loss.item()
            for k in ep_parts:
                ep_parts[k] += parts.get(k, 0.0)

        n = len(train_loader)
        val = validate(model, val_loaders, device, ignore_index=ignore_index)
        miou = sum(val[ds]["iou"] for ds in datasets) / len(datasets)
        dt = time.time() - t0
        msg = " | ".join(
            f"{ds} IoU {val[ds]['iou']:.4f} F1 {val[ds]['f1']:.4f} "
            f"P {val[ds]['precision']:.4f} R {val[ds]['recall']:.4f}"
            for ds in datasets)
        aux_msg = (f" aux {ep_parts['aux']/n:.4f} D {ep_parts['meanD']/n:.4f}"
                   if aux_on else "")
        print(f"ep {epoch:03d} loss {ep_loss/n:.4f}{aux_msg} | {msg} | "
              f"mIoU {miou:.4f} lr {lr:.2e} {dt:.0f}s", flush=True)
        row = [epoch, step, round(ep_loss/n, 5), round(ep_parts['bce']/n, 5),
               round(ep_parts['dice']/n, 5), round(ep_parts['boundary']/n, 5),
               round(ep_parts['aux']/n, 5), round(ep_parts['meanD']/n, 5),
               f"{lr:.3e}"]
        for ds in datasets:
            row += [round(val[ds]['iou'], 5), round(val[ds]['f1'], 5),
                    round(val[ds]['precision'], 5), round(val[ds]['recall'], 5)]
        row += [round(miou, 5), round(dt, 1)]
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

        # dataset-specific best checkpoints (+ mean for multi-dataset runs)
        scores = {ds: val[ds]["iou"] for ds in datasets}
        if len(datasets) > 1:
            scores["mean"] = miou
        for key, score in scores.items():
            if score > best[key]:
                best[key] = score
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "selected_by": key, "score": score, "val": val,
                            "config": cfg}, run_dir / f"best_{key}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "val": val, "config": cfg}, run_dir / "last.pt")

    (run_dir / "final.json").write_text(json.dumps(
        {"best_VAL_iou": best, "epochs": epochs,
         "note": "validation IoU only; test metrics live in "
                 "eval_<dataset>_test.json (added by src.eval)"}, indent=2))
    print("done. best " +
          " ".join(f"{k} {v:.4f}" for k, v in best.items()) + f" -> {run_dir}")


if __name__ == "__main__":
    main()
