"""Evaluate a trained checkpoint on its own dataset's test split (full tiles).

Usage:
    python -m src.eval --run /media/.../runs/dual_ufo --dataset ufo \
        [--split test] [--ckpt best_ufo.pt] [--save-preds]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.flood_dataset import FloodDataset
from .data.ext_datasets import Sen1Floods11Dataset, KuroSiwoDataset
from .metrics import SegMetrics
from .train import build_model

EXT_DS = {
    "sen1floods11": Sen1Floods11Dataset,
    "kurosiwo":     KuroSiwoDataset,
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["floodplanet", "ufo", "sen1floods11", "kurosiwo"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--dump-d-maps", type=int, default=8,
                    help="if model has disagreement, save (D,pred,gt) for the "
                         "first N tiles for scripts/disagreement_diag.py")
    args = ap.parse_args()

    run_dir = Path(args.run)
    ckpt_name = args.ckpt or f"best_{args.dataset}.pt"
    ckpt = torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg.get("model", {})).to(args.device)
    # migrate pre-refactor checkpoints: bare ADAC ("adapters.N.x") ->
    # Sequential stack ("adapters.N.0.x")
    sd = {re.sub(r"(adapters\.\d+)\.(?!\d)", r"\1.0.", k): v
          for k, v in ckpt["model"].items()}
    # fill buffers added after checkpoint was saved (wavelengths_buf, mean, std)
    # from the freshly-constructed model so load_state_dict doesn't error
    current_sd = model.state_dict()
    for k, v in current_sd.items():
        if k not in sd:
            sd[k] = v
    model.load_state_dict(sd)
    model.eval()

    ignore_index = None
    if args.dataset in EXT_DS:
        ds = EXT_DS[args.dataset](split=args.split, train=False)
        ignore_index = ds.IGNORE_INDEX
    else:
        ds = FloodDataset(datasets=(args.dataset,), split=args.split, train=False)
    loader = DataLoader(ds, batch_size=2, num_workers=4, pin_memory=True)
    m = SegMetrics(ignore_index=ignore_index)
    per_tile = []
    pred_dir = run_dir / f"preds_{args.split}"
    if args.save_preds:
        pred_dir.mkdir(exist_ok=True)

    # disagreement-aware fusion: export D maps + per-tile mean D for diagnostics
    disag_on = getattr(model, "disagreement_on", False)
    d_dir = run_dir / f"dmaps_{args.split}"
    n_dumped = 0
    if disag_on:
        d_dir.mkdir(exist_ok=True)

    for batch in loader:
        x = batch["image"].to(args.device, non_blocking=True)
        y = batch["mask"].to(args.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if disag_on:
                logits, extras = model(x, return_aux=True)
            else:
                logits = model(x)
        logits = logits.float()
        m.update(logits, y)
        for i, name in enumerate(batch["name"]):
            mt = SegMetrics(ignore_index=ignore_index)
            mt.update(logits[i:i+1], y[i:i+1])
            rec = {"tile": name, **{k: round(v, 5)
                   for k, v in mt.compute().items()}}
            if disag_on:
                Di = extras["D"][i, 0].float()
                rec["mean_D"] = round(Di.mean().item(), 6)
                if n_dumped < args.dump_d_maps:
                    np.save(d_dir / f"{name}_D.npy", Di.cpu().numpy())
                    np.save(d_dir / f"{name}_pred.npy",
                            torch.sigmoid(logits[i, 0]).cpu().numpy())
                    np.save(d_dir / f"{name}_gt.npy",
                            y[i, 0].float().cpu().numpy())
                    n_dumped += 1
            per_tile.append(rec)
            if args.save_preds:
                np.save(pred_dir / f"{name}.npy",
                        (logits[i, 0] >= 0).cpu().numpy().astype(np.uint8))

    res = {k: round(v, 5) for k, v in m.compute().items()}
    out = {"run": str(run_dir), "ckpt": ckpt_name, "epoch": ckpt.get("epoch"),
           "dataset": args.dataset, "split": args.split,
           "overall": res, "per_tile": per_tile}
    if disag_on:
        out["mean_D"] = round(
            float(np.mean([t["mean_D"] for t in per_tile])), 6)
    out_path = run_dir / f"eval_{args.dataset}_{args.split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    # mirror test metrics into final.json so val/test live side by side
    final_path = run_dir / "final.json"
    if final_path.exists() and args.split == "test":
        final = json.loads(final_path.read_text())
        final[f"TEST_{args.dataset}"] = res
        final_path.write_text(json.dumps(final, indent=2))
    print(f"{args.dataset} {args.split} ({ckpt_name}, ep {ckpt.get('epoch')}): {res}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
