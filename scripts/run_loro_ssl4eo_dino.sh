#!/bin/bash
# LORO for the SSL4EO-S12 DINO (ViT-S/16) comparative baseline, 33 folds, GPUs 1-3.
set -euo pipefail
# Activate your environment here, e.g.: conda activate floodduo
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSD="${FLOODDUO_RUNS:-./runs}/loro_comparative"
Q=$SSD/ssl4eo_dino_queue.txt
mkdir -p "$SSD/ssl4eo_dino"

if [ ! -s "$Q" ]; then
  : > "$Q"
  FP_REGS=(Bolivia US-Kansas Spain Somalia Ghana Cambodia Paraguay \
            US-Nebraska US-Alabama US-Carolina Nigeria Bangladesh US-Dakota \
            Uzbekistan Colombia US-Oklahoma US-Texas Nepal US-Arkansas)
  UFO_REGS=(HTX SPS NSW KTM GIL CMO PNE QUE DKA CTO SLC BNA MID BEI)
  for reg in "${FP_REGS[@]}"; do echo "floodplanet $reg" >> "$Q"; done
  for reg in "${UFO_REGS[@]}"; do echo "ufo $reg" >> "$Q"; done
fi

worker() {
  gpu=$1
  while true; do
    fold=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$fold" ] && break
    ds=$(echo "$fold"  | cut -d' ' -f1)
    reg=$(echo "$fold" | cut -d' ' -f2)
    ds_short=$([ "$ds" = "floodplanet" ] && echo "fp" || echo "ufo")
    cfg="configs/ssl4eo_dino_${ds_short}.yaml"
    outdir="$SSD/ssl4eo_dino/$ds/$reg"
    mkdir -p "$outdir"
    echo "[gpu$gpu] ssl4eo_dino $ds/$reg"
    env CUDA_VISIBLE_DEVICES=$gpu python -m src.loro_fold \
      --dataset "$ds" --region "$reg" \
      --epochs 80 --lr 5e-4 --batch-size 8 \
      --model-config "$cfg" \
      --seed 42 \
      --out-root "$SSD/ssl4eo_dino" \
      --device cuda:0 \
      > "$outdir/train.log" 2>&1
    echo "[gpu$gpu] done ssl4eo_dino $ds/$reg"
  done
  echo "[gpu$gpu] queue empty"
}

touch "$Q.lock"
for g in 1 2 3; do
  worker $g &
done
wait
echo "ALL LORO_SSL4EO_DINO FOLDS DONE"
