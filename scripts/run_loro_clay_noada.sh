#!/bin/bash
# LORO for the Clay-only (no-adapter) comparative baseline, 33 folds, GPUs 1-3.
set -euo pipefail
# Activate your environment here, e.g.: conda activate floodduo
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSD="${FLOODDUO_RUNS:-./runs}/loro_comparative"
Q=$SSD/queue_clay_noada.txt
mkdir -p "$SSD"

if [ ! -s "$Q" ]; then
  : > "$Q"
  for reg in Bolivia US-Kansas Spain Somalia Ghana Cambodia Paraguay \
             US-Nebraska US-Alabama US-Carolina Nigeria Bangladesh US-Dakota \
             Uzbekistan Colombia US-Oklahoma US-Texas Nepal US-Arkansas; do
    echo "floodplanet $reg" >> "$Q"
  done
  for reg in HTX SPS NSW KTM GIL CMO PNE QUE DKA CTO SLC BNA MID BEI; do
    echo "ufo $reg" >> "$Q"
  done
fi

worker() {
  gpu=$1
  while true; do
    fold=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$fold" ] && break
    ds=$(echo "$fold"  | cut -d' ' -f1)
    reg=$(echo "$fold" | cut -d' ' -f2)
    ds_short=$([ "$ds" = "floodplanet" ] && echo "fp" || echo "ufo")
    outdir="$SSD/clay_noada/$ds/$reg"
    mkdir -p "$outdir"
    echo "[gpu$gpu] clay_noada $ds/$reg"
    env CUDA_VISIBLE_DEVICES=$gpu python -m src.loro_fold \
      --dataset "$ds" --region "$reg" \
      --epochs 80 --lr 5e-4 --batch-size 8 \
      --model-config "configs/clay_noada_${ds_short}.yaml" \
      --seed 42 \
      --out-root "$SSD/clay_noada" \
      --device cuda:0 \
      > "$outdir/train.log" 2>&1
    echo "[gpu$gpu] done clay_noada $ds/$reg"
  done
  echo "[gpu$gpu] queue empty"
}

touch "$Q.lock"
for g in 1 2 3; do
  worker $g &
done
wait
echo "ALL LORO_CLAY_NOADA FOLDS DONE"
