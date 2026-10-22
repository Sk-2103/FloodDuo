#!/bin/bash
# U-Net LORO baseline, 33 folds, GPUs 1-3. 40 ep, lr=3e-4, bs=8, crop=512.
# Activate your environment here, e.g.: conda activate floodduo
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

R="${FLOODDUO_RUNS:-./runs}/loro_unet"
Q=$R/queue.txt
mkdir -p "$R"

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
    ds=$(echo "$fold" | cut -d' ' -f1)
    reg=$(echo "$fold" | cut -d' ' -f2)
    mkdir -p "$R/$ds/$reg"
    echo "[gpu$gpu] $ds/$reg"
    env CUDA_VISIBLE_DEVICES=$gpu python -m src.loro_fold \
      --dataset "$ds" --region "$reg" \
      --epochs 40 --lr 3e-4 --batch-size 8 \
      --model-config configs/unet_fp.yaml \
      --out-root "$R" \
      --device cuda:0 \
      > "$R/$ds/$reg/train.log" 2>&1
    echo "[gpu$gpu] done $ds/$reg"
  done
  echo "[gpu$gpu] queue empty"
}

touch "$Q.lock"
for g in 1 2 3; do
  worker $g &
done
wait
echo "ALL UNET LORO FOLDS DONE"
