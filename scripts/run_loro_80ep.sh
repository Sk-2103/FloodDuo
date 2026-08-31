#!/bin/bash
# LORO rerun: arch5 (DINOv3+Clay+EA+LinkNet) and UNet, 80 epochs, seed=42.
# lr=5e-4 (standing decision), GPUs 1-3.
# Outputs: outputs
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .

SSD=outputs/runs/loro_80ep
Q=$SSD/queue.txt
mkdir -p "$SSD"

if [ ! -s "$Q" ]; then
  : > "$Q"
  # arch5 folds
  for reg in Bolivia US-Kansas Spain Somalia Ghana Cambodia Paraguay \
             US-Nebraska US-Alabama US-Carolina Nigeria Bangladesh US-Dakota \
             Uzbekistan Colombia US-Oklahoma US-Texas Nepal US-Arkansas; do
    echo "arch5 floodplanet $reg" >> "$Q"
  done
  for reg in HTX SPS NSW KTM GIL CMO PNE QUE DKA CTO SLC BNA MID BEI; do
    echo "arch5 ufo $reg" >> "$Q"
  done
  # UNet folds
  for reg in Bolivia US-Kansas Spain Somalia Ghana Cambodia Paraguay \
             US-Nebraska US-Alabama US-Carolina Nigeria Bangladesh US-Dakota \
             Uzbekistan Colombia US-Oklahoma US-Texas Nepal US-Arkansas; do
    echo "unet floodplanet $reg" >> "$Q"
  done
  for reg in HTX SPS NSW KTM GIL CMO PNE QUE DKA CTO SLC BNA MID BEI; do
    echo "unet ufo $reg" >> "$Q"
  done
fi

worker() {
  gpu=$1
  while true; do
    fold=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$fold" ] && break
    model=$(echo "$fold" | cut -d' ' -f1)
    ds=$(echo "$fold"    | cut -d' ' -f2)
    reg=$(echo "$fold"   | cut -d' ' -f3)

    outdir="$SSD/$model/$ds/$reg"
    mkdir -p "$outdir"

    if [ "$model" = "arch5" ]; then
      cfg="configs/arch5_ea_linknet_fp.yaml"   # norm_mode overridden inside loro_fold
    else
      cfg="configs/unet_fp.yaml"
    fi

    echo "[gpu$gpu] $model $ds/$reg"
    env CUDA_VISIBLE_DEVICES=$gpu python -m src.loro_fold \
      --dataset "$ds" --region "$reg" \
      --epochs 80 --lr 5e-4 --batch-size 8 \
      --model-config "$cfg" \
      --seed 42 \
      --out-root "$SSD/$model" \
      --device cuda:0 \
      > "$outdir/train.log" 2>&1
    echo "[gpu$gpu] done $model $ds/$reg"
  done
  echo "[gpu$gpu] queue empty"
}

touch "$Q.lock"
for g in 1 2 3; do
  worker $g &
done
wait
echo "ALL LORO_80EP FOLDS DONE"
