#!/bin/bash
# LORO for the matched single-encoder controls (DINOv3-only, Clay-only),
# 66 folds (2 models x 33), GPUs 0-3.
set -uo pipefail
# Activate your environment here, e.g.: conda activate floodduo
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSD="${FLOODDUO_RUNS:-./runs}/loro_single_enc"
Q=$SSD/queue.txt; mkdir -p "$SSD"
FP_REGS=(Bolivia US-Kansas Spain Somalia Ghana Cambodia Paraguay US-Nebraska \
  US-Alabama US-Carolina Nigeria Bangladesh US-Dakota Uzbekistan Colombia \
  US-Oklahoma US-Texas Nepal US-Arkansas)
UFO_REGS=(BEI BNA CMO CTO DKA GIL HTX KTM MID NSW PNE QUE SLC SPS)
if [ ! -s "$Q" ]; then
  : > "$Q"
  for model in dinoonly clayonly; do
    for reg in "${FP_REGS[@]}"; do echo "$model floodplanet $reg" >> "$Q"; done
    for reg in "${UFO_REGS[@]}"; do echo "$model ufo $reg" >> "$Q"; done
  done
fi
touch "$Q.lock"
worker() {
  gpu=$1
  while true; do
    fold=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$fold" ] && break
    model=$(echo "$fold" | cut -d' ' -f1); ds=$(echo "$fold" | cut -d' ' -f2); reg=$(echo "$fold" | cut -d' ' -f3)
    short=fp; [ "$ds" = "ufo" ] && short=ufo
    cfg="configs/${model}_${short}.yaml"
    outdir="$SSD/$model/$ds/$reg"; mkdir -p "$outdir"
    echo "[gpu$gpu] $model $ds/$reg"
    env CUDA_VISIBLE_DEVICES=$gpu python -m src.loro_fold \
      --dataset "$ds" --region "$reg" --epochs 80 --lr 5e-4 --batch-size 8 \
      --model-config "$cfg" --seed 42 --out-root "$SSD/$model" --device cuda:0 \
      > "$outdir/train.log" 2>&1
    echo "[gpu$gpu] done $model $ds/$reg"
  done
  echo "[gpu$gpu] queue empty"
}
for g in 0 1 2 3; do worker $g & done
wait
echo "ALL SINGLE-ENCODER LORO FOLDS DONE"
