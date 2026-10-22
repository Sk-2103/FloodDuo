#!/bin/bash
# LORO: FloodDuo (arch6) vs its scalar-gate ablation (arch6_v0), FloodPlanet.
# 38 folds. 80 ep, lr=5e-4, seed=42, bs=8, GPUs 1-3.
set -euo pipefail
# Activate your environment here, e.g.: conda activate floodduo
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSD="${FLOODDUO_RUNS:-./runs}/loro_arch6"
Q=$SSD/queue.txt
mkdir -p "$SSD"

if [ ! -s "$Q" ]; then
  : > "$Q"
  FP_REGS=(Bolivia US-Kansas Spain Somalia Ghana Cambodia Paraguay \
            US-Nebraska US-Alabama US-Carolina Nigeria Bangladesh US-Dakota \
            Uzbekistan Colombia US-Oklahoma US-Texas Nepal US-Arkansas)
  for model in arch6 arch6_v0; do
    for reg in "${FP_REGS[@]}"; do echo "$model $reg" >> "$Q"; done
  done
fi

worker() {
  gpu=$1
  # don't grab GPUs until the random-split arch6 training has finished
  while pgrep -f "src.train --config configs/arch6" > /dev/null; do sleep 60; done
  while true; do
    fold=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$fold" ] && break
    model=$(echo "$fold" | cut -d' ' -f1)
    reg=$(echo "$fold"   | cut -d' ' -f2)
    cfg="configs/${model}_fp.yaml"
    outdir="$SSD/$model/floodplanet/$reg"
    mkdir -p "$outdir"
    echo "[gpu$gpu] $model floodplanet/$reg"
    env CUDA_VISIBLE_DEVICES=$gpu python -m src.loro_fold \
      --dataset floodplanet --region "$reg" \
      --epochs 80 --lr 5e-4 --batch-size 8 \
      --model-config "$cfg" --seed 42 \
      --out-root "$SSD/$model" --device cuda:0 \
      > "$outdir/train.log" 2>&1
    echo "[gpu$gpu] done $model floodplanet/$reg"
  done
  echo "[gpu$gpu] queue empty"
}

touch "$Q.lock"
for g in 1 2 3; do worker $g & done
wait
echo "ALL LORO_ARCH6 FOLDS DONE"
