#!/bin/bash
# LORO for the DOFA-L / Prithvi-EO2 / TerraMind-L comparative baselines,
# 99 folds (3 models x 33), GPUs 1-3.
set -euo pipefail
# Activate your environment here, e.g.: conda activate floodduo
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSD="${FLOODDUO_RUNS:-./runs}/loro_comparative"
Q=$SSD/queue.txt
mkdir -p "$SSD"

if [ ! -s "$Q" ]; then
  : > "$Q"
  FP_REGS=(Bolivia US-Kansas Spain Somalia Ghana Cambodia Paraguay \
            US-Nebraska US-Alabama US-Carolina Nigeria Bangladesh US-Dakota \
            Uzbekistan Colombia US-Oklahoma US-Texas Nepal US-Arkansas)
  UFO_REGS=(HTX SPS NSW KTM GIL CMO PNE QUE DKA CTO SLC BNA MID BEI)
  for model in dofa_noada prithvi terramind; do
    for reg in "${FP_REGS[@]}"; do
      echo "floodplanet $reg $model" >> "$Q"
    done
    for reg in "${UFO_REGS[@]}"; do
      echo "ufo $reg $model" >> "$Q"
    done
  done
fi

CONFIG_FOR() {
  case "$1" in
    dofa_noada) echo "configs/dofa_noada_${2}.yaml" ;;
    prithvi)    echo "configs/prithvi_${2}.yaml" ;;
    terramind)  echo "configs/terramind_${2}.yaml" ;;
  esac
}

worker() {
  gpu=$1
  while true; do
    fold=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$fold" ] && break
    ds=$(echo "$fold"    | cut -d' ' -f1)
    reg=$(echo "$fold"   | cut -d' ' -f2)
    model=$(echo "$fold" | cut -d' ' -f3)

    ds_short=$([ "$ds" = "floodplanet" ] && echo "fp" || echo "ufo")
    cfg=$(CONFIG_FOR "$model" "$ds_short")
    outdir="$SSD/$model/$ds/$reg"
    mkdir -p "$outdir"

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
echo "ALL LORO_COMPARATIVE FOLDS DONE"
