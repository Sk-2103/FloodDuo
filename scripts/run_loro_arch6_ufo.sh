#!/bin/bash
# LORO for arch6 (disagreement-gated fusion) on UFO — 14 folds.
# 80 ep, lr=5e-4, seed=42, bs=8, GPUs 1-3 (GPU 0 reserved). norm=per_image
# (forced by loro_fold). Output: loro_arch6/arch6/ufo/<region>/result.json
# (slots alongside the arch6 FP results).
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .

SSD=outputs/runs/loro_arch6
Q=$SSD/ufo_queue.txt
mkdir -p "$SSD/arch6/ufo"

if [ ! -s "$Q" ]; then
  : > "$Q"
  for reg in HTX SPS NSW KTM GIL CMO PNE QUE DKA CTO SLC BNA MID BEI; do
    echo "$reg" >> "$Q"
  done
fi

worker() {
  gpu=$1
  while true; do
    reg=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$reg" ] && break
    outdir="$SSD/arch6/ufo/$reg"
    mkdir -p "$outdir"
    echo "[gpu$gpu] arch6 ufo/$reg"
    env CUDA_VISIBLE_DEVICES=$gpu python -m src.loro_fold \
      --dataset ufo --region "$reg" \
      --epochs 80 --lr 5e-4 --batch-size 8 \
      --model-config configs/arch6_ufo.yaml --seed 42 \
      --out-root "$SSD/arch6" --device cuda:0 \
      > "$outdir/train.log" 2>&1
    echo "[gpu$gpu] done arch6 ufo/$reg"
  done
  echo "[gpu$gpu] queue empty"
}

touch "$Q.lock"
for g in 1 2 3; do worker $g & done
wait
echo "ALL ARCH6 UFO FOLDS DONE"
