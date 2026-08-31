#!/bin/bash
# Add ONE extra worker on GPU 0 to the already-running loro_ablation pool.
# Draws from the same queue.txt under the same flock lock as the gpu1-3 workers,
# so it never double-runs a fold and never disturbs the in-flight ones.
set -uo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .
SSD=outputs/runs/loro_ablation
Q=$SSD/queue.txt
gpu=0
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
