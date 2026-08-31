#!/bin/bash
# Launch the 2 Clay-only (no DINOv3) ablation runs.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .
R=outputs/runs
mkdir -p "$R"/clayonly_fp "$R"/clayonly_ufo

setsid nohup env CUDA_VISIBLE_DEVICES=0 \
  python -m src.train --config configs/clayonly_fp.yaml \
  > "$R/clayonly_fp/train.log" 2>&1 < /dev/null &
echo "launched clayonly_fp on GPU 0 (pid $!)"

setsid nohup env CUDA_VISIBLE_DEVICES=1 \
  python -m src.train --config configs/clayonly_ufo.yaml \
  > "$R/clayonly_ufo/train.log" 2>&1 < /dev/null &
echo "launched clayonly_ufo on GPU 1 (pid $!)"
