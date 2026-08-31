#!/bin/bash
# Launch the 2 Clay-variant runs (full-res patch-8 spectral branch).
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .
R=outputs/runs
mkdir -p "$R"/dual_clay_fp "$R"/dual_clay_ufo

setsid nohup env CUDA_VISIBLE_DEVICES=0 \
  python -m src.train --config configs/dual_clay_fp.yaml \
  > "$R/dual_clay_fp/train.log" 2>&1 < /dev/null &
echo "launched dual_clay_fp on GPU 0 (pid $!)"

setsid nohup env CUDA_VISIBLE_DEVICES=1 \
  python -m src.train --config configs/dual_clay_ufo.yaml \
  > "$R/dual_clay_ufo/train.log" 2>&1 < /dev/null &
echo "launched dual_clay_ufo on GPU 1 (pid $!)"
