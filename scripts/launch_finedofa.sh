#!/bin/bash
# Launch the 2 fine-grid-DOFA runs (input 2x upsampled -> 1/8-stride features).
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .
R=outputs/runs
mkdir -p "$R"/dual_finedofa_fp "$R"/dual_finedofa_ufo

setsid nohup env CUDA_VISIBLE_DEVICES=0 \
  python -m src.train --config configs/dual_finedofa_fp.yaml \
  > "$R/dual_finedofa_fp/train.log" 2>&1 < /dev/null &
echo "launched dual_finedofa_fp on GPU 0 (pid $!)"

setsid nohup env CUDA_VISIBLE_DEVICES=1 \
  python -m src.train --config configs/dual_finedofa_ufo.yaml \
  > "$R/dual_finedofa_ufo/train.log" 2>&1 < /dev/null &
echo "launched dual_finedofa_ufo on GPU 1 (pid $!)"
