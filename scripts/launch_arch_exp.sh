#!/bin/bash
# Launch architecture experiments 1 (UPerNet) and 2 (LinkNet) on GPUs 2/3.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .
R=outputs/runs
mkdir -p "$R"/arch1_upernet_fp "$R"/arch2_linknet_fp

setsid nohup env CUDA_VISIBLE_DEVICES=2 \
  python -m src.train --config configs/arch1_upernet_fp.yaml \
  > "$R/arch1_upernet_fp/train.log" 2>&1 < /dev/null &
echo "launched arch1_upernet_fp on GPU 2 (pid $!)"

setsid nohup env CUDA_VISIBLE_DEVICES=3 \
  python -m src.train --config configs/arch2_linknet_fp.yaml \
  > "$R/arch2_linknet_fp/train.log" 2>&1 < /dev/null &
echo "launched arch2_linknet_fp on GPU 3 (pid $!)"
