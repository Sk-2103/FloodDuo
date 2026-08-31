#!/bin/bash
# Launch the 4 per-image-normalization runs, one per GPU.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .
R=outputs/runs/normalize_results
mkdir -p "$R"/{dual_fp_pernorm,dual_ufo_pernorm,unet_fp_pernorm,unet_ufo_pernorm}

gpu=0
for cfg in dual_fp_pernorm dual_ufo_pernorm unet_fp_pernorm unet_ufo_pernorm; do
  setsid nohup env CUDA_VISIBLE_DEVICES=$gpu \
    python -m src.train --config "configs/${cfg}.yaml" \
    > "$R/$cfg/train.log" 2>&1 < /dev/null &
  echo "launched $cfg on GPU $gpu (pid $!)"
  gpu=$((gpu+1))
done
