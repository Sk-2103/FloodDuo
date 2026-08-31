#!/bin/bash
# Leave-one-region-out CV for Clay+DINOv3: 33 folds across 4 GPU workers.
# Workers pull folds from a shared queue (flock). GPUs 0/1 wait for the
# fine-DOFA runs to finish first.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .

R=outputs/runs
Q=$R/loro/queue.txt
mkdir -p "$R/loro"

if [ ! -s "$Q" ]; then
  : > "$Q"
  for reg in Bolivia US-Kansas Spain Somalia Ghana Cambodia Paraguay \
             US-Nebraska US-Alabama US-Carolina Nigeria Bangladesh US-Dakota \
             Uzbekistan Colombia US-Oklahoma US-Texas Nepal US-Arkansas; do
    echo "floodplanet $reg" >> "$Q"
  done
  for reg in HTX SPS NSW KTM GIL CMO PNE QUE DKA CTO SLC BNA MID BEI; do
    echo "ufo $reg" >> "$Q"
  done
fi

worker() {
  gpu=$1
  # GPUs 0/1: wait for fine-DOFA training to finish
  if [ "$gpu" -le 1 ]; then
    while pgrep -f "src.train --config configs/dual_finedofa" > /dev/null; do
      sleep 60
    done
  fi
  while true; do
    fold=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$fold" ] && break
    ds=$(echo "$fold" | cut -d' ' -f1)
    reg=$(echo "$fold" | cut -d' ' -f2)
    mkdir -p "$R/loro/$ds/$reg"
    echo "[gpu$gpu] starting $ds/$reg"
    env CUDA_VISIBLE_DEVICES=$gpu python -m src.loro_fold \
      --dataset "$ds" --region "$reg" --epochs 40 --device cuda:0 \
      > "$R/loro/$ds/$reg/train.log" 2>&1
    echo "[gpu$gpu] finished $ds/$reg"
  done
  echo "[gpu$gpu] queue empty, worker done"
}

touch "$Q.lock"
for g in 0 1 2 3; do
  worker $g &
done
wait
echo "ALL LORO FOLDS DONE"
