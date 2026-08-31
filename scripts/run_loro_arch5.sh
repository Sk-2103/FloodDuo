#!/bin/bash
# LORO CV with the arch5 champion recipe (EA + ADAC+PPAd + LinkNet, no stem).
# 33 folds across 4 GPU workers; results -> runs/loro_arch5/.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .

OUT=outputs/runs/loro_arch5
CFG=configs/arch5_ea_linknet_fp.yaml
Q=$OUT/queue.txt
mkdir -p "$OUT"

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
  while true; do
    fold=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$fold" ] && break
    ds=$(echo "$fold" | cut -d' ' -f1)
    reg=$(echo "$fold" | cut -d' ' -f2)
    mkdir -p "$OUT/$ds/$reg"
    echo "[gpu$gpu] starting $ds/$reg"
    env CUDA_VISIBLE_DEVICES=$gpu python -m src.loro_fold \
      --dataset "$ds" --region "$reg" --epochs 40 --device cuda:0 \
      --model-config "$CFG" --out-root "$OUT" \
      > "$OUT/$ds/$reg/train.log" 2>&1
    echo "[gpu$gpu] finished $ds/$reg"
  done
  echo "[gpu$gpu] queue empty, worker done"
}

touch "$Q.lock"
for g in 0 1 2 3; do
  worker $g &
done
wait
echo "ALL ARCH5 LORO FOLDS DONE"
