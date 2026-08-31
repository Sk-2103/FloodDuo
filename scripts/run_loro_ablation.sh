#!/bin/bash
# Per-component ablation LORO (ISPRS reviewer comment 2): additive ladder
#   Base(no adapters) -> +EA -> +EA+ADAC ; then arch6_v0(+PPA) & arch6(+gate) already exist.
# 3 models x (19 FP + 14 UFO) = 99 folds. 80 ep, lr 5e-4, seed 42, bs 8. GPUs 1-3.
set -uo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd .
SSD=outputs/runs/loro_ablation
Q=$SSD/queue.txt; mkdir -p "$SSD"
FP_REGS=(Bolivia US-Kansas Spain Somalia Ghana Cambodia Paraguay US-Nebraska \
  US-Alabama US-Carolina Nigeria Bangladesh US-Dakota Uzbekistan Colombia \
  US-Oklahoma US-Texas Nepal US-Arkansas)
UFO_REGS=(BEI BNA CMO CTO DKA GIL HTX KTM MID NSW PNE QUE SLC SPS)
if [ ! -s "$Q" ]; then
  : > "$Q"
  for model in abl_base abl_ea abl_ea_adac; do
    for reg in "${FP_REGS[@]}"; do echo "$model floodplanet $reg" >> "$Q"; done
    for reg in "${UFO_REGS[@]}"; do echo "$model ufo $reg" >> "$Q"; done
  done
fi
touch "$Q.lock"
worker() {
  gpu=$1
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
}
for g in 1 2 3; do worker $g & done
wait
echo "ALL LORO_ABLATION FOLDS DONE"
