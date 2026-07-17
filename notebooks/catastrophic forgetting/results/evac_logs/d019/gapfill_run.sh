#!/bin/bash
cd "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting" || exit 1
export HF_HOME=/scratch/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
SHARD="logs/gapfill_$(hostname).txt"; [ -f "$SHARD" ] || { echo "no shard"; exit 0; }
mkdir -p logs/gapfill_markers
idle=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, '{i=$1;gsub(/ /,"",i);u=$2;gsub(/ /,"",u);m=$3;gsub(/ /,"",m); if(u+0<5 && m+0<500) print i}')
for g in $idle; do
  while IFS= read -r ln; do
    rn=$(echo "$ln"|grep -oP -- '--run_name \K\S+'|head -1); [ -z "$rn" ] && continue
    [ -f "logs/gapfill_markers/$rn" ] && continue
    [ -f "results/$rn/summary.json" ] && continue
    pgrep -f "run_name $rn" >/dev/null 2>&1 && continue
    touch "logs/gapfill_markers/$rn"
    CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "$ln" > "logs/gapfill_${rn}.log" 2>&1 </dev/null &
    echo "launched $rn on GPU $g"; break
  done < "$SHARD"
done
