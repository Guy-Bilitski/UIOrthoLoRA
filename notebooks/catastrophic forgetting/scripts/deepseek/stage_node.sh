#!/bin/bash
# Stage everything a node needs for the DeepSeek run, from d001, THROTTLED (nice/ionice + bwlimit)
# so it never starves the sweep's disk/net. Run from d001. Idempotent (rsync skips up-to-date).
#   1. the DeepSeek model cache (~150 GB FP8)         2. MedMCQA + retention + WikiText datasets
#   3. the new pipeline code + medmcqa train/val json (nodes have the repo but not these new files)
#
# Usage:  bash scripts/deepseek/stage_node.sh <node>     e.g. stage_node.sh d002
set -euo pipefail
N="${1:?need node}"
WD="/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
HUB=/scratch/hf_cache/hub
DS=/scratch/hf_cache/datasets
RS=(rsync -a --info=progress2 --bwlimit=200000)   # 200 MB/s cap; bump if the sweep is idle
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

echo "[stage $N] ensuring dirs ..."
"${SSH[@]}" "ubuntu@$N" "mkdir -p $HUB $DS '$WD/scripts/deepseek' '$WD/repro/LLM-Adapters/ft-training_set' '$WD/fleet'"

echo "[stage $N] (1) model cache (~150 GB) ..."
nice -n 15 ionice -c3 "${RS[@]}" "$HUB/models--deepseek-ai--DeepSeek-V4-Flash" "ubuntu@$N:$HUB/"

echo "[stage $N] (2) datasets (medmcqa + retention + wikitext) ..."
for d in openlifescienceai___medmcqa cais___mmlu allenai___ai2_arc truthfulqa___truthful_qa \
         Salesforce___wikitext hails___mmlu_no_train openai___gsm8k lukaemon___bbh; do
  [ -d "$DS/$d" ] && nice -n 15 ionice -c3 "${RS[@]}" "$DS/$d" "ubuntu@$N:$DS/" || true
done
# hub-side dataset blobs (some loaders resolve via hub/)
for d in datasets--openlifescienceai--medmcqa datasets--cais--mmlu datasets--allenai--ai2_arc \
         datasets--truthfulqa--truthful_qa datasets--Salesforce--wikitext; do
  [ -d "$HUB/$d" ] && nice -n 15 ionice -c3 "${RS[@]}" "$HUB/$d" "ubuntu@$N:$HUB/" || true
done

echo "[stage $N] (3) pipeline code + medmcqa json ..."
"${RS[@]}" "$WD/scripts/deepseek/" "ubuntu@$N:$WD/scripts/deepseek/"
"${RS[@]}" "$WD/fleet/data_sanity.py" "$WD/fleet/monitor_tick.sh" "ubuntu@$N:$WD/fleet/"
"${RS[@]}" "$WD/repro/LLM-Adapters/ft-training_set/medmcqa_train.json" \
           "$WD/repro/LLM-Adapters/ft-training_set/medmcqa_val.json" \
           "ubuntu@$N:$WD/repro/LLM-Adapters/ft-training_set/"

echo "[stage $N] verify model shard count on node:"
"${SSH[@]}" "ubuntu@$N" "ls $HUB/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/*/model-*.safetensors 2>/dev/null | wc -l"
echo "[stage $N] DONE"
