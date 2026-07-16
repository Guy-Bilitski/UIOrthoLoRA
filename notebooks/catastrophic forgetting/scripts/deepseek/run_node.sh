#!/bin/bash
# Run ONE DeepSeek-V4-Flash adapter end-to-end on a drained node, using ALL 8 GPUs (the 284B
# base is sharded; unlike the 7B sweep this is one adapter/node, not one job/GPU — so DO NOT
# launch auto_dispatch.py for this). Sequential: train -> eval -> CE -> geometry. Idempotent
# (train skips if adapter complete; eval/CE skip if their output exists).
#
# Usage:  bash scripts/deepseek/run_node.sh <method> <lr> [run_name] [extra train flags...]
#   e.g.  bash scripts/deepseek/run_node.sh lora    3e-4 dsv4_lora_r16_lr3e4_s42
#         bash scripts/deepseek/run_node.sh milora  5e-4 dsv4_milora_r16_lr5e4_s42 --milora 1
#         bash scripts/deepseek/run_node.sh dora    2e-4 dsv4_dora_r16_lr2e4_s42   --use_dora 1
#         bash scripts/deepseek/run_node.sh lorawd  5e-4 dsv4_lorawd_r16_lr5e4_s42 --weight_decay 0.1
set -euo pipefail
METHOD="${1:?need method}"; METHOD_FLAG_METHOD="lora"; [ "$METHOD" = "clora" ] && METHOD_FLAG_METHOD="clora"
LR="${2:?need lr}"
RUN="${3:-dsv4_${1}_r16_lr${2//./}_s42}"
shift 3 || true
EXTRA=("$@")

WD="/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
cd "$WD"
PY=/home/guyb/UIOrthoLoRA/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/scratch/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
LOG="logs/dsv4_${RUN}.log"; mkdir -p logs
echo "==== [run_node] $RUN method=$METHOD lr=$LR extra=${EXTRA[*]:-} $(date -u +%FT%TZ) ====" | tee -a "$LOG"

echo "---- TRAIN ----" | tee -a "$LOG"
$PY scripts/deepseek/train_deepseek.py --method "$METHOD_FLAG_METHOD" --run_name "$RUN" \
    --learning_rate "$LR" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"

ADAPTER="/scratch/cf_models/$RUN"
echo "---- EVAL ----" | tee -a "$LOG"
if [ -f "results/$RUN/summary.json" ]; then
  echo "[run_node] summary exists — skip eval" | tee -a "$LOG"
else
  $PY scripts/deepseek/eval_deepseek.py --adapter "$ADAPTER" --run_name "$RUN" \
      --ret_suite broad --adapt_limit 1000 2>&1 | tee -a "$LOG"
fi

echo "---- CE ----" | tee -a "$LOG"
if [ -f "results/$RUN/forgetting.json" ]; then
  echo "[run_node] forgetting.json exists — skip CE" | tee -a "$LOG"
else
  $PY scripts/deepseek/ce_deepseek.py --adapter "$ADAPTER" --run_name "$RUN" \
      --max_length 1024 --max_blocks 40 2>&1 | tee -a "$LOG"
fi

echo "---- GEOMETRY (factor-only, cpu) ----" | tee -a "$LOG"
$PY scripts/deepseek/geo_deepseek.py --glob "$RUN" --adapters_root /scratch/cf_models 2>&1 | tee -a "$LOG"
echo "==== [run_node] $RUN DONE $(date -u +%FT%TZ) ====" | tee -a "$LOG"
