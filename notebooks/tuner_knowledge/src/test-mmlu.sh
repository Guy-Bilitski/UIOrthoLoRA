#!/bin/bash
set -euo pipefail

# Which GPU to use
export CUDA_VISIBLE_DEVICES=5

# Base HF model id
MODEL_ID="google/gemma-3-12b-it"

# Path to your trained LoRA adapter
PEFT_PATH="/home/guyb/UIOrthoLoRA/notebooks/tuner_knowledge/src/models/google_gemma-3-12b-it_lora_tr100_lora_r3_lr1e-3"

# Output directory
OUT_DIR="results/mmlu_single/gemma3_12b_it_lora_tr100_r3_lr1e-3"
mkdir -p "$OUT_DIR"

# Optional: fail fast if adapter path is wrong
if [ ! -d "$PEFT_PATH" ]; then
  echo "Adapter path does not exist: $PEFT_PATH" >&2
  exit 1
fi

# MMLU config
TASKS="mmlu"
NUM_FEWSHOT=5
BATCH_SIZE="auto"
# For full run leave empty, for debug use: LIMIT="--limit 100"
LIMIT=""

echo "========================================================"
echo "Running MMLU with lightweight script"
echo "  Base:   $MODEL_ID"
echo "  LoRA:   $PEFT_PATH"
echo "  GPU:    $CUDA_VISIBLE_DEVICES"
echo "  Output: $OUT_DIR"
echo "========================================================"

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_ID},peft=${PEFT_PATH},dtype=bfloat16,trust_remote_code=True,parallelize=True" \
  --tasks "${TASKS}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  ${LIMIT} \
  --output_path "${OUT_DIR}"

