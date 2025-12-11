#!/bin/bash
set -e

# --- Configuration ---
# Using the Pre-Trained (Vanilla) Gemma 3 12B
MODEL_ID="google/gemma-3-12b-pt"

# GPU Assignments
GPU_VERA=0
GPU_ORTHO=1

BASE_OUT_DIR="results_gemma_12b"
mkdir -p "$BASE_OUT_DIR"
mkdir -p "logs"

echo "=== Experiment Start: Gemma-3-12B Vanilla ==="
echo "VeRA GPU: $GPU_VERA | UIOrthoLoRA GPU: $GPU_ORTHO"

# Function to run Training AND Inference pipeline
run_pipeline() {
    local PEFT_TYPE=$1
    local GPU_ID=$2
    local LR=$3
    local EXTRA_ARGS=$4
    local LOG_FILE="logs/${PEFT_TYPE}_run.log"
    
    local ADAPTER_NAME="gemma-3-12b-${PEFT_TYPE}"
    local OUTPUT_DIR="${BASE_OUT_DIR}/${ADAPTER_NAME}"

    echo "[${PEFT_TYPE}] Starting Pipeline on GPU ${GPU_ID}..." > "$LOG_FILE"
    
    # 1. TRAIN
    echo "[${PEFT_TYPE}] Training..." | tee -a "$LOG_FILE"
    # Note: CUDA_VISIBLE_DEVICES handles the "1 GPU" isolation
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
        --model_id "$MODEL_ID" \
        --output_dir "$OUTPUT_DIR" \
        --peft_type "$PEFT_TYPE" \
        --learning_rate "$LR" \
        $EXTRA_ARGS >> "$LOG_FILE" 2>&1

    if [ $? -ne 0 ]; then
        echo "[${PEFT_TYPE}] ❌ Training Failed. Check $LOG_FILE" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "[${PEFT_TYPE}] ✅ Training Complete." | tee -a "$LOG_FILE"

    # 2. INFERENCE (MT-Bench)
    # We point fastchat to the ADAPTER directory.
    # It will auto-load the base model defined in adapter_config.json
    echo "[${PEFT_TYPE}] Running MT-Bench Inference..." | tee -a "$LOG_FILE"
    
    # We pass the base model explicitly to be safe with Gemma 3
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m fastchat.llm_judge.gen_model_answer \
        --model-path "$OUTPUT_DIR" \
        --model-base "$MODEL_ID" \
        --model-id "$ADAPTER_NAME" \
        --num-gpus-total 1 >> "$LOG_FILE" 2>&1

    if [ $? -ne 0 ]; then
        echo "[${PEFT_TYPE}] ❌ Inference Failed. Check $LOG_FILE" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "[${PEFT_TYPE}] ✅ Inference Complete. Answers saved." | tee -a "$LOG_FILE"
}

# --- Launch Jobs in Parallel ---

# Job 1: VeRA (Strict Table 9: Rank 1024, LR 4e-3)
(
    run_pipeline "vera" "$GPU_VERA" "4e-3" "--rank 1024"
) &

# Job 2: UIOrthoLoRA (Using 1e-4 as typical baseline, or match 4e-3 if desired)
(
    run_pipeline "uiortholora" "$GPU_ORTHO" "1e-4" "--svalues 256 --svectors 64"
) &

# Wait for both to finish
wait

echo "=========================================="
echo "All jobs finished."
echo "Check $BASE_OUT_DIR for adapters and MT-Bench answers."