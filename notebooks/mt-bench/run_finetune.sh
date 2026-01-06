#!/bin/bash
set -e

# --- Configuration ---
MODEL_ID="google/gemma-3-12b-pt"
BASE_OUT_DIR="results_gemma_12b_sweep"
GPU_VERA=0
GPU_ORTHO=1
mkdir -p "$BASE_OUT_DIR"
mkdir -p "logs"

# --- Hyperparameter Sweep Definitions ---
#VERA_RANK=1024
#VERA_LRS=("1e-4" "3e-4")  

ORTHO_SVAL=256
ORTHO_SVEC=64
ORTHO_LRS=("1e-4" "3e-4")
ORTHO_SVECS=("0" "32" "128")
ORTHO_SVALS=("128" "256" "512" "1024")


# --- Flag Parsing ---
DO_TRAIN=true
DO_INFERENCE=true


echo "=== Sweep Start ==="
echo "Mode: Train=${DO_TRAIN}, Inference=${DO_INFERENCE}"

# --- The Pipeline Function ---
run_pipeline() {
    local PEFT_TYPE=$1
    local GPU_ID=$2
    local LR=$3
    local EXTRA_ARGS=$4
    local ADAPTER_NAME=$5  # Passed explicitly now
    
    local LOG_FILE="logs/${ADAPTER_NAME}.log"
    local OUTPUT_DIR="${BASE_OUT_DIR}/${ADAPTER_NAME}"
    local MERGED_DIR="${OUTPUT_DIR}_merged"

    echo "[${ADAPTER_NAME}] Starting on GPU ${GPU_ID}..." > "$LOG_FILE"

    # 1. TRAIN
    if [ "$DO_TRAIN" = true ]; then
        echo "[${ADAPTER_NAME}] Training (LR=${LR})..." | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
            --model_id "$MODEL_ID" \
            --output_dir "$OUTPUT_DIR" \
            --peft_type "$PEFT_TYPE" \
            --learning_rate "$LR" \
            $EXTRA_ARGS >> "$LOG_FILE" 2>&1

        if [ $? -ne 0 ]; then
            echo "[${ADAPTER_NAME}] ❌ Training Failed." | tee -a "$LOG_FILE"
            exit 1
        fi
        echo "[${ADAPTER_NAME}] ✅ Training Complete. Loss saved to loss_metrics.csv" | tee -a "$LOG_FILE"
    else
        echo "[${ADAPTER_NAME}] ⏭️  Skipping Training." | tee -a "$LOG_FILE"
    fi

    # 2. INFERENCE
    if [ "$DO_INFERENCE" = true ]; then
        # Merge
        echo "[${ADAPTER_NAME}] Merging..." | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 merge.py \
            --base_model "$MODEL_ID" \
            --adapter_path "$OUTPUT_DIR" \
            --output_path "$MERGED_DIR" >> "$LOG_FILE" 2>&1

        if [ $? -ne 0 ]; then
            echo "[${ADAPTER_NAME}] ❌ Merge Failed." | tee -a "$LOG_FILE"
            exit 1
        fi

        # Gen Answers
        echo "[${ADAPTER_NAME}] Running MT-Bench..." | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m fastchat.llm_judge.gen_model_answer \
            --model-path "$MERGED_DIR" \
            --model-id "$ADAPTER_NAME" \
            --dtype bfloat16 \
            --num-gpus-total 1 >> "$LOG_FILE" 2>&1

        if [ $? -ne 0 ]; then
            echo "[${ADAPTER_NAME}] ❌ Inference Failed." | tee -a "$LOG_FILE"
            exit 1
        fi
        echo "[${ADAPTER_NAME}] ✅ Inference Complete." | tee -a "$LOG_FILE"
    fi
}

# --- Launch Sweeps in Parallel ---

# Loop for VeRA (Runs sequentially on GPU_VERA)
#(
#    for LR in "${VERA_LRS[@]}"; do
#        NAME="gemma-3-12b-vera-r${VERA_RANK}-lr${LR}"
#        run_pipeline "vera" "$GPU_VERA" "$LR" "--rank $VERA_RANK" "$NAME"
#    done
#) &

# Loop for UIOrthoLoRA (Runs sequentially on GPU_ORTHO)
#(
#    for LR in "${ORTHO_LRS[@]}"; do
#        NAME="gemma-3-12b-ortho-sv${ORTHO_SVAL}-vec${ORTHO_SVEC}-lr${LR}"
#        run_pipeline "uiortholora" "$GPU_ORTHO" "$LR" "--svalues $ORTHO_SVAL --svectors $ORTHO_SVEC" "$NAME"
#    done
#) &

(
    LR="1e-4"
    for ORTHO_SVEC in "${ORTHO_SVECS[@]}"; do
        for ORTHO_SVAL in "${ORTHO_SVALS[@]}"; do
            NAME="gemma-3-12b-ortho-sv${ORTHO_SVAL}-vec${ORTHO_SVEC}-lr${LR}"
            run_pipeline "uiortholora" "$GPU_ORTHO" "$LR" "--svalues $ORTHO_SVAL --svectors $ORTHO_SVEC" "$NAME"
        done
    done
) &

wait
echo "=========================================="
echo "Sweep Finished. Check $BASE_OUT_DIR for results."
