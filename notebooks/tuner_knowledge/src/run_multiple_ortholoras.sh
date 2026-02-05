#!/bin/bash

set -e

###############################################################################
#                         CONFIGURATION SECTION                               #
#     Adjust these parameters to control the entire pipeline behavior         #
###############################################################################

#------------------------------------------------------------------------------
# MODEL CONFIGURATION
#------------------------------------------------------------------------------
# Available models (uncomment one):
# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
# MODEL_ID="google/gemma-3-12b-it"
# MODEL_ID="mistralai/Ministral-3-14B-Instruct-2512"

#------------------------------------------------------------------------------
# DATASET CONFIGURATION
#------------------------------------------------------------------------------
# Available datasets: "triviaqa" or "hotpotqa"
# DATASET="triviaqa"
DATASET="hotpotqa"

#------------------------------------------------------------------------------
# ADAPTER CONFIGURATION
#------------------------------------------------------------------------------
# Learning rates to sweep (space-separated)
LEARNING_RATES="4e-3 6e-3 7e-3 8e-3 9e-3"

# UIorthoLoRA settings
ALPHA=32
DROPOUT=0.0

# UIorthoLoRA-specific settings - iterate over these
SVALUES_TO_RUN="1024 1280 1536"
SVECS_TO_RUN="0 128"

#------------------------------------------------------------------------------
# TRAINING CONFIGURATION
#------------------------------------------------------------------------------
NUM_EPOCHS=10
SEED=42
SC_NUMBER=10
INCLUDE_TRAINING=true

#------------------------------------------------------------------------------
# INTRINSIC EVALUATION (QA Inference on training dataset)
#------------------------------------------------------------------------------
RUN_QA_INFERENCE=true

#------------------------------------------------------------------------------
# EXTRINSIC EVALUATION (MMLU & BigBench)
#------------------------------------------------------------------------------
# MMLU settings
RUN_MMLU_EVAL=true
MMLU_NUM_FEWSHOT=0
MMLU_LIMIT=""

# BigBench settings
RUN_BIGBENCH_EVAL=true
BIGBENCH_TASKS="bigbench_analytic_entailment_multiple_choice,bigbench_cause_and_effect_multiple_choice,bigbench_conceptual_combinations_multiple_choice,bigbench_causal_judgment_multiple_choice,bigbench_analogical_similarity_multiple_choice,bigbench_common_morpheme_multiple_choice,bigbench_logical_deduction_multiple_choice,bigbench_logical_sequence_multiple_choice,bigbench_odd_one_out_multiple_choice"

#------------------------------------------------------------------------------
# GPU MAPPING (adapter -> GPU ID)
#------------------------------------------------------------------------------
GPU_DEVICE=3

#------------------------------------------------------------------------------
# SAMPLE RUN (for pipeline testing)
#------------------------------------------------------------------------------
SAMPLE_RUN=false
SAMPLE_SIZE=10

#------------------------------------------------------------------------------
# CLEANUP
#------------------------------------------------------------------------------
DELETE_MODEL_AFTER_EVAL=false

###############################################################################
#                     AUTO-DERIVED PATHS (do not modify)                      #
###############################################################################

# Create safe model name for file paths
MODEL_SAFE_NAME="${MODEL_ID//\//_}"

# Derive short model name for directory structure
case "$MODEL_ID" in
    *"Llama-3.1-8B"*)   MODEL_SHORT="llama-8b" ;;
    *"Llama-3.2-3B"*)   MODEL_SHORT="llama-3b" ;;
    *"gemma-3-12b"*)    MODEL_SHORT="gemma-12b" ;;
    *)                  MODEL_SHORT="${MODEL_SAFE_NAME}" ;;
esac

# Work directory and file paths based on model and dataset
WORK_DIR="results/${MODEL_SHORT}/${DATASET}/workdir"
WORK_FILE="${MODEL_SAFE_NAME}_scores"

PEFT_TYPE="uiortholora"

# Result JSONL for uiortholora
RESULTS_PATH="$WORK_DIR/$WORK_FILE-uiortholora.jsonl"

# Print configuration summary
echo "=============================================="
echo "           PIPELINE CONFIGURATION            "
echo "=============================================="
echo "Model:       $MODEL_ID"
echo "Dataset:     $DATASET"
echo "Adapter:     $PEFT_TYPE"
echo "SValues:     $SVALUES_TO_RUN"
echo "SVecs:       $SVECS_TO_RUN"
echo "Work dir:    $WORK_DIR"
echo "=============================================="

###############################################################################
# Helper functions for checking if steps are already completed
###############################################################################

# Check if model training is complete (adapter_config.json exists)
check_training_complete() {
    local model_path="$1"
    if [ -d "$model_path" ] && [ -f "${model_path}/adapter_config.json" ]; then
        return 0  # Training complete
    fi
    return 1  # Training not complete
}

# Check if QA inference results exist in JSONL for this specific config
check_inference_complete() {
    local results_file="$1"
    local model_path="$2"
    
    if [ ! -f "$results_file" ]; then
        return 1  # File doesn't exist
    fi
    
    # The adapter name is used as a key inside ft_evals
    local adapter_name=$(basename "$model_path")
    
    # Check if adapter appears as a key in ft_evals with a score value
    # Pattern: "adapter_name": {"score": ...}
    if grep -q "\"${adapter_name}\": {\"score\":" "$results_file" 2>/dev/null; then
        return 0  # Inference complete
    fi
    
    # Also check with spaces around the colon (in case of formatting variations)
    if grep -q "\"${adapter_name}\": { *\"score\":" "$results_file" 2>/dev/null; then
        return 0  # Inference complete
    fi
    
    return 1  # Inference not complete
}

check_mmlu_complete() {
    local mmlu_output_path="$1"
    local model_path="$2"
    
    if ! check_training_complete "$model_path"; then
        return 1
    fi
    
    if [ -d "$mmlu_output_path" ]; then
        if ls "${mmlu_output_path}"/results*.json 1>/dev/null 2>&1 || \
           ls "${mmlu_output_path}"/**/results*.json 1>/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

check_bigbench_complete() {
    local bigbench_output_path="$1"
    local model_path="$2"
    
    if ! check_training_complete "$model_path"; then
        return 1
    fi
    
    if [ -d "$bigbench_output_path" ]; then
        if ls "${bigbench_output_path}"/results*.json 1>/dev/null 2>&1 || \
           ls "${bigbench_output_path}"/**/results*.json 1>/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

###############################################################################
# Run identification and cleanup helper
###############################################################################

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_STATE_DIR="run_state"
mkdir -p "$RUN_STATE_DIR"

MAIN_PID="$$"
MAIN_PGID="$(ps -o pgid= "$MAIN_PID" | tr -d " ")"

echo "$MAIN_PGID" > "${RUN_STATE_DIR}/pgid_${RUN_ID}"
echo "$RUN_ID" > "${RUN_STATE_DIR}/latest"

BASE_LOG_DIR="logs"
LOG_ROOT="${BASE_LOG_DIR}/run_${RUN_ID}"
mkdir -p "$LOG_ROOT"

# Per run cleanup script
cat << EOF > "cleanup_run_${RUN_ID}.sh"
#!/bin/bash
set -euo pipefail
echo "Cleaning run ${RUN_ID}"
echo "Killing process group: ${MAIN_PGID}"
kill -9 -${MAIN_PGID} 2>/dev/null || true
echo "Removing logs for this run: ${LOG_ROOT}"
rm -rf "${LOG_ROOT}"
EOF

chmod +x "cleanup_run_${RUN_ID}.sh"
echo "Cleanup created: cleanup_run_${RUN_ID}.sh"
echo "Logs for this run will be stored under: ${LOG_ROOT}"

mkdir -p "$WORK_DIR" results logs models

export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
LOGFILE="logs/${PEFT_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo "=== [$PEFT_TYPE] Using GPU $GPU_DEVICE, logging to $LOGFILE ===" | tee -a "$LOGFILE"

for LEARNING_RATE in $LEARNING_RATES; do
    for SVECS in $SVECS_TO_RUN; do
        for SVALUES in $SVALUES_TO_RUN; do

            # Hardcoded identifier since we are training on All data
            TRAINING_LABEL="All" 
            PEFT_ARGS="--svalues $SVALUES --svectors $SVECS"
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_sv${SVALUES}_svec${SVECS}_lr${LEARNING_RATE}"

            if [ "$SAMPLE_RUN" = true ]; then
                SAMPLE_RUN_FLAG="--sample_run --sample_size $SAMPLE_SIZE"
            else
                SAMPLE_RUN_FLAG=""
            fi

            mkdir -p "$(dirname "$OUTPUT_PATH")"

            echo "[${PEFT_TYPE}] Train=${TRAINING_LABEL}, LR=${LEARNING_RATE}, SValues=${SVALUES}, SVecs=${SVECS}" | tee -a "$LOGFILE"

            # Check if training is needed
            SKIP_TRAINING=false
            if check_training_complete "$OUTPUT_PATH"; then
                echo "⏭️  [SKIP] Training already complete: $OUTPUT_PATH" | tee -a "$LOGFILE"
                SKIP_TRAINING=true
            fi

            # Check if inference is needed
            SKIP_INFERENCE=false
            if check_inference_complete "$RESULTS_PATH" "$OUTPUT_PATH"; then
                echo "⏭️  [SKIP] Inference results already exist in: $RESULTS_PATH" | tee -a "$LOGFILE"
                SKIP_INFERENCE=true
            fi

            # Run training/inference if needed
            if [ "$SKIP_TRAINING" = true ] && [ "$SKIP_INFERENCE" = true ]; then
                echo "⏭️  [SKIP] Both training and inference already complete, skipping train.py" | tee -a "$LOGFILE"
            else
                # Determine which flags to pass
                if [ "$SKIP_TRAINING" = true ]; then
                    INCLUDE_TRAINING_FLAG=""
                elif [ "$INCLUDE_TRAINING" = true ]; then
                    INCLUDE_TRAINING_FLAG="--include_training"
                else
                    INCLUDE_TRAINING_FLAG=""
                fi

                if [ "$SKIP_INFERENCE" = true ]; then
                    RUN_QA_INFERENCE_FLAG=""
                elif [ "$RUN_QA_INFERENCE" = true ]; then
                    RUN_QA_INFERENCE_FLAG="--run_qa_inference"
                else
                    RUN_QA_INFERENCE_FLAG=""
                fi

                # Only run if at least one step is needed
                if [ -n "$INCLUDE_TRAINING_FLAG" ] || [ -n "$RUN_QA_INFERENCE_FLAG" ]; then
                    python3 train.py \
                        --model_id "$MODEL_ID" \
                        --peft_type "$PEFT_TYPE" \
                        --alpha "$ALPHA" \
                        --dropout "$DROPOUT" \
                        --output_path "$OUTPUT_PATH" \
                        --num_epochs "$NUM_EPOCHS" \
                        --learning_rate "$LEARNING_RATE" \
                        --seed "$SEED" \
                        --results_path "$RESULTS_PATH" \
                        --dataset "$DATASET" \
                        --sc_number "$SC_NUMBER" \
                        $INCLUDE_TRAINING_FLAG \
                        $RUN_QA_INFERENCE_FLAG \
                        $SAMPLE_RUN_FLAG \
                        --model_path "$OUTPUT_PATH" \
                        $PEFT_ARGS 2>&1 | tee -a "$LOGFILE"

                    TRAIN_EXIT_CODE=${PIPESTATUS[0]}

                    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
                        echo "✗ Training/inference failed with exit code $TRAIN_EXIT_CODE" | tee -a "$LOGFILE"
                        continue
                    fi

                    echo "✓ Training/inference completed successfully" | tee -a "$LOGFILE"
                fi
            fi

            # Run MMLU evaluation if enabled
            if [ "$RUN_MMLU_EVAL" = true ]; then
                MMLU_OUTPUT_PATH="results/mmlu/${DATASET}/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_lr${LEARNING_RATE}_sv${SVALUES}_svec${SVECS}"

                if check_mmlu_complete "$MMLU_OUTPUT_PATH" "$OUTPUT_PATH"; then
                    echo "⏭️  [SKIP] MMLU evaluation already complete: $MMLU_OUTPUT_PATH" | tee -a "$LOGFILE"
                else
                    echo "" | tee -a "$LOGFILE"
                    echo "==========================================" | tee -a "$LOGFILE"
                    echo "Running MMLU Evaluation (SValues=${SVALUES}, SVecs=${SVECS})" | tee -a "$LOGFILE"
                    echo "==========================================" | tee -a "$LOGFILE"

                    mkdir -p "$(dirname "$MMLU_OUTPUT_PATH")"

                    MMLU_CMD="lm_eval --model hf --model_args pretrained=${OUTPUT_PATH},tokenizer=${MODEL_ID},dtype=bfloat16,trust_remote_code=True --tasks mmlu --num_fewshot $MMLU_NUM_FEWSHOT --batch_size auto --output_path $MMLU_OUTPUT_PATH"

                    if [ -n "$MMLU_LIMIT" ]; then
                        MMLU_CMD="$MMLU_CMD --limit $MMLU_LIMIT"
                    fi

                    echo "Running: $MMLU_CMD" | tee -a "$LOGFILE"
                    MMLU_START_TIME=$(date +%s)

                    eval $MMLU_CMD 2>&1 | tee -a "$LOGFILE"
                    MMLU_EXIT_CODE=${PIPESTATUS[0]}

                    MMLU_END_TIME=$(date +%s)
                    MMLU_DURATION=$((MMLU_END_TIME - MMLU_START_TIME))

                    if [ $MMLU_EXIT_CODE -eq 0 ]; then
                        echo "✓ MMLU evaluation completed successfully" | tee -a "$LOGFILE"
                        echo "Results saved to: $MMLU_OUTPUT_PATH" | tee -a "$LOGFILE"
                        echo "MMLU evaluation took: ${MMLU_DURATION} seconds" | tee -a "$LOGFILE"
                    else
                        echo "✗ MMLU evaluation failed with exit code $MMLU_EXIT_CODE" | tee -a "$LOGFILE"
                    fi
                fi        
            else
                echo "MMLU evaluation disabled (RUN_MMLU_EVAL=false)" | tee -a "$LOGFILE"
            fi
            
            # Run BigBench evaluation if enabled
            if [ "$RUN_BIGBENCH_EVAL" = true ]; then
                BIGBENCH_OUTPUT_PATH="results/bigbench/${DATASET}/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_lr${LEARNING_RATE}_sv${SVALUES}_svec${SVECS}"

                if check_bigbench_complete "$BIGBENCH_OUTPUT_PATH" "$OUTPUT_PATH"; then
                    echo "⏭️  [SKIP] BigBench evaluation already complete: $BIGBENCH_OUTPUT_PATH" | tee -a "$LOGFILE"
                else
                    echo "" | tee -a "$LOGFILE"
                    echo "==========================================" | tee -a "$LOGFILE"
                    echo "Running BigBench Evaluation (SValues=${SVALUES}, SVecs=${SVECS})" | tee -a "$LOGFILE"
                    echo "==========================================" | tee -a "$LOGFILE"

                    mkdir -p "$BIGBENCH_OUTPUT_PATH"

                    BIGBENCH_CMD="lm_eval --model hf \
                        --model_args pretrained=${OUTPUT_PATH},tokenizer=${MODEL_ID},dtype=bfloat16,trust_remote_code=True \
                        --tasks $BIGBENCH_TASKS \
                        --batch_size auto \
                        --output_path $BIGBENCH_OUTPUT_PATH"

                    echo "Running: $BIGBENCH_CMD" | tee -a "$LOGFILE"
                    BIGBENCH_START_TIME=$(date +%s)

                    eval $BIGBENCH_CMD 2>&1 | tee -a "$LOGFILE"
                    BIGBENCH_EXIT_CODE=${PIPESTATUS[0]}

                    BIGBENCH_END_TIME=$(date +%s)
                    BIGBENCH_DURATION=$((BIGBENCH_END_TIME - BIGBENCH_START_TIME))

                    if [ $BIGBENCH_EXIT_CODE -eq 0 ]; then
                        echo "✓ BigBench evaluation completed successfully" | tee -a "$LOGFILE"
                        echo "Results saved to: $BIGBENCH_OUTPUT_PATH" | tee -a "$LOGFILE"
                        echo "BigBench evaluation took: ${BIGBENCH_DURATION} seconds" | tee -a "$LOGFILE"
                    else
                        echo "✗ BigBench evaluation failed with exit code $BIGBENCH_EXIT_CODE" | tee -a "$LOGFILE"
                    fi
                fi
            else
                echo "BigBench evaluation disabled (RUN_BIGBENCH_EVAL=false)" | tee -a "$LOGFILE"
            fi

            # Delete model after evaluation if enabled
            if [ "$DELETE_MODEL_AFTER_EVAL" = true ]; then
                echo "" | tee -a "$LOGFILE"
                echo "Deleting model: $OUTPUT_PATH" | tee -a "$LOGFILE"
                rm -rf "$OUTPUT_PATH"
                echo "✓ Model deleted" | tee -a "$LOGFILE"
            fi

        done # End SVecs Loop
    done # End SValues Loop
done # End Learning Rate Loop

echo "=== [$PEFT_TYPE] Finished. Logs saved to $LOGFILE ===" | tee -a "$LOGFILE"

echo "✅ All UIorthoLoRA configurations completed. Check logs/ for detailed output."
