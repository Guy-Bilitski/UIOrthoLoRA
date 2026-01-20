#!/bin/bash

set -e

###############################################################################
# Helper functions for checking if steps are already completed
###############################################################################

check_training_complete() {
    local model_path="$1"
    if [ -d "$model_path" ] && [ -f "${model_path}/adapter_config.json" ]; then
        return 0
    fi
    return 1
}

check_inference_complete() {
    local results_file="$1"
    local model_path="$2"
    
    if [ ! -f "$results_file" ]; then
        return 1
    fi
    
    local adapter_name=$(basename "$model_path")
    
    if grep -q "\"${adapter_name}\":" "$results_file" 2>/dev/null; then
        return 0
    fi
    
    return 1
}

check_mmlu_complete() {
    local mmlu_output_path="$1"
    
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

###############################################################################
# Configuration
###############################################################################

MODEL_ID="google/gemma-3-12b-it"

ALPHA=32
DROPOUT=0.0
NUM_EPOCHS=10
SEED=42
SC_NUMBER=10
INCLUDE_TRAINING=true
RUN_QA_INFERENCE=true

# UIorthoLoRA sweep parameters
LEARNING_RATES="7e-3 9e-3 1e-2 2e-2 4e-2"
SVECTORS_LIST="512 256 128 64 32 0"
SVALUES_LIST="1024"

# GPU mapping: one GPU per svectors value
declare -A GPU_MAP=(
    [512]=2
    [256]=3
    [128]=4
    [64]=5
    [32]=6
    [0]=7
)

# MMLU Evaluation settings
RUN_MMLU_EVAL=true
MMLU_NUM_FEWSHOT=0
MMLU_LIMIT=""

DELETE_MODEL_AFTER_EVAL=false

# Sample run settings
SAMPLE_RUN=false
SAMPLE_SIZE=10

# BigBench
RUN_BIGBENCH_EVAL=true
BIGBENCH_TASKS="bigbench_analytic_entailment_multiple_choice,bigbench_cause_and_effect_multiple_choice,bigbench_conceptual_combinations_multiple_choice,bigbench_causal_judgment_multiple_choice,bigbench_analogical_similarity_multiple_choice,bigbench_common_morpheme_multiple_choice,bigbench_logical_deduction_multiple_choice,bigbench_logical_sequence_multiple_choice,bigbench_odd_one_out_multiple_choice"

WORK_DIR="results/gemma-12b/workdir"
WORK_FILE="google_gemma-3-12b-it-hotspotqa_scores"

mkdir -p results logs models

MODEL_SAFE_NAME="${MODEL_ID//\//_}"
PEFT_TYPE="uiortholora"
TRAINING_LABEL="All"

###############################################################################
# Launch one process per svectors value (each on its own GPU)
###############################################################################

for SVECS in $SVECTORS_LIST; do
(
    export CUDA_VISIBLE_DEVICES="${GPU_MAP[$SVECS]}"
    LOGFILE="${LOG_ROOT}/${PEFT_TYPE}_svecs${SVECS}_$(date +%Y%m%d_%H%M%S).log"

    echo "=== [UIorthoLoRA svecs=$SVECS] Using GPU ${GPU_MAP[$SVECS]}, logging to $LOGFILE ===" | tee -a "$LOGFILE"

    RESULTS_PATH="$WORK_DIR/${WORK_FILE}-uiortholora.jsonl"

    for SVALUES in $SVALUES_LIST; do
        for LEARNING_RATE in $LEARNING_RATES; do
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_s${SVALUES}_v${SVECS}_lr${LEARNING_RATE}"

            if [ "$SAMPLE_RUN" = true ]; then
                SAMPLE_RUN_FLAG="--sample_run --sample_size $SAMPLE_SIZE"
            else
                SAMPLE_RUN_FLAG=""
            fi

            mkdir -p "$(dirname "$OUTPUT_PATH")"
            mkdir -p "$(dirname "$RESULTS_PATH")"

            echo "" | tee -a "$LOGFILE"
            echo "==========================================" | tee -a "$LOGFILE"
            echo "[UIorthoLoRA] svalues=${SVALUES}, svecs=${SVECS}, LR=${LEARNING_RATE}" | tee -a "$LOGFILE"
            echo "==========================================" | tee -a "$LOGFILE"

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
                        --sc_number "$SC_NUMBER" \
                        $INCLUDE_TRAINING_FLAG \
                        $RUN_QA_INFERENCE_FLAG \
                        $SAMPLE_RUN_FLAG \
                        --model_path "$OUTPUT_PATH" \
                        --svalues "$SVALUES" \
                        --svectors "$SVECS" 2>&1 | tee -a "$LOGFILE"

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
                MMLU_OUTPUT_PATH="results/mmlu/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_s${SVALUES}_v${SVECS}_lr${LEARNING_RATE}"

                if check_mmlu_complete "$MMLU_OUTPUT_PATH"; then
                    echo "⏭️  [SKIP] MMLU evaluation already complete: $MMLU_OUTPUT_PATH" | tee -a "$LOGFILE"
                else
                    echo "" | tee -a "$LOGFILE"
                    echo "Running MMLU Evaluation" | tee -a "$LOGFILE"

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
                        echo "✓ MMLU evaluation completed (${MMLU_DURATION}s)" | tee -a "$LOGFILE"
                    else
                        echo "✗ MMLU evaluation failed with exit code $MMLU_EXIT_CODE (${MMLU_DURATION}s)" | tee -a "$LOGFILE"
                    fi
                fi
            fi

            # Run BigBench evaluation if enabled
            if [ "$RUN_BIGBENCH_EVAL" = true ]; then
                BIGBENCH_OUTPUT_PATH="results/bigbench/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_s${SVALUES}_v${SVECS}_lr${LEARNING_RATE}"

                if check_bigbench_complete "$BIGBENCH_OUTPUT_PATH"; then
                    echo "⏭️  [SKIP] BigBench evaluation already complete: $BIGBENCH_OUTPUT_PATH" | tee -a "$LOGFILE"
                else
                    echo "" | tee -a "$LOGFILE"
                    echo "Running BigBench Evaluation" | tee -a "$LOGFILE"

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
                        echo "✓ BigBench evaluation completed (${BIGBENCH_DURATION}s)" | tee -a "$LOGFILE"
                    else
                        echo "✗ BigBench evaluation failed with exit code $BIGBENCH_EXIT_CODE (${BIGBENCH_DURATION}s)" | tee -a "$LOGFILE"
                    fi
                fi
            fi

            # Delete model after evaluation if enabled
            if [ "$DELETE_MODEL_AFTER_EVAL" = true ]; then
                echo "Deleting model: $OUTPUT_PATH" | tee -a "$LOGFILE"
                rm -rf "$OUTPUT_PATH"
                echo "✓ Model deleted" | tee -a "$LOGFILE"
            fi

        done
    done

    echo "=== [UIorthoLoRA svecs=$SVECS] Finished. Logs saved to $LOGFILE ===" | tee -a "$LOGFILE"
) &
done

wait
echo "✅ All UIorthoLoRA configurations completed. Check ${LOG_ROOT}/ for detailed output."