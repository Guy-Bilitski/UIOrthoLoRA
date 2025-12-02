#!/bin/bash
set -e

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

# Per run cleanup script (only kills this run and removes only this run logs)
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
RUN_QA_INFERENCE=false

LORA_RANK=3
VERA_RANK=1024
RANDLORA_RANK=512
SVALUES=256
SVECS=64

# Check before training if model dir exists and reuse it
CHECK_IF_TRAINED=true

# MMLU evaluation settings
RUN_MMLU_EVAL=true
MMLU_NUM_FEWSHOT=5
MMLU_LIMIT=""   # set numeric value to limit examples, empty means full

DELETE_MODEL_AFTER_EVAL=true

# GPU mapping per adapter
declare -A GPU_MAP=(
    [uiortholora]=4
    [lora]=5
    [vera]=6
    [randlora]=7
)

WORK_DIR="results/gemma-12b/workdir"
WORK_FILE="google_gemma-3-12b-it_scores"

declare -A RESULT_MAP=(
    [uiortholora]="${WORK_DIR}/${WORK_FILE}-uiortholora.jsonl"
    [lora]="${WORK_DIR}/${WORK_FILE}-lora.jsonl"
    [vera]="${WORK_DIR}/${WORK_FILE}-vera.jsonl"
    [randlora]="${WORK_DIR}/${WORK_FILE}-randlora.jsonl"
)

mkdir -p results models "$WORK_DIR" "$BASE_LOG_DIR"

MODEL_SAFE_NAME="${MODEL_ID//\//_}"

PEFT_TYPES="lora uiortholora vera randlora"
TRAINING_NUMBERS="100 500 1000 2000 3000 4000 5000"
LEARNING_RATES="1e-3 1e-4 1e-5"

# Flags
if [ "$INCLUDE_TRAINING" = true ]; then
    INCLUDE_TRAINING_FLAG="--include_training"
else
    INCLUDE_TRAINING_FLAG=""
fi

if [ "$RUN_QA_INFERENCE" = true ]; then
    RUN_QA_INFERENCE_FLAG="--run_qa_inference"
else
    RUN_QA_INFERENCE_FLAG=""
fi

###############################################################################
# Main loop: per PEFT type in parallel
###############################################################################

for PEFT_TYPE in $PEFT_TYPES; do
(
    GPU_ID="${GPU_MAP[$PEFT_TYPE]:-}"
    if [ -z "$GPU_ID" ]; then
        echo "[${PEFT_TYPE}] No GPU mapping defined, skipping" | tee -a "${LOG_ROOT}/${PEFT_TYPE}_error.log"
        exit 0
    fi

    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    RESULTS_PATH="${RESULT_MAP[$PEFT_TYPE]}"
    LOGFILE="${LOG_ROOT}/${PEFT_TYPE}_$(date +%Y%m%d_%H%M%S).log"

    echo "=== [${PEFT_TYPE}] Using GPU ${GPU_ID}, logging to ${LOGFILE} ===" | tee -a "$LOGFILE"

    for LEARNING_RATE in $LEARNING_RATES; do
        for TRAINING_NUMBER in $TRAINING_NUMBERS; do

            # Build output path and PEFT specific args
            case "$PEFT_TYPE" in
                uiortholora)
                    PEFT_ARGS=(--svalues "$SVALUES" --svectors "$SVECS")
                    OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_uiortholora_s${SVALUES}_v${SVECS}_lr${LEARNING_RATE}"
                    ;;
                lora)
                    PEFT_ARGS=(--lora_rank "$LORA_RANK")
                    OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_lora_r${LORA_RANK}_lr${LEARNING_RATE}"
                    ;;
                vera)
                    PEFT_ARGS=(--vera_rank "$VERA_RANK")
                    OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_vera_r${VERA_RANK}_lr${LEARNING_RATE}"
                    ;;
                randlora)
                    PEFT_ARGS=(--rand_lora_rank "$RANDLORA_RANK")
                    OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_randlora_r${RANDLORA_RANK}_lr${LEARNING_RATE}"
                    ;;
                *)
                    echo "[${PEFT_TYPE}] Unknown PEFT type, skipping" | tee -a "$LOGFILE"
                    continue
                    ;;
            esac

            mkdir -p "$(dirname "$OUTPUT_PATH")"

            echo "[${PEFT_TYPE}] Train=${TRAINING_NUMBER}, LR=${LEARNING_RATE}" | tee -a "$LOGFILE"

            # Track whether the model existed before this run
            MODEL_PREEXISTING=0
            SKIP_TRAINING=0

            if [ "$CHECK_IF_TRAINED" = true ] && [ -d "$OUTPUT_PATH" ]; then
                MODEL_PREEXISTING=1
                SKIP_TRAINING=1
                echo "[${PEFT_TYPE}] Model already exists at ${OUTPUT_PATH}" | tee -a "$LOGFILE"
                echo "[${PEFT_TYPE}] Skipping training and going directly to evaluation" | tee -a "$LOGFILE"
            fi

            # Training block (unless we intentionally skip)
            if [ "$SKIP_TRAINING" -eq 0 ] && [ "$INCLUDE_TRAINING" = true ]; then
                echo "[${PEFT_TYPE}] Training model (directory does not exist or CHECK_IF_TRAINED=false)" | tee -a "$LOGFILE"

                python3 train.py \
                    --model_id "$MODEL_ID" \
                    --peft_type "$PEFT_TYPE" \
                    --alpha "$ALPHA" \
                    --dropout "$DROPOUT" \
                    --training_number "$TRAINING_NUMBER" \
                    --output_path "$OUTPUT_PATH" \
                    --num_epochs "$NUM_EPOCHS" \
                    --learning_rate "$LEARNING_RATE" \
                    --seed "$SEED" \
                    --results_path "$RESULTS_PATH" \
                    --sc_number "$SC_NUMBER" \
                    $INCLUDE_TRAINING_FLAG \
                    $RUN_QA_INFERENCE_FLAG \
                    --model_path "$OUTPUT_PATH" \
                    "${PEFT_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"

                TRAIN_EXIT_CODE=${PIPESTATUS[0]}

                if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
                    echo "[${PEFT_TYPE}] Training failed with exit code ${TRAIN_EXIT_CODE}" | tee -a "$LOGFILE"
                    # On train failure we do not attempt eval
                    continue
                fi

                echo "[${PEFT_TYPE}] Training completed successfully" | tee -a "$LOGFILE"

            else
                # If we skip training, consider train as success from the perspective of eval
                TRAIN_EXIT_CODE=0
                echo "[${PEFT_TYPE}] Training step skipped" | tee -a "$LOGFILE"
            fi

            # MMLU evaluation
            if [ "$RUN_MMLU_EVAL" = true ]; then
                echo "" | tee -a "$LOGFILE"
                echo "==========================================" | tee -a "$LOGFILE"
                echo "[${PEFT_TYPE}] Running MMLU evaluation" | tee -a "$LOGFILE"
                echo "Model path: ${OUTPUT_PATH}" | tee -a "$LOGFILE"
                echo "==========================================" | tee -a "$LOGFILE"

                MMLU_OUTPUT_PATH="results/mmlu/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_lr${LEARNING_RATE}"
                mkdir -p "$(dirname "$MMLU_OUTPUT_PATH")"

                MMLU_ARGS=(
                    lm_eval
                    --model hf
                    --model_args "pretrained=${MODEL_ID},peft=${OUTPUT_PATH}"
                    --tasks mmlu
                    --num_fewshot "${MMLU_NUM_FEWSHOT}"
                    --batch_size auto
                    --output_path "${MMLU_OUTPUT_PATH}"
                )

                if [ -n "$MMLU_LIMIT" ]; then
                    MMLU_ARGS+=(--limit "${MMLU_LIMIT}")
                fi

                echo "Running: ${MMLU_ARGS[*]}" | tee -a "$LOGFILE"
                MMLU_START_TIME=$(date +%s)

                "${MMLU_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
                MMLU_EXIT_CODE=${PIPESTATUS[0]}

                MMLU_END_TIME=$(date +%s)
                MMLU_DURATION=$((MMLU_END_TIME - MMLU_START_TIME))

                if [ "$MMLU_EXIT_CODE" -eq 0 ]; then
                    echo "[${PEFT_TYPE}] MMLU evaluation completed successfully" | tee -a "$LOGFILE"
                    echo "Results saved to: ${MMLU_OUTPUT_PATH}" | tee -a "$LOGFILE"
                    echo "MMLU evaluation took: ${MMLU_DURATION} seconds" | tee -a "$LOGFILE"
                else
                    echo "[${PEFT_TYPE}] MMLU evaluation failed with exit code ${MMLU_EXIT_CODE}" | tee -a "$LOGFILE"
                    echo "MMLU evaluation took: ${MMLU_DURATION} seconds" | tee -a "$LOGFILE"
                fi

                # Delete model only if we created it in this run
                if [ "$DELETE_MODEL_AFTER_EVAL" = true ]; then
                    if [ "$MODEL_PREEXISTING" -eq 1 ]; then
                        echo "[${PEFT_TYPE}] Model existed before this run, not deleting: ${OUTPUT_PATH}" | tee -a "$LOGFILE"
                    else
                        echo "[${PEFT_TYPE}] Deleting model directory created in this run: ${OUTPUT_PATH}" | tee -a "$LOGFILE"
                        rm -rf "$OUTPUT_PATH"
                        echo "[${PEFT_TYPE}] Model directory deleted" | tee -a "$LOGFILE"
                    fi
                fi
            else
                echo "[${PEFT_TYPE}] MMLU evaluation disabled (RUN_MMLU_EVAL=false)" | tee -a "$LOGFILE"
            fi

        done
    done

    echo "=== [${PEFT_TYPE}] Finished. Logs saved to ${LOGFILE} ===" | tee -a "$LOGFILE"
) &
done

wait
echo "All adapters completed. Per run logs are under ${LOG_ROOT}"
