#!/bin/bash

set -e

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
    # e.g., "ft_evals": {"google_gemma-3-12b-it_lora_trAll_lora_r3_lr1e-4": {"train": true, "score": 1.0}}
    local adapter_name=$(basename "$model_path")
    
    # Check if adapter appears as a key in ft_evals (with quotes and colon)
    if grep -q "\"${adapter_name}\":" "$results_file" 2>/dev/null; then
        return 0  # Inference complete
    fi
    
    return 1  # Inference not complete
}

# Check if MMLU evaluation is complete (results JSON exists)
check_mmlu_complete() {
    local mmlu_output_path="$1"
    
    if [ -d "$mmlu_output_path" ]; then
        if ls "${mmlu_output_path}"/results*.json 1>/dev/null 2>&1 || \
           ls "${mmlu_output_path}"/**/results*.json 1>/dev/null 2>&1; then
            return 0  # MMLU complete
        fi
    fi
    return 1  # MMLU not complete
}

# Check if BigBench evaluation is complete (results JSON exists)
check_bigbench_complete() {
    local bigbench_output_path="$1"
    
    if [ -d "$bigbench_output_path" ]; then
        if ls "${bigbench_output_path}"/results*.json 1>/dev/null 2>&1 || \
           ls "${bigbench_output_path}"/**/results*.json 1>/dev/null 2>&1; then
            return 0  # BigBench complete
        fi
    fi
    return 1  # BigBench not complete
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

### Finished blocked

# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
#MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
MODEL_ID="google/gemma-3-12b-it"
#MODEL_ID="mistralai/Ministral-3-14B-Instruct-2512"

ALPHA=32
DROPOUT=0.0
NUM_EPOCHS=10
SEED=42
SC_NUMBER=10
INCLUDE_TRAINING=true
RUN_QA_INFERENCE=true
LORA_RANK=3
VERA_RANK=1024
RANDLORA_RANK=512
SVALUES=1024
SVECS=0

# MMLU Evaluation settings
RUN_MMLU_EVAL=true
# RUN_MMLU_EVAL=true
MMLU_NUM_FEWSHOT=0
MMLU_LIMIT="" # Set to empty string or remove --limit flag to run all


DELETE_MODEL_AFTER_EVAL=false

# Sample run settings (for pipeline testing)
SAMPLE_RUN=false
SAMPLE_SIZE=10

# Big bench
RUN_BIGBENCH_EVAL=true
BIGBENCH_TASKS="bigbench_analytic_entailment_multiple_choice,bigbench_cause_and_effect_multiple_choice,bigbench_conceptual_combinations_multiple_choice,bigbench_causal_judgment_multiple_choice,bigbench_analogical_similarity_multiple_choice,bigbench_common_morpheme_multiple_choice,bigbench_logical_deduction_multiple_choice,bigbench_logical_sequence_multiple_choice,bigbench_odd_one_out_multiple_choice"

# GPU mapping per adapter
declare -A GPU_MAP=(
    [uiortholora]=0
    [lora]=1
    [vera]=2
    [randlora]=3
)

# WORK_DIR="results/llama-8b/workdir"
#WORK_DIR="results/llama-3b/workdir"
WORK_DIR="results/gemma-12b/workdir"
#WORK_DIR="results/mistral-14b/workdir"

# WORK_FILE="meta-llama_Llama-3.1-8B-Instruct_scores"
#WORK_FILE="meta-llama_Llama-3.2-3B-Instruct_scores"
# WORK_FILE="google_gemma-3-12b-it_scores"
WORK_FILE="google_gemma-3-12b-it-hotspotqa_scores"
#WORK_FILE="Ministral-3-14B-Instruct-2512"

# Result JSONL per adapter
declare -A RESULT_MAP=(
    [uiortholora]="$WORK_DIR/$WORK_FILE-uiortholora.jsonl"
    [lora]="$WORK_DIR/$WORK_FILE-lora.jsonl"
    [vera]="$WORK_DIR/$WORK_FILE-vera.jsonl"
    [randlora]="$WORK_DIR/$WORK_FILE-randlora.jsonl"
)

mkdir -p results logs models

MODEL_SAFE_NAME="${MODEL_ID//\//_}"

PEFT_TYPES="lora uiortholora vera"
LEARNING_RATES="7e-5 1e-4 5e-4 7e-4 1e-3"

# Set flags based on config
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

# Launch one process per adapter
for PEFT_TYPE in $PEFT_TYPES; do
(
    export CUDA_VISIBLE_DEVICES="${GPU_MAP[$PEFT_TYPE]}"
    RESULTS_PATH="${RESULT_MAP[$PEFT_TYPE]}"
    LOGFILE="logs/${PEFT_TYPE}_$(date +%Y%m%d_%H%M%S).log"

    echo "=== [$PEFT_TYPE] Using GPU ${GPU_MAP[$PEFT_TYPE]}, logging to $LOGFILE ===" | tee -a "$LOGFILE"

    for LEARNING_RATE in $LEARNING_RATES; do
        
        # Hardcoded identifier since we are training on All data
        TRAINING_LABEL="All" 

        if [ "$PEFT_TYPE" = "uiortholora" ]; then
            PEFT_ARGS="--svalues $SVALUES --svectors $SVECS"
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_uiortholora_s${SVALUES}_v${SVECS}_lr${LEARNING_RATE}"
        elif [ "$PEFT_TYPE" = "lora" ]; then
            PEFT_ARGS="--lora_rank $LORA_RANK"
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_lora_r${LORA_RANK}_lr${LEARNING_RATE}"
        elif [ "$PEFT_TYPE" = "vera" ]; then
            PEFT_ARGS="--vera_rank $VERA_RANK"
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_vera_r${VERA_RANK}_lr${LEARNING_RATE}"
        elif [ "$PEFT_TYPE" = "randlora" ]; then
            PEFT_ARGS="--rand_lora_rank $RANDLORA_RANK"
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_randlora_r${RANDLORA_RANK}_lr${LEARNING_RATE}"
        fi

        if [ "$SAMPLE_RUN" = true ]; then
            SAMPLE_RUN_FLAG="--sample_run --sample_size $SAMPLE_SIZE"
        else
            SAMPLE_RUN_FLAG=""
        fi

        mkdir -p "$(dirname "$OUTPUT_PATH")"

        echo "[${PEFT_TYPE}] Train=${TRAINING_LABEL}, LR=${LEARNING_RATE}" | tee -a "$LOGFILE"

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
            MMLU_OUTPUT_PATH="results/mmlu/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_lr${LEARNING_RATE}"

            if check_mmlu_complete "$MMLU_OUTPUT_PATH"; then
                echo "⏭️  [SKIP] MMLU evaluation already complete: $MMLU_OUTPUT_PATH" | tee -a "$LOGFILE"
            else
                echo "" | tee -a "$LOGFILE"
                echo "==========================================" | tee -a "$LOGFILE"
                echo "Running MMLU Evaluation" | tee -a "$LOGFILE"
                echo "==========================================" | tee -a "$LOGFILE"

                MMLU_OUTPUT_PATH="results/mmlu/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_lr${LEARNING_RATE}"
                mkdir -p "$(dirname "$MMLU_OUTPUT_PATH")"

                # Build the lm_eval command
                MMLU_CMD="lm_eval --model hf --model_args pretrained=${OUTPUT_PATH},tokenizer=${MODEL_ID},dtype=bfloat16,trust_remote_code=True --tasks mmlu --num_fewshot $MMLU_NUM_FEWSHOT --batch_size auto --output_path $MMLU_OUTPUT_PATH"

                # Add limit if specified
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
                    echo "MMLU evaluation took: ${MMLU_DURATION} seconds" | tee -a "$LOGFILE"
                fi
            fi         
        else
            echo "MMLU evaluation disabled (RUN_MMLU_EVAL=false)" | tee -a "$LOGFILE"
        fi
            


        # Run BigBench evaluation if enabled
        if [ "$RUN_BIGBENCH_EVAL" = true ]; then
            BIGBENCH_OUTPUT_PATH="results/bigbench/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_lr${LEARNING_RATE}"

            if check_bigbench_complete "$BIGBENCH_OUTPUT_PATH"; then
                echo "⏭️  [SKIP] BigBench evaluation already complete: $BIGBENCH_OUTPUT_PATH" | tee -a "$LOGFILE"
            else
                echo "" | tee -a "$LOGFILE"
                echo "==========================================" | tee -a "$LOGFILE"
                echo "Running BigBench Evaluation" | tee -a "$LOGFILE"
                echo "==========================================" | tee -a "$LOGFILE"

                BIGBENCH_OUTPUT_PATH="results/bigbench/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_LABEL}_lr${LEARNING_RATE}"
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
                    echo "BigBench evaluation took: ${BIGBENCH_DURATION} seconds" | tee -a "$LOGFILE"
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

    done

    echo "=== [$PEFT_TYPE] Finished. Logs saved to $LOGFILE ===" | tee -a "$LOGFILE"
) &
done

wait
echo "✅ All adapters completed. Check logs/ for detailed output."
