#!/bin/bash
set -e

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
ALPHA=32
DROPOUT=0.0
NUM_EPOCHS=10
SEED=42
SC_NUMBER=10
INCLUDE_TRAINING=true
RUN_QA_INFERENCE=false
LORA_RANK=3
VERA_RANK=3
RANDLORA_RANK=512
SVALUES=256
SVECS=64

# MMLU Evaluation settings
RUN_MMLU_EVAL=true
MMLU_NUM_FEWSHOT=5
MMLU_LIMIT=10  # Set to empty string or remove --limit flag to run all
DELETE_MODEL_AFTER_EVAL=true

# GPU mapping per adapter
declare -A GPU_MAP=(
    [uiortholora]=0
    [lora]=1
    [vera]=2
    [randlora]=3
)

# Result JSONL per adapter
declare -A RESULT_MAP=(
    [uiortholora]="results/meta-llama_Llama-3.1-8B-Instruct_scores-uiortholora.jsonl"
    [lora]="results/meta-llama_Llama-3.1-8B-Instruct_scores-lora.jsonl"
    [vera]="results/meta-llama_Llama-3.1-8B-Instruct_scores-vera.jsonl"
    [randlora]="results/meta-llama_Llama-3.1-8B-Instruct_scores-randlora.jsonl"
)

mkdir -p results logs models

MODEL_SAFE_NAME="${MODEL_ID//\//_}"

# TRAINING_NUMBERS="100 500 1000 2000 3000 4000 5000"
# LEARNING_RATES="1e-3 1e-4 1e-5"

PEFT_TYPES="lora"
TRAINING_NUMBERS="1"
LEARNING_RATES="1e-3"

if [ "$INCLUDE_TRAINING" = true ]; then
    INCLUDE_TRAINING_FLAG="--include_training"
else
    INCLUDE_TRAINING_FLAG=""
fi

# Launch one process per adapter
for PEFT_TYPE in $PEFT_TYPES; do
(
    export CUDA_VISIBLE_DEVICES="${GPU_MAP[$PEFT_TYPE]}"
    RESULTS_PATH="${RESULT_MAP[$PEFT_TYPE]}"
    LOGFILE="logs/${PEFT_TYPE}_$(date +%Y%m%d_%H%M%S).log"

    echo "=== [$PEFT_TYPE] Using GPU ${GPU_MAP[$PEFT_TYPE]}, logging to $LOGFILE ==="

    for LEARNING_RATE in $LEARNING_RATES; do
        for TRAINING_NUMBER in $TRAINING_NUMBERS; do

            if [ "$PEFT_TYPE" = "uiortholora" ]; then
                PEFT_ARGS="--svalues $SVALUES --svectors $SVECS"
                OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_uiortholora_s${SVALUES}_v${SVECS}_lr${LEARNING_RATE}"
            elif [ "$PEFT_TYPE" = "lora" ]; then
                PEFT_ARGS="--lora_rank $LORA_RANK"
                OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_lora_r${LORA_RANK}_lr${LEARNING_RATE}"
            elif [ "$PEFT_TYPE" = "vera" ]; then
                PEFT_ARGS="--vera_rank $VERA_RANK"
                OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_vera_r${VERA_RANK}_lr${LEARNING_RATE}"
            elif [ "$PEFT_TYPE" = "randlora" ]; then
                PEFT_ARGS="--rand_lora_rank $RANDLORA_RANK"
                OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_randlora_r${RANDLORA_RANK}_lr${LEARNING_RATE}"
            fi

            mkdir -p "$(dirname "$OUTPUT_PATH")"

            echo "[${PEFT_TYPE}] Train=${TRAINING_NUMBER}, LR=${LEARNING_RATE}"
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
                $PEFT_ARGS >>"$LOGFILE" 2>&1

            # Run MMLU evaluation if enabled
            if [ "$RUN_MMLU_EVAL" = true ] && [ "$INCLUDE_TRAINING" = true ]; then
                echo ""
                echo "=========================================="
                echo "Running MMLU Evaluation"
                echo "=========================================="
                
                MMLU_OUTPUT_PATH="results/mmlu/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}.json"
                mkdir -p "$(dirname "$MMLU_OUTPUT_PATH")"
                
                # Build the lm_eval command
                MMLU_CMD="lm_eval --model hf --model_args pretrained=$OUTPUT_PATH --tasks mmlu --num_fewshot $MMLU_NUM_FEWSHOT --batch_size auto --output_path $MMLU_OUTPUT_PATH"
                
                # Add limit if specified
                if [ -n "$MMLU_LIMIT" ]; then
                    MMLU_CMD="$MMLU_CMD --limit $MMLU_LIMIT"
                fi
                
                echo "Running: $MMLU_CMD"
                MMLU_START_TIME=$(date +%s)
                eval $MMLU_CMD
                MMLU_END_TIME=$(date +%s)
                MMLU_DURATION=$((MMLU_END_TIME - MMLU_START_TIME))
                
                if [ $? -eq 0 ]; then
                    echo "✓ MMLU evaluation completed successfully"
                    echo "Results saved to: $MMLU_OUTPUT_PATH"
                    echo "MMLU evaluation took: ${MMLU_DURATION} seconds"
                else
                    echo "✗ MMLU evaluation failed"
                    echo "MMLU evaluation took: ${MMLU_DURATION} seconds"
                fi
                
                # Delete model after evaluation if enabled
                if [ "$DELETE_MODEL_AFTER_EVAL" = true ]; then
                    echo ""
                    echo "Deleting model: $OUTPUT_PATH"
                    rm -rf "$OUTPUT_PATH"
                    echo "✓ Model deleted"
                fi
            fi
        done
    done

    echo "=== [$PEFT_TYPE] Finished. Logs saved to $LOGFILE ===") &
done

wait
echo "✅ All adapters completed. Check logs/ for detailed output."

