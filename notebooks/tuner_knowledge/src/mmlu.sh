#!/bin/bash

set -e

### block to control the ps tree


#!/bin/bash
set -e

# Create a unique run ID for this execution
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="run_state"
mkdir -p "$RUN_DIR"

# Record the main script PID and process group
MAIN_PID=$$
MAIN_PGID=$(ps -o pgid= $MAIN_PID | tr -d ' ')

# Save run state
echo "$MAIN_PGID" > "$RUN_DIR/pgid_$RUN_ID"
echo "$RUN_ID" > "$RUN_DIR/latest"

# Create cleanup script for this run
cat << EOF > cleanup_run_$RUN_ID.sh
#!/bin/bash
echo "Cleaning run $RUN_ID"
echo "Killing process group: $MAIN_PGID"
kill -9 -$MAIN_PGID 2>/dev/null || true
echo "Deleting logs/"
rm -rf logs
EOF

chmod +x cleanup_run_$RUN_ID.sh
echo "Cleanup created: cleanup_run_$RUN_ID.sh"
### Finished blocked

#MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID="google/gemma-3-12b-it"
ALPHA=32
DROPOUT=0.0
NUM_EPOCHS=10
SEED=42
SC_NUMBER=10
INCLUDE_TRAINING=true
RUN_QA_INFERENCE=false
#RUN_QA_INFERENCE=true
LORA_RANK=3
VERA_RANK=1024
RANDLORA_RANK=512
SVALUES=256
SVECS=64

# MMLU Evaluation settings
RUN_MMLU_EVAL=true
#RUN_MMLU_EVAL=false
MMLU_NUM_FEWSHOT=5
MMLU_LIMIT=  # Set to empty string or remove --limit flag to run all
DELETE_MODEL_AFTER_EVAL=true

# GPU mapping per adapter
declare -A GPU_MAP=(
    [uiortholora]=4
    [lora]=5
    [vera]=6
    [randlora]=7
)

#WORK_DIR="results/llama-8b/workdir"
WORK_DIR="results/gemma-12b/workdir"

#WORK_FILE="meta-llama_Llama-3.1-8B-Instruct_scores"
WORK_FILE="google_gemma-3-12b-it_scores"

# Result JSONL per adapter
declare -A RESULT_MAP=(
    [uiortholora]="$WORK_DIR/$WORK_FILE-uiortholora.jsonl"
    [lora]="$WORK_DIR/$WORK_FILE-lora.jsonl"
    [vera]="$WORK_DIR/$WORK_FILE-vera.jsonl"
    [randlora]="$WORK_DIR/$WORK_FILE-randlora.jsonl"
)

mkdir -p results logs models

MODEL_SAFE_NAME="${MODEL_ID//\//_}"

PEFT_TYPES="lora uiortholora vera randlora"
TRAINING_NUMBERS="100 500 1000 2000 3000 4000 5000"
LEARNING_RATES="1e-3 1e-4 1e-5"


#PEFT_TYPES="randlora"
#TRAINING_NUMBERS="100"
#LEARNING_RATES="1e-3"


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

            echo "[${PEFT_TYPE}] Train=${TRAINING_NUMBER}, LR=${LEARNING_RATE}" | tee -a "$LOGFILE"
            
            # Run training
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
                $PEFT_ARGS 2>&1 | tee -a "$LOGFILE"

            TRAIN_EXIT_CODE=${PIPESTATUS[0]}
            
            if [ $TRAIN_EXIT_CODE -ne 0 ]; then
                echo "✗ Training failed with exit code $TRAIN_EXIT_CODE" | tee -a "$LOGFILE"
                continue
            fi

            echo "✓ Training completed successfully" | tee -a "$LOGFILE"

            # Run MMLU evaluation if enabled
            if [ "$RUN_MMLU_EVAL" = true ]; then
                echo "" | tee -a "$LOGFILE"
                echo "==========================================" | tee -a "$LOGFILE"
                echo "Running MMLU Evaluation" | tee -a "$LOGFILE"
                echo "==========================================" | tee -a "$LOGFILE"
                
                MMLU_OUTPUT_PATH="results/mmlu/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_lr${LEARNING_RATE}"
                mkdir -p "$(dirname "$MMLU_OUTPUT_PATH")"
                
                # Build the lm_eval command
                MMLU_CMD="lm_eval --model hf --model_args pretrained=$MODEL_ID,peft=$OUTPUT_PATH --tasks mmlu --num_fewshot $MMLU_NUM_FEWSHOT --batch_size auto --output_path $MMLU_OUTPUT_PATH"
                
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
                
                # Delete model after evaluation if enabled
                if [ "$DELETE_MODEL_AFTER_EVAL" = true ]; then
                    echo "" | tee -a "$LOGFILE"
                    echo "Deleting model: $OUTPUT_PATH" | tee -a "$LOGFILE"
                    rm -rf "$OUTPUT_PATH"
                    echo "✓ Model deleted" | tee -a "$LOGFILE"
                fi
            else
                echo "MMLU evaluation disabled (RUN_MMLU_EVAL=false)" | tee -a "$LOGFILE"
            fi
        done
    done

    echo "=== [$PEFT_TYPE] Finished. Logs saved to $LOGFILE ===" | tee -a "$LOGFILE"
) &
done

wait
echo "✅ All adapters completed. Check logs/ for detailed output."
