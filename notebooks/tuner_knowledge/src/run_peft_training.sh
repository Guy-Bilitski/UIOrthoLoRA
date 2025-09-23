#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES=0

MODEL_ID="meta-llama/Llama-3.2-3B"
RESULTS_PATH="results/llama/meta-llama_Llama-3.2-3B_scores.jsonl"
ALPHA=8
DROPOUT=0.0
NUM_EPOCHS=5
LEARNING_RATE=1e-4
SEED=42
SC_NUMBER=5
INCLUDE_TRAINING=true

if [ "$INCLUDE_TRAINING" = true ]; then
    INCLUDE_TRAINING_FLAG="--include_training"
else
    INCLUDE_TRAINING_FLAG=""
fi

MODEL_SAFE_NAME="${MODEL_ID//\//_}"

for PEFT_TYPE in vera; do
    for TRAINING_NUMBER in 100 500; do
        
        OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_r${RANK}_tr${TRAINING_NUMBER}"
        
        # MODEL_PATH dynamically matches the number of training samples
        MODEL_PATH="$OUTPUT_PATH"

        echo "=== Running: PEFT_TYPE=$PEFT_TYPE, TRAINING_NUMBER=$TRAINING_NUMBER ==="
        echo "Output path: $OUTPUT_PATH"
        echo "Model path:  $MODEL_PATH"

        mkdir -p "$(dirname "$OUTPUT_PATH")"

        python train.py \
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
            --model_path "$MODEL_PATH"
    done
done
