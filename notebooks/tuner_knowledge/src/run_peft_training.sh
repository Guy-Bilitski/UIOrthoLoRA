#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES=3

MODEL_ID="meta-llama/Llama-3.2-3B"
RESULTS_PATH="results/llama/meta-llama_Llama-3.2-3B_scores_merged.jsonl"
ALPHA=8
DROPOUT=0.0
NUM_EPOCHS=10
LEARNING_RATE=1e-4
SEED=42
SC_NUMBER=5
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

MODEL_SAFE_NAME="${MODEL_ID//\//_}"

# PEFT_TYPES="vera randlora"
# TRAINING_NUMBERS="3000 4000 5000"

PEFT_TYPES="lora"
TRAINING_NUMBERS="1"

for PEFT_TYPE in $PEFT_TYPES; do
    for TRAINING_NUMBER in $TRAINING_NUMBERS; do

        # Conditional args
        if [ "$PEFT_TYPE" = "uiortholora" ]; then
            PEFT_ARGS="--svalues $SVALUES --svectors $SVECS"
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_uiortholora_s${SVALUES}_v${SVECS}"
        elif [ "$PEFT_TYPE" = "lora" ]; then
            PEFT_ARGS="--lora_rank $LORA_RANK"
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_lora_r${LORA_RANK}"
        elif [ "$PEFT_TYPE" = "vera" ]; then
            PEFT_ARGS="--vera_rank $VERA_RANK"
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_vera_r${VERA_RANK}"
        elif [ "$PEFT_TYPE" = "randlora" ]; then
            PEFT_ARGS="--rand_lora_rank $RANDLORA_RANK"
            OUTPUT_PATH="models/${MODEL_SAFE_NAME}_${PEFT_TYPE}_tr${TRAINING_NUMBER}_randlora_r${RANDLORA_RANK}"
        else
            PEFT_ARGS=""
        fi

        # MODEL_PATH dynamically matches the number of training samples
        MODEL_PATH="$OUTPUT_PATH"

        echo "=== Running: PEFT_TYPE=$PEFT_TYPE, TRAINING_NUMBER=$TRAINING_NUMBER ==="
        echo "Output path: $OUTPUT_PATH"
        echo "Model path:  $MODEL_PATH"

        mkdir -p "$(dirname "$OUTPUT_PATH")"

        # Train the model
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
            $RUN_QA_INFERENCE_FLAG \
            --model_path "$MODEL_PATH" \
            $PEFT_ARGS

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
        
        echo ""
        echo "=========================================="
        echo ""
    done
done
