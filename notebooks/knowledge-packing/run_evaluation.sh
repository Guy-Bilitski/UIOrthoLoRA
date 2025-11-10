#!/bin/bash

# Evaluation script for trained LoRA models
# This generates predictions needed for the analysis notebook

# export CUDA_VISIBLE_DEVICES=0
# export NVIDIA_VISIBLE_DEVICES=0

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

# Configuration
BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_PATH="./lora_r1a1/HighKnown/lora1_onlyproj_bs8_LR0.001_seed42_trained_on_1Unknown_0HighKnown"
LOG_FILE="evaluation_$(date +%Y%m%d_%H%M%S).log"

echo "Starting evaluation..."
echo "Logs will be saved to: $LOG_FILE"
echo "Track progress with: tail -f $LOG_FILE"

# Run evaluation with nohup
nohup python -u ./evaluate_lora.py \
    --lora_path "$LORA_PATH" \
    --base_model "$BASE_MODEL" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "To monitor: tail -f $LOG_FILE"
echo "To check if running: ps -p $PID"

