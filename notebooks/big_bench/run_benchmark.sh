#!/bin/bash

# 1. Define the base relative path exactly as it worked in your manual command
RELATIVE_PATH_PREFIX="../tuner_knowledge/src/models"

# 2. Define the list of models
MODELS=(
    # --- Base Model ---
    #"google/gemma-3-12b-it"

    # --- LoRA Models ---
    #"google_gemma-3-12b-it_lora_trAll_lora_r3_lr1e-4"
    #"google_gemma-3-12b-it_lora_trAll_lora_r3_lr5e-4"
    #"google_gemma-3-12b-it_lora_trAll_lora_r3_lr5e-5"

    # --- RandLoRA Models ---
    #"google_gemma-3-12b-it_randlora_trAll_randlora_r512_lr1e-4"
    #"google_gemma-3-12b-it_randlora_trAll_randlora_r512_lr5e-4"
    #"google_gemma-3-12b-it_randlora_trAll_randlora_r512_lr5e-5"

    # --- VeRA Models ---
    #"google_gemma-3-12b-it_vera_trAll_vera_r1024_lr1e-4"
    #"google_gemma-3-12b-it_vera_trAll_vera_r1024_lr5e-4"
    #"google_gemma-3-12b-it_vera_trAll_vera_r1024_lr5e-5"

    # --- UIOrthoLoRA Models ---
    #"google_gemma-3-12b-it_uiortholora_trAll_uiortholora_s512_v16_lr5e-4"
    "google_gemma-3-12b-it_uiortholora_trAll_uiortholora_s1024_v16_lr1e-4"
    #"google_gemma-3-12b-it_uiortholora_trAll_uiortholora_s1024_v16_lr5e-4"
    #"google_gemma-3-12b-it_uiortholora_trAll_uiortholora_s1024_v16_lr5e-5"
)

# 3. Tasks list
TASKS="bigbench_analytic_entailment_multiple_choice,bigbench_cause_and_effect_multiple_choice,bigbench_conceptual_combinations_multiple_choice,bigbench_causal_judgment_multiple_choice,bigbench_analogical_similarity_multiple_choice,bigbench_common_morpheme_multiple_choice,bigbench_logical_deduction_multiple_choice,bigbench_logical_sequence_multiple_choice,bigbench_odd_one_out_multiple_choice"

# 4. Loop
for MODEL_NAME in "${MODELS[@]}"; do
    
    echo "========================================================"
    echo "Starting evaluation for: $MODEL_NAME"

    # Define Output Directory
    if [[ "$MODEL_NAME" == "google/gemma-3-12b-it" ]]; then
        OUTPUT_DIR="./gemma-12b-results/base_model"
    else
        OUTPUT_DIR="./gemma-12b-results/${MODEL_NAME}"
    fi
    mkdir -p "$OUTPUT_DIR"

    # Construct the model path argument exactly like your working command
    if [[ "$MODEL_NAME" == "google/gemma-3-12b-it" ]]; then
        # Hugging Face Hub Path
        MODEL_PATH="$MODEL_NAME"
    else
        # Local Relative Path (No readlink, just string concatenation)
        MODEL_PATH="${RELATIVE_PATH_PREFIX}/${MODEL_NAME}"
    fi

    echo "Using Path: $MODEL_PATH"
    
    # Run using the exact arguments that worked manually:
    # pretrained=PATH (handles both base and adapters if config is present)
    accelerate launch --num_processes 8 -m lm_eval \
        --model hf \
	--model_args "pretrained=${MODEL_PATH},tokenizer=google/gemma-3-12b-it,dtype=bfloat16,trust_remote_code=True" \
        --tasks $TASKS \
        --batch_size auto \
        --output_path "$OUTPUT_DIR"

    echo "Finished evaluation for $MODEL_NAME"
    echo "--------------------------------------------------------"
    echo ""
done
