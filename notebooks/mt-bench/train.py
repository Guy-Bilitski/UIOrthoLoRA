import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    VeraConfig, 
    UIOrthoLoRAConfig, 
    TaskType
)
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3-12B Vanilla with VeRA/UIOrthoLoRA")
    
    # Model & Paths
    parser.add_argument("--model_id", type=str, required=True, help="Path or ID of base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the adapter")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned")
    
    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    # Keeping strictly to VeRA paper (Table 9)
    parser.add_argument("--batch_size", type=int, default=4)     # Paper: 4
    parser.add_argument("--grad_accum", type=int, default=4)     # Paper: 4
    
    # PEFT Configuration
    parser.add_argument("--peft_type", type=str, choices=["vera", "uiortholora"], required=True)
    parser.add_argument("--rank", type=int, default=1024)        # Paper: 1024
    parser.add_argument("--svalues", type=int, default=256)
    parser.add_argument("--svectors", type=int, default=64)
    
    return parser.parse_args()

def formatting_prompts_func(example):
    """
    Formats Alpaca for Gemma 3. 
    Using standard ChatML/Llama style is often safer for generic benchmarking 
    unless you specifically want Gemma's <start_of_turn> tokens.
    """
    output_texts = []
    for instruction, input_text, output in zip(example['instruction'], example['input'], example['output']):
        user_content = f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction
        text = (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>\n"
        )
        output_texts.append(text)
    return output_texts

def main():
    args = parse_args()
    print(f"--- Starting Training: {args.peft_type} on {args.model_id} ---")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # 2. Load Base Model (Full bfloat16, No Quantization)
    print(f"Loading model (bfloat16) to device...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
        attn_implementation="flash_attention_2"
    )

    # 3. Configure PEFT
    # Gemma 3 uses SigLIP for vision, but we target the text transformer linear layers.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if args.peft_type == "vera":
        peft_config = VeraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            target_modules=target_modules,
            vera_dropout=0.05,
            save_projection=True, 
        )
    elif args.peft_type == "uiortholora":
        peft_config = UIOrthoLoRAConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            num_svalues_to_adapt=args.svalues,
            num_svectors_to_adapt=args.svectors,
            uiortholora_dropout=0.05,
            initial_scaler=0.1,
            initial_sigma=0.1
        )

    # 4. Training Config (Optimized for Speed)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        
        # Strict VeRA Paper Specs
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        
        # Speed & Precision
        bf16=True,                       
        gradient_checkpointing=False,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=2048,
        dataset_text_field="text",
        packing=False,
        report_to="none"
    )

    # 5. Train
    dataset = load_dataset(args.dataset_name, split="train")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()