import torch
import argparse
import os
import shutil
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # --- SAFETY CLEANUP ---
    # If the folder exists, remove it to ensure we don't mix old/new files
    if os.path.exists(args.output_path):
        print(f"Removing existing merged directory: {args.output_path}")
        shutil.rmtree(args.output_path)

    print(f"Loading Base: {args.base_model}")
    # FIX: Use torch_dtype=torch.bfloat16 to avoid NaN/Inf in Gemma-3
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading Adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging...")
    model = model.merge_and_unload()

    print(f"Saving to {args.output_path}")
    model.save_pretrained(args.output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)
    print("Done.")

if __name__ == "__main__":
    main()
