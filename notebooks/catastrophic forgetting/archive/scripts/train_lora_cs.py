"""
Reproduction: LoRA (rank 32) on LLM-Adapters commonsense_170k, LLaMA-2-7B.

Faithfully replicates the LLM-Adapters `finetune.py` commonsense recipe
(prompt template, cutoff, LoRA config, effective batch 16, lr 3e-4, 3 epochs)
but runs in the modern env (transformers 5.x, peft 0.19.1) in bf16 instead of
fp16 (B200 / sm_100 has native bf16; the old fp16+int8 stack can't run here).

The number is determined by the data + prompt template + hyperparameters, not
the Trainer wrapper, so those are copied verbatim from the reference repo.
"""
import os
import argparse

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "repro/LLM-Adapters/ft-training_set/commonsense_170k.json")


# --- EXACT prompt template copied from LLM-Adapters/finetune.py:generate_prompt ---
def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--data_path", default=DEFAULT_DATA)
    ap.add_argument("--output_dir", default=os.path.join(HERE, "models/lora_cs_l2-7b_r32"))
    ap.add_argument("--cutoff_len", type=int, default=256)
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=16)          # effective (optimizer) batch
    ap.add_argument("--micro_batch_size", type=int, default=16)    # per-device; B200 has the memory
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--train_on_inputs", action="store_true", default=True)  # LLM-Adapters default
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full set; >0 for a quick smoke test")
    args = ap.parse_args()

    grad_accum = max(1, args.batch_size // args.micro_batch_size)
    target_modules = args.target_modules.split(",")
    print(f"[config] {vars(args)}\n[config] grad_accum={grad_accum} target_modules={target_modules}", flush=True)

    # --- tokenizer (match reference: pad=0/unk, left padding) ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=args.cutoff_len,
                           padding=False, return_tensors=None)
        if (result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < args.cutoff_len and add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            user_len = len(tokenize(user_prompt, add_eos_token=False)["input_ids"])
            tokenized["labels"] = [-100] * user_len + tokenized["labels"][user_len:]
        return tokenized

    data = load_dataset("json", data_files=args.data_path)["train"]
    if args.max_samples > 0:
        data = data.select(range(min(args.max_samples, len(data))))
    train_data = data.shuffle(seed=args.seed).map(generate_and_tokenize_prompt,
                                                  remove_columns=data.column_names)
    print(f"[data] {len(train_data)} training examples", flush=True)

    # --- model + LoRA ---
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
                      lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            save_strategy="no",
            output_dir=args.output_dir,
            report_to="none",
            seed=args.seed,
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8,
                                             return_tensors="pt", padding=True),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
