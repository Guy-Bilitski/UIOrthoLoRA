
import os
import math
import argparse
import inspect
import csv
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# --- CRITICAL IMPORT FOR GEMMA-3 PATCH ---
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer
from peft import VeraConfig, UIOrthoLoRAConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ------------------------------------------------------------------
# PATCH: Fix for Gemma-3 / SigLIP crash in SFTTrainer
# SFTTrainer calls get_input_embeddings() on all modules.
# SiglipVisionTransformer raises NotImplementedError by default.
# We patch it to return None, preventing the training crash.
# ------------------------------------------------------------------
def _dummy_get_input_embeddings(self):
    return None

if hasattr(SiglipVisionTransformer, "get_input_embeddings"):
    SiglipVisionTransformer.get_input_embeddings = _dummy_get_input_embeddings
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune (PEFT) with TRL SFTTrainer")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned")
    p.add_argument("--learning_rate", type=float, required=True)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--peft_type", type=str, choices=["vera", "uiortholora"], required=True)
    p.add_argument("--rank", type=int, default=1024)
    p.add_argument("--svalues", type=int, default=256)
    p.add_argument("--svectors", type=int, default=64)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def alpaca_to_text(ex):
    instruction = ex["instruction"]
    input_text = ex.get("input") or ""
    output = ex["output"]
    if input_text.strip():
        prompt = f"### Instruction:\n{instruction}\n\nInput:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return {"text": prompt + output}


def is_gemma3_model(model_id: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        mt = (getattr(cfg, "model_type", "") or "").lower()
        if "gemma3" in mt: return True
    except: pass
    return "gemma-3" in model_id.lower() or "gemma3" in model_id.lower()


def make_sft_config(**kwargs):
    # Normalize arguments for TRL
    sig = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sig and "max_seq_length" in kwargs:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    if "max_seq_length" in sig and "max_length" in kwargs:
        kwargs["max_seq_length"] = kwargs.pop("max_length")

    # CRITICAL: Disable safetensors to support Orthogonal PEFT tensor views
    kwargs["save_safetensors"] = False
    return SFTConfig(**kwargs)


def tokenize_example(example, tokenizer, max_len, add_token_type_ids):
    enc = tokenizer(example["text"], truncation=True, max_length=max_len, padding=False)
    input_ids = enc["input_ids"]
    out = {"input_ids": input_ids, "attention_mask": enc["attention_mask"], "labels": input_ids.copy()}
    if add_token_type_ids:
        out["token_type_ids"] = [0] * len(input_ids)
    return out


@dataclass
class DataCollatorForCausalLMWithOptionalTTI:
    tokenizer: Any
    label_pad_token_id: int = -100
    add_token_type_ids: bool = False

    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        token_type_ids = [f.pop("token_type_ids") for f in features] if "token_type_ids" in features[0] else None

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for lab in labels:
            # Pad labels with -100 to ignore padding in loss
            padded_labels.append(lab + [self.label_pad_token_id] * (max_len - len(lab)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        if token_type_ids is not None:
            padded_tti = []
            for tti in token_type_ids:
                padded_tti.append(tti + [0] * (max_len - len(tti)))
            batch["token_type_ids"] = torch.tensor(padded_tti, dtype=torch.long)
        elif self.add_token_type_ids:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

        return batch


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"--- Starting: {args.peft_type} | LR: {args.learning_rate} ---")

    requires_token_type_ids = is_gemma3_model(args.model_id)
    if requires_token_type_ids:
        print("Detected Gemma-3 style model, will include token_type_ids")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,  # Used 'dtype' as requested
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map=None
    )
    model.config.use_cache = False
    model.to("cuda")

    # Gemma modules typically targeted
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if args.peft_type == "vera":
        peft_config = VeraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            target_modules=target_modules,
            vera_dropout=0.05,
            save_projection=True
        )
    else:
        peft_config = UIOrthoLoRAConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            num_svalues_to_adapt=args.svalues,
            num_svectors_to_adapt=args.svectors,
            uiortholora_dropout=0.05,
            initial_scaler=0.1,
            initial_sigma=0.1
        )

    # --- Dataset Loading & Validation Split ---
    print(f"Loading and splitting dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.map(alpaca_to_text, remove_columns=dataset.column_names)

    # 95% Train, 5% Evaluation
    dataset = dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print("Tokenizing...")
    train_dataset = train_dataset.map(lambda x: tokenize_example(x, tokenizer, args.max_len, requires_token_type_ids), remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(lambda x: tokenize_example(x, tokenizer, args.max_len, requires_token_type_ids), remove_columns=eval_dataset.column_names)

    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(0.1 * total_steps)

    training_args = make_sft_config(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",  # Enable evaluation every epoch
        save_strategy="epoch",
        report_to="none",
        packing=False,
        max_length=args.max_len,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        data_collator=DataCollatorForCausalLMWithOptionalTTI(tokenizer, add_token_type_ids=requires_token_type_ids)
    )

    trainer.train()

    print(f"Saving adapter to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    # Safe save ensures we don't crash on non-contiguous tensors
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- Save Loss Metrics (Train & Eval) to CSV ---
    log_history = trainer.state.log_history
    csv_file = os.path.join(args.output_dir, "loss_metrics.csv")

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "train_loss", "eval_loss"])

        # log_history is a mixed list. Some entries have 'loss', some have 'eval_loss'.
        for log in log_history:
            epoch = log.get("epoch", "")
            step = log.get("step", "")
            train_loss = log.get("loss", "")
            eval_loss = log.get("eval_loss", "")

            if train_loss != "" or eval_loss != "":
                writer.writerow([epoch, step, train_loss, eval_loss])

    print(f"Metrics saved to {csv_file}")
    print("Done.")

if __name__ == "__main__":
    main()
