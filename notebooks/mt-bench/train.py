import os
import math
import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import VeraConfig, UIOrthoLoRAConfig, TaskType
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune (PEFT) with TRL SFTTrainer (Gemma-3 safe)")
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
        if "gemma3" in mt:
            return True
    except Exception:
        pass
    mid = model_id.lower()
    return ("gemma-3" in mid) or ("gemma3" in mid)


def make_sft_config(**kwargs):
    sig = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sig and "max_seq_length" in kwargs:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    if "max_seq_length" in sig and "max_length" in kwargs:
        kwargs["max_seq_length"] = kwargs.pop("max_length")
    return SFTConfig(**kwargs)


def tokenize_example(
    example: Dict[str, Any],
    tokenizer: Any,
    max_len: int,
    add_token_type_ids: bool,
) -> Dict[str, Any]:
    enc = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.copy(),
    }
    if add_token_type_ids:
        out["token_type_ids"] = [0] * len(input_ids)
    return out


@dataclass
class DataCollatorForCausalLMWithOptionalTTI:
    tokenizer: Any
    label_pad_token_id: int = -100
    add_token_type_ids: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]

        token_type_ids = None
        if "token_type_ids" in features[0]:
            token_type_ids = [f.pop("token_type_ids") for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        padded_labels: List[List[int]] = []
        for lab in labels:
            pad_len = max_len - len(lab)
            padded_labels.append(lab + [self.label_pad_token_id] * pad_len)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        if token_type_ids is not None:
            padded_tti: List[List[int]] = []
            for tti in token_type_ids:
                pad_len = max_len - len(tti)
                padded_tti.append(tti + [0] * pad_len)
            batch["token_type_ids"] = torch.tensor(padded_tti, dtype=torch.long)
        elif self.add_token_type_ids:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

        return batch


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"--- Starting Training: {args.peft_type} on {args.model_id} ---")

    requires_token_type_ids = is_gemma3_model(args.model_id)
    if requires_token_type_ids:
        print("Detected Gemma-3 style model, will include token_type_ids")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map=None,
    )
    model.config.use_cache = False
    model.to("cuda")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if args.peft_type == "vera":
        peft_config = VeraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            target_modules=target_modules,
            vera_dropout=0.05,
            save_projection=True,
        )
    else:
        peft_config = UIOrthoLoRAConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            num_svalues_to_adapt=args.svalues,
            num_svectors_to_adapt=args.svectors,
            uiortholora_dropout=0.05,
            initial_scaler=0.1,
            initial_sigma=0.1,
        )

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.map(alpaca_to_text, remove_columns=dataset.column_names)

    print("Tokenizing dataset (required because we use a custom collator)...")
    dataset = dataset.map(
        lambda x: tokenize_example(x, tokenizer, args.max_len, requires_token_type_ids),
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(0.1 * total_steps)

    training_args = make_sft_config(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        packing=False,
        max_length=args.max_len,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForCausalLMWithOptionalTTI(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        add_token_type_ids=requires_token_type_ids,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    print(f"Saving adapter to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)

    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
