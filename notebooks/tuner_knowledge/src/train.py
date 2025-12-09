import os
import sys
sys.path.append(os.path.expanduser("/home/guy.bilitski/UIOrthoLoRA/notebooks/tuner_knowledge"))
sys.path.append(os.path.expanduser("/home/guyb/projects/UIOrthoLoRA/notebooks/tuner_knowledge"))

import torch
import json
import random
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, VeraConfig, PeftConfig, PeftModel, RandLoraConfig, UIOrthoLoRAConfig
from shared_prompt import SYSTEM_PROMPT
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
from triviaQA_load import take_first_n, stream_triviaqa_rc
from argument_parser import parse_arguments
from datasets import Dataset
from inference import evaluate_self_consistency, prepare_sc_inputs, get_prompt_template_and_parser
from typing import Any, Dict, List, Optional, Iterable
from tqdm import tqdm
from itertools import islice

FT_MODEL_ID_DEFAULT = "ft-A"


def is_gemma3_model(model_id: str) -> bool:
    """Check if the model is a Gemma3 model that requires token_type_ids."""
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(config, 'model_type', '').lower()
        return 'gemma3' in model_type
    except Exception:
        # Fallback to string matching
        return 'gemma-3' in model_id.lower() or 'gemma3' in model_id.lower()


def count_file_rows_and_duplicates(path: str) -> tuple:
    """Returns (total_rows, unique_ids, duplicate_count)"""
    with open(path, "r", encoding="utf-8") as f:
        ids = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    ids.append(row.get("id"))
                except json.JSONDecodeError:
                    pass
    
    total = len(ids)
    unique = len(set(ids))
    duplicates = total - unique
    return total, unique, duplicates


def deduplicate_jsonl_file(path: str) -> int:
    """
    Remove duplicate rows from JSONL file, keeping the last occurrence of each ID.
    Returns the number of duplicates removed.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    original_count = len(rows)
    
    # Keep last occurrence of each ID (later entries may have more data)
    seen_ids = {}
    for i, row in enumerate(rows):
        row_id = row.get("id")
        seen_ids[row_id] = i
    
    # Build deduplicated list preserving original order
    unique_indices = sorted(seen_ids.values())
    deduped_rows = [rows[i] for i in unique_indices]
    
    removed_count = original_count - len(deduped_rows)
    
    if removed_count > 0:
        print(f"[DEDUP] Removing {removed_count} duplicates from {path}")
        with open(path, "w", encoding="utf-8") as f:
            for row in deduped_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[DEDUP] File now has {len(deduped_rows)} rows")
    
    return removed_count


def log_file_state(path: str, context: str):
    """Simple one-line file state logging"""
    total, unique, dups = count_file_rows_and_duplicates(path)
    status = "✓" if dups == 0 else f"⚠️ {dups} DUPLICATES"
    print(f"[FILE STATE] {context}: rows={total}, unique_ids={unique} {status}", flush=True)


def stream_jsonl_batches(
    path: str,
    batch_size: int = 32,
    limit: int = None
) -> Iterable[List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = []
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping line {i+1}: JSON decode error: {e}")
                continue
            if limit and len(rows) >= limit:
                break

    print(f"[STREAM] Loaded {len(rows)} rows, will yield in batches of {batch_size}")
    
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def _score_is_zero(score: Any) -> bool:
    try:
        return float(score) == 0.0
    except (TypeError, ValueError):
        return False


def _is_target_row(row: Dict[str, Any]) -> bool:
    if row.get("is_validation", False):
        return False
    base = row.get("base_eval", {})
    return _score_is_zero(base.get("score", None))


def _is_eligible(row: Dict[str, Any], ft_model_id: str) -> bool:
    if not _is_target_row(row):
        return False
    train = (
        row.get("ft_evals", {})
           .get(ft_model_id, {})
           .get("train", False)
    )
    return not bool(train)


def _ensure_ft_entry(row: Dict[str, Any], ft_model_id: str) -> None:
    ft_evals = row.get("ft_evals")
    if not isinstance(ft_evals, dict):
        ft_evals = {}
        row["ft_evals"] = ft_evals
    if ft_model_id not in ft_evals or not isinstance(ft_evals[ft_model_id], dict):
        ft_evals[ft_model_id] = {}


def mark_and_return_number_to_train_inplace(
    jsonl_path: str,
    number_to_train: int,
    ft_model_id: str = FT_MODEL_ID_DEFAULT,
    seed: Optional[int] = 42,
    initialize_missing_flag: bool = True
) -> Dataset:
    
    print(f"\n[MARK] Starting mark_and_return_number_to_train_inplace")
    print(f"[MARK] ft_model_id={ft_model_id}, number_to_train={number_to_train}")
    
    log_file_state(jsonl_path, "before marking")

    if number_to_train is None:
        raise ValueError("number_to_train is required")

    if seed is not None:
        random.seed(seed)

    # Load all rows
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"[MARK] Loaded {len(rows)} rows")

    # Initialize missing train flags
    if initialize_missing_flag:
        for row in rows:
            if _is_target_row(row):
                _ensure_ft_entry(row, ft_model_id)
                if "train" not in row["ft_evals"][ft_model_id]:
                    row["ft_evals"][ft_model_id]["train"] = False

    # Find eligible indices
    eligible_idx = [i for i, r in enumerate(rows) if _is_eligible(r, ft_model_id)]
    print(f"[MARK] Eligible candidates: {len(eligible_idx)}, sampling: {number_to_train}")

    batch_idx = random.sample(eligible_idx, min(number_to_train, len(eligible_idx)))

    # Mark selected rows
    selected_rows = []
    for i in batch_idx:
        row = rows[i]
        _ensure_ft_entry(row, ft_model_id)
        row["ft_evals"][ft_model_id]["train"] = True
        selected_rows.append(row)

    # Write back
    print(f"[MARK] Writing {len(rows)} rows back to file")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log_file_state(jsonl_path, "after marking")

    formatted_examples = [format_prompt(example) for example in selected_rows]
    return Dataset.from_list(formatted_examples)


def format_prompt(example):
    answer = example["answer"]["normalized_value"]
    prompt = SYSTEM_PROMPT.replace("{question}", example["question"])
    full = prompt + " " + answer
    return {
        "text": full,
        "input": prompt,
        "label": answer
    }


def tokenize_fn(example, tokenizer, add_token_type_ids=False):
    """
    Tokenize example for causal LM training.
    
    Args:
        example: Dict with 'text' and 'input' keys
        tokenizer: HuggingFace tokenizer
        add_token_type_ids: If True, add token_type_ids (required for Gemma3)
    """
    full_text = example["text"]
    prompt_text = example["input"]

    full_tokens = tokenizer(full_text, truncation=True)
    prompt_tokens = tokenizer(prompt_text, truncation=True)

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()
    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

    while len(labels) < len(input_ids):
        labels.append(-100)

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
    # Add token_type_ids for Gemma3 (all zeros for causal LM)
    if add_token_type_ids:
        result["token_type_ids"] = [0] * len(input_ids)
    
    return result


def write_sc_score_FT_to_jsonl_batch(batch_data, sc_scores, jsonl_file_path, ft_model_id):
    """Update self-consistency scores in the JSONL file."""
    
    if len(batch_data) != len(sc_scores):
        raise ValueError(f"Mismatch: {len(batch_data)} examples but {len(sc_scores)} scores")

    # Build id->score mapping
    id_to_score = {}
    for example, score in zip(batch_data, sc_scores):
        question_id = example.get("id")
        if question_id is None:
            raise ValueError(f"Missing 'id' in batch_data example")
        id_to_score[question_id] = score

    # Read file
    jsonl_path = Path(jsonl_file_path)
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Update rows
    updated_count = 0
    for row in rows:
        row_id = row.get("id")
        if row_id in id_to_score:
            _ensure_ft_entry(row, ft_model_id)
            row["ft_evals"][ft_model_id]["score"] = id_to_score[row_id]
            updated_count += 1

    # If mismatch, we have duplicates - dedupe and retry
    if updated_count != len(id_to_score):
        print(f"[WRITE] Mismatch: updated {updated_count}, expected {len(id_to_score)}. Deduplicating...")
        deduplicate_jsonl_file(jsonl_file_path)
        
        # Reload and retry
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        
        updated_count = 0
        for row in rows:
            row_id = row.get("id")
            if row_id in id_to_score:
                _ensure_ft_entry(row, ft_model_id)
                row["ft_evals"][ft_model_id]["score"] = id_to_score[row_id]
                updated_count += 1

    # Write back
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if updated_count != len(id_to_score):
        raise ValueError(f"Expected to update {len(id_to_score)} entries but updated {updated_count}")


def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def set_contiguous(model):
    for m in model.modules():
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            base = m.parametrizations.weight[0].base
            if not base.is_contiguous():
                base.data = base.data.contiguous()


def build_peft_config(args):
    if args.peft_type == "lora":
        return LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"
        )
    elif args.peft_type == "vera":
        return VeraConfig(
            task_type="CAUSAL_LM",
            r=args.vera_rank,
            vera_dropout=args.dropout,
            target_modules=["q_proj", "v_proj"],
        )
    elif args.peft_type == "randlora":
        return RandLoraConfig(
            task_type="CAUSAL_LM",
            r=args.rand_lora_rank,
            randlora_alpha=args.alpha,
            randlora_dropout=args.dropout,
            target_modules=["q_proj", "v_proj"],
        )
    elif args.peft_type == "uiortholora":
        return UIOrthoLoRAConfig(
            num_svalues_to_adapt=args.svalues,
            num_svectors_to_adapt=args.svectors,
            uiortholora_alpha=args.alpha,
            uiortholora_dropout=args.dropout,
            fan_in_fan_out=False,
            initial_scaler=0.1,
            initial_sigma=0.1,
            target_modules=["q_proj", "v_proj"],
        )
    else:
        raise ValueError(f"Unknown PEFT type: {args.peft_type}")


def load_peft_model(model_id, peft_config):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model


def get_trainer(model, tokenized_dataset, data_collator, args):
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="none"
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )


def tokenize_dataset(dataset, tokenizer, add_token_type_ids=False):
    """
    Tokenize dataset with optional token_type_ids support.
    
    Args:
        dataset: HuggingFace Dataset
        tokenizer: HuggingFace tokenizer
        add_token_type_ids: If True, add token_type_ids (required for Gemma3)
    """
    return dataset.map(
        lambda x: tokenize_fn(x, tokenizer, add_token_type_ids=add_token_type_ids),
        remove_columns=["label", "text", "input"]
    )


@dataclass
class DataCollatorWithTokenTypeIds:
    """
    Custom data collator that handles token_type_ids for Gemma3 models.
    Falls back to standard behavior for other models.
    """
    tokenizer: Any
    label_pad_token_id: int = -100
    add_token_type_ids: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate labels from features
        labels = None
        if "labels" in features[0]:
            labels = [f.pop("labels") for f in features]
        
        # Separate token_type_ids if present
        token_type_ids = None
        if "token_type_ids" in features[0]:
            token_type_ids = [f.pop("token_type_ids") for f in features]
        
        # Pad the rest using tokenizer
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        # Handle labels separately with proper padding
        if labels is not None:
            max_len = batch["input_ids"].shape[1]
            padded_labels = []
            for label in labels:
                padding_len = max_len - len(label)
                # Pad on the left since padding_side="left"
                padded_labels.append([self.label_pad_token_id] * padding_len + label)
            batch["labels"] = torch.tensor(padded_labels)
        
        # Handle token_type_ids with proper padding
        if token_type_ids is not None:
            max_len = batch["input_ids"].shape[1]
            padded_token_type_ids = []
            for tti in token_type_ids:
                padding_len = max_len - len(tti)
                # Pad on the left with 0s
                padded_token_type_ids.append([0] * padding_len + tti)
            batch["token_type_ids"] = torch.tensor(padded_token_type_ids)
        elif self.add_token_type_ids:
            # Create token_type_ids if required but not in features
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        
        return batch


def get_data_collator(tokenizer, model_id, add_token_type_ids=False):
    """
    Get appropriate data collator based on model type.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model_id: Model identifier string
        add_token_type_ids: If True, use custom collator with token_type_ids support
    """
    if add_token_type_ids:
        return DataCollatorWithTokenTypeIds(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            add_token_type_ids=True
        )
    else:
        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model_id,
            padding=True,
            label_pad_token_id=-100
        )


def process_with_dynamic_batch_size(
    jsonl_path,
    initial_batch_size=30,
    min_batch_size=1,
    prompt_template=None,
    parser=None,
    model=None,
    tokenizer=None,
    args=None,
    ft_model_id=None,
):
    print(f"\n[PROCESS] Starting evaluation")
    log_file_state(jsonl_path, "at start of evaluation")
    
    batch_num = 0
    for batch in tqdm(
        stream_jsonl_batches(jsonl_path, batch_size=initial_batch_size),
        desc="Evaluating batches"
    ):
        batch_num += 1
        current_chunk_size = initial_batch_size
        success = False

        while current_chunk_size >= min_batch_size and not success:
            try:
                batch_sc_scores = []
                for i in range(0, len(batch), current_chunk_size):
                    sub_batch = batch[i : i + current_chunk_size]
                    questions, ground_truths = prepare_sc_inputs(sub_batch)
                    sc_scores = evaluate_self_consistency(
                        questions,
                        ground_truths,
                        prompt_template,
                        parser,
                        model,
                        tokenizer,
                        args.sc_number
                    )
                    batch_sc_scores.extend(sc_scores)
                    print(f"\n* Processed chunk of size {len(sub_batch)}", flush=True)
                    torch.cuda.empty_cache()

                print(f"[BATCH {batch_num}] Writing {len(batch)} items with {len(batch_sc_scores)} scores")
                write_sc_score_FT_to_jsonl_batch(batch, batch_sc_scores, jsonl_path, ft_model_id)
                print(f"** Successfully processed batch {batch_num}", flush=True)
                success = True

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_chunk_size = current_chunk_size // 2
                if new_chunk_size < min_batch_size:
                    break
                print(f"! OOM with chunk size {current_chunk_size}, retrying with {new_chunk_size}", flush=True)
                current_chunk_size = new_chunk_size

        if not success:
            raise RuntimeError(f"Cannot process batch even with minimum chunk size of {min_batch_size}.")

    log_file_state(jsonl_path, "at end of evaluation")


def main():
    args = parse_arguments()
    ft_model_id = args.model_path.split('/')[-1]

    print(f"\n{'='*60}")
    print(f"[MAIN] Starting")
    print(f"[MAIN] peft_type={args.peft_type}, ft_model_id={ft_model_id}")
    print(f"[MAIN] training_number={args.training_number}")
    print(f"{'='*60}\n")

    log_file_state(args.results_path, "initial")

    model_id = args.model_id
    
    # Check if model requires token_type_ids (e.g., Gemma3)
    requires_token_type_ids = is_gemma3_model(model_id)
    if requires_token_type_ids:
        print(f"[MAIN] Detected Gemma3 model - will include token_type_ids")

    if args.include_training:
        training_dataset = mark_and_return_number_to_train_inplace(
            args.results_path,
            args.training_number,
            ft_model_id=ft_model_id,
            seed=args.seed,
            initialize_missing_flag=True
        )
        tokenizer = get_tokenizer(model_id)
        tokenized_dataset = tokenize_dataset(
            training_dataset, 
            tokenizer, 
            add_token_type_ids=requires_token_type_ids
        )
        data_collator = get_data_collator(
            tokenizer, 
            model_id, 
            add_token_type_ids=requires_token_type_ids
        )

        peft_config = build_peft_config(args)
        model = load_peft_model(model_id, peft_config)

        print("Starting training...")
        print(f"=== Using PEFT type: {args.peft_type} ===")
        trainer = get_trainer(model, tokenized_dataset, data_collator, args)
        trainer.train()
        set_contiguous(model)

        print(f"Saving model to {args.output_path}")
        os.makedirs(args.output_path, exist_ok=True)
        trainer.save_model(args.output_path)
        tokenizer.save_pretrained(args.output_path)

    else:
        print("Loading pre-trained model for evaluation only...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        peft_model_path = args.model_path
        config = PeftConfig.from_pretrained(peft_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="cuda",
            dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, peft_model_path)

    model.eval()
    torch.compile(model)

    if args.run_qa_inference:
        prompt_template, parser = get_prompt_template_and_parser()
        process_with_dynamic_batch_size(
            args.results_path,
            initial_batch_size=500,
            min_batch_size=1,
            prompt_template=prompt_template,
            parser=parser,
            model=model,
            tokenizer=tokenizer,
            args=args,
            ft_model_id=ft_model_id
        )
    else:
        print("\nSkipping Q&A inference evaluation (--run_qa_inference not set)")

    log_file_state(args.results_path, "final")
    print(f"\n[MAIN] Completed")


if __name__ == "__main__":
    main()