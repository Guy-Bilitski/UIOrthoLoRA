import os
import sys
sys.path.append(os.path.expanduser("/home/guy.bilitski/UIOrthoLoRA/notebooks/tuner_knowledge"))
sys.path.append(os.path.expanduser("/home/guyb/projects/UIOrthoLoRA/notebooks/tuner_knowledge"))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, VeraConfig, PeftConfig, PeftModel, RandLoraConfig, UIOrthoLoRAConfig
import torch
from src.shared_prompt import SYSTEM_PROMPT
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
from triviaQA_load import take_first_n, stream_triviaqa_rc
from src.argument_parser import parse_arguments
from datasets import Dataset
from inference import evaluate_self_consistency, prepare_sc_inputs, get_prompt_template_and_parser
import math
from typing import Any, Dict, List, Optional
from tqdm import tqdm

FT_MODEL_ID_DEFAULT = "ft-A"

import json
from typing import List, Dict, Iterable
from itertools import islice

def stream_jsonl_batches_memory_efficient(
    path: str,
    batch_size: int = 32,
    limit: int = None
) -> Iterable[List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        iterator = (json.loads(line) for line in f if line.strip())
        count = 0

        while True:
            if limit and count >= limit:
                break

            batch = list(islice(iterator, batch_size))
            if not batch:
                break

            yield batch
            count += len(batch)

def stream_jsonl_batches(
    path: str,
    batch_size: int = 32,
    limit: int = None
) -> Iterable[List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        # Read all lines and parse valid JSON objects
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

    # Yield in batches
    total = len(rows)
    for i in range(0, total, batch_size):
        yield rows[i:i + batch_size]



def _score_is_zero(score: Any) -> bool:
    try:
        return float(score) == 0.0
    except (TypeError, ValueError):
        return False

def _is_target_row(row: Dict[str, Any]) -> bool:
    """
    Target rows are those with:
      - is_validation == False
      - base_eval.score == 0 (accepts 0, 0.0, or "0")
    """
    if row.get("is_validation", False):
        return False
    base = row.get("base_eval", {})
    return _score_is_zero(base.get("score", None))

def _is_eligible(row: Dict[str, Any], ft_model_id: str) -> bool:
    """
    Eligible rows are target rows that are NOT already marked train==True for the given ft model.
    Missing 'train' counts as False.
    """
    if not _is_target_row(row):
        return False
    train = (
        row.get("ft_evals", {})
           .get(ft_model_id, {})
           .get("train", False)
    )
    return not bool(train)

def _ensure_ft_entry(row: Dict[str, Any], ft_model_id: str) -> None:
    """
    Ensure row has ft_evals[ft_model_id] dict.
    """
    ft_evals = row.get("ft_evals")
    if not isinstance(ft_evals, dict):
        ft_evals = {}
        row["ft_evals"] = ft_evals
    if ft_model_id not in ft_evals or not isinstance(ft_evals[ft_model_id], dict):
        ft_evals[ft_model_id] = {}

def mark_and_return_number_to_train_inplace(
    jsonl_path: str,
    number_to_train: float,
    ft_model_id: str = FT_MODEL_ID_DEFAULT,
    seed: Optional[int] = 42,
    initialize_missing_flag: bool = True
) -> List[Dict[str, Any]]:
    """
    - Reads a single JSONL file.
    - Optionally initializes missing ft_evals[ft_model_id].train=False for *target* rows.
    - Computes eligible rows (target & not already train==True).
    - Randomly selects N% of eligible rows (ceil; clamped to [0, len(eligible)]).
    - Marks those rows train=True for ft_model_id.
    - Rewrites the SAME file (no temp file).
    - Returns the FULL selected rows (list of dicts).

    percent: 0..100 (values <0 treated as 0, >100 treated as 100).
    """
    # clamp number_to_train
    if number_to_train is None:
        raise ValueError("number_to_train is required (0..100)")

    if seed is not None:
        random.seed(seed)

    # 1) Load all rows
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # 2) Optionally initialize missing train flag as False for target rows
    if initialize_missing_flag:
        for row in rows:
            if _is_target_row(row):
                _ensure_ft_entry(row, ft_model_id)
                if "train" not in row["ft_evals"][ft_model_id]:
                    row["ft_evals"][ft_model_id]["train"] = False

    # 3) Find eligible indices (target & not already train==True)
    eligible_idx = [i for i, r in enumerate(rows) if _is_eligible(r, ft_model_id)]
    batch_idx = random.sample(eligible_idx, number_to_train)

    # 5) Mark selected rows as train=True and collect full rows for return
    selected_rows: List[Dict[str, Any]] = []
    for i in batch_idx:
        row = rows[i]
        _ensure_ft_entry(row, ft_model_id)
        row["ft_evals"][ft_model_id]["train"] = True
        selected_rows.append(row)

    # 6) Rewrite the SAME file
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(selected_rows[0:5])  # Debugging output

    formatted_examples = [
        format_prompt(example)
        for example in selected_rows
    ]


    # 7) Return the selected rows
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

def tokenize_fn(example, tokenizer):
    full_text = example["text"]
    prompt_text = example["input"]

    # Tokenize
    full_tokens = tokenizer(full_text, truncation=True)
    prompt_tokens = tokenizer(prompt_text, truncation=True)

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Build labels
    labels = input_ids.copy()
    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

    # 🔑 Pad labels to match input length (required!)
    while len(labels) < len(input_ids):
        labels.append(-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def write_sc_score_FT_to_jsonl_batch(batch_data, sc_scores, jsonl_file_path, ft_model_id):
    """
    Update self-consistency scores for fine-tuned models in an existing JSONL file.    
    Args:
        batch_data (List[dict]): Original batch data from TriviaQA containing question_id, question, answer
        sc_scores (List[float]): Self-consistency scores for each question in the batch
        jsonl_file_path (str): Path to the existing JSONL file to update
        ft_model_id (str): Fine-tuned model ID (e.g., "ft-A", "ft-B")
    """
    
    if len(batch_data) != len(sc_scores):
        raise ValueError(f"Mismatch: {len(batch_data)} examples but {len(sc_scores)} scores")

    # Create a mapping from question_id to new score for quick lookup
    id_to_score = {}
    for example, score in zip(batch_data, sc_scores):
        question_id = example.get("id")
        if question_id is None:
            raise ValueError(f"Missing 'id' in batch_data example: {example}")
        id_to_score[question_id] = score
    
    # Read all entries from the JSONL file
    jsonl_path = Path(jsonl_file_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")
    
    # Load all rows into memory
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line}")
                    continue
    
    # Update the relevant rows
    updated_count = 0
    for row in rows:
        row_id = row.get("id")
        if row_id in id_to_score:
            # Ensure ft_evals structure exists
            _ensure_ft_entry(row, ft_model_id)
            
            # Update the score (preserve existing train flag if it exists)
            row["ft_evals"][ft_model_id]["score"] = id_to_score[row_id]
            updated_count += 1
    
    # Write all rows back to the file
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Report any missing IDs - this ensures all batch data gets processed
    jsonl_ids = {row.get("id") for row in rows}
    missing_ids = set(id_to_score.keys()) - jsonl_ids
    if missing_ids:
        print(f"Error: Could not find entries for {len(missing_ids)} IDs: {missing_ids}")
        print(f"Batch IDs looking for: {list(id_to_score.keys())[:5]}...")  # Show first 5 for debugging
        print(f"JSONL IDs available: {list(jsonl_ids)[:5]}...")  # Show first 5 for debugging
        raise ValueError(f"Missing entries in JSONL for batch IDs: {missing_ids}")
    
    # Verify all batch entries were processed
    if updated_count != len(id_to_score):
        raise ValueError(f"Expected to update {len(id_to_score)} entries but only updated {updated_count}")


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
        return _build_lora_config(args.lora_rank, args.alpha, args.dropout)
    elif args.peft_type == "vera":
        return _build_vera_config(args.vera_rank, args.alpha, args.dropout)
    elif args.peft_type == "randlora":
        return _build_randlora_config(args.rand_lora_rank, args.alpha, args.dropout)
    elif args.peft_type == "uiortholora":
        return _build_uiortholora_config(args.svalues, args.svectors, args.alpha, args.dropout)
    else:
        raise ValueError(f"Unknown PEFT type: {args.peft_type}")

def _build_lora_config(r, lora_alpha, lora_dropout):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

def _build_vera_config(rank, alpha, dropout):
    return VeraConfig(
        task_type="CAUSAL_LM",           
        r=rank,                                            
        vera_dropout=dropout,                     
        target_modules=["q_proj", "v_proj"], 
    )

def _build_randlora_config(rank, alpha, dropout):
    return RandLoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        randlora_alpha=alpha,
        randlora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],
    )

def _build_uiortholora_config(svalues, svectors, alpha, dropout):
    return UIOrthoLoRAConfig(
        num_svalues_to_adapt=svalues,
        num_svectors_to_adapt=svectors,
        uiortholora_alpha=alpha,
        uiortholora_dropout=dropout,
        fan_in_fan_out=False,
        initial_scaler=0.1,
        initial_sigma=0.1,
        target_modules=["q_proj", "v_proj"],
    )

def load_peft_model(model_id, peft_config):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
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
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
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

def get_training_dataset_mock(batch_num=1000, batch_size=10):
    dataset=stream_triviaqa_rc(batch_size=batch_size, split="train")
    raw_ds = take_first_n(dataset, batch_num)
    formatted_examples = [
        format_prompt(example)
        for batch in raw_ds
        for example in batch
    ]
    return Dataset.from_list(formatted_examples)

def tokenize_dataset(dataset, tokenizer):
    return dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        remove_columns=["label", "text", "input"]
    )

def get_data_collator(tokenizer, model_id):
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model_id,
        padding=True,
        label_pad_token_id=-100
    )

import torch
from tqdm import tqdm

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
    """
    Processes a JSONL file in batches, dynamically reducing the batch size by
    chunking if a CUDA OutOfMemoryError is encountered.
    """
    # The outer loop streams large batches from the file
    for batch in tqdm(
        stream_jsonl_batches(jsonl_path, batch_size=initial_batch_size),
        desc="Evaluating batches"
    ):
        current_chunk_size = initial_batch_size
        success = False  # Flag to indicate if the batch was processed successfully

        # This loop retries the entire batch with a smaller chunk size upon OOM error
        while current_chunk_size >= min_batch_size and not success:
            try:
                batch_sc_scores = []
                # Process the large batch in smaller chunks
                for i in range(0, len(batch), current_chunk_size):
                    sub_batch = batch[i : i + current_chunk_size]

                    # Perform evaluation on the smaller chunk
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
                    # Clear cache after processing each chunk to free memory
                    torch.cuda.empty_cache()

                # If all chunks are processed, write the collected results for the entire batch
                write_sc_score_FT_to_jsonl_batch(batch, batch_sc_scores, jsonl_path, ft_model_id)
                print(f"** Successfully processed batch of size {len(batch)} with chunk size {current_chunk_size}", flush=True)
                success = True  # Mark the entire batch as successfully processed

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()  # Free up memory before retrying
                new_chunk_size = current_chunk_size // 2
                if new_chunk_size < min_batch_size:
                    break # Avoids retrying with a size smaller than the minimum
                print(f"! OOM with chunk size {current_chunk_size}, retrying with {new_chunk_size}", flush=True)
                current_chunk_size = new_chunk_size

        # If the batch could not be processed even with the minimum chunk size
        if not success:
            raise RuntimeError(f"!!! Cannot process batch even with minimum chunk size of {min_batch_size}.")


def main():
    args = parse_arguments()
    ft_model_id = args.model_path.split('/')[-1]

    # === Train the model on the training dataset and save it ===

    # Get model type and tokenizer
    model_id = args.model_id

    if args.include_training:
        training_dataset = mark_and_return_number_to_train_inplace(args.results_path,
                                                        args.training_number, ft_model_id=ft_model_id, seed=args.seed, initialize_missing_flag=True)
        tokenizer = get_tokenizer(model_id)

        # Prepare the dataset
        tokenized_dataset = tokenize_dataset(training_dataset, tokenizer)
        data_collator = get_data_collator(tokenizer, model_id)

        # Build PEFT config and load the model
        peft_config = build_peft_config(args)
        model = load_peft_model(model_id, peft_config)

        # Train the model
        print("Starting training...")
        print(f"=== Using PEFT type: {args.peft_type} ===")
        trainer = get_trainer(model, tokenized_dataset, data_collator, args)
        trainer.train()
        set_contiguous(model)

        # # Save the fine-tuned model
        print(f"Saving model to {args.output_path}")
        os.makedirs(args.output_path, exist_ok=True)
        trainer.save_model(args.output_path)
        tokenizer.save_pretrained(args.output_path)
    
    else:
        print("Loading pre-trained model for evaluation only...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # 2. Load PEFT config
        peft_model_path = args.model_path
        config = PeftConfig.from_pretrained(peft_model_path)

        # 3. Load base model (must match what was used during fine-tuning)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, 
            device_map="cuda", 
            torch_dtype=torch.float16
        )

        # 4. Attach the adapter
        model = PeftModel.from_pretrained(base_model, peft_model_path)


    model.eval()  # Set model to evaluation mode
    torch.compile(model)
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
        

if __name__ == "__main__":
    main()
