import os
import sys
sys.path.append(os.path.expanduser("/home/guy.bilitski/UIOrthoLoRA/notebooks/tuner_knowledge"))
sys.path.append(os.path.expanduser("/home/guyb/projects/UIOrthoLoRA/notebooks/tuner_knowledge"))

import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, VeraConfig, PeftConfig, PeftModel, RandLoraConfig, UIOrthoLoRAConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from argument_parser import parse_arguments
from datasets import Dataset
from inference import evaluate_self_consistency, prepare_sc_inputs, get_prompt_template
from typing import Any, Dict, List, Optional, Iterable
from tqdm import tqdm

FT_MODEL_ID_DEFAULT = "ft-A"

# Templates for DataCollatorForCompletionOnlyLM
# The collator masks everything from INSTRUCTION_TEMPLATE up to (not including) RESPONSE_TEMPLATE
# Loss is computed only on tokens AFTER RESPONSE_TEMPLATE
INSTRUCTION_TEMPLATE = "Question:"
RESPONSE_TEMPLATE = "\nAnswer:"


# =============================================================================
# FILE UTILITIES
# =============================================================================

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
    """Remove duplicate rows from JSONL file, keeping the last occurrence of each ID."""
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
    seen_ids = {}
    for i, row in enumerate(rows):
        seen_ids[row.get("id")] = i
    
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


def stream_jsonl_batches(path: str, batch_size: int = 32, limit: int = None) -> Iterable[List[Dict]]:
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


# =============================================================================
# ROW SELECTION UTILITIES  
# =============================================================================

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
    train = row.get("ft_evals", {}).get(ft_model_id, {}).get("train", False)
    return not bool(train)


def _ensure_ft_entry(row: Dict[str, Any], ft_model_id: str) -> None:
    ft_evals = row.get("ft_evals")
    if not isinstance(ft_evals, dict):
        row["ft_evals"] = {}
    if ft_model_id not in row["ft_evals"] or not isinstance(row["ft_evals"][ft_model_id], dict):
        row["ft_evals"][ft_model_id] = {}


def get_all_zero_score_samples(
    jsonl_path: str,
    ft_model_id: str = FT_MODEL_ID_DEFAULT,
) -> Dataset:
    """Get ALL samples where base model scored 0 and return as Dataset.
    
    This ensures all adapters train on identical samples - no random selection.
    """
    print(f"\n[SELECT] Getting all zero-score samples for training")
    print(f"[SELECT] ft_model_id={ft_model_id}")
    log_file_state(jsonl_path, "before selection")

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"[SELECT] Loaded {len(rows)} total rows")

    # Select ALL rows where base model scored 0 (excluding validation)
    selected_rows = [r for r in rows if _is_target_row(r)]
    
    # Mark them as training samples for this ft_model_id
    for row in selected_rows:
        _ensure_ft_entry(row, ft_model_id)
        row["ft_evals"][ft_model_id]["train"] = True

    print(f"[SELECT] Selected {len(selected_rows)} zero-score samples (100% of eligible)")

    # Write back with train markers
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log_file_state(jsonl_path, "after selection")
    return Dataset.from_list(selected_rows)

# def mark_and_return_number_to_train_inplace(
#     jsonl_path: str,
#     number_to_train: int,
#     ft_model_id: str = FT_MODEL_ID_DEFAULT,
#     seed: Optional[int] = 42,
#     initialize_missing_flag: bool = True
# ) -> Dataset:
#     """Select training examples from JSONL, mark them, and return as Dataset."""
#     print(f"\n[MARK] Starting mark_and_return_number_to_train_inplace")
#     print(f"[MARK] ft_model_id={ft_model_id}, number_to_train={number_to_train}")
#     log_file_state(jsonl_path, "before marking")

#     if number_to_train is None:
#         raise ValueError("number_to_train is required")

#     if seed is not None:
#         random.seed(seed)

#     rows = []
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 rows.append(json.loads(line))

#     print(f"[MARK] Loaded {len(rows)} rows")

#     if initialize_missing_flag:
#         for row in rows:
#             if _is_target_row(row):
#                 _ensure_ft_entry(row, ft_model_id)
#                 if "train" not in row["ft_evals"][ft_model_id]:
#                     row["ft_evals"][ft_model_id]["train"] = False

#     eligible_idx = [i for i, r in enumerate(rows) if _is_eligible(r, ft_model_id)]
#     print(f"[MARK] Eligible candidates: {len(eligible_idx)}, sampling: {number_to_train}")

#     batch_idx = random.sample(eligible_idx, min(number_to_train, len(eligible_idx)))

#     selected_rows = []
#     for i in batch_idx:
#         row = rows[i]
#         _ensure_ft_entry(row, ft_model_id)
#         row["ft_evals"][ft_model_id]["train"] = True
#         selected_rows.append(row)

#     print(f"[MARK] Writing {len(rows)} rows back to file")
#     with open(jsonl_path, "w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

#     log_file_state(jsonl_path, "after marking")
#     return Dataset.from_list(selected_rows)


# =============================================================================
# DATA FORMATTING FOR SFTTrainer
# =============================================================================

def format_for_sft(example: Dict) -> Dict:
    question = example["question"]
    answer = example["answer"]["normalized_value"]
    
    text = f"Question: {question}\nAnswer: {answer}"
    return {"text": text}


def format_dataset_for_sft(dataset: Dataset) -> Dataset:
    """Format all examples for SFTTrainer."""
    return dataset.map(
        format_for_sft, 
        remove_columns=[col for col in dataset.column_names if col != "text"],
        desc="Formatting for SFT"
    )


# =============================================================================
# LOGGING / VERIFICATION
# =============================================================================

def log_tokenizer_info(tokenizer):
    """Log tokenizer configuration."""
    print("\n" + "=" * 70)
    print("TOKENIZER INFO")
    print("=" * 70)
    print(f"  Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model max length: {tokenizer.model_max_length}")
    print(f"  Padding side: {tokenizer.padding_side}")
    print(f"  PAD token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"  EOS token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"  BOS token: {repr(tokenizer.bos_token)} (id={tokenizer.bos_token_id})")
    print("=" * 70 + "\n")


def log_template_tokenization(tokenizer):
    """Verify how templates tokenize - critical for DataCollatorForCompletionOnlyLM."""
    print("\n" + "=" * 70)
    print("TEMPLATE TOKENIZATION CHECK")
    print("=" * 70)
    
    # Check instruction template
    instr_tokens = tokenizer.encode(INSTRUCTION_TEMPLATE, add_special_tokens=False)
    print(f"  INSTRUCTION_TEMPLATE: {repr(INSTRUCTION_TEMPLATE)}")
    print(f"    Token IDs: {instr_tokens}")
    print(f"    Decoded: {repr(tokenizer.decode(instr_tokens))}")
    
    # Check response template
    resp_tokens = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    print(f"  RESPONSE_TEMPLATE: {repr(RESPONSE_TEMPLATE)}")
    print(f"    Token IDs: {resp_tokens}")
    print(f"    Decoded: {repr(tokenizer.decode(resp_tokens))}")
    
    # Check a full example
    sample = f"Question: What is 2+2?{RESPONSE_TEMPLATE} 4"
    sample_tokens = tokenizer.encode(sample, add_special_tokens=True)
    print(f"\n  Sample text: {repr(sample)}")
    print(f"    Full token IDs: {sample_tokens}")
    print(f"    Token count: {len(sample_tokens)}")
    
    # Find where response template appears
    for i in range(len(sample_tokens) - len(resp_tokens) + 1):
        if sample_tokens[i:i+len(resp_tokens)] == resp_tokens:
            print(f"    Response template found at position: {i}")
            print(f"    Tokens before (prompt): {sample_tokens[:i]}")
            print(f"    Tokens after (response): {sample_tokens[i+len(resp_tokens):]}")
            break
    else:
        print("    ⚠️ WARNING: Response template NOT found as exact subsequence!")
        print("    This may cause issues with DataCollatorForCompletionOnlyLM")
    
    print("=" * 70 + "\n")


def verify_collator_output(tokenizer, dataset, data_collator, n_examples: int = 3):
    """
    Verify that the data collator correctly creates labels.
    This is THE critical check before training.
    """
    print("\n" + "=" * 70)
    print("DATA COLLATOR VERIFICATION")
    print("=" * 70)
    
    # Manually tokenize examples (simulating what SFTTrainer does)
    examples = []
    for i in range(min(n_examples, len(dataset))):
        text = dataset[i]["text"]
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None
        )
        examples.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        })
    
    # Apply collator - this is what happens during training
    batch = data_collator(examples)
    
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    all_valid = True
    
    for i in range(len(examples)):
        print(f"\n{'─' * 60}")
        print(f"EXAMPLE {i + 1}")
        print(f"{'─' * 60}")
        
        input_ids = batch["input_ids"][i].tolist()
        labels = batch["labels"][i].tolist()
        attention_mask = batch["attention_mask"][i].tolist()
        
        # Count tokens
        n_pad = sum(1 for t in input_ids if t == tokenizer.pad_token_id)
        n_masked_labels = sum(1 for l in labels if l == -100)
        n_loss_labels = sum(1 for l in labels if l != -100)
        
        print(f"\n[COUNTS]")
        print(f"  Total tokens: {len(input_ids)}")
        print(f"  Padding tokens: {n_pad}")
        print(f"  Masked labels (-100): {n_masked_labels}")
        print(f"  Loss labels: {n_loss_labels}")
        
        # Decode full input (without padding)
        non_pad_ids = [t for t, m in zip(input_ids, attention_mask) if m == 1]
        print(f"\n[FULL INPUT] ({len(non_pad_ids)} tokens)")
        print(tokenizer.decode(non_pad_ids))
        
        # Decode ONLY the loss region
        loss_token_ids = [t for t, l in zip(input_ids, labels) if l != -100]
        print(f"\n[LOSS REGION] ({len(loss_token_ids)} tokens)")
        print(f"  Text: {repr(tokenizer.decode(loss_token_ids))}")
        print(f"  Token IDs: {loss_token_ids}")
        
        # Show token-by-token at boundary
        print(f"\n[TOKEN-BY-TOKEN BOUNDARY]")
        # Find first loss token
        first_loss_idx = None
        for j, l in enumerate(labels):
            if l != -100:
                first_loss_idx = j
                break
        
        if first_loss_idx is not None and first_loss_idx > 0:
            # Show a few tokens before and after boundary
            start = max(0, first_loss_idx - 3)
            end = min(len(input_ids), first_loss_idx + 4)
            
            print(f"  {'Pos':<5} {'Token ID':<10} {'Token':<20} {'Label':<10} {'Loss?'}")
            print(f"  {'-'*5} {'-'*10} {'-'*20} {'-'*10} {'-'*5}")
            for j in range(start, end):
                tok_id = input_ids[j]
                tok_str = repr(tokenizer.decode([tok_id]))
                label = labels[j]
                label_str = "MASKED" if label == -100 else str(label)
                loss_indicator = "" if label == -100 else "← LOSS"
                marker = ">>>" if j == first_loss_idx else "   "
                print(f"  {marker}{j:<2} {tok_id:<10} {tok_str:<20} {label_str:<10} {loss_indicator}")
        
        # Validation checks
        print(f"\n[VALIDATION]")
        
        # Check 1: Loss region should not be empty
        if n_loss_labels == 0:
            print(f"  ❌ FAIL: No loss tokens! Model will learn nothing.")
            all_valid = False
        else:
            print(f"  ✓ Has {n_loss_labels} loss tokens")
        
        # Check 2: Loss region should contain the answer
        original_text = dataset[i]["text"]
        answer_start = original_text.find(RESPONSE_TEMPLATE) + len(RESPONSE_TEMPLATE)
        expected_answer = original_text[answer_start:].strip()
        actual_loss_text = tokenizer.decode(loss_token_ids).strip()
        
        # Remove EOS if present for comparison
        if actual_loss_text.endswith(tokenizer.eos_token):
            actual_loss_text = actual_loss_text[:-len(tokenizer.eos_token)].strip()
        
        if expected_answer in actual_loss_text or actual_loss_text in expected_answer:
            print(f"  ✓ Loss region contains expected answer")
        else:
            print(f"  ⚠️ WARNING: Loss text '{actual_loss_text}' vs expected '{expected_answer}'")
        
        # Check 3: EOS should be in loss region (model should learn to stop)
        if tokenizer.eos_token_id in loss_token_ids:
            print(f"  ✓ EOS token in loss region")
        else:
            print(f"  ⚠️ EOS token NOT in loss region (may affect generation stopping)")
    
    print(f"\n{'=' * 70}")
    if all_valid:
        print("✓ ALL CHECKS PASSED - Collator is working correctly")
    else:
        print("❌ SOME CHECKS FAILED - Review output above")
    print("=" * 70 + "\n")
    
    return all_valid


class SFTLoggingCallback(TrainerCallback):
    """
    Custom callback to log training internals.
    Logs actual batch data periodically to verify training is proceeding correctly.
    """
    
    def __init__(self, tokenizer, log_every_n_steps: int = 50):
        self.tokenizer = tokenizer
        self.log_every_n_steps = log_every_n_steps
        self.logged_first_batch = False
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Log at specific intervals."""
        # Always log first step
        if state.global_step == 0:
            self._should_log_next = True
        elif state.global_step % self.log_every_n_steps == 0:
            self._should_log_next = True
        else:
            self._should_log_next = False
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log training metrics."""
        if state.global_step % 10 == 0:  # Every 10 steps, basic metrics
            if state.log_history:
                latest = state.log_history[-1]
                loss = latest.get("loss", "N/A")
                lr = latest.get("learning_rate", "N/A")
                print(f"[STEP {state.global_step}] loss={loss}, lr={lr}", flush=True)
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Log at start of training."""
        print("\n" + "=" * 70)
        print("TRAINING STARTED")
        print("=" * 70)
        
        if model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Trainable %: {100 * trainable_params / total_params:.4f}%")
        
        print(f"  Epochs: {args.num_train_epochs}")
        print(f"  Batch size: {args.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Warmup ratio: {args.warmup_ratio}")
        print("=" * 70 + "\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log at end of training."""
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"  Total steps: {state.global_step}")
        print(f"  Final loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")
        print("=" * 70 + "\n")


def log_first_batch(trainer, tokenizer):
    """
    Log the very first batch to see exactly what goes into the model.
    Call this AFTER trainer is created but BEFORE training.
    """
    print("\n" + "=" * 70)
    print("FIRST BATCH INSPECTION")
    print("=" * 70)
    
    # Get the dataloader
    dataloader = trainer.get_train_dataloader()
    
    # Get first batch
    first_batch = next(iter(dataloader))
    
    print(f"\nBatch keys: {list(first_batch.keys())}")
    print(f"Batch size: {first_batch['input_ids'].shape[0]}")
    print(f"Sequence length: {first_batch['input_ids'].shape[1]}")
    
    # Examine first example in batch
    input_ids = first_batch["input_ids"][0].tolist()
    labels = first_batch["labels"][0].tolist()
    attention_mask = first_batch["attention_mask"][0].tolist()
    
    n_masked = sum(1 for l in labels if l == -100)
    n_loss = sum(1 for l in labels if l != -100)
    
    print(f"\n[FIRST EXAMPLE IN BATCH]")
    print(f"  Input length: {len(input_ids)}")
    print(f"  Masked labels: {n_masked}")
    print(f"  Loss labels: {n_loss}")
    
    # Full sequence
    print(f"\n[FULL INPUT]")
    print(tokenizer.decode(input_ids))
    
    # Loss region
    loss_tokens = [t for t, l in zip(input_ids, labels) if l != -100]
    print(f"\n[LOSS REGION]")
    print(repr(tokenizer.decode(loss_tokens)))
    
    print("\n" + "=" * 70 + "\n")


# =============================================================================
# MODEL AND TRAINING
# =============================================================================

def get_tokenizer(model_id: str):
    """Load tokenizer with appropriate settings for training."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Standard for causal LM training
    return tokenizer


def build_peft_config(args):
    """Build PEFT configuration based on args."""
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.peft_type == "lora":
        return LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM"
        )
    elif args.peft_type == "vera":
        return VeraConfig(
            task_type="CAUSAL_LM",
            r=args.vera_rank,
            vera_dropout=args.dropout,
            target_modules=target_modules,
        )
    elif args.peft_type == "randlora":
        return RandLoraConfig(
            task_type="CAUSAL_LM",
            r=args.rand_lora_rank,
            randlora_alpha=args.alpha,
            randlora_dropout=args.dropout,
            target_modules=target_modules,
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
            target_modules=target_modules,
        )
    else:
        raise ValueError(f"Unknown PEFT type: {args.peft_type}")


def load_base_model(model_id: str):
    """Load base model for training."""
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,  # Using dtype, not torch_dtype (deprecated)
        device_map="cuda",
    )


def set_contiguous(model):
    """Ensure parametrized weights are contiguous (needed for some PEFT methods)."""
    for m in model.modules():
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            base = m.parametrizations.weight[0].base
            if not base.is_contiguous():
                base.data = base.data.contiguous()


def create_sft_trainer(model, tokenizer, train_dataset, peft_config, args):
    """
    Create SFTTrainer with DataCollatorForCompletionOnlyLM.
    """
    
    # Create completion-only collator
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        instruction_template=INSTRUCTION_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training configuration
    sft_config = SFTConfig(
        output_dir=args.output_path,
        
        # Batch settings
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        
        # Training duration
        num_train_epochs=args.num_epochs,
        
        # Optimizer settings
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        optim="adamw_torch",
        
        # Precision
        bf16=True,
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        
        # Saving
        save_strategy="no",
        
        # Sequence length
        max_seq_length=1024,
        
        # Dataset
        dataset_text_field="text",
        packing=False,
        
        # Other
        report_to="none",
        remove_unused_columns=True,
    )
    
    # Create logging callback
    logging_callback = SFTLoggingCallback(tokenizer, log_every_n_steps=50)
    
    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[logging_callback],
    )


# =============================================================================
# EVALUATION
# =============================================================================

def write_sc_score_FT_to_jsonl_batch(batch_data, sc_scores, jsonl_file_path, ft_model_id):
    """Update self-consistency scores in the JSONL file."""
    if len(batch_data) != len(sc_scores):
        raise ValueError(f"Mismatch: {len(batch_data)} examples but {len(sc_scores)} scores")

    id_to_score = {}
    for example, score in zip(batch_data, sc_scores):
        question_id = example.get("id")
        if question_id is None:
            raise ValueError(f"Missing 'id' in batch_data example")
        id_to_score[question_id] = score

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

    updated_count = 0
    for row in rows:
        row_id = row.get("id")
        if row_id in id_to_score:
            _ensure_ft_entry(row, ft_model_id)
            row["ft_evals"][ft_model_id]["score"] = id_to_score[row_id]
            updated_count += 1

    if updated_count != len(id_to_score):
        print(f"[WRITE] Mismatch: updated {updated_count}, expected {len(id_to_score)}. Deduplicating...")
        deduplicate_jsonl_file(jsonl_file_path)
        
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

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if updated_count != len(id_to_score):
        raise ValueError(f"Expected to update {len(id_to_score)} entries but updated {updated_count}")


def process_with_dynamic_batch_size(
    jsonl_path,
    initial_batch_size=30,
    min_batch_size=1,
    prompt_template=None,
    model=None,
    tokenizer=None,
    args=None,
    ft_model_id=None,
):
    """Run evaluation with dynamic batch sizing to handle OOM."""
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_arguments()
    ft_model_id = args.model_path.split('/')[-1]

    print(f"\n{'=' * 70}")
    print(f"SFT TRAINING PIPELINE")
    print(f"{'=' * 70}")
    print(f"  PEFT type: {args.peft_type}")
    print(f"  Model ID: {ft_model_id}")
    print(f"  Training samples: {args.training_number}")
    print(f"{'=' * 70}\n")

    log_file_state(args.results_path, "initial")
    model_id = args.model_id

    if args.include_training:
        # =================================================================
        # STEP 1: DATA PREPARATION
        # =================================================================
        print("\n[STEP 1] Data Preparation")
        print("-" * 40)
        
        # raw_dataset = mark_and_return_number_to_train_inplace(
        #     args.results_path,
        #     args.training_number,
        #     ft_model_id=ft_model_id,
        #     seed=args.seed,
        #     initialize_missing_flag=True
        # )
        raw_dataset = get_all_zero_score_samples(
            args.results_path,
            ft_model_id=ft_model_id,
        )
        train_dataset = format_dataset_for_sft(raw_dataset)
        
        print(f"\nDataset size: {len(train_dataset)} examples")
        print("\n[SAMPLE DATA]")
        for i in range(min(3, len(train_dataset))):
            print(f"  Example {i+1}: {train_dataset[i]['text'][:100]}...")
        
        # =================================================================
        # STEP 2: TOKENIZER SETUP
        # =================================================================
        print("\n[STEP 2] Tokenizer Setup")
        print("-" * 40)
        
        tokenizer = get_tokenizer(model_id)
        log_tokenizer_info(tokenizer)
        log_template_tokenization(tokenizer)
        
        # =================================================================
        # STEP 3: MODEL LOADING
        # =================================================================
        print("\n[STEP 3] Model Loading")
        print("-" * 40)
        
        model = load_base_model(model_id)
        peft_config = build_peft_config(args)
        
        print(f"  Base model loaded: {model_id}")
        print(f"  PEFT config: {args.peft_type}")
        
        # =================================================================
        # STEP 4: TRAINER CREATION & VERIFICATION
        # =================================================================
        print("\n[STEP 4] Trainer Creation & Verification")
        print("-" * 40)
        
        trainer = create_sft_trainer(model, tokenizer, train_dataset, peft_config, args)
        
        # Critical: Verify the collator is working
        collator_ok = verify_collator_output(
            tokenizer, 
            train_dataset, 
            trainer.data_collator, 
            n_examples=3
        )
        
        if not collator_ok:
            raise RuntimeError(
                "Data collator verification FAILED!\n"
                "The response template may not be found in your data.\n"
                f"Expected template: {repr(RESPONSE_TEMPLATE)}"
            )
        
        # Log first actual batch
        log_first_batch(trainer, tokenizer)
        
        # =================================================================
        # STEP 5: TRAINING
        # =================================================================
        print("\n[STEP 5] Training")
        print("-" * 40)
        
        trainer.train()
        
        # Post-training cleanup
        set_contiguous(trainer.model)
        
        # =================================================================
        # STEP 6: SAVE MODEL
        # =================================================================
        print("\n[STEP 6] Saving Model")
        print("-" * 40)
        
        os.makedirs(args.output_path, exist_ok=True)
        trainer.save_model(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        print(f"  Model saved to: {args.output_path}")
        
        model = trainer.model

    else:
        # =================================================================
        # LOAD PRE-TRAINED MODEL
        # =================================================================
        print("\n[LOADING] Pre-trained model for evaluation")
        print("-" * 40)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # For generation

        config = PeftConfig.from_pretrained(args.model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="cuda",
            dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)

    # =================================================================
    # EVALUATION
    # =================================================================
    model.eval()
    
    try:
        model = torch.compile(model)
        print("  torch.compile: Success")
    except Exception as e:
        print(f"  torch.compile: Failed ({e}), continuing without")

    if args.run_qa_inference:
        print("\n[EVALUATION] Running Q&A inference")
        print("-" * 40)
        
        tokenizer.padding_side = "left"  # For generation
        
        prompt_template = get_prompt_template()
        process_with_dynamic_batch_size(
            args.results_path,
            initial_batch_size=500,
            min_batch_size=1,
            prompt_template=prompt_template,
            model=model,
            tokenizer=tokenizer,
            args=args,
            ft_model_id=ft_model_id
        )
    else:
        print("\n[SKIP] Q&A inference (--run_qa_inference not set)")

    log_file_state(args.results_path, "final")
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETED")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()