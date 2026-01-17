import gc
import os
import sys
# Adjust paths as necessary
sys.path.append(os.path.expanduser("/home/guy.bilitski/UIOrthoLoRA/notebooks/tuner_knowledge"))
sys.path.append(os.path.expanduser("/home/guyb/projects/UIOrthoLoRA/notebooks/tuner_knowledge"))

import torch
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, Gemma3ForConditionalGeneration
from peft import LoraConfig, VeraConfig, PeftConfig, PeftModel, RandLoraConfig, UIOrthoLoRAConfig
from trl import SFTTrainer, SFTConfig
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from datasets import Dataset
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer
# Custom imports
from argument_parser import parse_arguments
from inference import evaluate_self_consistency, prepare_sc_inputs
from shared_prompt import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

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

FT_MODEL_ID_DEFAULT = "ft-A"

# =============================================================================
# CONSTANTS & TEMPLATES
# =============================================================================

# CRITICAL: This must match the token that signifies the start of the Assistant's turn.
# For Gemma/Llama-3, this is usually "<start_of_turn>model".
# The DataCollator will search for this token sequence and mask everything before it.
RESPONSE_TEMPLATE = "<start_of_turn>model"


# =============================================================================
# FILE UTILITIES
# =============================================================================

def count_file_rows_and_duplicates(path: str) -> tuple:
    """Returns (total_rows, unique_ids, duplicate_count)"""
    if not os.path.exists(path):
        return 0, 0, 0
    
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
    if not os.path.exists(path):
        return 0

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
    if not os.path.exists(path):
        print(f"[FILE STATE] {context}: File {path} not found.")
        return
        
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
    """Get ALL samples where base model scored 0 and return as Dataset."""
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


# =============================================================================
# DATA FORMATTING FOR SFTTrainer
# =============================================================================
class Gemma3DataCollator(DataCollatorForLanguageModeling):
    """
    Wrapper that ensures token_type_ids are padded along with input_ids.
    Fixes the crash for Gemma 3 / SigLIP models in SFTTrainer.
    """
    def __call__(self, examples):
        # 1. Let the standard collator handle input_ids, labels, and attention_mask
        batch = super().__call__(examples)
        
        # 2. WE must manually pad token_type_ids to match the shape of input_ids
        if "token_type_ids" in examples[0]:
            # Convert lists to tensors
            type_ids = [torch.tensor(e["token_type_ids"]) for e in examples]
            
            # Pad them to the longest sequence in this batch (creates a rectangle)
            padded = torch.nn.utils.rnn.pad_sequence(type_ids, batch_first=True, padding_value=0)
            
            # 3. Alignment Check:
            # The standard collator might pad to a multiple of 8 (e.g. 50 -> 56).
            # We need to make sure our token_type_ids are also length 56.
            diff = batch["input_ids"].shape[1] - padded.shape[1]
            if diff > 0:
                padded = torch.nn.functional.pad(padded, (0, diff), value=0)
                
            batch["token_type_ids"] = padded
            
        return batch

def get_response_template_for_model(model_id: str):
    """get response template based on model type."""
    if "gemma" in model_id.lower():
        return "<start_of_turn>model"
    elif "llama" in model_id.lower():
        return "<|start_header_id|>assistant<|end_header_id|>"
    
def build_completion_mask(input_ids: List[int], response_token_ids: List[int]) -> List[int]:
    """Build completion mask based on response token IDs."""
    start_index = -1
    n = len(response_token_ids)
    
    # Scan the input_ids to find where the response template occurs
    for i in range(len(input_ids) - n + 1):
        if input_ids[i : i + n] == response_token_ids:
            start_index = i + n  # The answer starts AFTER the template
            break
            
    if start_index == -1:
        # Fallback: If template not found, mask everything (train on nothing)
        return [0] * len(input_ids)
    else:
        return [0] * start_index + [1] * (len(input_ids) - start_index)


def format_for_sft(example: Dict, model_id: str, tokenizer=None) -> Dict:
    """
    Format, Tokenize, and Mask data for Packing.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be passed to format_for_sft")

    # --- 1. Extract the Answer ---
    answer_field = example['answer']
    if isinstance(answer_field, dict):
        # TriviaQA format: {"normalized_value": "...", "aliases": [...], ...}
        raw_answer = answer_field.get('normalized_value', answer_field.get('value', str(answer_field)))
    else:
        # HotpotQA format: answer is directly a string
        raw_answer = str(answer_field)

    # --- 2. Create Full Text with Chat Template ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {example['question']}"},
        {"role": "assistant", "content": raw_answer}
    ]
    
    # Generate the full string (e.g. "<start_of_turn>user...<start_of_turn>model...")
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # --- 3. Tokenization ---
    # We must tokenize now to calculate indices.
    tokenized_full = tokenizer(full_text, add_special_tokens=False)
    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    token_type_ids = [0] * len(input_ids)

    # --- 4. Build the Completion Mask ---
    # We need to find the token sequence for "<start_of_turn>model"
    # Note: Use add_special_tokens=False to avoid adding BOS tokens to the template itself
    response_template = get_response_template_for_model(model_id)
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    completion_mask = build_completion_mask(
        input_ids,
        response_token_ids
    )

    # --- 5. Return the formatted example ---
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "token_type_ids": token_type_ids,
    }


def format_dataset_for_sft(dataset: Dataset, tokenizer, model_id) -> Dataset:
    """Format all examples for SFTTrainer."""
    formatted_dataset = dataset.map(
        format_for_sft,
        fn_kwargs={"tokenizer": tokenizer, "model_id": model_id},
        remove_columns=dataset.column_names,
        desc="Formatting for SFT"
    )
    print(f"[FORMAT] Formatted dataset with {len(formatted_dataset)} examples for SFTTrainer")
    print(f"[FORMAT] Sample formatted example keys: {list(formatted_dataset[0].keys())}")
    return formatted_dataset


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
    print("=" * 70 + "\n")


def log_template_tokenization(tokenizer):
    """Verify how templates tokenize - critical for DataCollatorForLanguageModeling."""
    print("\n" + "=" * 70)
    print("TEMPLATE TOKENIZATION CHECK")
    print("=" * 70)
    
    # Check response template
    resp_tokens = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    print(f"  RESPONSE_TEMPLATE: {repr(RESPONSE_TEMPLATE)}")
    print(f"    Token IDs: {resp_tokens}")
    print(f"    Decoded: {repr(tokenizer.decode(resp_tokens))}")
    
    # Check a full example
    sample_msgs = [
        {"role": "user", "content": "Test Q"},
        {"role": "assistant", "content": "Test A"}
    ]
    # We try to use the chat template to see if our RESPONSE_TEMPLATE exists in it
    try:
        sample = tokenizer.apply_chat_template(sample_msgs, tokenize=False)
        sample_tokens = tokenizer.encode(sample, add_special_tokens=True)
        print(f"\n  Sample chat text: {repr(sample)}")
        
        # Find where response template appears
        found = False
        for i in range(len(sample_tokens) - len(resp_tokens) + 1):
            if sample_tokens[i:i+len(resp_tokens)] == resp_tokens:
                print(f"    Response template found at token index: {i}")
                print(f"    Everything BEFORE index {i}, which is token {sample_tokens[i-1]} will be MASKED (Loss = 0)")
                print(f"    Everything FROM index {i}, which is token {sample_tokens[i]} onwards will be TRAINED (Loss > 0)")
                found = True
                break
        
        if not found:
            print("    ⚠️ WARNING: Response template NOT found as exact subsequence in default chat template!")
            print("    This is expected if your template adds extra spaces or newlines.")
            print("    The DataCollator will attempt to match the tokens flexibly.")
    except Exception as e:
        print(f"    Could not run template check: {e}")
    
    print("=" * 70 + "\n")


def verify_collator_output(tokenizer, dataset, data_collator, n_examples: int = 3):
    print("\n" + "=" * 70)
    print("DATA COLLATOR VERIFICATION")
    print("=" * 70)

    examples = []
    for i in range(min(n_examples, len(dataset))):
        examples.append({
            "input_ids": dataset[i]["input_ids"],
            "attention_mask": dataset[i]["attention_mask"],
            "completion_mask": dataset[i]["completion_mask"],
        })

    batch = data_collator(examples)

    for i in range(len(examples)):
        input_ids = batch["input_ids"][i].tolist()
        labels = batch["labels"][i].tolist()

        n_masked = sum(l == -100 for l in labels)
        n_loss = len(labels) - n_masked

        print(f"\nExample {i+1}: total={len(labels)} masked={n_masked} loss={n_loss}")

        first_loss_idx = next((j for j, l in enumerate(labels) if l != -100), None)
        print("first_loss_idx:", first_loss_idx)

        cm = batch.get("completion_mask", None)
        if cm is not None:
            cm_i = cm[i].tolist()
            print("completion_mask zeros:", cm_i.count(0), "ones:", cm_i.count(1))

        loss_token_ids = [t for t, l in zip(input_ids, labels) if l != -100]
        print("LOSS TEXT (decoded):", repr(tokenizer.decode(loss_token_ids)))

    return True



class SFTLoggingCallback(TrainerCallback):
    def __init__(self, tokenizer, log_every_n_steps: int = 50):
        self.tokenizer = tokenizer
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            if state.log_history:
                latest = state.log_history[-1]
                loss = latest.get("loss", "N/A")
                lr = latest.get("learning_rate", "N/A")
                print(f"[STEP {state.global_step}] loss={loss}, lr={lr}", flush=True)
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        print("\n" + "=" * 70)
        print("TRAINING STARTED")
        print("=" * 70)
        if model is not None:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Epochs: {args.num_train_epochs}")
        print(f"  Learning rate: {args.learning_rate}")
        print("=" * 70 + "\n")


# =============================================================================
# MODEL AND TRAINING
# =============================================================================

def get_tokenizer(model_id: str):
    """Load tokenizer with appropriate settings for training."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # SFT requires right padding for batching
    return tokenizer


def build_peft_config(args):
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
    if "gemma-3" in model_id.lower():
        print(f"[LOAD] Loading Gemma 3 as ConditionalGeneration model: {model_id}")
        return Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cuda",
            dtype=torch.bfloat16
        )
    
    # Fallback for standard LLMs
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.bfloat16
    )


def set_contiguous(model):
    for m in model.modules():
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            base = m.parametrizations.weight[0].base
            if not base.is_contiguous():
                base.data = base.data.contiguous()


def create_sft_trainer(model, tokenizer, train_dataset, peft_config, args):
    data_collator = Gemma3DataCollator(
        tokenizer.pad_token_id,
        completion_only_loss=True,
    )
    
    sft_config = SFTConfig(
        output_dir=args.output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        bf16=True,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="no",
        max_length=1024,
        packing=False,
        remove_unused_columns=False,
    )
    
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
# EVALUATION & POST-PROCESSING
# =============================================================================

def write_sc_score_FT_to_jsonl_batch(batch_data, sc_scores, jsonl_file_path, ft_model_id):
    if len(batch_data) != len(sc_scores):
        raise ValueError(f"Mismatch: {len(batch_data)} examples but {len(sc_scores)} scores")

    id_to_score = {}
    for example, score in zip(batch_data, sc_scores):
        question_id = example.get("id")
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
        print("### Warning: duplication found it the data!!  ###")
        deduplicate_jsonl_file(jsonl_file_path)
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
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


def process_with_dynamic_batch_size(jsonl_path, initial_batch_size, min_batch_size, model, tokenizer, args, ft_model_id):
    print(f"\n[PROCESS] Starting evaluation")
    batch_num = 0
    for batch in tqdm(stream_jsonl_batches(jsonl_path, batch_size=initial_batch_size), desc="Evaluating batches"):
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
                        questions, ground_truths, model, tokenizer, SYSTEM_PROMPT, [], args.sc_number, debug=True
                    )
                    batch_sc_scores.extend(sc_scores)
                    torch.cuda.empty_cache()

                write_sc_score_FT_to_jsonl_batch(batch, batch_sc_scores, jsonl_path, ft_model_id)
                success = True

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_chunk_size = current_chunk_size // 2
                if new_chunk_size < min_batch_size:
                    break
                print(f"! OOM, retrying with {new_chunk_size}", flush=True)
                current_chunk_size = new_chunk_size

        if not success:
            raise RuntimeError(f"Cannot process batch.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_arguments()

    # Sample run banner
    if args.sample_run:
        print("\n" + "!" * 70)
        print("!!! SAMPLE RUN MODE - Testing pipeline with minimal data !!!")
        print("!" * 70 + "\n")

    ft_model_id = args.model_path.split('/')[-1]
    model_id = args.model_id

    print(f"\n{'=' * 70}", flush=True)
    print(f"SFT TRAINING PIPELINE", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  PEFT type: {args.peft_type}", flush=True)
    print(f"  Model ID: {ft_model_id}", flush=True)
    print(f"  Base Model: {model_id}", flush=True)
    print(f"  Response Template (Masking Trigger): {repr(get_response_template_for_model(model_id))}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    log_file_state(args.results_path, "initial")

    if args.include_training:
        # =================================================================
        # STEP 1: TOKENIZER SETUP
        # =================================================================
        print("\n[STEP 1] Tokenizer Setup", flush=True)
        print("-" * 40, flush=True)
        
        tokenizer = get_tokenizer(model_id)
        log_tokenizer_info(tokenizer)
        log_template_tokenization(tokenizer)
        
        # =================================================================
        # STEP 2: DATA PREPARATION
        # =================================================================
        print("\n[STEP 2] Data Preparation", flush=True)
        print("-" * 40, flush=True)
        
        # Get data based on zero scores
        raw_dataset = get_all_zero_score_samples(
            args.results_path,
            ft_model_id=ft_model_id,
        )

        if args.sample_run:
            sample_size = min(args.sample_size, len(raw_dataset))
            raw_dataset = raw_dataset.select(range(sample_size))
            print(f"[SAMPLE RUN] Limited to {sample_size} samples")
            # Override epochs for quick testing
            args.num_epochs = 1
            print(f"[SAMPLE RUN] Epochs set to 1")
        
        # Format using the tokenizer's chat template
        train_dataset = format_dataset_for_sft(raw_dataset, tokenizer, model_id)
        
        print(f"\nDataset size: {len(train_dataset)} examples", flush=True)
        print("\n[SAMPLE DATA - Formatted]", flush=True)
        for i in range(min(1, len(train_dataset))):
            decoded_sample = tokenizer.decode(train_dataset[i]['input_ids'], skip_special_tokens=False)
            print(f"{decoded_sample}...", flush=True)
        
        # =================================================================
        # STEP 3: MODEL LOADING
        # =================================================================
        print("\n[STEP 3] Model Loading", flush=True)
        print("-" * 40, flush=True)
        
        model = load_base_model(model_id)
        peft_config = build_peft_config(args)
        
        print(f"  Base model loaded: {model_id}", flush=True)
        
        # =================================================================
        # STEP 4: TRAINER CREATION & VERIFICATION
        # =================================================================
        print("\n[STEP 4] Trainer Creation & Verification", flush=True)
        print("-" * 40, flush=True)
        
        trainer = create_sft_trainer(model, tokenizer, train_dataset, peft_config, args)
        
        # CRITICAL VERIFICATION: Ensure masking is correct
        collator_ok = verify_collator_output(
            tokenizer, 
            train_dataset, 
            trainer.data_collator, 
            n_examples=3
        )
        
        if not collator_ok:
            raise RuntimeError("Data collator verification FAILED! Aborting training.")
        
        # =================================================================
        # STEP 5: TRAINING
        # =================================================================
        print("\n[STEP 5] Training", flush=True)
        print("-" * 40, flush=True)
        
        trainer.train()
        
        if args.peft_type == "uiortholora":
            set_contiguous(trainer.model)
        
        # =================================================================
        # STEP 6: SAVE MODEL
        # =================================================================
        print("\n[STEP 6] Saving Model", flush=True)
        print("-" * 40, flush=True)
        
        os.makedirs(args.output_path, exist_ok=True)

        # # Delete optimizer states (biggest memory hog after model)
        # trainer.optimizer = None
        # trainer.lr_scheduler = None

        # # Clear all gradients
        # for param in trainer.model.parameters():
        #     param.grad = None

        # # Force garbage collection
        # gc.collect()
        # torch.cuda.empty_cache()

        trainer.save_model(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        print(f"  Model saved to: {args.output_path}", flush=True)

        model = trainer.model

    else:
        # LOAD PRE-TRAINED MODEL
        print("\n[LOADING] Pre-trained model for evaluation", flush=True)
        tokenizer = get_tokenizer(model_id)
        tokenizer.padding_side = "left" # For inference
        
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
    
    # try:
    #     model = torch.compile(model)
    #     print("  torch.compile: Success")
    # except Exception as e:
    #     print(f"  torch.compile: Failed ({e}), continuing without")

    if args.run_qa_inference:
        print("\n[EVALUATION] Running Q&A inference")
        tokenizer.padding_side = "left" # For inference
        process_with_dynamic_batch_size(
            args.results_path, 250, 1, model, tokenizer, args, ft_model_id
        )
    else:
        print("\n[SKIP] Q&A inference (--run_qa_inference not set)")

    log_file_state(args.results_path, "final")
    print(f"\n{'=' * 70}\nPIPELINE COMPLETED\n{'=' * 70}\n")


if __name__ == "__main__":
    main()