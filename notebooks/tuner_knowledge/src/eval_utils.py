# eval_utils.py
# Shared utilities for QA evaluation across datasets (TriviaQA, HotpotQA, etc.)

import re
import string
import unicodedata
import torch
from typing import List
from rapidfuzz import fuzz
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
FUZZY_MATCH_THRESHOLD = 85

# ==============================================================================
# 1. ANSWER NORMALIZATION & SCORING
# ==============================================================================

def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation/articles/whitespace, and strip accents."""
    
    def remove_accents(text):
        return ''.join(c for c in unicodedata.normalize('NFD', text) 
                       if unicodedata.category(c) != 'Mn')

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(remove_accents(s)))))


def get_numbers(text: str) -> set:
    """Returns a set of all numbers found in the text."""
    return set(re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text))


def check_answer_correctness(pred: str, ground_truths: List[str]) -> bool:
    """
    Robust matching strategy:
    1. Digit Safety: If pred has numbers, they MUST match numbers in GT.
    2. Exact Match: Standard normalized exact match.
    3. Token Overlap: Handles reordering/subsets.
    4. Fuzzy Match: Handles typos ("Colombia" vs "Columbia").
    """
    pred_norm = normalize_answer(pred)
    pred_nums = get_numbers(pred) 
    
    for gt in ground_truths:
        gt_norm = normalize_answer(gt)
        
        # --- A. Digit Safety Check ---
        gt_nums = get_numbers(gt)
        if pred_nums or gt_nums:
            if pred_nums != gt_nums:
                continue

        # --- B. Exact Match ---
        if pred_norm == gt_norm:
            return True
            
        # --- C. Token Overlap ---
        pred_tokens = set(pred_norm.split())
        gt_tokens = set(gt_norm.split())
        
        if pred_tokens:
            common = pred_tokens.intersection(gt_tokens)
            # Pred tokens subset of GT tokens
            if len(common) == len(pred_tokens) and len(pred_tokens) > 0:
                return True 
            # GT tokens subset of Pred tokens (allow small chatty buffer)
            if len(common) == len(gt_tokens) and len(gt_tokens) > 0:
                if len(pred_tokens) <= len(gt_tokens) + 2:
                    return True

        # --- D. Fuzzy Match ---
        if fuzz.token_sort_ratio(pred_norm, gt_norm) >= FUZZY_MATCH_THRESHOLD:
            return True

    return False


def batch_sc_score(
    preds_per_q: List[List[str]],
    gts_per_q: List[List[str]]
) -> List[float]:
    """Compute self-consistency (SC) score per question."""
    scores = []
    for preds, aliases in zip(preds_per_q, gts_per_q):
        correct = sum(1 for p in preds if check_answer_correctness(p, aliases))
        scores.append(correct / len(preds))
    return scores


# ==============================================================================
# 2. MODEL & TOKENIZER
# ==============================================================================

def get_tokenizer_and_model(model_id: str):
    """Load tokenizer and model for evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda", 
        dtype=torch.bfloat16
    )
    model.eval()
    return tokenizer, model


# ==============================================================================
# 3. PROMPT BUILDING & GENERATION
# ==============================================================================

def build_chat_messages(question: str, few_shot_examples: list, system_prompt: str) -> list:
    """Build messages list for chat template."""
    messages = [{"role": "system", "content": system_prompt}]
    
    for ex in few_shot_examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    
    messages.append({"role": "user", "content": question})
    return messages


def evaluate_self_consistency(
        questions: List[str], 
        ground_truths: List[List[str]],
        model, 
        tokenizer,
        system_prompt: str,
        few_shot_examples: list,
        n_gen: int = 10, 
        debug: bool = False,
        num_debug_samples: int = 1
    ) -> List[float]:
    """
    Evaluate questions using self-consistency scoring.
    
    Returns a list of SC scores (0.0-1.0) for each question.
    """
    prompts = []
    for q in questions:
        messages = build_chat_messages(q, few_shot_examples, system_prompt)
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    input_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.5, 
            top_p=0.9,
            max_new_tokens=20,
            num_return_sequences=n_gen,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    preds_per_q = [
        decoded[i * n_gen:(i + 1) * n_gen] for i in range(len(questions))
    ]

    # Parse predictions (strip "Answer:" prefix if present)
    parsed_per_q = []
    for seqs in preds_per_q:
        group = []
        for text in seqs:
            cleaned_answer = text.strip()
            if cleaned_answer.lower().startswith("answer:"):
                cleaned_answer = cleaned_answer[7:].strip()
            group.append(cleaned_answer)
        parsed_per_q.append(group)

    scores = batch_sc_score(parsed_per_q, ground_truths)

    if debug:
        _print_debug_info(questions, ground_truths, parsed_per_q, scores, num_debug_samples)

    return scores


def _print_debug_info(questions, ground_truths, parsed_per_q, scores, num_debug_samples):
    """Print detailed debug information for evaluation."""
    num_to_debug = min(num_debug_samples, len(questions))
    for idx in range(num_to_debug):
        print("\n" + "="*60)
        print(f"🔵 Sample {idx+1}/{num_to_debug}")
        print(f"🔵 Question: {questions[idx]}")
        print(f"🔵 Ground Truth: {ground_truths[idx]}")
        
        aliases_norm = [normalize_answer(a) for a in ground_truths[idx]]
        print(f"🔵 Normalized: {aliases_norm}")
        print(f"🔵 Model Predictions: {parsed_per_q[idx]}")
        
        print("🔵 Per-prediction results:")
        for i, pred in enumerate(parsed_per_q[idx]):
            pred_norm = normalize_answer(pred)
            is_correct = check_answer_correctness(pred, ground_truths[idx])
            status = "✅" if is_correct else "❌"
            print(f"   [{i+1}] '{pred}' -> normalized: '{pred_norm}' | {status}")
        
        print(f"🔵 Final SC Score: {scores[idx]:.2f}")
        print("="*60 + "\n")