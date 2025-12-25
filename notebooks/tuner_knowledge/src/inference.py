import os
import re
import string
import json
import torch
from typing import List, Set
from tqdm import tqdm
from rapidfuzz import fuzz
from transformers import AutoModelForCausalLM, AutoTokenizer
from triviaQA_load import stream_triviaqa_rc
from shared_prompt import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

# Configuration
NUM_DEBUG_SAMPLES = 1
FUZZY_MATCH_THRESHOLD = 85

# ==============================================================================
# 1. SCORING LOGIC
# ==============================================================================

import unicodedata

def normalize_answer(s):
    """Lower text, remove punctuation/articles/whitespace, and STRIP ACCENTS."""
    
    def remove_accents(text):
        # NFD decomposition splits characters (e.g., 'é' becomes 'e' + '´')
        # We then filter out the non-spacing mark categories ('Mn')
        return ''.join(c for c in unicodedata.normalize('NFD', text) 
                       if unicodedata.category(c) != 'Mn')

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        # Keep the space replacement we fixed earlier!
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    # Apply accent removal FIRST, then the rest
    return white_space_fix(remove_articles(remove_punc(lower(remove_accents(s)))))

def get_numbers(text):
    """Returns a set of all numbers found in the text."""
    # Matches digits, allowing for commas/decimals (e.g. 1,000 or 3.14)
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
        
        # --- A. Digit Safety Check (CRITICAL) ---
        # This prevents "Apollo 7" matching "Apollo 8" even if fuzzy score is high
        gt_nums = get_numbers(gt)
        if pred_nums or gt_nums:
            if pred_nums != gt_nums:
                continue # Numbers don't match, this alias is invalid.

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

        # --- D. Fuzzy Match (The Fix) ---
        # We only get here if numbers matched (or there were no numbers).
        # This catches "Colombia" vs "Columbia" (score ~88)
        if fuzz.token_sort_ratio(pred_norm, gt_norm) >= FUZZY_MATCH_THRESHOLD:
            return True

    return False

def batch_sc_score_triviaqa(
    preds_per_q: List[List[str]],
    gts_per_q: List[List[str]]
) -> List[float]:
    """
    Compute self-consistency (SC) score per question using robust matching.
    """
    scores = []
    for preds, aliases in zip(preds_per_q, gts_per_q):
        correct = sum(1 for p in preds if check_answer_correctness(p, aliases))
        scores.append(correct / len(preds))
    return scores

# ==============================================================================
# 2. EVALUATION & INFRASTRUCTURE
# ==============================================================================

def build_chat_messages(question: str, few_shot_examples: list, system_prompt: str):
    """Build messages list for chat template."""
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add few-shot examples as user/assistant turns
    for ex in few_shot_examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    
    # Add the actual question
    messages.append({"role": "user", "content": question})
    return messages

def prepare_sc_inputs(batch):
    questions = []
    ground_truths = []

    for example in batch:
        q = example["question"]
        normalized_aliases = example["answer"].get("normalized_aliases", [])
        normalized_value = example["answer"].get("normalized_value", "")

        if normalized_value and normalized_value not in normalized_aliases:
            normalized_aliases.append(normalized_value)

        questions.append(q)
        ground_truths.append(normalized_aliases)

    return questions, ground_truths

def get_tokenizer_and_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", dtype=torch.bfloat16)
    model.eval()
    # model = torch.compile(model)
    return tokenizer, model

def evaluate_self_consistency(
        questions: List[str], 
        ground_truths: List[List[str]],
        model, 
        tokenizer,
        system_prompt: str,
        few_shot_examples: list,
        n_gen: int = 10, 
        debug: bool = False
    ) -> List[float]:
    
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

    parsed_per_q = []
    for seqs in preds_per_q:
        group = []
        for text in seqs:
            cleaned_answer = text.strip()
            if cleaned_answer.lower().startswith("answer:"):
                 cleaned_answer = cleaned_answer[7:].strip()
            group.append(cleaned_answer)
        parsed_per_q.append(group)

    scores = batch_sc_score_triviaqa(parsed_per_q, ground_truths)

    if debug:
        num_to_debug = min(NUM_DEBUG_SAMPLES, len(questions))
        for idx in range(num_to_debug):
            print("\n" + "="*60)
            print(f"🔵 Sample {idx+1}/{num_to_debug}")
            print(f"🔵 Question: {questions[idx]}")
            print(f"🔵 Ground Truth Aliases: {ground_truths[idx]}")
            
            # RESTORED: Detailed normalization info for debugging
            aliases_norm = [normalize_answer(a) for a in ground_truths[idx]]
            print(f"🔵 Normalized Aliases: {aliases_norm}")
            print(f"🔵 Model Predictions: {parsed_per_q[idx]}")
            
            print("🔵 Per-prediction results:")
            for i, pred in enumerate(parsed_per_q[idx]):
                pred_norm = normalize_answer(pred)
                is_correct = check_answer_correctness(pred, ground_truths[idx])
                status = "✅" if is_correct else "❌"
                print(f"   [{i+1}] '{pred}' -> normalized: '{pred_norm}' | {status}")
            
            print(f"🔵 Final SC Score: {scores[idx]:.2f}")
            print("="*60 + "\n")

    return scores

def write_sc_scores_to_jsonl_batch(batch_data, sc_scores, output_file_path, model_name="llama-3.2-3b"):
    """
    Write self-consistency scores to a JSONL file in batch mode.
    """
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    batch_results = []
    for i, (example, score) in enumerate(zip(batch_data, sc_scores)):
        result_entry = {
            "id": example["question_id"], 
            "question": example["question"],
            "answer": example["answer"], 
            "is_validation": False, 
            "base_eval": {
                "model_id": model_name,
                "score": score
            }
        }
        batch_results.append(result_entry)

    with open(output_file_path, "a", encoding="utf-8") as outfile:
        for result in batch_results:
            outfile.write(json.dumps(result) + "\n")

def main():
    debug = True
    n_gen = 10
    split = "train"
    model_id = "google/gemma-3-12b-it"

    output_file_path = f"results/{model_id.replace('/', '_')}_scores.jsonl"

    tokenizer, model = get_tokenizer_and_model(model_id)

    for batch in tqdm(stream_triviaqa_rc(split, batch_size=50), desc="Evaluating batches"):
        questions, ground_truths = prepare_sc_inputs(batch)
        sc_scores = evaluate_self_consistency(
            questions, 
            ground_truths, 
            model, 
            tokenizer, 
            SYSTEM_PROMPT,
            FEW_SHOT_EXAMPLES,
            n_gen, 
            debug
        )
        write_sc_scores_to_jsonl_batch(batch, sc_scores, output_file_path, model_id.split('/')[-1])

if __name__ == "__main__":
    main()