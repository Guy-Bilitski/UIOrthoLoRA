# triviaqa_eval.py
import os
import json
from typing import List
from tqdm import tqdm

from triviaQA_load import stream_triviaqa_rc
from shared_prompt import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from eval_utils import (
    get_tokenizer_and_model,
    evaluate_self_consistency,
)

# Configuration
NUM_DEBUG_SAMPLES = 1


def prepare_sc_inputs(batch):
    """
    Prepare inputs for self-consistency evaluation.
    Supports multiple answer formats:
    - TriviaQA: dict with normalized_aliases and normalized_value
    - HotspotQA: simple string answer
    """
    questions = []
    ground_truths = []

    for example in batch:
        q = example["question"]
        answer = example["answer"]
        
        # Handle different answer formats
        if isinstance(answer, dict):
            # TriviaQA format: dict with normalized_aliases and normalized_value
            normalized_aliases = answer.get("normalized_aliases", [])
            normalized_value = answer.get("normalized_value", "")

            if normalized_value and normalized_value not in normalized_aliases:
                normalized_aliases.append(normalized_value)
            ground_truths.append(normalized_aliases)
        else:
            # HotspotQA / simple string format
            ground_truths.append([str(answer)] if answer else [])

        questions.append(q)

    return questions, ground_truths


def write_sc_scores_to_jsonl_batch(
    batch_data, 
    sc_scores, 
    output_file_path, 
    model_name: str = "gemma-3-12b-it"
):
    """Write self-consistency scores to a JSONL file."""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    batch_results = []
    for example, score in zip(batch_data, sc_scores):
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

    output_file_path = f"results/triviaqa_{model_id.replace('/', '_')}_scores.jsonl"

    tokenizer, model = get_tokenizer_and_model(model_id)

    for batch in tqdm(stream_triviaqa_rc(split, batch_size=50), desc="Evaluating TriviaQA"):
        questions, ground_truths = prepare_sc_inputs(batch)
        sc_scores = evaluate_self_consistency(
            questions, 
            ground_truths, 
            model, 
            tokenizer, 
            SYSTEM_PROMPT,
            FEW_SHOT_EXAMPLES,
            n_gen, 
            debug,
            NUM_DEBUG_SAMPLES
        )
        write_sc_scores_to_jsonl_batch(
            batch, sc_scores, output_file_path, model_id.split('/')[-1]
        )


if __name__ == "__main__":
    main()