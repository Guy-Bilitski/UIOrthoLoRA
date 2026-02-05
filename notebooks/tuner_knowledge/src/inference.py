# qa_eval.py
import os
import json
import argparse
from tqdm import tqdm

from triviaQA_load import stream_triviaqa_rc
from hotpotqa_load import stream_hotpotqa  # Assuming similar loader exists
from shared_prompt import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from eval_utils import (
    get_tokenizer_and_model,
    evaluate_self_consistency,
)

# Configuration
NUM_DEBUG_SAMPLES = 1

# Supported configurations
SUPPORTED_DATASETS = ["triviaqa", "hotpotqa"]
SUPPORTED_MODELS = {
    "gemma-12b": "google/gemma-3-12b-it",
    "gemma-3b": "google/gemma-3-3b-it",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def get_dataset_loader(dataset: str):
    """Return the appropriate dataset loader based on dataset name."""
    loaders = {
        "triviaqa": stream_triviaqa_rc,
        "hotpotqa": stream_hotpotqa,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {SUPPORTED_DATASETS}")
    return loaders[dataset]


def prepare_sc_inputs(batch, dataset: str):
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
        
        # Handle different answer formats based on dataset
        if dataset == "triviaqa" and isinstance(answer, dict):
            # TriviaQA format: dict with normalized_aliases and normalized_value
            normalized_aliases = answer.get("normalized_aliases", [])
            normalized_value = answer.get("normalized_value", "")

            if normalized_value and normalized_value not in normalized_aliases:
                normalized_aliases.append(normalized_value)
            ground_truths.append(normalized_aliases)
        else:
            # HotspotQA / simple string format
            if isinstance(answer, list):
                ground_truths.append([str(a) for a in answer if a])
            else:
                ground_truths.append([str(answer)] if answer else [])

        questions.append(q)

    return questions, ground_truths


def get_output_path(base_dir: str, model_name: str, dataset: str) -> str:
    """
    Generate output path following the structure:
    {model_name}/{dataset}/workdir/{model_name}_{dataset}_scores.jsonl
    """
    # Extract short model name (e.g., "gemma-12b" from "google/gemma-3-12b-it")
    model_short = model_name.split("/")[-1]
    
    output_dir = os.path.join(base_dir, dataset, "workdir")
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, f"{model_short}_{dataset}_scores-final.jsonl")


def write_sc_scores_to_jsonl_batch(
    batch_data, 
    sc_scores, 
    output_file_path, 
    model_name: str,
    dataset: str
):
    """Write self-consistency scores to a JSONL file."""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    batch_results = []
    for example, score in zip(batch_data, sc_scores):
        # Handle different ID field names across datasets
        example_id = example.get("question_id") or example.get("id") or example.get("_id")
        
        result_entry = {
            "id": example_id, 
            "question": example["question"],
            "answer": example["answer"], 
            "dataset": dataset,
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


def run_evaluation(
    dataset: str,
    model_key: str,
    split: str = "train",
    batch_size: int = 50,
    n_gen: int = 10,
    debug: bool = False,
    base_output_dir: str = "results"
):
    """Run evaluation for a specific dataset and model combination."""
    
    # Resolve model ID
    if model_key in SUPPORTED_MODELS:
        model_id = SUPPORTED_MODELS[model_key]
    else:
        # Assume it's a full model path
        model_id = model_key
    
    print(f"Running evaluation:")
    print(f"  Dataset: {dataset}")
    print(f"  Model: {model_id}")
    print(f"  Split: {split}")
    print(f"  Debug: {debug}")
    
    # Setup paths following the structure from the image
    # e.g., gemma-12b/hotspotqa/workdir/
    model_short = model_key if model_key in SUPPORTED_MODELS else model_id.split("/")[-1]
    output_dir = os.path.join(base_output_dir, model_short, dataset, "workdir")
    output_file_path = os.path.join(output_dir, f"{model_id.replace('/', '_')}_scores.jsonl")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing output file if starting fresh
    if os.path.exists(output_file_path) and not debug:
        print(f"Warning: Output file exists: {output_file_path}")
        print("Appending to existing file. Delete manually if you want to restart.")
    
    print(f"  Output: {output_file_path}")
    
    # Load model and tokenizer
    tokenizer, model = get_tokenizer_and_model(model_id)
    
    # Get dataset loader
    dataset_loader = get_dataset_loader(dataset)
    
    # Run evaluation
    for batch in tqdm(
        dataset_loader(split, batch_size=batch_size), 
        desc=f"Evaluating {dataset} with {model_short}"
    ):
        questions, ground_truths = prepare_sc_inputs(batch, dataset)
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
            batch, sc_scores, output_file_path, model_id.split('/')[-1], dataset
        )
    
    print(f"Evaluation complete. Results saved to: {output_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run QA evaluation with self-consistency scoring"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help=f"Model to use. Shortcuts: {list(SUPPORTED_MODELS.keys())} or full HF model path"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=50,
        help="Batch size for processing (default: 50)"
    )
    parser.add_argument(
        "--n-gen", 
        type=int, 
        default=10,
        help="Number of generations for self-consistency (default: 10)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode with limited samples"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Base output directory (default: results)"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        dataset=args.dataset,
        model_key=args.model,
        split=args.split,
        batch_size=args.batch_size,
        n_gen=args.n_gen,
        debug=args.debug,
        base_output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()