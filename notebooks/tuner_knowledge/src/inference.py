import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from triviaQA_load import stream_triviaqa_rc

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
import torch
import string
import json
from langchain.output_parsers import RegexParser
from langchain.schema import OutputParserException
from tqdm import tqdm
import re
from word2number import w2n
from shared_prompt import SYSTEM_PROMPT
from rapidfuzz import fuzz
from typing import List
import datetime
from tqdm import tqdm


def get_prompt_template_and_parser():
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=SYSTEM_PROMPT,
    )

    parser = RegexParser(
        regex=r"Question:.*?\nAnswer:\s*(?P<answer>.+?)(?:\n|$)",  # non-greedy match
        output_keys=["answer"]
    )

    return prompt_template, parser


def prepare_sc_inputs(batch):
    """
    Convert a batch from TriviaQA-RC to questions and normalized ground truths.

    Parameters
    ----------
    batch : List[dict]
        A batch of examples from TriviaQA-RC.

    Returns
    -------
    questions : List[str]
        List of questions.
    ground_truths : List[List[str]]
        List of lists of normalized ground-truth aliases.
    """
    questions = []
    ground_truths = []

    for example in batch:
        q = example["question"]
        normalized_aliases = example["answer"].get("normalized_aliases", [])
        normalized_value = example["answer"].get("normalized_value", "")

        # Include normalized_value only if it's not already in the list
        if normalized_value and normalized_value not in normalized_aliases:
            normalized_aliases.append(normalized_value)

        questions.append(q)
        ground_truths.append(normalized_aliases)

    return questions, ground_truths


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# TODO: Ensure this is good enough!
def batch_sc_score_triviaqa(
    preds_per_q: List[List[str]],
    gts_per_q: List[List[str]],
    threshold: int = 85
) -> List[float]:
    """
    Compute self-consistency (SC) score per question:
    For each group of predictions (e.g., multiple generations), compute the proportion
    of predictions that match any of the normalized gold aliases.
    """
    scores = []

    for preds, aliases in zip(preds_per_q, gts_per_q):
        aliases_norm = [normalize_answer(a) for a in aliases]
        correct = sum(
            any(fuzz.token_sort_ratio(normalize_answer(p), a) >= threshold for a in aliases_norm)
            for p in preds
        )
        scores.append(correct / len(preds))

    return scores

def get_tokenizer_and_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)
    model.eval()
    model = torch.compile(model)
    return tokenizer, model


def evaluate_self_consistency(
        questions, ground_truths,
        prompt_template, parser,
        model, tokenizer, n_gen=5, debug=False):

    prompts = [prompt_template.format(question=q) for q in questions]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            max_new_tokens=10,
            num_return_sequences=n_gen,
            pad_token_id=tokenizer.eos_token_id,
        )

    if debug:
        print("🔵 Questions:", questions)
        print()
        print("🔵 ground truth:", ground_truths)
        print()

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # group into [batch, n_gen]
    preds_per_q = [
        decoded[i * n_gen:(i + 1) * n_gen] for i in range(len(questions))
    ]

    # parse answers (fallback to raw text)
    parsed_per_q = []
    for seqs in preds_per_q:
        group = []
        for text in seqs:
            try:
                group.append(parser.parse(text)["answer"].strip())
            except OutputParserException:
                group.append(text)
            except Exception as e:
                print(f"Error parsing text '{text}': {e}")
                group.append("N/A")
        parsed_per_q.append(group)

    return batch_sc_score_triviaqa(parsed_per_q, ground_truths)


def write_sc_scores_to_jsonl_batch(batch_data, sc_scores, output_file_path, model_name="llama-3.2-3b"):
    """
    Write self-consistency scores to a JSONL file in batch mode.
    
    Args:
        batch_data (List[dict]): Original batch data from TriviaQA
        sc_scores (List[float]): Self-consistency scores for each question in the batch
        output_file_path (str): Path to the output JSONL file
        model_name (str): Name of the model used for evaluation
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Prepare the data to write
    batch_results = []
    for i, (example, score) in enumerate(zip(batch_data, sc_scores)):
        # Create a new entry with the desired structure
        result_entry = {
            "id": example["question_id"],  # Use the actual question_id from dataset
            "question": example["question"],
            "answer": example["answer"],  # Keep the full answer object from dataset
            "is_validation": False,  # Set to False for now as requested
            "base_eval": {
                "model_id": model_name,
                "score": score
            }
        }
        batch_results.append(result_entry)
    
    # Write batch to file (append mode)
    with open(output_file_path, "a", encoding="utf-8") as outfile:
        for result in batch_results:
            outfile.write(json.dumps(result) + "\n")
    

def main():
    debug=False
    n_gen=5
    split="train"
    model_id = "meta-llama/Llama-3.2-3B"
    
    # Add output file path
    output_file_path = f"results/{model_id.replace('/', '_')}_scores.jsonl"
    
    # Comment out model loading for testing
    tokenizer, model = get_tokenizer_and_model(model_id)
    prompt_template, parser = get_prompt_template_and_parser()

    for batch in tqdm(stream_triviaqa_rc(split, batch_size=30), desc="Evaluating batches"):
        questions, ground_truths = prepare_sc_inputs(batch)
        sc_scores = evaluate_self_consistency(questions, ground_truths, prompt_template, parser, model, tokenizer, n_gen, debug)
        # Write the sc scores to the json file (batch processing)
        write_sc_scores_to_jsonl_batch(batch, sc_scores, output_file_path, model_id.split('/')[-1])



if __name__ == "__main__":
    main()
