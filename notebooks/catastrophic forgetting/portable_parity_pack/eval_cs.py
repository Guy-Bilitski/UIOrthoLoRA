"""
Reproduction eval: commonsense generation accuracy, matching LLM-Adapters
`commonsense_evaluate.py` (prompt template, beam search num_beams=4 /
max_new_tokens=32, answer extraction) but in the modern env (peft 0.19.1,
transformers 5.x) in bf16. Evaluates a trained adapter on one dataset.
"""
import os
import re
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

import run_lib

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "repro/LLM-Adapters/dataset")


# --- EXACT eval prompt template from LLM-Adapters/commonsense_evaluate.py ---
def generate_prompt(instruction, inp=None):
    if inp:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {inp}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def extract_answer(dataset, sentence):
    s = sentence.strip()
    pats = {
        "boolq": r"true|false",
        "piqa": r"solution1|solution2",
        "social_i_qa": r"answer1|answer2|answer3|answer4|answer5",
        "ARC-Challenge": r"answer1|answer2|answer3|answer4|answer5",
        "ARC-Easy": r"answer1|answer2|answer3|answer4|answer5",
        "openbookqa": r"answer1|answer2|answer3|answer4|answer5",
        "hellaswag": r"ending1|ending2|ending3|ending4",
        "winogrande": r"option1|option2",
    }[dataset]
    found = re.findall(pats, s)
    return found[0] if found else ""


def run_eval(model, tokenizer, dataset, batch_size=32, num_beams=4, max_new_tokens=32, limit=0,
             device="cuda:0"):
    """Evaluate an in-memory model on one CS dataset. Returns (acc, correct, total, records)."""
    test_path = os.path.join(DATASET_DIR, dataset, "test.json")
    with open(test_path) as f:
        data = json.load(f)
    if limit > 0:
        data = data[:limit]
    gen_cfg = GenerationConfig(num_beams=num_beams)
    correct, total, records = 0, 0, []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        prompts = [generate_prompt(d["instruction"], d.get("input") or None) for d in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 generation_config=gen_cfg, max_new_tokens=max_new_tokens)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for d, full in zip(batch, decoded):
            resp = full.split("### Response:")[1].strip() if "### Response:" in full else full
            pred = extract_answer(dataset, resp)
            ok = (pred == d.get("answer"))
            correct += int(ok)
            total += 1
            records.append({**d, "pred": pred, "resp": resp[:200], "flag": ok})
        print(f"\r[eval] {dataset} {total}/{len(data)} | acc {correct/total:.4f}", end="", flush=True)
    print(f"\n[eval] {dataset} accuracy = {correct/total:.4f} ({correct}/{total})", flush=True)
    return correct / total, correct, total, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--lora_weights", required=True)
    ap.add_argument("--dataset", default="boolq")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set; >0 for a quick check")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.lora_weights, dtype=torch.bfloat16,
                                      device_map={"": 0})
    model.eval()
    print(f"[eval] {args.dataset} | adapter={args.lora_weights}", flush=True)
    acc, correct, total, records = run_eval(model, tokenizer, args.dataset, args.batch_size,
                                            args.num_beams, args.max_new_tokens, args.limit)

    run_name = os.path.basename(os.path.normpath(args.lora_weights))
    # infer method from the adapter config
    method = "unknown"
    try:
        with open(os.path.join(args.lora_weights, "adapter_config.json")) as f:
            method = json.load(f).get("peft_type", "unknown")
    except Exception:
        pass

    out_path = args.out or os.path.join(HERE, "results", run_name, f"eval_{args.dataset}.json")
    run_lib.write_json(out_path, {"run_name": run_name, "method": method, "dataset": args.dataset,
                                  "adapter": args.lora_weights, "accuracy": acc, "correct": correct,
                                  "total": total, "num_beams": args.num_beams, "records": records})
    summary = {"run_name": run_name, "method": method, "dataset": args.dataset, "accuracy": acc,
               "correct": correct, "total": total, "num_beams": args.num_beams,
               "git_commit": run_lib.git_commit(), "evaluated_at": run_lib.now_iso()}
    run_lib.append_registry("eval_registry.jsonl", summary)
    print(f"[eval] wrote {out_path} | registry updated", flush=True)


if __name__ == "__main__":
    main()
