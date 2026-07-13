"""
Shared helpers for the reproduction campaign: prompt template, tokenization,
and structured run logging. Both train_cs.py and eval_cs.py import from here so
LoRA / UIOrthoLoRA / future methods go through identical data + logging paths.

Logging layout (all under notebooks/catastrophic forgetting/):
  models/<run_name>/                 adapter weights + tokenizer
  models/<run_name>/run_config.json  full resolved config + trainable params + git + timing
  results/<run_name>/eval_<ds>.json  per-dataset eval detail
  results/train_registry.jsonl       one line per training run
  results/eval_registry.jsonl        one line per eval
"""
import os
import json
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# --- EXACT prompt templates from LLM-Adapters (train has output; eval does not) ---
def train_prompt(dp):
    if dp["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {dp["instruction"]}

                ### Input:
                {dp["input"]}

                ### Response:
                {dp["output"]}"""  # noqa: E501
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {dp["instruction"]}

                ### Response:
                {dp["output"]}"""  # noqa: E501


def eval_prompt(instruction, inp=None):
    if inp:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {inp}

                ### Response:
                """  # noqa: E501
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def count_trainable(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return tr, tot


def append_registry(name, record):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "a") as f:
        f.write(json.dumps(record) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
