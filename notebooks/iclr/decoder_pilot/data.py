"""Pinned GSM8K inputs, chat formatting, completion-only targets and answer scoring.

No downloads happen here; ``pilot.py prepare`` materializes the pinned files
through the campaign's sealed-source helper. The official test split is the
held-aside evaluation and is never used for selection or tuning; a fixed seeded
10% of the official train split is the inner-selection set.
"""

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch

SYSTEM_PROMPT = "Solve the math word problem. Show your reasoning briefly, then give the final numeric answer on the last line as '#### <number>'."
SPLIT_SEED = 271828
SELECTION_FRACTION = 0.1
ANSWER_PATTERN = re.compile(r"####\s*(-?\$?[\d,]*\.?\d+)")
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")


def read_gsm8k(directory):
    """Read the sealed pinned parquet shards; returns {'train': [...], 'test': [...]} with stable IDs."""
    from notebooks.iclr.campaign.preparation import verify_source

    manifest = verify_source(directory, "dataset")
    import pyarrow.parquet as pq

    splits = {}
    for split in ("train", "test"):
        files = [f for f in manifest["source"]["files"] if f.startswith("main/") and f"/{split}-" in f]
        if len(files) != 1:
            raise ValueError(f"Expected exactly one pinned main/{split} shard")
        rows = pq.read_table(Path(directory) / files[0]).to_pylist()
        splits[split] = [dict(sample_id=f"gsm8k:{split}:{i}", question=r["question"], answer=r["answer"]) for i, r in enumerate(rows)]
    return splits


def inner_split(train_rows, selection_fraction=SELECTION_FRACTION, seed=SPLIT_SEED):
    """Seeded disjoint train/selection split of the official train rows (no labels to stratify on)."""
    n = len(train_rows)
    count = max(1, min(n - 1, int(round(n * selection_fraction))))
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    selection = sorted(order[:count])
    train = sorted(order[count:])
    return [train_rows[i] for i in train], [train_rows[i] for i in selection]


def gold_answer(answer_text):
    match = ANSWER_PATTERN.search(answer_text)
    if not match:
        raise ValueError("GSM8K solution without '#### <answer>' line")
    return normalize_number(match.group(1))


def normalize_number(text):
    cleaned = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    return str(value.normalize()) if value != value.to_integral() else str(int(value))


def extract_prediction(generated_text):
    """Last '#### <number>' if present, else the last number in the text; None if neither exists."""
    matches = ANSWER_PATTERN.findall(generated_text)
    if matches:
        return normalize_number(matches[-1]), "hash_marker"
    numbers = NUMBER_PATTERN.findall(generated_text)
    if numbers:
        return normalize_number(numbers[-1]), "last_number_fallback"
    return None, "no_number"


def is_correct(prediction, gold):
    return prediction is not None and gold is not None and prediction == gold


def chat_prompt(tokenizer, question, system_prompt=SYSTEM_PROMPT):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer, row, max_length, system_prompt=SYSTEM_PROMPT):
    """Prompt tokens are masked (-100); the completion is the reference solution plus the end token."""
    prompt = chat_prompt(tokenizer, row["question"], system_prompt)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion = row["answer"] + tokenizer.eos_token
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    truncated = len(ids) > max_length
    if truncated:
        ids, labels = ids[:max_length], labels[:max_length]
    return dict(
        sample_id=row["sample_id"],
        input_ids=ids,
        labels=labels,
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
        truncated=truncated,
        gold=gold_answer(row["answer"]),
    )


def mask_completion(prompt_ids, completion_ids, max_length):
    """Pure masking helper used by tests: returns (ids, labels, truncated)."""
    ids = list(prompt_ids) + list(completion_ids)
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    truncated = len(ids) > max_length
    return ids[:max_length], labels[:max_length], truncated


class EncodedExamples:
    """Right-padded tensor batches with a content fingerprint (mirrors campaign TensorExamples)."""

    def __init__(self, encoded, pad_token_id):
        if not encoded:
            raise ValueError("Empty example set")
        self.rows = list(encoded)
        self.sample_ids = tuple(r["sample_id"] for r in self.rows)
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Duplicate sample IDs")
        self.pad_token_id = pad_token_id
        self.size = len(self.rows)
        digest = hashlib.sha256(json.dumps(self.sample_ids).encode())
        for r in self.rows:
            digest.update(json.dumps([r["input_ids"], r["labels"]]).encode())
        self.fingerprint = digest.hexdigest()
        self.completion_tokens = sum(sum(1 for t in r["labels"] if t != -100) for r in self.rows)

    def batch(self, indices, device):
        rows = [self.rows[int(i)] for i in indices]
        width = max(len(r["input_ids"]) for r in rows)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            n = len(r["input_ids"])
            ids[i, :n] = torch.tensor(r["input_ids"])
            labels[i, :n] = torch.tensor(r["labels"])
            mask[i, :n] = 1
        return dict(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))


def completion_nll(logits, labels):
    """Sum of next-token NLL over completion positions and the number of scored tokens."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    losses = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction="none"
    ).view(shift_labels.shape)
    scored = shift_labels != -100
    per_example_sum = (losses * scored).sum(1)
    per_example_count = scored.sum(1)
    return per_example_sum, per_example_count
