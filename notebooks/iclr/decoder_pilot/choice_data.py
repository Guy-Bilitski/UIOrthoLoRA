"""CommonsenseQA answer-label data and five-choice scoring.

Design source: `DECODER_FIVE_DAY_REVIEW_20260919.md` section 3. The point of this task is that answer
accuracy and answer probability refer to the same finite set of choices, so a reference rationale's style
cannot dominate the evaluated loss the way it does on GSM8K.

Protocol, fixed before any outcome is seen:

* Training scores exactly ONE token per example, the answer label, with ordinary full-vocabulary
  cross-entropy. The prompt is masked and no end-of-sequence token is scored. There is no rationale
  training and no trainable classification head.
* The five labels A-E must each be a single token at the answer position; ``verify_label_tokens`` checks
  that against the real tokenizer and refuses otherwise, because a multi-token label would silently change
  what "choice NLL" means.
* Outcomes, all exported together: accuracy from the largest of the five choice logits; NLL of the correct
  choice after renormalizing over those five; the full-vocabulary NLL of the label token; and the total
  probability mass the model puts on the five labels. The last one exists so that improved output
  formatting is not mistaken for improved choice discrimination.
* The 1,221-example public **validation** split is reserved for final evaluation and is called held-out
  validation. It is not the official test split, whose labels are not public.
"""

import hashlib
import json
from pathlib import Path

import torch

CHOICE_LABELS = ("A", "B", "C", "D", "E")
SELECTION_FRACTION = 0.10
SPLIT_SEED = 271828
SYSTEM_PROMPT = (
    "Answer the multiple-choice question. Reply with the single letter of the correct option and nothing else."
)


def read_commonsense_qa(directory):
    """Rows from the sealed parquet files, verified against the sealed source manifest."""
    import pyarrow.parquet as pq

    from notebooks.iclr.campaign.preparation import verify_source

    directory = Path(directory)
    verify_source(directory, "dataset")
    splits = {}
    for name, filename in (("train", "train-00000-of-00001.parquet"), ("validation", "validation-00000-of-00001.parquet")):
        rows = pq.read_table(directory / "data" / filename).to_pylist()
        cleaned = []
        for index, row in enumerate(rows):
            labels = list(row["choices"]["label"])
            texts = list(row["choices"]["text"])
            if labels != list(CHOICE_LABELS) or len(texts) != len(CHOICE_LABELS):
                raise ValueError(f"{name} row {index} does not have the five labelled choices A-E: {labels}")
            if row["answerKey"] not in CHOICE_LABELS:
                raise ValueError(f"{name} row {index} has answer key {row['answerKey']!r}")
            cleaned.append(dict(sample_id=f"csqa:{name}:{row['id']}", question=row["question"], choices=texts,
                                answer_label=row["answerKey"], answer_index=CHOICE_LABELS.index(row["answerKey"])))
        splits[name] = cleaned
    return splits


def inner_split(train_rows, selection_fraction=SELECTION_FRACTION, seed=SPLIT_SEED):
    """A fixed, disjoint inner selection split carved out of train. The validation split is never touched."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_rows), generator=generator).tolist()
    cut = int(round(len(train_rows) * selection_fraction))
    selection = [train_rows[i] for i in sorted(order[:cut])]
    remaining = [train_rows[i] for i in sorted(order[cut:])]
    if len({r["sample_id"] for r in selection} & {r["sample_id"] for r in remaining}):
        raise ValueError("Inner selection split overlaps the training remainder")
    return remaining, selection


def choice_block(choices):
    return "\n".join(f"{label}. {text}" for label, text in zip(CHOICE_LABELS, choices))


def chat_prompt(tokenizer, row, system_prompt=SYSTEM_PROMPT):
    user = f"{row['question']}\n{choice_block(row['choices'])}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def verify_label_tokens(tokenizer):
    """Each of A-E must be one token at the answer position, or 'choice NLL' would not mean what it says."""
    mapping, offenders = {}, []
    for label in CHOICE_LABELS:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            offenders.append((label, ids))
        else:
            mapping[label] = ids[0]
    if offenders:
        raise ValueError("Answer labels must be single tokens at the answer position: " + repr(offenders))
    if len(set(mapping.values())) != len(CHOICE_LABELS):
        raise ValueError("Answer labels must map to distinct tokens: " + repr(mapping))
    return mapping


def encode_example(tokenizer, row, max_length, label_tokens, system_prompt=SYSTEM_PROMPT):
    """Prompt masked; exactly one scored token, the answer label. No EOS is scored."""
    prompt = chat_prompt(tokenizer, row, system_prompt)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    truncated = len(prompt_ids) + 1 > max_length
    if truncated:
        # Keep the tail of the prompt, which carries the choices and the generation marker.
        prompt_ids = prompt_ids[-(max_length - 1):]
    answer_id = label_tokens[row["answer_label"]]
    ids = prompt_ids + [answer_id]
    labels = [-100] * len(prompt_ids) + [answer_id]
    return dict(sample_id=row["sample_id"], input_ids=ids, labels=labels, prompt_length=len(prompt_ids),
                completion_length=1, truncated=truncated, answer_index=row["answer_index"],
                answer_label=row["answer_label"])


class ChoiceExamples:
    """Right-padded batches plus the answer position, with a content fingerprint."""

    def __init__(self, encoded, pad_token_id):
        if not encoded:
            raise ValueError("Empty example set")
        self.rows = list(encoded)
        self.sample_ids = tuple(r["sample_id"] for r in self.rows)
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Duplicate sample IDs")
        self.pad_token_id = pad_token_id
        self.size = len(self.rows)
        self.fingerprint = hashlib.sha256(json.dumps(self.sample_ids).encode()).hexdigest()

    def batch(self, indices, device):
        rows = [self.rows[int(i)] for i in indices]
        width = max(len(r["input_ids"]) for r in rows)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        answer_position = torch.zeros(len(rows), dtype=torch.long)
        for i, row in enumerate(rows):
            n = len(row["input_ids"])
            ids[i, :n] = torch.tensor(row["input_ids"])
            labels[i, :n] = torch.tensor(row["labels"])
            mask[i, :n] = 1
            answer_position[i] = row["prompt_length"] - 1  # position whose logits predict the label
        return dict(input_ids=ids.to(device), attention_mask=mask.to(device)), labels.to(device), answer_position.to(device)


@torch.no_grad()
def score_choices(logits, answer_position, answer_index, choice_token_ids):
    """Per-example five-choice outcomes and the full-vocabulary label loss, from one forward pass."""
    rows = torch.arange(logits.shape[0], device=logits.device)
    at_answer = logits[rows, answer_position.to(logits.device)].float()
    log_probs = torch.log_softmax(at_answer, dim=-1)
    choice_ids = torch.tensor(choice_token_ids, device=logits.device)
    choice_logits = at_answer[:, choice_ids]
    choice_log_probs = torch.log_softmax(choice_logits, dim=-1)
    target = answer_index.to(logits.device)
    return dict(
        choice_logits=choice_logits.cpu(),
        predicted_index=choice_logits.argmax(dim=-1).cpu(),
        correct=(choice_logits.argmax(dim=-1) == target).cpu(),
        choice_nll=(-choice_log_probs.gather(1, target[:, None]).squeeze(1)).cpu(),
        full_vocabulary_label_nll=(-log_probs.gather(1, choice_ids[target][:, None]).squeeze(1)).cpu(),
        choice_probability_mass=log_probs[:, choice_ids].exp().sum(dim=-1).cpu(),
    )


def summarize_choices(scored):
    """Aggregate the four registered quantities. Accuracy and choice NLL are co-primary."""
    n = len(scored["correct"])
    return dict(
        examples=n,
        accuracy=float(scored["correct"].double().mean()) if n else None,
        correct=int(scored["correct"].sum()),
        choice_nll_mean=float(scored["choice_nll"].double().mean()) if n else None,
        full_vocabulary_label_nll_mean=float(scored["full_vocabulary_label_nll"].double().mean()) if n else None,
        choice_probability_mass_mean=float(scored["choice_probability_mass"].double().mean()) if n else None,
        choice_probability_mass_min=float(scored["choice_probability_mass"].min()) if n else None,
        note=("Accuracy and choice NLL are co-primary. Full-vocabulary label NLL and choice probability mass "
              "are retained so that better output formatting is not mistaken for better choice discrimination."),
    )
