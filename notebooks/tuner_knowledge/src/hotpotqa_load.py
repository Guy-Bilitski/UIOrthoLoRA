# hotpotqa_load.py
# -----------------------------------------------------------
# pip install datasets
# -----------------------------------------------------------
from pathlib import Path
from itertools import islice
import json
from typing import Iterable, List, Dict, Sequence

from datasets import load_dataset


# ────────────────────────────────────────────────────────────
# 1. STREAM THE DATA
# ────────────────────────────────────────────────────────────


def stream_hotpotqa(
    split: str = "train",
    batch_size: int = 32,
    limit: int = None,
    config: str = "distractor",
    level: str = None
) -> Iterable[List[Dict]]:
    """
    Lazily iterate over HotpotQA in batches without downloading the full dataset.

    Parameters
    ----------
    split : {"train", "validation"}
        Which part of the dataset to stream.
    batch_size : int
        Number of examples per batch.
    limit : int | None
        Maximum number of batches to yield (None = unlimited).
    config : {"distractor", "fullwiki"}
        HotpotQA configuration:
        - "distractor": includes gold paragraphs + distractor paragraphs
        - "fullwiki": full Wikipedia setting (harder)
    level : {"easy", "medium", "hard"} | None
        Filter by difficulty level. None returns all levels.

    Yields
    ------
    List[dict]
        A batch of examples. Each example contains:
        - id: unique identifier
        - question: the question string
        - answer: the answer string
        - type: "comparison" or "bridge"
        - level: "easy", "medium", or "hard"
        - supporting_facts: dict with 'title' and 'sent_id' lists
        - context: dict with 'title' and 'sentences' lists
    """
    stream = load_dataset(
        "hotpot_qa", config, split=split, streaming=True
    )
    
    if level:
        stream = stream.filter(lambda x: x["level"] == level)

    iterator = iter(stream)
    count = 0

    while True:
        if limit and count >= limit:
            break
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
        count += 1


# ────────────────────────────────────────────────────────────
# 2. TAKE THE FIRST N ROWS
# ────────────────────────────────────────────────────────────
def take_first_n(
    stream: Iterable[Dict], n: int, columns: Sequence[str] | None = None
) -> List[Dict]:
    """
    Collect the first *n* items from an iterable,
    optionally keeping only specific columns.
    """
    records = list(islice(stream, n))
    if columns:
        records = [{k: rec[k] for k in columns if k in rec} for rec in records]
    return records


# ────────────────────────────────────────────────────────────
# 3. SAVE TO JSON‑LINES
# ────────────────────────────────────────────────────────────
def to_jsonl(records: List[Dict], out_path: str | Path) -> None:
    """Save a list of dictionaries to a JSON‑Lines file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    print(f"✅ Saved {len(records)} rows → {out_path}")


# ────────────────────────────────────────────────────────────
# 4. DRIVER
# ────────────────────────────────────────────────────────────
def main(
    n_samples: int = 10,
    out_file: str = "hotpotqa_first10.jsonl",
    split: str = "train",
    columns: Sequence[str] | None = None,
) -> None:
    """
    End‑to‑end pipeline: stream → slice → (optionally filter) → save.
    """
    stream = stream_hotpotqa(split)
    first_n = take_first_n(stream, n_samples, columns)
    to_jsonl(first_n, out_file)


if __name__ == "__main__":
    # Example: get the first 10 rows with only the question & answer fields
    main(
        n_samples=10,
        out_file="hotpotqa_q&a_first10.jsonl",
        columns=["id", "question", "answer", "type", "level"],
    )