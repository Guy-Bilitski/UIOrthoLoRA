# triviaqa_first_n_filtered.py
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


def stream_triviaqa_rc(
    split: str = "train",
    batch_size: int = 32,
    limit=None
) -> Iterable[List[Dict]]:
    """
    Lazily iterate over TriviaQA‑RC in batches without downloading the full dataset.

    Parameters
    ----------
    split : {"train", "validation", "test"}
        Which part of the dataset to stream.
    batch_size : int
        Number of examples per batch.

    Yields
    ------
    List[dict]
        A batch of examples.
    """
    stream = load_dataset(
        "mandarjoshi/trivia_qa", "rc", split=split, streaming=True
    )

    iterator = iter(stream)
    count = 0

    while True:
        if limit and count > limit:
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

    Parameters
    ----------
    stream : Iterable[dict]
        Streaming dataset iterator.
    n : int
        Number of items to grab.
    columns : list[str] | None
        Subset of keys to keep (e.g. ["question", "answer"]).

    Returns
    -------
    list[dict]
        Exactly *n* records in the original order.
    """
    records = list(islice(stream, n))
    if columns:
        records = [{k: rec[k] for k in columns if k in rec} for rec in records]
    return records


# ────────────────────────────────────────────────────────────
# 3. SAVE TO JSON‑LINES
# ────────────────────────────────────────────────────────────
def to_jsonl(records: List[Dict], out_path: str | Path) -> None:
    """
    Save a list of dictionaries to a JSON‑Lines file.

    Parameters
    ----------
    records : list[dict]
        Data to write.
    out_path : str | Path
        Destination file (e.g., "triviaqa_first10.jsonl").
    """
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
    out_file: str = "triviaqa_first10.jsonl",
    split: str = "train",
    columns: Sequence[str] | None = None,
) -> None:
    """
    End‑to‑end pipeline: stream → slice → (optionally filter) → save.

    Parameters
    ----------
    n_samples : int
        How many examples to keep.
    out_file : str
        Output JSON‑Lines file.
    split : str
        Dataset split ("train", "validation", or "test").
    columns : list[str] | None
        Keep only these columns (None = keep all).
    """
    stream = stream_triviaqa_rc(split)
    first_n = take_first_n(stream, n_samples, columns)
    to_jsonl(first_n, out_file)


if __name__ == "__main__":
    # Example: get the first 10 rows with only the question & answer fields
    main(
        n_samples=10,
        out_file="triviaqa_q&a_first10.jsonl",
        columns=["question", "answer"],
    )
