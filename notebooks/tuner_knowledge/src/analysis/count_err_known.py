#!/usr/bin/env python3
"""
Recompute |D_err| and |D_known| (Table 1) from raw self-consistency score files.

Definitions (recovered from the pipeline; see train.py:_is_target_row and
process_results.py:_kg):
    D_err   = questions whose base self-consistency score == 0.0   (the error /
              zero-score set, i.e. the training set = tr_actual)
    D_known = questions whose base score  > 0.0                    (everything the
              base model got at least partly right)
    D_err and D_known PARTITION the evaluated set: |D_err| + |D_known| = total.

Confirmed against the preserved llama-3b TriviaQA file:
    32,799 (score==0) + 43,724 (score>0) = 76,523.

No GPU needed: scores already live in base_eval.score inside the JSONL files. This
script only counts. When the same qid appears more than once (appended re-evals),
the LAST occurrence wins (matches deduplicate_jsonl_file's keep-last behaviour).

Usage
-----
    python count_err_known.py                          # auto-discover known files
    python count_err_known.py --jsonl A.jsonl B.jsonl  # count specific files
    python count_err_known.py --jsonl F --label gemma-12b/triviaqa
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

# Reported Table 1 values, keyed by "<model>/<dataset>".
PAPER_TABLE = {
    "llama-3b/triviaqa": (32799, 43724),
    "llama-3b/hotpotqa": (11323, 4612),
    "gemma-12b/triviaqa": (20469, 28147),
    "gemma-12b/hotpotqa": (10938, 4723),
}

# Where raw score files may live, relative to this script's directory (src/analysis).
# Any *_scores*.jsonl works: base_eval is identical across adapter files.
SEARCH_ROOTS = [
    "../results",                    # repo layout used by process_results.py
    "~/results",                     # local server copy
    "~/archive/UIOrthoLoRA/notebooks/tuner_knowledge/src/results",
]
MODEL_TOKENS = {
    "llama-3b": "Llama-3.2-3B",
    "gemma-12b": "gemma-3-12b",
}


def load_scores(path: str) -> dict:
    """Return {qid: rounded_score} for a JSONL file, keeping the last write per qid.

    Malformed lines are skipped (some archived files have a few corrupt rows).
    """
    scores = {}
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            s = (o.get("base_eval") or {}).get("score")
            if s is None:
                continue
            scores[o.get("id")] = round(float(s), 1)
    if bad:
        print(f"    (skipped {bad} malformed line(s))")
    return scores


def summarize(label: str, scores: dict) -> None:
    hist = Counter(scores.values())
    total = len(scores)
    d_err = hist[0.0]
    d_known = sum(v for k, v in hist.items() if k > 0.0)  # the recovered definition

    print(f"\n=== {label} ===")
    print(f"  file rows (unique qids): {total:,}")
    print(f"  {'score':>5} {'count':>9} {'pct':>7}")
    for k in sorted(hist):
        print(f"  {k:>5} {hist[k]:>9,} {100*hist[k]/total:>6.2f}%")

    print(f"  |D_err|   (score == 0)  = {d_err:,}")
    print(f"  |D_known| (score  > 0)  = {d_known:,}")
    print(f"  check: err + known      = {d_err + d_known:,}  (should equal rows)")
    # alternative cutoffs, for reference
    print(f"    [alt] score == 1.0    = {hist[1.0]:,}")
    print(f"    [alt] score >= 0.8    = {sum(v for k, v in hist.items() if k >= 0.8):,}")

    if label in PAPER_TABLE:
        p_err, p_known = PAPER_TABLE[label]
        oe = "PASS" if d_err == p_err else "DIFF"
        ok = "PASS" if d_known == p_known else "DIFF"
        print(f"  paper Table 1: |D_err|={p_err:,}  |D_known|={p_known:,}")
        print(f"  -> D_err  {oe} (paper {p_err:,} vs computed {d_err:,})")
        print(f"  -> D_known {ok} (paper {p_known:,} vs computed {d_known:,})")
        if oe == "DIFF" or ok == "DIFF":
            print("     NOTE: a DIFF usually means this file is a later/partial eval "
                  "snapshot than the one used for the paper, not a wrong definition.")


def discover(model: str, dataset: str) -> str | None:
    """Find one score file for model x dataset. Returns first match or None."""
    tok = MODEL_TOKENS[model]
    here = Path(__file__).resolve().parent
    patterns = []
    for root in SEARCH_ROOTS:
        base = Path(root).expanduser()
        if not base.is_absolute():
            base = (here / base).resolve()
        # prefer dataset-scoped paths, then fall back to model-only dirs
        patterns += [
            f"{base}/{model}/{dataset}/**/*scores*.jsonl",
            f"{base}/{model}/{dataset}/*scores*.jsonl",
            f"{base}/{model}/**/*{tok}*scores*.jsonl",
        ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", nargs="+", help="explicit score file(s) to count")
    ap.add_argument("--label", help="label for a single --jsonl file "
                                    "(e.g. gemma-12b/triviaqa) to compare vs Table 1")
    args = ap.parse_args()

    if args.jsonl:
        for i, path in enumerate(args.jsonl):
            label = args.label if (args.label and len(args.jsonl) == 1) else path
            summarize(label, load_scores(path))
        return

    # auto-discovery mode
    print("Auto-discovering score files (dataset split lives in the path, not the "
          "rows; HotpotQA files may be absent locally)...")
    for model in ("llama-3b", "gemma-12b"):
        for dataset in ("triviaqa", "hotpotqa"):
            label = f"{model}/{dataset}"
            path = discover(model, dataset)
            if path is None:
                print(f"\n=== {label} ===\n  NO FILE FOUND (searched {SEARCH_ROOTS})")
                continue
            print(f"\n[{label}] using: {path}")
            summarize(label, load_scores(path))


if __name__ == "__main__":
    main()
