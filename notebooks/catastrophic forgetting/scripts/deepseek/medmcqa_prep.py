"""Build MedMCQA train/val JSON in the campaign's {instruction,input,output,answer} schema.

MedMCQA (openlifescienceai/medmcqa): question + 4 options (opa..opd) + cop (0-3 correct idx).
This is the DeepSeek generalization run's ADAPT task (narrow medical domain, clear headroom).
Same instruction/answer format the 7B commonsense trainer uses (run_lib.train_prompt), so
train_deepseek.py / eval_deepseek.py reuse the identical prompt machinery.

  train  -> a subsample of the train split (labeled), used to fine-tune.
  val    -> the validation split (labeled; MedMCQA test cop is hidden), used for adapt accuracy.

Usage:
  HF_HOME=/scratch/hf_cache python3 scripts/deepseek/medmcqa_prep.py --n_train 30000
Writes:
  repro/LLM-Adapters/ft-training_set/medmcqa_train.json
  repro/LLM-Adapters/ft-training_set/medmcqa_val.json
"""
import argparse, json, os, random

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(HERE, "repro/LLM-Adapters/ft-training_set")
LETTERS = ["A", "B", "C", "D"]


def to_row(ex):
    """MedMCQA example -> {instruction,input,output,answer}. answer letter = LETTERS[cop]."""
    opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
    cop = int(ex["cop"])
    if not (0 <= cop < 4):
        return None
    body = ex["question"].strip() + "\n" + "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, opts))
    instr = (body + "\n\nAnswer format: A/B/C/D")
    letter = LETTERS[cop]
    return {"instruction": instr, "input": "", "output": f"the correct answer is {letter}",
            "answer": letter}


def build(split, n, seed):
    from datasets import load_dataset
    d = load_dataset("openlifescienceai/medmcqa", split=split)
    rows = [r for r in (to_row(ex) for ex in d) if r is not None]
    random.Random(seed).shuffle(rows)
    if n and n > 0:
        rows = rows[:n]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=30000, help="subsample size (0=all ~182k)")
    ap.add_argument("--n_val", type=int, default=0, help="val cap (0=all ~4.2k)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    tr = build("train", args.n_train, args.seed)
    va = build("validation", args.n_val, args.seed)
    for name, rows in (("medmcqa_train.json", tr), ("medmcqa_val.json", va)):
        p = os.path.join(OUT_DIR, name)
        json.dump(rows, open(p, "w"))
        # class balance sanity (should be ~uniform over A/B/C/D if data is clean)
        from collections import Counter
        c = Counter(r["answer"] for r in rows)
        print(f"[medmcqa] wrote {len(rows)} rows -> {p}  balance={dict(sorted(c.items()))}", flush=True)


if __name__ == "__main__":
    main()
