"""Reconstruct data/metamathqa_395k.json from the public MetaMathQA dataset.

The campaign's file is the LLM-Adapters instruction format built from
meta-math/MetaMathQA (fields query -> instruction, response -> output, input="").
The original file's SHA256 (first 32 hex chars): 13c5920ac97dc4afa9d4420701533f5b
If you can copy the original file from the campaign host/backup instead, prefer that
and verify the checksum; this reconstruction is believed identical but field-order/
row-order differences would change the hash while remaining semantically equivalent
(training samples are shuffled by the seeded loader anyway).
"""
import hashlib
import json
import os

from datasets import load_dataset


def main():
    os.makedirs("data", exist_ok=True)
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    rows = [{"instruction": r["query"], "input": "", "output": r["response"]} for r in ds]
    print(f"rows: {len(rows)} (expected 395000)")
    out = "data/metamathqa_395k.json"
    with open(out, "w") as f:
        json.dump(rows, f)
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()[:32]
    print(f"wrote {out}; sha256[:32]={h}")
    print("original file sha256[:32]=13c5920ac97dc4afa9d4420701533f5b "
          "(hash mismatch does NOT imply semantic difference — see docstring)")


if __name__ == "__main__":
    main()
