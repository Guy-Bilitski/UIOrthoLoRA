"""Build the FULL MetaMathQA 395K training set (faithful CLoRA Table 3 setting), in the
alpaca schema train_cs.py expects. Keeps `original_question` + `type` so we can hash-dedup
against GSM8K/MATH test (contamination check) before training. Distinct filename from the
100K subset so nothing collides."""
import json
import os
from datasets import load_dataset

OUT = "repro/LLM-Adapters/ft-training_set/metamathqa_395k.json"

ds = load_dataset("meta-math/MetaMathQA", split="train")  # cached from the 100K build
rows = [{"instruction": r["query"], "input": "", "output": r["response"], "answer": "",
         "original_question": r.get("original_question", ""), "type": r.get("type", "")}
        for r in ds]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(rows, f)
print(f"wrote {len(rows)} rows -> {OUT}")
# schema check vs the 100K file
ref = json.load(open("repro/LLM-Adapters/ft-training_set/metamathqa_100k.json"))
print("100K keys:", sorted(ref[0].keys()))
print("395K keys:", sorted(rows[0].keys()))
from collections import Counter
print("type distribution:", dict(Counter(r["type"] for r in rows)))
print("sample instruction:", rows[0]["instruction"][:90])
print("sample output tail :", rows[0]["output"][-80:])
