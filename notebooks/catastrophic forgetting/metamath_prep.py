"""Prepare the math adaptation training set: MetaMathQA -> alpaca schema JSON, matching
the commonsense_170k format ({instruction,input,output,answer}) so train_cs.py runs it
unchanged. ~100k subset to keep the training budget comparable to commonsense (170k, 3ep).
Adapt eval is GSM8K (eval_one_gpu --adapt_task gsm8k); retention suite is unchanged."""
import json
import os
from datasets import load_dataset

OUT = "repro/LLM-Adapters/ft-training_set/metamathqa_100k.json"
N = 100000

ds = load_dataset("meta-math/MetaMathQA", split="train")
ds = ds.shuffle(seed=42).select(range(min(N, len(ds))))
rows = [{"instruction": r["query"], "input": "", "output": r["response"], "answer": ""} for r in ds]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(rows, f)
print(f"wrote {len(rows)} rows -> {OUT}")
print("sample instruction:", rows[0]["instruction"][:90])
print("sample output     :", rows[0]["output"][:90])
