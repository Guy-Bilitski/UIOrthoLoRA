"""Build the Hendrycks MATH test set (all 7 subjects, ~5000 problems) in the alpaca schema,
with the GOLD answer extracted from the \\boxed{...} in the reference solution using the
canonical Hendrycks helpers. Faithful CLoRA Table 3 MATH eval target."""
import json
import os
from datasets import load_dataset, concatenate_datasets

CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
           "number_theory", "prealgebra", "precalculus"]
OUT = "repro/LLM-Adapters/dataset/MATH/test.json"


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    if s[:len(left)] == left and s[-1] == "}":
        return s[len(left):-1]
    return None


parts = []
for c in CONFIGS:
    parts.append(load_dataset("EleutherAI/hendrycks_math", c, split="test"))
ds = concatenate_datasets(parts)

rows, no_box = [], 0
for r in ds:
    gold = remove_boxed(last_boxed_only_string(r["solution"]))
    if gold is None:
        no_box += 1
        continue
    rows.append({"instruction": r["problem"], "input": "", "output": "",
                 "answer": gold, "level": r["level"], "type": r["type"]})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(rows, f)
print(f"wrote {len(rows)} MATH test problems -> {OUT}  (dropped {no_box} with no boxed answer)")
print("sample problem:", rows[0]["instruction"][:80])
print("sample gold answer:", rows[0]["answer"])
