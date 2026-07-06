"""Faithful math eval matching CLoRA / LLM-Adapters (0-shot Alpaca instruction template +
answer extraction), for the reproduction track. Replaces the lm-eval gsm8k (5-shot Q:/A:
strict-match) train/eval mismatch.

- GSM8K: extract the LAST number from the response, correct iff |pred - gold| <= 1e-3
  (matches repro/LLM-Adapters/evaluate.py extract_answer_number).
- Hendrycks MATH: extract the model's \\boxed{...} (or "The answer is:" tail) and compare to
  the gold boxed answer via the CANONICAL Hendrycks is_equiv (LaTeX normalization), NOT a
  numeric match (MATH answers are fractions/expressions/sets).

Uses run_lib.eval_prompt so the eval template == the training template (run_lib.train_prompt).
In-process (per uiortholora-phase1-gotchas: avoids the PEFT reload bug).
"""
import os
import re
import json

import torch
from transformers import GenerationConfig

import run_lib

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "repro/LLM-Adapters/dataset")


def _generate(model, tokenizer, instructions, batch_size, max_new_tokens, num_beams, device="cuda:0"):
    outs = []
    gen_cfg = GenerationConfig(num_beams=num_beams, do_sample=False)
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        prompts = [run_lib.eval_prompt(x) for x in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 generation_config=gen_cfg, max_new_tokens=max_new_tokens)
        for full in tokenizer.batch_decode(out, skip_special_tokens=True):
            resp = full.split("### Response:")[1].strip() if "### Response:" in full else full
            outs.append(resp)
        print(f"\r[math_eval] generated {min(i + batch_size, len(instructions))}/{len(instructions)}",
              end="", flush=True)
    print("", flush=True)
    return outs


# ------------------------------- GSM8K -------------------------------
def _extract_number(text):
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def run_gsm8k_faithful(model, tokenizer, limit=0, batch_size=16, num_beams=4, max_new_tokens=256):
    data = json.load(open(os.path.join(DATASET_DIR, "gsm8k", "test.json")))
    if limit > 0:
        data = data[:limit]
    resps = _generate(model, tokenizer, [d["instruction"] for d in data],
                      batch_size, max_new_tokens, num_beams)
    correct = 0
    for d, resp in zip(data, resps):
        pred = _extract_number(resp)
        try:
            gold = float(str(d["answer"]).replace(",", ""))
        except (ValueError, KeyError):
            gold = None
        if pred is not None and gold is not None and abs(pred - gold) <= 1e-3:
            correct += 1
    return correct / len(data), correct, len(data)


# --------------------- Hendrycks MATH (canonical is_equiv) ---------------------
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
        if s[:len(left)] == left:
            return s[len(left):]
    left = "\\boxed{"
    if s[:len(left)] == left and s[-1] == "}":
        return s[len(left):-1]
    return None


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        if string != "{}/{}".format(a, b):
            return string
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace(r"\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1, str2):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


def _extract_math_answer(resp):
    boxed = last_boxed_only_string(resp)
    if boxed is not None:
        inner = remove_boxed(boxed)
        if inner is not None:
            return inner
    for marker in ("The answer is:", "The answer is", "answer is:", "answer is"):
        if marker in resp:
            return resp.split(marker)[-1].strip().rstrip(".").strip()
    return None


def run_math_hendrycks(model, tokenizer, limit=0, batch_size=8, num_beams=1, max_new_tokens=512):
    data = json.load(open(os.path.join(DATASET_DIR, "MATH", "test.json")))
    if limit > 0:
        data = data[:limit]
    resps = _generate(model, tokenizer, [d["instruction"] for d in data],
                      batch_size, max_new_tokens, num_beams)
    correct, parse_fail = 0, 0
    for d, resp in zip(data, resps):
        pred = _extract_math_answer(resp)
        if pred is None:
            parse_fail += 1
            continue
        if is_equiv(pred, d["answer"]):
            correct += 1
    return correct / len(data), correct, len(data), parse_fail


if __name__ == "__main__":  # CPU self-test of the extraction + equivalence logic (no model)
    assert _extract_number("...so the total is 42") == 42.0
    assert _extract_number("The answer is: 1,024") == 1024.0
    assert _extract_math_answer("blah \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"
    assert _extract_math_answer("The answer is: \\sqrt{5}.") == "\\sqrt{5}"
    assert is_equiv("\\frac{1}{2}", "\\dfrac{1}{2}")
    assert is_equiv("0.5", "\\frac{1}{2}")
    assert is_equiv("2", "2")
    assert not is_equiv("2", "3")
    assert is_equiv("\\frac12", "\\frac{1}{2}")
    print("MATH_EVAL SELF-TEST OK")
