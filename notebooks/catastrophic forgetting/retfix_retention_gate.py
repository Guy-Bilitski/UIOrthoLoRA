"""retfix_retention_gate.py — 1-GPU diagnostics for the retention-axis fixes.
TO BE RUN AT THE POOL RESTART WINDOW ONLY (needs a GPU). Standalone: no live script
imports this file. Three modes; each writes JSON + raw sample dumps under
results/retfix_diag/ and never touches campaign_summary.jsonl or any summary.json.

  # Gate 1 (MMLU-Pro failure-mode confirmation + tolerant-extraction check), ~20-30 min:
  CUDA_VISIBLE_DEVICES=0 python retfix_retention_gate.py --mode mmlupro \
      --adapter /scratch/cf_models/frm_lora_lr3e4_c256_s42

  # Gate 2 (PiSSA BBH collapse: format vs forgetting), ~20-30 min:
  CUDA_VISIBLE_DEVICES=0 python retfix_retention_gate.py --mode pissa \
      --adapter /scratch/cf_models/frm_pissa_lr3e4_c256_s42

  # Gate 3 (pin the full-set base BBH ceiling under the CURRENT harness), ~1-2 h:
  CUDA_VISIBLE_DEVICES=0 python retfix_retention_gate.py --mode base_bbh

PASS CRITERIA (see handoff/22_RETENTION_FIX.md):
  mmlupro : stock extraction fails on >50% of docs whose generation contains
            "answer is:"-style output; tolerant extraction lands in [10, 19]
            OR the letter is provably never emitted (then MMLU-Pro must be dropped).
  pissa   : verdict printed = FORMAT (content right, exact_match wrong) vs
            DEGENERATE (content wrong/empty) from the saved generations.
  base_bbh: full-set answer-only BBH within ~1 pp of 33.10 -> confirms the paper
            ceiling holds under gen_cap=1024/max_len=4096 + normalized exact_match,
            i.e. old (n=49) and frepro cells share ONE BBH axis.
"""
import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

import bbh_metric_fix  # live module; imported BY us (allowed) — never the reverse

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.join(HERE, "results", "retfix_diag")

# Stock lm-eval mmlu_pro extraction + the tolerant fallback chain we are validating.
STOCK_RE = re.compile(r"answer is \(?([ABCDEFGHIJ])\)?")
TOLERANT_RES = [
    re.compile(r"answer is:?\s*\(?([ABCDEFGHIJ])\)?\b"),        # 'answer is: C' / 'answer is (C)'
    re.compile(r"[Aa]nswer:\s*\(?([ABCDEFGHIJ])\)?\b"),          # 'Answer: C'
    re.compile(r"\b([ABCDEFGHIJ])\)?\s*(?:is|was)\s+correct"),   # 'C is correct'
    re.compile(r"answer is:?\s*\(?([ABCDEFGHIJ])\)?\s*[.,\n]"),  # trailing punctuation form
]


def load_model(base_model, adapter, gen_cap, max_len):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token_id = 0
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="cuda:0")
    m.generation_config.max_new_tokens = gen_cap
    if adapter:
        m = PeftModel.from_pretrained(m, adapter, device_map={"": 0})
    m.eval()
    return HFLM(pretrained=m, tokenizer=tok, batch_size="auto", max_length=max_len), tok


def dump_samples(res, out_path):
    """Flatten lm-eval --log_samples output to one jsonl of (task, doc, target, generation)."""
    n = 0
    with open(out_path, "w") as f:
        for task, samples in (res.get("samples") or {}).items():
            for s in samples:
                resps = s.get("resps") or []
                gen = resps[0][0] if resps and resps[0] else ""
                filt = s.get("filtered_resps")
                f.write(json.dumps({
                    "task": task, "target": s.get("target"),
                    "generation": gen, "filtered": filt,
                    "exact_match": s.get("exact_match")}) + "\n")
                n += 1
    return n


def mode_mmlupro(args):
    lm, _ = load_model(args.base_model, args.adapter, args.gen_cap, args.max_len)
    res = simple_evaluate(model=lm, tasks=["mmlu_pro"], bootstrap_iters=0,
                          limit=args.limit, log_samples=True,
                          gen_kwargs=f"max_gen_toks={args.gen_cap}")
    run = os.path.basename(os.path.normpath(args.adapter or "base"))
    os.makedirs(os.path.join(DIAG, run), exist_ok=True)
    sp = os.path.join(DIAG, run, "mmlu_pro_samples.jsonl")
    n = dump_samples(res, sp)

    # offline: stock vs tolerant extraction over the saved generations
    stats = {"n": 0, "stock_hit": 0, "tolerant_hit": 0, "tolerant_correct": 0,
             "has_answer_is_colon": 0, "letter_never_emitted": 0}
    for line in open(sp):
        s = json.loads(line)
        gen, tgt = s["generation"] or "", (s["target"] or "").strip()
        stats["n"] += 1
        if STOCK_RE.search(gen):
            stats["stock_hit"] += 1
        m = next((r.search(gen) for r in TOLERANT_RES if r.search(gen)), None)
        if m:
            stats["tolerant_hit"] += 1
            if m.group(1) == tgt:
                stats["tolerant_correct"] += 1
        if re.search(r"answer is:", gen, re.I):
            stats["has_answer_is_colon"] += 1
        if not re.search(r"\b[ABCDEFGHIJ]\b", gen):
            stats["letter_never_emitted"] += 1
    stats["stock_score_pct"] = round(100 * next(
        (v for k, v in res["results"].get("mmlu_pro", {}).items()
         if k.startswith("exact_match") and "stderr" not in k), 0), 2)
    stats["tolerant_score_pct"] = round(100 * stats["tolerant_correct"] / max(stats["n"], 1), 2)
    out = os.path.join(DIAG, run, "mmlupro_gate.json")
    json.dump(stats, open(out, "w"), indent=2)
    print(f"[retfix/mmlupro] {n} samples -> {sp}")
    print(f"[retfix/mmlupro] {json.dumps(stats, indent=2)}")
    print(f"[retfix/mmlupro] PASS if tolerant_score_pct in [10,19] "
          f"(fix recoverable) OR letter_never_emitted/n > 0.5 (drop MMLU-Pro, final).")


def mode_pissa(args):
    lm, _ = load_model(args.base_model, args.adapter, args.gen_cap, args.max_len)
    res = simple_evaluate(model=lm, tasks=["bbh_fewshot"], bootstrap_iters=0,
                          limit=args.limit, log_samples=True,
                          gen_kwargs=f"max_gen_toks={args.gen_cap}")
    run = os.path.basename(os.path.normpath(args.adapter))
    os.makedirs(os.path.join(DIAG, run), exist_ok=True)
    sp = os.path.join(DIAG, run, "bbh_samples.jsonl")
    n = dump_samples(res, sp)

    stats = {"n": 0, "exact_match": 0, "target_in_gen": 0, "empty_gen": 0,
             "metamath_style": 0}  # 'The answer is: ...' leakage into BBH answers
    for line in open(sp):
        s = json.loads(line)
        gen, tgt = (s["generation"] or "").strip(), (s["target"] or "").strip()
        stats["n"] += 1
        if s.get("exact_match"):
            stats["exact_match"] += 1
        if tgt and tgt.lower() in gen.lower():
            stats["target_in_gen"] += 1
        if not gen:
            stats["empty_gen"] += 1
        if re.search(r"answer is:", gen, re.I) or gen.startswith("####"):
            stats["metamath_style"] += 1
    frac_recover = (stats["target_in_gen"] - stats["exact_match"]) / max(stats["n"], 1)
    verdict = ("EVAL-ARTIFACT (format): correct content present but exact_match misses it"
               if frac_recover > 0.15 else
               "REAL-FORGETTING: generations do not contain the targets")
    stats["verdict"] = verdict
    out = os.path.join(DIAG, run, "pissa_gate.json")
    json.dump(stats, open(out, "w"), indent=2)
    print(f"[retfix/pissa] {n} samples -> {sp}")
    print(f"[retfix/pissa] {json.dumps(stats, indent=2)}")


def mode_base_bbh(args):
    lm, _ = load_model(args.base_model, None, args.gen_cap, args.max_len)
    res = simple_evaluate(model=lm, tasks=["bbh_fewshot"], bootstrap_iters=0,
                          limit=None, gen_kwargs=f"max_gen_toks={args.gen_cap}")
    v = next((v for k, v in res["results"].get("bbh_fewshot", {}).items()
              if k.startswith("exact_match") and "stderr" not in k), None)
    score = round(100 * v, 2) if v is not None else None
    os.makedirs(DIAG, exist_ok=True)
    out = os.path.join(DIAG, "base_bbh_fullset_current_harness.json")
    json.dump({"base_model": args.base_model, "bbh_fewshot_fullset": score,
               "gen_cap": args.gen_cap, "max_len": args.max_len,
               "paper_ceiling": 33.10, "spotcheck_lim40": 36.57,
               "metric_fix_applied": True}, open(out, "w"), indent=2)
    print(f"[retfix/base_bbh] full-set answer-only BBH = {score} "
          f"(paper ceiling 33.10; lim-40 spot-check 36.57) -> {out}")
    print(f"[retfix/base_bbh] PASS if |score - 33.10| <~ 1.0 => ONE shared BBH axis.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mmlupro", "pissa", "base_bbh"], required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--limit", type=int, default=10, help="docs per subtask (diag modes)")
    ap.add_argument("--gen_cap", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=4096)
    args = ap.parse_args()
    bbh_metric_fix.ensure_bbh_fewshot_metric_fix()
    if args.mode == "mmlupro":
        assert args.adapter, "--adapter required for mmlupro mode"
        mode_mmlupro(args)
    elif args.mode == "pissa":
        assert args.adapter, "--adapter required for pissa mode"
        mode_pissa(args)
    else:
        mode_base_bbh(args)


if __name__ == "__main__":
    main()
