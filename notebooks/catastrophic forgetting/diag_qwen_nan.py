"""Diagnose the Qwen bf16 NaN (2026-08-27). Standalone: does NOT touch train_cs.py.

Observed: Qwen2.5-7B + LoRA on commonsense_170k NaNs mid-warmup on this box
(H200 / torch 2.12.0+cu130 / transformers 5.10.2), while Llama-2-7B is clean.
Onset is data-order-locked: seed 43 dies at epoch 0.006572 for BOTH lorawd(wd=0.3)
and milora(wd=0) — the same batch — and seed 44 dies at 0.0291. So it is a
specific batch, not a method or a weight-decay effect.

This replays the SAME data pipeline (same tokenisation, same shuffle seed, same
batch size / cutoff) and, per configuration, either
  (a) --scan : forward-only over the first N batches to find the batch whose
      forward pass first produces a non-finite loss/logit (no optimiser), or
  (b) --train: a short LoRA train to see whether a config survives past the
      known death step.
Configurations: attn_implementation in {sdpa, eager} x dtype {bf16, fp32-master}.

Usage:
  python diag_qwen_nan.py --scan --attn sdpa
  python diag_qwen_nan.py --scan --attn eager
  python diag_qwen_nan.py --train --attn eager --steps 260
"""
import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "repro/LLM-Adapters/ft-training_set/commonsense_170k.json")
PROMPT = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Response:\n")


def batches(tok, seed, bs, cutoff, n_batches):
    """Same shuffle seed + same prompt/cutoff as train_cs.py's loader."""
    from datasets import load_dataset
    ds = load_dataset("json", data_files=DATA)["train"].shuffle(seed=seed)
    buf = []
    for i, row in enumerate(ds):
        text = PROMPT.format(instruction=row["instruction"])
        if row.get("input"):
            text += row["input"] + "\n"
        text += str(row.get("output", ""))
        buf.append(text)
        if len(buf) == bs:
            enc = tok(buf, return_tensors="pt", padding=True, truncation=True,
                      max_length=cutoff)
            yield (i + 1) // bs - 1, enc, buf
            buf = []
            if (i + 1) // bs >= n_batches:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--cutoff", type=int, default=256)
    ap.add_argument("--n_batches", type=int, default=260)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[diag] {args.model} attn={args.attn} seed={args.seed}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation=args.attn)
    print(f"[diag] loaded; attn in use = {model.config._attn_implementation}", flush=True)

    if args.train:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        model.train()

    worst = 0.0
    for step, enc, texts in batches(tok, args.seed, args.bs, args.cutoff, args.n_batches):
        ids = enc["input_ids"].to("cuda:0")
        mask = enc["attention_mask"].to("cuda:0")
        if args.train:
            warm = min(1.0, (step + 1) / 100)          # same warmup shape as the recipe
            for g in opt.param_groups:
                g["lr"] = args.lr * warm
            out = model(input_ids=ids, attention_mask=mask, labels=ids)
            loss = out.loss
        else:
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask, labels=ids)
                loss = out.loss
        mx = float(out.logits.abs().max())
        worst = max(worst, mx)
        bad = (not torch.isfinite(loss)) or (not torch.isfinite(out.logits).all())
        if bad or step % 20 == 0:
            print(f"[diag] step {step:4d} loss={float(loss):.4f} max|logit|={mx:9.1f} "
                  f"len={ids.shape[1]:4d} {'<-- NON-FINITE' if bad else ''}", flush=True)
        if bad:
            print(f"[diag] FIRST NON-FINITE at step {step}. Longest text in batch:", flush=True)
            print("   ", max(texts, key=len)[:300].replace("\n", " | "), flush=True)
            json.dump({"attn": args.attn, "seed": args.seed, "step": step,
                       "max_logit_seen": worst, "mode": "train" if args.train else "scan"},
                      open(os.path.join(HERE, "logs",
                                        f"diag_qwen_{args.attn}_{args.seed}.json"), "w"), indent=1)
            return
        if args.train:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(gn):
                print(f"[diag] NON-FINITE GRAD at step {step} (loss was finite)", flush=True)
                return
            opt.step()
            opt.zero_grad(set_to_none=True)
    print(f"[diag] SURVIVED {args.n_batches} batches, max|logit| seen = {worst:.1f}", flush=True)


if __name__ == "__main__":
    main()
