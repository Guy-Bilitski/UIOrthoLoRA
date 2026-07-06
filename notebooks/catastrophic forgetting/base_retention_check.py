"""Alignment check: run the BASE Llama-2-7B through our retention suite (bbh_fewshot + mmlu_pro)
and compare to CLoRA's reported base out-domain reference (BBH 34.91, MMLU-Pro 18.56). If ours is
close, our retention eval is measuring the same thing CLoRA's out-domain column does."""
import sys
import torch
import bbh_metric_fix
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

bbh_metric_fix.ensure_bbh_fewshot_metric_fix()
BASE = "meta-llama/Llama-2-7b-hf"
lim = int(sys.argv[1]) if len(sys.argv) > 1 else 40

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token_id = 0
tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
lm = HFLM(pretrained=m, tokenizer=tok, batch_size="auto", max_length=4096)
r = simple_evaluate(model=lm, tasks=["bbh_fewshot", "mmlu_pro"], bootstrap_iters=0,
                    limit=lim, gen_kwargs="max_gen_toks=1024")


def metric(t):
    row = r["results"].get(t, {})
    for p in ("exact_match", "acc_norm", "acc"):
        v = next((v for k, v in row.items() if k.startswith(p) and "stderr" not in k), None)
        if v is not None:
            return round(100 * v, 2)
    return None


print(f"\n==== BASE Llama-2-7B retention (limit={lim}/task) ====", flush=True)
print(f"BBH={metric('bbh_fewshot')} (CLoRA ref 34.91)   MMLU-Pro={metric('mmlu_pro')} (CLoRA ref 18.56)", flush=True)
