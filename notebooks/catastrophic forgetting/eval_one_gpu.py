"""
Evaluate ONE trained adapter (LoRA / CLoRA) fully on ONE GPU, in-process: load the
adapter, then 8-dataset commonsense acc + retention (answer-only BBH + MMLU-Pro via
in-memory lm-eval) + F-delta. Same legs and summary format as uio_inprocess.py, so
all methods are directly comparable. Designed to run many adapters in parallel
(one per GPU) via gpu_pool rather than sharding a single adapter across GPUs.

    CUDA_VISIBLE_DEVICES=6 python eval_one_gpu.py --adapter /scratch/cf_models/clora_cs_k2048
"""
import os
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import run_lib
import eval_cs
from uio_inprocess import fdelta_inprocess, CS_DATASETS

HERE = run_lib.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--ret_max_gen", type=int, default=0)
    ap.add_argument("--ret_limit", type=int, default=0)
    args = ap.parse_args()
    run_name = args.run_name or os.path.basename(os.path.normpath(args.adapter))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter, device_map={"": 0})
    model.eval()
    method = "unknown"
    try:
        import json
        method = json.load(open(os.path.join(args.adapter, "adapter_config.json"))).get("peft_type", "unknown")
    except Exception:
        pass
    print(f"==== EVAL {run_name} (method={method}) ====", flush=True)

    # in-domain commonsense
    cs = {}
    for ds in CS_DATASETS:
        acc, _, _, _ = eval_cs.run_eval(model, tokenizer, ds, batch_size=32, num_beams=4,
                                        max_new_tokens=32, limit=args.eval_limit)
        cs[ds] = round(100 * acc, 2)
    cs_avg = round(sum(cs.values()) / len(cs), 2)
    print(f"[{run_name}] CS={cs} CS_AVG={cs_avg}", flush=True)

    # F-delta
    try:
        fd = fdelta_inprocess(model, tokenizer)
    except Exception as e:
        print(f"[{run_name}] fdelta failed: {e}", flush=True); fd = {}

    # retention (in-memory)
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
    _rl = args.ret_limit if args.ret_limit > 0 else (args.eval_limit if args.eval_limit > 0 else None)
    res = simple_evaluate(model=lm, tasks=["bbh_fewshot", "mmlu_pro"], bootstrap_iters=0,
                          limit=_rl,
                          gen_kwargs=(f"max_gen_toks={args.ret_max_gen}" if args.ret_max_gen > 0 else None))
    ret = {}
    for t in ["bbh_fewshot", "mmlu_pro"]:
        row = res["results"].get(t, {})
        em = next((v for k, v in row.items() if k.startswith("exact_match") and "stderr" not in k), None)
        ret["bbh" if t == "bbh_fewshot" else "mmlu_pro"] = round(100 * em, 2) if em is not None else None
    ret_mean = round((ret.get("bbh", 0) + ret.get("mmlu_pro", 0)) / 2, 2)

    headline = {"cs_avg": cs_avg, "bbh": ret.get("bbh"), "mmlu_pro": ret.get("mmlu_pro"),
                "retention_mean": ret_mean, "fdelta": fd.get("fdelta_token_weighted"),
                "dw_sv_max": fd.get("dw_sv_max"), "dw_sv_mean": fd.get("dw_sv_mean")}
    summary = {"run_name": run_name, "method": method, "adapter": args.adapter,
               "per_dataset": cs, "fdelta": fd, "headline": headline,
               "git_commit": run_lib.git_commit(), "evaluated_at": run_lib.now_iso()}
    run_lib.write_json(os.path.join(HERE, "results", run_name, "summary.json"), summary)
    run_lib.append_registry("campaign_summary.jsonl", {"run_name": run_name, "method": method,
                                                       **headline, "evaluated_at": run_lib.now_iso()})
    print(f"==== {run_name} HEADLINE: {headline} ====", flush=True)


if __name__ == "__main__":
    main()
