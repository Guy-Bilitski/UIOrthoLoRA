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
    ap.add_argument("--gen_cap", type=int, default=1024,
                    help="Hard ceiling on generated tokens for ALL generate_until evals "
                         "(retention + gsm8k adapt), set via generation_config.max_new_tokens. "
                         "Bounds runaway base-model generations (e.g. Qwen2.5 ships "
                         "max_new_tokens=2048, making ~17%% of questions ramble to the limit "
                         "without producing a parseable answer). 512 leaves genuine CoT intact.")
    ap.add_argument("--max_len", type=int, default=4096,
                    help="HFLM max_length (context+gen budget). Default 4096 = Llama-2's window. "
                         "Qwen2.5's native 32768 makes lm-eval's auto batch-sizer reserve memory "
                         "for 32k-token seqs -> picks batch=1 -> ~10x slower eval. Capping to 4096 "
                         "matches Llama-2 (better cross-model comparability) and lets the batcher "
                         "use ~32x batch; mmlu_pro/bbh 5-shot contexts (~2.5k tok) fit in 4096-512.")
    ap.add_argument("--ret_suite", choices=["core", "broad"], default="core",
                    help="core=BBH+MMLU-Pro (backward-comparable retention_mean); "
                         "broad adds MMLU+ARC-c+TruthfulQA -> retention_broad.")
    ap.add_argument("--adapt_task", choices=["cs", "gsm8k"], default="cs",
                    help="in-domain adaptation metric: cs=8-task commonsense (default); "
                         "gsm8k=math (lm-eval gsm8k exact_match) for the 2nd adaptation domain. "
                         "cs_avg holds whichever is selected so the adapt axis stays uniform.")
    args = ap.parse_args()
    run_name = args.run_name or os.path.basename(os.path.normpath(args.adapter))

    # Self-heal the bbh_fewshot exact_match normalization (idempotent). Without it a
    # base model's leading-space answers score 0 on BBH (Qwen bbh 0.00 -> 0.54); no-op
    # for Llama-2. See bbh_metric_fix.py / handoff/16.
    import bbh_metric_fix
    bbh_metric_fix.ensure_bbh_fewshot_metric_fix()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # Only fall back to id 0 when the tokenizer has no declared pad token.
    # Llama-2 has pad=None -> 0 (=<unk>, harmless). Qwen2.5 DOES declare a pad
    # (<|endoftext|>=151643); its token 0 decodes to "!", so forcing pad=0 made
    # batch-padding of finished sequences render as "!!!!" in the decoded response,
    # which corrupted BBH's greedy answer regex ("the answer is 24.!!!!") -> bbh=0.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cuda:0")
    # Hard generation ceiling (set on the BASE model before the PEFT wrap, so the
    # base generation_config that .generate() actually uses is the one we cap).
    # Qwen2.5 ships generation_config.max_new_tokens=2048, which overrides HFLM's
    # max_length so ~17%% of generate_until questions ramble to 2048 tokens (~8x
    # slower, ~24h/cell) without ever emitting a parseable answer. Capping bounds
    # those ramblers (they score 0 either way) while leaving genuine CoT (mean
    # ~140-240 tok) untouched. No-op for models that already stop early
    # (Llama-2: max_new_tokens=None). Verified: bio 0.75->0.75, math 0.42->0.44.
    model.generation_config.max_new_tokens = args.gen_cap
    model = PeftModel.from_pretrained(model, args.adapter, device_map={"": 0})
    model.eval()
    print(f"==== generation cap: max_new_tokens={args.gen_cap} ====", flush=True)
    method = "unknown"
    try:
        import json
        method = json.load(open(os.path.join(args.adapter, "adapter_config.json"))).get("peft_type", "unknown")
    except Exception:
        pass
    print(f"==== EVAL {run_name} (method={method}) ====", flush=True)

    # in-domain adaptation: commonsense (8-task) or math (gsm8k)
    if args.adapt_task == "gsm8k":
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        _lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto", max_length=args.max_len)
        _r = simple_evaluate(model=_lm, tasks=["gsm8k"], bootstrap_iters=0,
                             limit=(args.eval_limit or None),
                             gen_kwargs=f"max_gen_toks={args.gen_cap}")
        _row = _r["results"].get("gsm8k", {})
        _em = next((v for k, v in _row.items() if k.startswith("exact_match") and "stderr" not in k), None)
        cs = {"gsm8k": round(100 * _em, 2) if _em is not None else None}
        cs_avg = cs["gsm8k"]
    else:
        cs = {}
        for ds in CS_DATASETS:
            acc, _, _, _ = eval_cs.run_eval(model, tokenizer, ds, batch_size=32, num_beams=4,
                                            max_new_tokens=32, limit=args.eval_limit)
            cs[ds] = round(100 * acc, 2)
        cs_avg = round(sum(cs.values()) / len(cs), 2)
    print(f"[{run_name}] adapt={args.adapt_task} CS={cs} CS_AVG={cs_avg}", flush=True)

    # F-delta
    try:
        fd = fdelta_inprocess(model, tokenizer)
    except Exception as e:
        print(f"[{run_name}] fdelta failed: {e}", flush=True); fd = {}

    # retention (in-memory). retention_mean stays BBH+MMLU-Pro so it's directly
    # comparable to the entire existing campaign; --ret_suite broad adds 3 tasks
    # (MMLU, ARC-challenge, TruthfulQA-mc2) reported individually + as retention_broad.
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto", max_length=args.max_len)
    _rl = args.ret_limit if args.ret_limit > 0 else (args.eval_limit if args.eval_limit > 0 else None)
    CORE = ["bbh_fewshot", "mmlu_pro"]
    EXTRA = ["mmlu", "arc_challenge", "truthfulqa_mc2"]
    tasks = CORE + (EXTRA if args.ret_suite == "broad" else [])
    # max_gen_toks here keeps HFLM's context-budgeting (max_ctx_len = max_length -
    # max_gen_toks) consistent with the generation_config.max_new_tokens cap above;
    # the cap is what actually bounds generation (group-task gen_kwargs don't bind).
    res = simple_evaluate(model=lm, tasks=tasks, bootstrap_iters=0, limit=_rl,
                          gen_kwargs=f"max_gen_toks={args.gen_cap}")

    def _metric(task):
        row = res["results"].get(task, {})
        for pref in ("exact_match", "acc_norm", "acc"):
            v = next((v for k, v in row.items() if k.startswith(pref) and "stderr" not in k), None)
            if v is not None:
                return round(100 * v, 2)
        return None
    LABEL = {"bbh_fewshot": "bbh", "mmlu_pro": "mmlu_pro", "mmlu": "mmlu",
             "arc_challenge": "arc_c", "truthfulqa_mc2": "truthfulqa"}
    ret = {LABEL[t]: _metric(t) for t in tasks}
    ret_mean = round(((ret.get("bbh") or 0) + (ret.get("mmlu_pro") or 0)) / 2, 2)
    bvals = [v for v in ret.values() if v is not None]
    ret_broad = round(sum(bvals) / len(bvals), 2) if (args.ret_suite == "broad" and bvals) else None

    headline = {"cs_avg": cs_avg, "adapt_task": args.adapt_task, **ret,
                "retention_mean": ret_mean, "retention_broad": ret_broad,
                "fdelta": fd.get("fdelta_token_weighted"),
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
