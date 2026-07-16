"""Evaluate ONE DeepSeek-V4-Flash adapter sharded across a node's 8 GPUs, in-process.

Same three legs and the SAME summary.json schema as eval_one_gpu.py (so 284B rows merge with
the 7B campaign): adapt-task accuracy + retention suite + magnitude (fdelta). The only changes:
  - base loaded dequant FP8->bf16, device_map="auto" (sharded), frozen; adapter injected onto it.
  - adapt task = MedMCQA (medical MC): letter-accuracy over the labeled val split.
  - retention (bbh_fewshot/mmlu_pro/mmlu/arc_challenge/truthfulqa_mc2) and fdelta reuse the exact
    7B machinery (lm_eval HFLM + uio_inprocess.fdelta_inprocess) — both are per-module and work
    unchanged on the sharded model.

Usage (drained 8-GPU node):
  HF_HOME=/scratch/hf_cache HF_HUB_OFFLINE=1 python3 scripts/deepseek/eval_deepseek.py \
    --adapter /scratch/cf_models/dsv4_lora_r16_lr3e4_s42 --ret_suite broad --adapt_limit 1000
"""
import os, re, sys, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
import run_lib
from uio_inprocess import fdelta_inprocess

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
VAL = os.path.join(HERE, "repro/LLM-Adapters/ft-training_set/medmcqa_val.json")
LETTERS = ["A", "B", "C", "D"]
_LET_RE = re.compile(r"\b([ABCD])\b")


@torch.no_grad()
def medmcqa_accuracy(model, tokenizer, limit, gen_cap=8, batch_size=8):
    """Greedy-generate the answer letter for each val prompt (same prompt template as training)
    and compare to the gold letter. Returns (acc, n_correct, n_total)."""
    rows = json.load(open(VAL))
    if limit and limit > 0:
        rows = rows[:limit]
    in_dev = model.get_input_embeddings().weight.device
    correct = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        prompts = [run_lib.train_prompt({**r, "output": ""}) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(in_dev)
        out = model.generate(**enc, max_new_tokens=gen_cap, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        for j, r in enumerate(batch):
            gen = tokenizer.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = _LET_RE.search(gen.upper())
            pred = m.group(1) if m else None
            correct += int(pred == r["answer"])
    n = len(rows)
    return round(100 * correct / n, 2), correct, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="", help="adapter dir; empty/'none' = base ceiling")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--adapt_limit", type=int, default=1000, help="MedMCQA val examples (0=all ~4.2k)")
    ap.add_argument("--ret_limit", type=int, default=0)
    ap.add_argument("--ret_suite", choices=["core", "broad"], default="broad")
    ap.add_argument("--ret_batch", default="1", help="HFLM batch_size (int or 'auto')")
    ap.add_argument("--gen_cap", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=4096)
    args = ap.parse_args()
    base_only = args.adapter in ("", "none")
    if base_only and not args.run_name:
        ap.error("--run_name required in base-only mode")
    run_name = args.run_name or os.path.basename(os.path.normpath(args.adapter))

    import bbh_metric_fix
    bbh_metric_fix.ensure_bbh_fewshot_metric_fix()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from transformers import FineGrainedFP8Config
    print(f"[load] {MODEL} dequant->bf16 device_map=auto ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        quantization_config=FineGrainedFP8Config(dequantize=True),
        low_cpu_mem_usage=True, trust_remote_code=True)
    model.generation_config.max_new_tokens = args.gen_cap
    method = "BASE"
    if not base_only:
        model = PeftModel.from_pretrained(model, args.adapter)  # LoRA injects onto sharded base
        try:
            method = json.load(open(os.path.join(args.adapter, "adapter_config.json"))).get("peft_type", "unknown")
        except Exception:
            method = "unknown"
    model.eval()
    print(f"==== EVAL {run_name} (method={method}) ====", flush=True)

    acc, c, t = medmcqa_accuracy(model, tokenizer, args.adapt_limit)
    cs = {"medmcqa": acc}
    cs_avg = acc
    print(f"[{run_name}] MedMCQA={acc} ({c}/{t})", flush=True)

    if base_only:
        fd = {"fdelta_token_weighted": 0.0, "dw_sv_max": 0.0, "dw_sv_mean": 0.0, "n_matrices": 0}
    else:
        try:
            fd = fdelta_inprocess(model, tokenizer)
        except Exception as e:
            print(f"[{run_name}] fdelta failed: {e}", flush=True); fd = {}
        # F_Delta on the ADAPT distribution (MedMCQA) — the 7B semantics are "F_Delta on
        # adapt-task inputs"; the default prompts above are the 7B CS suite, which is
        # off-distribution here. Report both (adversarial-review fix 2026-07-16, flaw #4).
        try:
            rows = json.load(open(VAL))[:100]
            mprompts = [run_lib.train_prompt({**r, "output": ""}) for r in rows]
            fd_a = fdelta_inprocess(model, tokenizer, prompts=mprompts)
            fd["fdelta_adapt"] = fd_a.get("fdelta_token_weighted")
        except Exception as e:
            print(f"[{run_name}] fdelta_adapt failed: {e}", flush=True)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    bs = args.ret_batch if args.ret_batch == "auto" else int(args.ret_batch)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs, max_length=args.max_len)
    CORE = ["bbh_fewshot", "mmlu_pro"]
    EXTRA = ["mmlu", "arc_challenge", "truthfulqa_mc2"]
    tasks = CORE + (EXTRA if args.ret_suite == "broad" else [])
    res = simple_evaluate(model=lm, tasks=tasks, bootstrap_iters=0,
                          limit=(args.ret_limit or None), gen_kwargs=f"max_gen_toks={args.gen_cap}")

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
    ret_bbh = ret.get("bbh")
    bvals = [v for v in ret.values() if v is not None]
    ret_broad = round(sum(bvals) / len(bvals), 2) if (args.ret_suite == "broad" and bvals) else None

    headline = {"cs_avg": cs_avg, "adapt_task": "medmcqa", **ret, "medmcqa": acc,
                "retention_mean": ret_mean, "retention_bbh": ret_bbh, "retention_broad": ret_broad,
                "fdelta": fd.get("fdelta_token_weighted"), "fdelta_adapt": fd.get("fdelta_adapt"),
                "dw_sv_max": fd.get("dw_sv_max"), "dw_sv_mean": fd.get("dw_sv_mean")}
    # Keep the FULL per-subtask lm_eval rows (esp. MMLU medical subtasks) so retention can be
    # recomputed with/without adapt-domain overlap post-hoc — irrecoverable otherwise
    # (adversarial-review fix 2026-07-16, flaw #5).
    def _clean(d):
        try:
            return json.loads(json.dumps(d, default=str))
        except Exception:
            return {}
    summary = {"run_name": run_name, "method": method, "adapter": args.adapter or None,
               "per_dataset": cs, "fdelta": fd, "headline": headline,
               "lm_eval_results": _clean(res.get("results", {})),
               "lm_eval_groups": _clean(res.get("groups", {})),
               "git_commit": run_lib.git_commit(), "evaluated_at": run_lib.now_iso()}
    run_lib.write_json(os.path.join(HERE, "results", run_name, "summary.json"), summary)
    run_lib.append_registry("campaign_summary.jsonl", {"run_name": run_name, "method": method,
                                                       **headline, "evaluated_at": run_lib.now_iso()})
    print(f"==== {run_name} HEADLINE: {headline} ====", flush=True)


if __name__ == "__main__":
    main()
