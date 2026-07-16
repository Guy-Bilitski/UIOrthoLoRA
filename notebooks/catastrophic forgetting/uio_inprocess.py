"""
Self-contained UIOrthoLoRA run: TRAIN then EVALUATE the same in-memory model, with
NO save/reload. This is required because the UIOrthoLoRA PEFT checkpoint cannot
round-trip (the trained orthogonal rotators are dropped on reload due to PEFT's
adapter-name remapping of the parametrization keys; the SVD basis was also dropped
until fixed). Evaluating in-process guarantees eval == the trained model.

Runs entirely on ONE GPU: train (3 epochs CS-170K) -> 8-dataset commonsense acc ->
retention (answer-only BBH + MMLU-Pro CoT via lm-eval, in-memory model) -> F-delta.
Writes results/<run>/summary.json. Nothing but JSON is persisted.

    CUDA_VISIBLE_DEVICES=5 python uio_inprocess.py --k_val 2048 --k_vec 410 --run_name uio_kval2048_kvec410
"""
import os
import json
import time
import argparse

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import get_peft_model, UIOrthoLoRAConfig
from peft.tuners.tuners_utils import BaseTunerLayer

import run_lib
import eval_cs

HERE = run_lib.HERE
DEFAULT_DATA = os.path.join(HERE, "repro/LLM-Adapters/ft-training_set/commonsense_170k.json")
CS_DATASETS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
               "ARC-Easy", "ARC-Challenge", "openbookqa"]


def fdelta_inprocess(model, tokenizer, n_inputs=100, device="cuda:0", prompts=None, deltas=None):
    """deltas: optional {module_name: dW tensor} for non-PEFT updates (e.g. full FT, E2 arm) —
    hooks attach to those named modules instead of PEFT BaseTunerLayer discovery."""
    layers, hooks, accum = {}, [], {}

    def make_hook(name):
        def pre_hook(module, inputs):
            x = inputs[0]
            dw = layers[name]["dw"]
            xf = x.reshape(-1, x.shape[-1]).to(dw.dtype)
            xn = xf.norm(dim=-1)
            dwxn = torch.matmul(xf, dw.T).norm(dim=-1)
            m = xn > 1e-6
            s, c = accum.get(name, (0.0, 0))
            accum[name] = (s + (dwxn[m] / xn[m]).sum().item(), c + int(m.sum().item()))
        return pre_hook

    if deltas is not None:
        mods = dict(model.named_modules())
        for name, dw in deltas.items():
            mod = mods[name]
            dw = dw.detach().to(next(mod.parameters()).device)
            sv = torch.linalg.svdvals(dw.float())[0].item()
            layers[name] = {"dw": dw, "sv": sv}
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, BaseTunerLayer) and hasattr(mod, "get_delta_weight"):
                try:
                    dw = mod.get_delta_weight("default").detach()
                except Exception:
                    continue
                sv = torch.linalg.svdvals(dw.float())[0].item()
                layers[name] = {"dw": dw, "sv": sv}
                hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    if prompts is None:  # default: the 7B CS adaptation distribution (original behavior)
        prompts = []
        for ds in ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "openbookqa"]:
            data = json.load(open(os.path.join(HERE, "repro/LLM-Adapters/dataset", ds, "test.json")))
            for d in data[: (n_inputs // 7) + 2]:
                prompts.append(run_lib.eval_prompt(d["instruction"], d.get("input") or None))
    prompts = prompts[:n_inputs]
    with torch.no_grad():
        for i in range(0, len(prompts), 8):
            enc = tokenizer(prompts[i:i + 8], return_tensors="pt", padding=True, truncation=True,
                            max_length=256).to(device)
            model(**enc)
    for h in hooks:
        h.remove()
    tot_s = sum(s for s, c in accum.values()); tot_c = sum(c for s, c in accum.values())
    svs = [layers[n]["sv"] for n in layers]
    return {"fdelta_token_weighted": round(tot_s / tot_c, 4) if tot_c else None,
            "dw_sv_mean": round(sum(svs) / len(svs), 4), "dw_sv_max": round(max(svs), 4),
            "n_matrices": len(layers)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--k_val", type=int, default=2048)
    ap.add_argument("--k_vec", type=int, default=410)
    ap.add_argument("--use_de", type=int, default=1)
    ap.add_argument("--drop_major", type=int, default=0,
                    help="1 = paper-correct: major/preserved band contributes 0 to ΔW (true-identity top subspace). exp A5.")
    ap.add_argument("--lambda_E", type=float, default=0.0, help="directional-leakage penalty weight (left/E side). exp B2 / clean D1 knob.")
    ap.add_argument("--lambda_D", type=float, default=0.0, help="directional-leakage penalty weight (right/D side).")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--initial_scaler", type=float, default=0.1)
    ap.add_argument("--initial_sigma", type=float, default=0.1)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--data_path", default=DEFAULT_DATA)
    ap.add_argument("--cutoff_len", type=int, default=256)
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--micro_batch_size", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--eval_limit", type=int, default=0, help=">0 caps CS+retention examples (smoke only)")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_retention", action="store_true", help="CS+F-delta only (fast, for LR/hparam sweeps)")
    ap.add_argument("--retention_tasks", default="", help="override retention tasks, e.g. bbh_fewshot,mmlu_pro")
    ap.add_argument("--ret_max_gen", type=int, default=0, help=">0 caps MMLU-Pro CoT gen length (speed)")
    ap.add_argument("--ret_limit", type=int, default=0, help=">0 subsamples retention only (CS stays full)")
    ap.add_argument("--no_leakage", action="store_true", help="skip orthogonality-leakage thermometers")
    ap.add_argument("--no_leakage_drift", action="store_true", help="skip the expensive drift SVD in leakage")
    args = ap.parse_args()

    grad_accum = max(1, args.batch_size // args.micro_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos=True):
        r = tokenizer(prompt, truncation=True, max_length=args.cutoff_len, padding=False, return_tensors=None)
        if r["input_ids"][-1] != tokenizer.eos_token_id and len(r["input_ids"]) < args.cutoff_len and add_eos:
            r["input_ids"].append(tokenizer.eos_token_id); r["attention_mask"].append(1)
        r["labels"] = r["input_ids"].copy()
        return r

    data = load_dataset("json", data_files=args.data_path)["train"]
    if args.max_samples > 0:
        data = data.select(range(min(args.max_samples, len(data))))
    train_data = data.shuffle(seed=args.seed).map(lambda dp: tokenize(run_lib.train_prompt(dp)),
                                                  remove_columns=data.column_names)

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cuda:0")
    cfg = UIOrthoLoRAConfig(target_modules=args.target_modules.split(","),
                            num_svalues_to_adapt=args.k_val, num_svectors_to_adapt=args.k_vec,
                            uiortholora_alpha=args.alpha, uiortholora_dropout=args.dropout,
                            use_de=bool(args.use_de), initial_scaler=args.initial_scaler,
                            initial_sigma=args.initial_sigma, drop_major=bool(args.drop_major))
    model = get_peft_model(model, cfg)
    trainable, total = run_lib.count_trainable(model)
    model.config.use_cache = False
    print(f"[{args.run_name}] UIOrthoLoRA k_val={args.k_val} k_vec={args.k_vec} use_de={args.use_de} "
          f"trainable={trainable:,} ({100*trainable/total:.3f}%)", flush=True)

    # penalty-aware trainer: adds the grad-enabled directional-leakage penalty (exp B2 / clean D1)
    class PenTrainer(transformers.Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            out = model(**inputs)
            loss = out.loss
            import leakage as _lkp
            pen = _lkp.leakage_penalty(model, args.lambda_E, args.lambda_D)
            if pen is not None:
                loss = loss + pen.to(loss.dtype)
            return (loss, out) if return_outputs else loss

    TrainerCls = PenTrainer if (args.lambda_E > 0 or args.lambda_D > 0) else transformers.Trainer
    if TrainerCls is PenTrainer:
        print(f"[{args.run_name}] leakage penalty ON: lambda_E={args.lambda_E} lambda_D={args.lambda_D}", flush=True)

    t0 = time.time()
    if not args.skip_train:
        trainer = TrainerCls(
            model=model, train_dataset=train_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=args.micro_batch_size, gradient_accumulation_steps=grad_accum,
                warmup_steps=args.warmup_steps, num_train_epochs=args.num_epochs,
                learning_rate=args.learning_rate, bf16=True, logging_steps=50, optim="adamw_torch",
                lr_scheduler_type="linear", save_strategy="no", output_dir="/tmp/uio_trash",
                report_to="none", seed=args.seed),
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))
        from norm_trace import NormTraceCallback
        _ntc = NormTraceCallback(); trainer.add_callback(_ntc)
        tr = trainer.train()
        train_loss = tr.training_loss
        norm_trace = _ntc.trace
    else:
        train_loss = None; norm_trace = []
    train_s = round(time.time() - t0, 1)
    model.eval()
    model.config.use_cache = True

    # ---- in-domain commonsense (8 datasets) ----
    cs = {}
    for ds in CS_DATASETS:
        acc, _, _, _ = eval_cs.run_eval(model, tokenizer, ds, batch_size=32, num_beams=4,
                                        max_new_tokens=32, limit=args.eval_limit)
        cs[ds] = round(100 * acc, 2)
    cs_avg = round(sum(cs.values()) / len(cs), 2)
    print(f"[{args.run_name}] CS per-dataset={cs}  CS_AVG={cs_avg}", flush=True)

    # ---- F-delta (before retention to reuse model on GPU) ----
    try:
        fd = fdelta_inprocess(model, tokenizer)
    except Exception as e:
        print(f"[{args.run_name}] fdelta failed: {e}", flush=True); fd = {}

    # ---- orthogonality-leakage thermometers (paper App. B.1) ----
    leak = {}
    if not args.no_leakage:
        try:
            import leakage as _lk
            leak = _lk.model_leakage(model, with_drift=not args.no_leakage_drift)
            print(f"[{args.run_name}] leakage mu_E={leak.get('mu_E')} nu_D={leak.get('nu_D')} "
                  f"leak11={leak.get('leak11')} offtail_F={leak.get('offtail_F')} "
                  f"driftU={leak.get('drift_U')} driftV={leak.get('drift_V')}", flush=True)
        except Exception as e:
            print(f"[{args.run_name}] leakage failed: {e}", flush=True)

    # ---- method-agnostic spectral forensics (UᵀΔWV) — same axes as CLoRA/LoRA ----
    fx = {}
    if not args.no_leakage:
        try:
            import forensics as _fx
            fx = _fx.model_forensics(model)
            print(f"[{args.run_name}] forensics out_top_0.5={fx.get('out_top_0.5')} "
                  f"sigma_resp={fx.get('sigma_resp')} in_com={fx.get('in_com')}", flush=True)
        except Exception as e:
            print(f"[{args.run_name}] forensics failed: {e}", flush=True)

    # ---- out-domain retention (answer-only BBH + MMLU-Pro CoT), in-memory ----
    ret, ret_mean = {}, None
    if not args.skip_retention:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        ret_tasks = (args.retention_tasks.split(",") if args.retention_tasks else ["bbh_fewshot", "mmlu_pro"])
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
        gen_kwargs = (f"max_gen_toks={args.ret_max_gen}" if args.ret_max_gen > 0 else None)
        ret_limit = args.ret_limit if args.ret_limit > 0 else (args.eval_limit if args.eval_limit > 0 else None)
        res = simple_evaluate(model=lm, tasks=ret_tasks, bootstrap_iters=0,
                              limit=ret_limit, gen_kwargs=gen_kwargs)
        for t in ret_tasks:
            row = res["results"].get(t, {})
            em = next((v for k, v in row.items() if k.startswith("exact_match") and "stderr" not in k), None)
            ret[("bbh" if "bbh" in t else "mmlu_pro" if "mmlu" in t else t)] = round(100 * em, 2) if em is not None else None
        ret_mean = round((ret.get("bbh", 0) + ret.get("mmlu_pro", 0)) / 2, 2)
        print(f"[{args.run_name}] retention bbh={ret.get('bbh')} mmlu_pro={ret.get('mmlu_pro')} mean={ret_mean}", flush=True)

    headline = {"cs_avg": cs_avg, "bbh": ret.get("bbh"), "mmlu_pro": ret.get("mmlu_pro"),
                "retention_mean": ret_mean, "fdelta": fd.get("fdelta_token_weighted"),
                "dw_sv_max": fd.get("dw_sv_max"), "dw_sv_mean": fd.get("dw_sv_mean"),
                "mu_E": leak.get("mu_E"), "nu_D": leak.get("nu_D"), "leak11": leak.get("leak11"),
                "offtail_F": leak.get("offtail_F"), "drift_U": leak.get("drift_U"), "drift_V": leak.get("drift_V")}
    summary = {"run_name": args.run_name, "method": "uiortholora",
               "config": {"k_val": args.k_val, "k_vec": args.k_vec, "use_de": bool(args.use_de),
                          "drop_major": bool(args.drop_major), "learning_rate": args.learning_rate,
                          "initial_sigma": args.initial_sigma, "initial_scaler": args.initial_scaler,
                          "seed": args.seed, "lambda_E": args.lambda_E, "lambda_D": args.lambda_D},
               "trainable_params": trainable, "trainable_pct": round(100 * trainable / total, 4),
               "train_loss": train_loss, "train_s": train_s, "per_dataset": cs, "fdelta": fd,
               "leakage": leak, "forensics": fx, "norm_trace": norm_trace, "headline": headline,
               "git_commit": run_lib.git_commit(), "evaluated_at": run_lib.now_iso()}
    run_lib.write_json(os.path.join(HERE, "results", args.run_name, "summary.json"), summary)
    run_lib.append_registry("campaign_summary.jsonl", {"run_name": args.run_name, **headline,
                                                       "trainable": trainable, "evaluated_at": run_lib.now_iso()})
    print(f"==== {args.run_name} HEADLINE: {headline} ====", flush=True)


if __name__ == "__main__":
    main()
