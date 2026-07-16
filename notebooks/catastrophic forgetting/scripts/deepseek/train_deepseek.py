"""Sharded LoRA trainer for DeepSeek-V4-Flash (284B FP8 MoE) — the generalization run.

Mirrors train_cs.py's data / prompt / tokenization / Trainer / method surface EXACTLY (so the
adapters are comparable across methods and to the 7B campaign); the only differences are the
things a 284B FP8 MoE forces:

  1. LOAD: the base is dequantized FP8->bf16 (FineGrainedFP8Config(dequantize=True)) and sharded
     across the node's 8 GPUs with device_map="auto" (naive model-parallel). Dequant-to-bf16 is
     deliberate and load-bearing: the residual-init methods (milora/sclora/lora_null/corda)
     overwrite base.weight = W_res and SVD it — that requires a real, mutable [out,in] matrix,
     which a frozen FP8 base cannot provide. Under device_map="auto" each weight lives WHOLE on
     one GPU, so every per-matrix op (residual SVD, get_delta_weight, fdelta hooks) stays correct;
     only the load line changes vs the 7B path. Base is frozen; only bf16 LoRA adapters train.
  2. TARGETS: DeepSeek uses MLA attention — module names are q_a_proj/q_b_proj/kv_proj/o_b_proj
     (NOT q/k/v_proj). Attention-only first (dense, clean for spectral-spread analysis; the 256
     FP4 experts are left untouched). --target_modules is auto-verified against named_modules().
  3. gradient checkpointing ON (activation memory across 43 pipelined layers).

Method surface is identical to train_cs.py: --method lora + toggle flags
  dora=--use_dora 1 · lorawd=--weight_decay X · milora/sclora/lora_null/corda=residual flags ·
  clora=--method clora. Fairness: same base/r/alpha/targets/data/seed/schedule for every method;
  only the method + its (7B-derived) LR vary.

Usage (on a drained 8-GPU node):
  HF_HOME=/scratch/hf_cache CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 scripts/deepseek/train_deepseek.py --method lora --run_name dsv4_lora_r16_lr3e4_s42 \
      --learning_rate 3e-4
"""
import os, sys, json, time, argparse
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)  # working dir (…/catastrophic forgetting) — reuse the 7B code
import run_lib
from train_cs import build_adapter, _nq_train_windows, CLoRARegularizer, CLoRATrainer

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
# MLA attention projections (verified from modeling_deepseek_v4.py); auto-checked at runtime.
DEFAULT_TARGETS = "q_a_proj,q_b_proj,kv_proj,o_b_proj"
DEFAULT_DATA = os.path.join(HERE, "repro/LLM-Adapters/ft-training_set/medmcqa_train.json")


def load_base(dtype=torch.bfloat16):
    """FP8->bf16 dequantized, sharded across all visible GPUs. Returns a frozen base model."""
    from transformers import FineGrainedFP8Config
    qcfg = FineGrainedFP8Config(dequantize=True)  # materialize bf16 weights (mutable; residual-safe)
    # The FP8->bf16 conversion of the (unused) MTP expert weights intermittently trips
    # transformers' strict loading-report (~1 in 4 loads, nondeterministic under device_map);
    # a fresh from_pretrained succeeds. Retry rather than fail the cell.
    last = None
    for attempt in range(4):
        try:
            return AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=dtype, device_map="auto", quantization_config=qcfg,
                low_cpu_mem_usage=True, trust_remote_code=True)
        except RuntimeError as e:
            last = e
            print(f"[load_base] attempt {attempt+1}/4 failed ({str(e)[:90]}); retrying", flush=True)
            import gc; gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    raise last


def resolve_targets(model, requested):
    """Keep only requested suffixes that actually exist as nn.Linear leaves; report the rest."""
    import torch.nn as nn
    want = [t.strip() for t in requested.split(",") if t.strip()]
    present = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) or mod.__class__.__name__.endswith("Linear"):
            leaf = name.split(".")[-1]
            if leaf in want:
                present.add(leaf)
    ok = [t for t in want if t in present]
    missing = [t for t in want if t not in present]
    if missing:
        print(f"[targets] WARNING requested but absent: {missing}; using {ok}", flush=True)
    if not ok:
        raise SystemExit(f"[targets] none of {want} found among Linear leaves — check MLA names")
    print(f"[targets] LoRA target_modules = {ok}", flush=True)
    return ",".join(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["lora", "uiortholora", "clora"])
    ap.add_argument("--run_name", default="")
    ap.add_argument("--out_root", default="/scratch/cf_models")
    ap.add_argument("--base_model", default=MODEL)  # logged; load uses MODEL constant
    ap.add_argument("--data_path", default=DEFAULT_DATA)
    ap.add_argument("--cutoff_len", type=int, default=512)
    ap.add_argument("--num_epochs", type=float, default=1.0)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--micro_batch_size", type=int, default=1)  # naive MP: keep activations small
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--target_modules", default=DEFAULT_TARGETS)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--train_on_inputs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=0)
    # method flags (identical semantics to train_cs.py)
    ap.add_argument("--use_dora", type=int, default=0)
    ap.add_argument("--corda", type=int, default=0)
    ap.add_argument("--corda_calib_size", type=int, default=128)
    ap.add_argument("--milora", type=int, default=0)
    ap.add_argument("--sclora", type=int, default=0)
    ap.add_argument("--sclora_beta", type=float, default=0.5)
    ap.add_argument("--sclora_calib_size", type=int, default=128)
    ap.add_argument("--lora_null", type=int, default=0)
    ap.add_argument("--lora_null_calib_size", type=int, default=128)
    ap.add_argument("--lora_null_dim", type=int, default=0)
    ap.add_argument("--lora_null_freeze_a", type=int, default=0)
    ap.add_argument("--calib_source", choices=["nq_open", "eval_matched"], default="nq_open")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)  # alpha = 2r (7B campaign convention)
    ap.add_argument("--clora_k", type=int, default=512)
    ap.add_argument("--clora_lambda", type=float, default=1.0)
    # UIOrthoLoRA (not used for this run, kept for build_adapter compatibility)
    ap.add_argument("--k_val", type=int, default=256)
    ap.add_argument("--k_vec", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--use_de", type=int, default=1)
    ap.add_argument("--initial_scaler", type=float, default=0.1)
    ap.add_argument("--initial_sigma", type=float, default=0.1)
    args = ap.parse_args()
    args.use_de = bool(args.use_de)
    args.train_on_inputs = bool(args.train_on_inputs)
    # residual methods need per-matrix SVD -> not implemented via cordapp here
    args._cordapp_patterns = None

    run_name = args.run_name or f"dsv4_{args.method}_r{args.lora_r}_s{args.seed}"
    out_root = args.out_root if os.path.isdir(os.path.dirname(args.out_root) or "/") else os.path.join(HERE, "models")
    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "run_config.json")) and \
       os.path.exists(os.path.join(out_dir, "adapter_model.safetensors")):
        print(f"[train_ds] {run_name}: complete adapter already at {out_dir} — SKIP", flush=True)
        return
    grad_accum = max(1, args.batch_size // args.micro_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        r = tokenizer(prompt, truncation=True, max_length=args.cutoff_len, padding=False, return_tensors=None)
        if (r["input_ids"][-1] != tokenizer.eos_token_id and len(r["input_ids"]) < args.cutoff_len
                and add_eos_token):
            r["input_ids"].append(tokenizer.eos_token_id)
            r["attention_mask"].append(1)
        r["labels"] = r["input_ids"].copy()
        return r

    def gen_and_tok(dp):
        tok = tokenize(run_lib.train_prompt(dp))
        if not args.train_on_inputs:
            ul = len(tokenize(run_lib.train_prompt({**dp, "output": ""}), add_eos_token=False)["input_ids"])
            tok["labels"] = [-100] * ul + tok["labels"][ul:]
        return tok

    data = load_dataset("json", data_files=args.data_path)["train"]
    if args.max_samples > 0:
        data = data.select(range(min(args.max_samples, len(data))))
    train_data = data.shuffle(seed=args.seed).map(gen_and_tok, remove_columns=data.column_names)

    print(f"[load] {MODEL} dequant->bf16, device_map=auto ...", flush=True)
    t_load = time.time()
    model = load_base()
    print(f"[load] done in {time.time()-t_load:.0f}s; hf_device_map spans "
          f"{len(set(model.hf_device_map.values()))} devices", flush=True)
    args.target_modules = resolve_targets(model, args.target_modules)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    model, adapter_desc = build_adapter(args.method, model, args)
    trainable, total = run_lib.count_trainable(model)
    print(f"[run {run_name}] method={args.method} trainable={trainable:,} "
          f"({100*trainable/total:.4f}%) adapter={adapter_desc}", flush=True)

    apply_residual_inits(model, args, data, tokenizer)

    clora_reg = None
    if args.method == "clora":
        clora_reg = CLoRARegularizer(model, k=args.clora_k, lambda_=args.clora_lambda, seed=args.seed)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum, warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs, learning_rate=args.learning_rate, bf16=True,
        weight_decay=args.weight_decay, logging_steps=10, optim="adamw_torch",
        lr_scheduler_type="linear", save_strategy="no", output_dir=out_dir,
        report_to="none", seed=args.seed, gradient_checkpointing=False)  # enabled on model above
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    Trainer = CLoRATrainer if args.method == "clora" else transformers.Trainer
    kw = {"clora_reg": clora_reg} if args.method == "clora" else {}
    trainer = Trainer(model=model, train_dataset=train_data, args=training_args, data_collator=collator, **kw)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tr_out = trainer.train()
    dt = time.time() - t0
    peak_gb = round(max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30, 2) \
        if torch.cuda.is_available() else None
    print(f"[mem] peak per-GPU alloc {peak_gb} GB", flush=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    if _is_residual(args):
        import residual_save as Rs
        # capture happened pre-train; convert saved adapter to W0-relative rank-2r
        n, r0 = Rs.convert_saved_to_w0_relative(out_dir, main.init_adapter)
        print(f"[residual_save] converted {n} layers to W0-relative rank-{2*r0} adapter", flush=True)

    cfg = {"run_name": run_name, "method": args.method, "task": "medmcqa",
           "base_model": MODEL, "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
           "adapter": adapter_desc, "trainable_params": trainable, "total_params": total,
           "trainable_pct": round(100 * trainable / total, 4), "grad_accum": grad_accum,
           "effective_batch": args.micro_batch_size * grad_accum, "train_runtime_s": round(dt, 1),
           "final_train_loss": tr_out.training_loss, "peak_mem_gb": peak_gb,
           "git_commit": run_lib.git_commit(), "finished_at": run_lib.now_iso()}
    run_lib.write_json(os.path.join(out_dir, "run_config.json"), cfg)
    run_lib.append_registry("train_registry.jsonl", cfg)
    print(f"[run {run_name}] done in {dt:.0f}s | loss {tr_out.training_loss:.4f} | saved {out_dir}", flush=True)


def _is_residual(args):
    return bool(getattr(args, "corda", 0) or getattr(args, "milora", 0)
                or getattr(args, "sclora", 0) or getattr(args, "lora_null", 0))


def apply_residual_inits(model, args, data, tokenizer):
    """Run the requested residual-init (mutates base.weight=W_res on the bf16 base) and snapshot
    the init adapter for W0-relative conversion at save. Same logic/modules as train_cs.py; the
    calibration forwards run through the device_map-sharded model (inputs -> model.device)."""
    if getattr(args, "milora", 0):
        import milora_init as Mi
        err = Mi.apply_milora(model, r=args.lora_r)
        print(f"[milora] minor-SVD init applied; loss-preserving err={err:.2e}", flush=True)
    if getattr(args, "corda", 0):
        import corda_init as Ci
        cov = Ci.collect_corda_cov(model, _nq_train_windows(args.corda_calib_size), tokenizer,
                                   calib_size=args.corda_calib_size, max_len=2048, bs=1)
        err = Ci.apply_corda(model, cov, r=args.lora_r)
        print(f"[corda] KPA init applied to {len(cov)} layers; err={err:.2e}", flush=True)
    if getattr(args, "sclora", 0):
        import sclora_init as Si
        dplus = [run_lib.train_prompt(dp) for dp in data.select(range(min(args.sclora_calib_size, len(data))))]
        dminus = _nq_train_windows(args.sclora_calib_size, source=args.calib_source)
        M = Si.collect_sclora_M(model, dplus, dminus, tokenizer, beta=args.sclora_beta, max_len=2048)
        err = Si.apply_sclora(model, M, r=args.lora_r)
        print(f"[sclora] beta={args.sclora_beta} init applied; err={err:.2e}", flush=True)
    if getattr(args, "lora_null", 0):
        import lora_null_init as Ni
        cov = Ni.collect_lora_null_cov(model, _nq_train_windows(args.lora_null_calib_size, source=args.calib_source),
                                       tokenizer, calib_size=args.lora_null_calib_size, max_len=2048, bs=1)
        nd = args.lora_null_dim if args.lora_null_dim > 0 else None
        err = Ni.apply_lora_null(model, cov, r=args.lora_r, null_dim=nd)
        if args.lora_null_freeze_a:
            for _n, _m in model.named_modules():
                if "default" in getattr(_m, "lora_A", {}):
                    _m.lora_A["default"].weight.requires_grad_(False)
        print(f"[lora_null] null-space init applied to {len(cov)} layers; err={err:.2e}", flush=True)
    if _is_residual(args):
        import residual_save as Rs
        main.init_adapter = Rs.capture_init_adapter(model)


if __name__ == "__main__":
    main.init_adapter = None
    main()
