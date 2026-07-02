"""
Generalized commonsense trainer for the reproduction campaign.

Same data / prompt template / tokenization / Trainer for every method — only the
adapter config differs — so numbers are comparable. Currently supports:
  --method lora        (validated: BoolQ 69.97% on LLaMA-2-7B)
  --method uiortholora (our method)

Everything is logged: models/<run_name>/run_config.json (full config + trainable
param count + git commit + timing) and a line in results/train_registry.jsonl.
"""
import os
import json
import time
import argparse

import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

import run_lib


class CLoRARegularizer:
    """CLoRA orthogonal subspace penalty (Lu et al., ACL 2025).

    Ports github.com/sutakori/CLoRA exactly: one frozen random-orthonormal pair per
    target module — P_u (out x k), P_v (in x k) — penalty
        lambda * sum_modules ( 0.5*||A @ P_v||_F^2 + 0.5*||B^T @ P_u||_F^2 )
    summed over the LoRA A/B of every target module (vanilla LoRA init).
    """

    def __init__(self, model, k, lambda_=1.0, seed=42):
        self.k = k
        self.lambda_ = lambda_
        self.layers = []  # (LoraLayer, P_v[in,k], P_u[out,k])
        g = torch.Generator(device="cpu").manual_seed(seed)
        for name, mod in model.named_modules():
            if isinstance(mod, LoraLayer) and "default" in mod.lora_A:
                w = mod.get_base_layer().weight  # (out, in)
                out_f, in_f = w.shape
                Pv = self._rand_orthonormal(in_f, k, g, w.device).to(dtype=w.dtype)
                Pu = self._rand_orthonormal(out_f, k, g, w.device).to(dtype=w.dtype)
                self.layers.append((mod, Pv, Pu))
        print(f"[clora] registered {len(self.layers)} target-module P pairs (k={k}, lambda={lambda_})",
              flush=True)

    @staticmethod
    def _rand_orthonormal(dim, k, g, device):
        # k orthonormal columns in R^dim (matches nn.init.orthogonal_ semantics).
        # randn on CPU (deterministic via generator), QR on GPU to avoid thrashing
        # the 128-core CPU when many runs build P concurrently.
        a = torch.randn(dim, k, generator=g).to(device=device, dtype=torch.float32)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # sign-stabilize
        return q.detach()

    def loss(self):
        reg = 0.0
        for mod, Pv, Pu in self.layers:
            A = mod.lora_A["default"].weight  # (r, in)  (peft keeps these fp32)
            B = mod.lora_B["default"].weight  # (out, r)
            reg = reg + torch.norm(A @ Pv.to(A.dtype), p="fro") ** 2 / 2
            reg = reg + torch.norm(B.T @ Pu.to(B.dtype), p="fro") ** 2 / 2
        return self.lambda_ * reg


class CLoRATrainer(transformers.Trainer):
    def __init__(self, *args, clora_reg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.clora_reg = clora_reg

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        loss, outputs = out
        if self.clora_reg is not None and model.training:
            loss = loss + self.clora_reg.loss()
        return (loss, outputs) if return_outputs else loss

HERE = run_lib.HERE
DEFAULT_DATA = os.path.join(HERE, "repro/LLM-Adapters/ft-training_set/commonsense_170k.json")


def build_adapter(method, model, args):
    """Return peft-wrapped model + a dict describing the adapter config (for logging)."""
    targets = args.target_modules.split(",")
    if method in ("lora", "clora"):
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=targets,
                         lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
                         use_dora=bool(getattr(args, "use_dora", 0)))
        desc = {"r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.dropout,
                "use_dora": bool(getattr(args, "use_dora", 0))}
        if method == "clora":
            desc.update({"clora_k": args.clora_k, "clora_lambda": args.clora_lambda})
    elif method == "uiortholora":
        from peft import UIOrthoLoRAConfig
        cfg = UIOrthoLoRAConfig(
            target_modules=targets,
            num_svalues_to_adapt=args.k_val,
            num_svectors_to_adapt=args.k_vec,
            uiortholora_alpha=args.alpha,
            uiortholora_dropout=args.dropout,
            use_de=args.use_de,
            initial_scaler=args.initial_scaler,
            initial_sigma=args.initial_sigma,
        )
        desc = {"k_val": args.k_val, "k_vec": args.k_vec, "alpha": args.alpha,
                "dropout": args.dropout, "use_de": args.use_de,
                "initial_scaler": args.initial_scaler, "initial_sigma": args.initial_sigma}
    else:
        raise ValueError(f"unknown method {method}")
    return get_peft_model(model, cfg), {"target_modules": targets, **desc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["lora", "uiortholora", "clora"])
    ap.add_argument("--run_name", default="")
    ap.add_argument("--out_root", default="/scratch/cf_models")
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--data_path", default=DEFAULT_DATA)
    ap.add_argument("--cutoff_len", type=int, default=256)
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="AdamW decay on adapter params = subspace-free MAGNITUDE knob (CLoRA control).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--micro_batch_size", type=int, default=16)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--train_on_inputs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=0)
    # LoRA
    ap.add_argument("--use_dora", type=int, default=0, help="DoRA: decouple magnitude (m) from direction. Test the adaptation-per-||dW|| frontier vs retention (DoRA never eval'd for retention).")
    ap.add_argument("--corda", type=int, default=0, help="CorDA-KPA data-aware init (faithful port; needs --lora_alpha==--lora_r). corda_init.py.")
    ap.add_argument("--corda_calib_size", type=int, default=256)
    ap.add_argument("--milora", type=int, default=0, help="MiLoRA minor (bottom-r) SVD init; no calib. milora_init.py. Needs --lora_alpha==--lora_r.")
    ap.add_argument("--sclora", type=int, default=0, help="SC-LoRA data-aware init (D+/D- covariance). sclora_init.py. Needs --lora_alpha==--lora_r.")
    ap.add_argument("--sclora_beta", type=float, default=0.5, help="SC-LoRA balance: top-r eigvecs of (1-beta)Cov+ - beta*Cov- (swept knob).")
    ap.add_argument("--sclora_calib_size", type=int, default=256)
    # LoRA-Null (null-space init; lora_null_init.py). Needs --lora_alpha==--lora_r.
    ap.add_argument("--lora_null", type=int, default=0, help="LoRA-Null: init adapter in the null space of knowledge-input activations (calib=nq_open). lora_null_init.py.")
    ap.add_argument("--lora_null_calib_size", type=int, default=256)
    ap.add_argument("--lora_null_dim", type=int, default=0, help="Null-space dimensionality; 0 -> rank-matched (=lora_r). See fidelity flag in lora_null_init.py.")
    ap.add_argument("--lora_null_freeze_a", type=int, default=0, help="Freeze A during training (paper's best-preservation variant); default 0 trains both (head-to-head fairness).")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    # CLoRA
    ap.add_argument("--clora_k", type=int, default=512)
    ap.add_argument("--clora_lambda", type=float, default=1.0)
    # UIOrthoLoRA
    ap.add_argument("--k_val", type=int, default=256)
    ap.add_argument("--k_vec", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--use_de", type=int, default=1)
    ap.add_argument("--initial_scaler", type=float, default=0.1)
    ap.add_argument("--initial_sigma", type=float, default=0.1)
    args = ap.parse_args()
    args.use_de = bool(args.use_de)
    args.train_on_inputs = bool(args.train_on_inputs)

    run_name = args.run_name or f"{args.method}_cs_l2-7b_s{args.seed}"
    # checkpoints go on the big root volume (/scratch): UIOrthoLoRA stores full SVD
    # buffers (~14.5GB/adapter), too big for the 44G /home.
    out_root = args.out_root if os.path.isdir(os.path.dirname(args.out_root) or "/") else os.path.join(HERE, "models")
    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    grad_accum = max(1, args.batch_size // args.micro_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        r = tokenizer(prompt, truncation=True, max_length=args.cutoff_len, padding=False,
                      return_tensors=None)
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

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model, adapter_desc = build_adapter(args.method, model, args)
    trainable, total = run_lib.count_trainable(model)
    model.config.use_cache = False
    print(f"[run {run_name}] method={args.method} trainable={trainable:,} ({100*trainable/total:.3f}%) "
          f"adapter={adapter_desc}", flush=True)

    if getattr(args, "corda", 0):
        import corda_init as Ci
        from datasets import load_dataset as _ld
        assert args.lora_alpha == args.lora_r, "CorDA needs scaling=1 -> set --lora_alpha == --lora_r"
        # CorDA-KPA freezes the directions most responsive to the calibration data (the knowledge to
        # preserve). Paper/repo default = QA knowledge (nq_open), NOT general LM text. FIXED 2026-06-29
        # (was wikitext-2 -> preserved the wrong subspace; matches sclora/lora_null which use nq_open).
        try:
            nq = _ld("google-research-datasets/nq_open", split="validation")
            cprompts = [q for q in nq["question"][:args.corda_calib_size]]
        except Exception as e:
            print(f"[corda] nq_open load failed ({e}); falling back to wikitext calib", flush=True)
            wt = _ld("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            cprompts = [t for t in wt["text"] if len(t.strip()) > 50][:args.corda_calib_size]
        cov = Ci.collect_corda_cov(model, cprompts, tokenizer, calib_size=args.corda_calib_size)
        err = Ci.apply_corda(model, cov, r=args.lora_r)
        print(f"[corda] KPA init (nq_open calib) applied to {len(cov)} layers; loss-preserving err={err:.2e}", flush=True)

    if getattr(args, "milora", 0):
        import milora_init as Mi
        assert args.lora_alpha == args.lora_r, "MiLoRA needs scaling=1 -> set --lora_alpha == --lora_r"
        err = Mi.apply_milora(model, r=args.lora_r)
        print(f"[milora] minor-SVD init applied; loss-preserving err={err:.2e}", flush=True)

    if getattr(args, "sclora", 0):
        import sclora_init as Si
        from datasets import load_dataset as _ld
        assert args.lora_alpha == args.lora_r, "SC-LoRA needs scaling=1 -> set --lora_alpha == --lora_r"
        # D+ = the finetuning task; D- = world knowledge to preserve. Repo uses NQ-open (nq_open).
        dplus = [run_lib.train_prompt(dp) for dp in data.select(range(min(args.sclora_calib_size, len(data))))]
        try:
            nq = _ld("google-research-datasets/nq_open", split="validation")
            dminus = [q for q in nq["question"][:args.sclora_calib_size]]
        except Exception as e:
            print(f"[sclora] nq_open load failed ({e}); falling back to wikitext D-", flush=True)
            wt = _ld("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            dminus = [t for t in wt["text"] if len(t.strip()) > 50][:args.sclora_calib_size]
        # output-side balanced 2nd-moment with beta+sign folded in (repo-faithful); then eigh top-r.
        M = Si.collect_sclora_M(model, dplus, dminus, tokenizer, beta=args.sclora_beta,
                                max_len=args.cutoff_len)
        err = Si.apply_sclora(model, M, r=args.lora_r)
        print(f"[sclora] beta={args.sclora_beta} (output-side, NQ-open D-) init applied; "
              f"loss-preserving err={err:.2e}", flush=True)

    if getattr(args, "lora_null", 0):
        import lora_null_init as Ni
        from datasets import load_dataset as _ld
        assert args.lora_alpha == args.lora_r, "LoRA-Null needs scaling=1 -> set --lora_alpha == --lora_r"
        # Null space of KNOWLEDGE-input activations to preserve (repo default: nq_open, 256 samples).
        try:
            nq = _ld("google-research-datasets/nq_open", split="validation")
            kprompts = [q for q in nq["question"][:args.lora_null_calib_size]]
        except Exception as e:
            print(f"[lora_null] nq_open load failed ({e}); falling back to wikitext calib", flush=True)
            wt = _ld("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            kprompts = [t for t in wt["text"] if len(t.strip()) > 50][:args.lora_null_calib_size]
        cov = Ni.collect_lora_null_cov(model, kprompts, tokenizer, calib_size=args.lora_null_calib_size,
                                       max_len=args.cutoff_len)
        nd = args.lora_null_dim if args.lora_null_dim > 0 else None
        err = Ni.apply_lora_null(model, cov, r=args.lora_r, null_dim=nd)
        if args.lora_null_freeze_a:
            for _n, _m in model.named_modules():
                if "default" in getattr(_m, "lora_A", {}):
                    _m.lora_A["default"].weight.requires_grad_(False)
        print(f"[lora_null] null-space init applied to {len(cov)} layers "
              f"(null_dim={nd or args.lora_r}, freeze_a={bool(args.lora_null_freeze_a)}); "
              f"loss-preserving err={err:.2e}", flush=True)

    # CorDA/MiLoRA/SC-LoRA/LoRA-Null overwrite base.weight=W_res but PEFT saves only the adapter ->
    # snapshot the init adapter now so we can persist a W0-relative (rank-2r) adapter at save.
    residual_method = bool(getattr(args, "corda", 0) or getattr(args, "milora", 0)
                           or getattr(args, "sclora", 0) or getattr(args, "lora_null", 0))
    init_adapter = None
    if residual_method:
        import residual_save as Rs
        init_adapter = Rs.capture_init_adapter(model)

    clora_reg = None
    if args.method == "clora":
        clora_reg = CLoRARegularizer(model, k=args.clora_k, lambda_=args.clora_lambda, seed=args.seed)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum, warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs, learning_rate=args.learning_rate, bf16=True,
        weight_decay=args.weight_decay,
        logging_steps=10, optim="adamw_torch", lr_scheduler_type="linear",
        save_strategy="no", output_dir=out_dir, report_to="none", seed=args.seed)
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    if args.method == "clora":
        trainer = CLoRATrainer(model=model, train_dataset=train_data, args=training_args,
                               data_collator=collator, clora_reg=clora_reg)
    else:
        trainer = transformers.Trainer(model=model, train_dataset=train_data, args=training_args,
                                       data_collator=collator)
    from norm_trace import NormTraceCallback
    _ntc = NormTraceCallback(); trainer.add_callback(_ntc)
    t0 = time.time()
    tr_out = trainer.train()
    dt = time.time() - t0
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    if residual_method:
        import residual_save as Rs
        n, r0 = Rs.convert_saved_to_w0_relative(out_dir, init_adapter)
        print(f"[residual_save] converted {n} layers to W0-relative rank-{2*r0} adapter "
              f"(was rank-{r0}); eval now uses correct W0 base", flush=True)

    cfg = {
        "run_name": run_name, "method": args.method, "task": "commonsense_170k",
        "base_model": args.base_model, "args": vars(args), "adapter": adapter_desc,
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "grad_accum": grad_accum, "effective_batch": args.micro_batch_size * grad_accum,
        "train_runtime_s": round(dt, 1), "final_train_loss": tr_out.training_loss,
        "norm_trace": _ntc.trace,
        "git_commit": run_lib.git_commit(), "finished_at": run_lib.now_iso(),
    }
    run_lib.write_json(os.path.join(out_dir, "run_config.json"), cfg)
    run_lib.append_registry("train_registry.jsonl", cfg)
    print(f"[run {run_name}] done in {dt:.0f}s | loss {tr_out.training_loss:.4f} | "
          f"trainable {trainable:,} | saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
