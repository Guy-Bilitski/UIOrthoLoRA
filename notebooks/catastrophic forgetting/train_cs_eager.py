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
        # NB: under device_map sharding (DeepSeek) each target Linear can live on a
        # different GPU, so accumulate every per-layer term onto a single device
        # (reg's device = first term's) before summing. On the single-GPU 7B path this
        # is a no-op. Also pin Pv/Pu to each A/B's device, not just dtype.
        reg = None
        for mod, Pv, Pu in self.layers:
            A = mod.lora_A["default"].weight  # (r, in)  (peft keeps these fp32)
            B = mod.lora_B["default"].weight  # (out, r)
            t = torch.norm(A @ Pv.to(A.device, A.dtype), p="fro") ** 2 / 2 \
              + torch.norm(B.T @ Pu.to(B.device, B.dtype), p="fro") ** 2 / 2
            reg = t if reg is None else reg + t.to(reg.device)
        if reg is None:
            return 0.0
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


def _nq_train_windows(calib_size, seqlen=2048, seed=233, source="nq_open"):
    """REPO-FAITHFUL calibration text for the data-aware inits (LoRA-Null / CorDA / SC-LoRA D-).

    HungerPWAY/LoRA-Null get_calib_data "nqopen" (== iboing/CorDA == CoffeePot1206/SC-LoRA D-;
    all three share this exact loader): concatenate ALL nq_open TRAIN questions into one blob,
    then draw `calib_size` random windows of seqlen*10 chars each (the tokenizer later truncates
    to `seqlen` tokens) => ~calib_size*seqlen tokens => C is generically FULL-RANK at every layer.
    Sampling a handful of short individual questions instead (the old loader) leaves C rank-deficient
    and the extracted subspace an arbitrary basis of the unsampled input space -- verified against
    LoRA-Null commit 1e6808a / handoff/23 sec 1.5, 3, 4. seed 233 = repo default calib seed;
    the wikitext fallback preserves old behavior ONLY if nq_open is unreachable (never observed).

    source="eval_matched" (B4 arm, PI-approved): SAME windowed mechanism, but the text blob is MMLU
    auxiliary_train (question + choices + answer; disjoint from every MMLU/MMLU-Pro TEST split)
    instead of nq_open train. Tests whether a data-aware method's retention deviation is a
    calibration-corpus artifact: the "knowledge to preserve" is drawn from the same distribution
    the retention eval probes."""
    import random as _rnd
    from datasets import load_dataset as _ld
    _rnd.seed(seed)
    if source == "eval_matched":
        mm = _ld("cais/mmlu", "auxiliary_train", split="train")
        rows = []
        for ex in mm:
            ex = ex.get("train", ex)  # cais/mmlu auxiliary_train nests each row under a 'train' key
            ch = list(ex["choices"])
            rows.append(ex["question"] + "\n" + "\n".join(ch) + "\nAnswer: " + ch[int(ex["answer"])])
        tot_text = "\n\n".join(rows)
        print(f"[calib] eval_matched source: mmlu auxiliary_train, {len(rows)} rows, "
              f"{len(tot_text)} chars", flush=True)
    else:
        try:
            nq = _ld("google-research-datasets/nq_open", split="train")
            tot_text = "\n\n".join(nq["question"])
        except Exception as e:
            print(f"[calib] nq_open load failed ({e}); falling back to wikitext calib", flush=True)
            wt = _ld("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            tot_text = "\n\n".join(wt["text"])
    prompts = []
    hi = max(0, len(tot_text) - seqlen - 1)
    for _ in range(calib_size):
        i = _rnd.randint(0, hi)
        prompts.append(tot_text[i:i + seqlen * 10])  # char window; tokenizer truncates to seqlen tokens
    return prompts


def build_adapter(method, model, args):
    """Return peft-wrapped model + a dict describing the adapter config (for logging)."""
    targets = args.target_modules.split(",")
    if method in ("lora", "clora"):
        # CorDA++ per-layer dynamic ranks: rank_pattern/alpha_pattern were precomputed on the raw model
        # (before this wrap) and stashed on args by main(); they hold scaling=alpha/r at every layer.
        _cpp = getattr(args, "_cordapp_patterns", None)
        _kw = dict(rank_pattern=_cpp[0], alpha_pattern=_cpp[1]) if _cpp else {}
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=targets,
                         lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
                         use_dora=bool(getattr(args, "use_dora", 0)), **_kw)
        desc = {"r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.dropout,
                "use_dora": bool(getattr(args, "use_dora", 0))}
        if method == "clora":
            desc.update({"clora_k": args.clora_k, "clora_lambda": args.clora_lambda})
        if _cpp:
            desc.update({"cordapp": 1, "cordapp_layers": len(_cpp[0])})
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
    # E5 replay baseline (adversarial-review menu 2026-07-16): mix a fraction of
    # general-knowledge QA (nq_open train, instruction-formatted) into the adapt set —
    # the practitioner's standard forgetting mitigation. Tests whether replay beats /
    # moves the magnitude relation.
    ap.add_argument("--replay_frac", type=float, default=0.0,
                    help=">0: add replay_frac*len(adapt_data) nq_open QA examples to training")
    # E2 full-FT anchor (adversarial-review menu 2026-07-16): no adapter — train ALL params;
    # F_Delta computed from the dense dW on the target modules and stored in
    # <out_dir>/fdelta_fullft.json (eval_one_gpu --adapter none picks it up). Scope test:
    # is the magnitude relation a LoRA-family artifact or does dense FT sit on the same curve?
    ap.add_argument("--full_ft", type=int, default=0,
                    help="1 = full fine-tune (no PEFT); saves the full model to out_root/run_name")
    # LoRA
    ap.add_argument("--use_dora", type=int, default=0, help="DoRA: decouple magnitude (m) from direction. Test the adaptation-per-||dW|| frontier vs retention (DoRA never eval'd for retention).")
    ap.add_argument("--corda", type=int, default=0, help="CorDA-KPA data-aware init (faithful port; residual_save preserves any alpha/r, so alpha=2r is fine). corda_init.py.")
    ap.add_argument("--corda_calib_size", type=int, default=256)
    ap.add_argument("--milora", type=int, default=0, help="MiLoRA minor (bottom-r) SVD init; no calib. milora_init.py.")
    ap.add_argument("--pissa", type=int, default=0, help="PiSSA major (top-r) SVD init; no calib. data_aware_init.pissa_BAR. Residual method; residual_save preserves any scaling (alpha!=r OK).")
    ap.add_argument("--sclora", type=int, default=0, help="SC-LoRA data-aware init (D+/D- covariance). sclora_init.py. residual_save preserves any alpha/r (alpha=2r OK).")
    ap.add_argument("--sclora_beta", type=float, default=0.5, help="SC-LoRA balance: top-r eigvecs of (1-beta)Cov+ - beta*Cov- (swept knob).")
    ap.add_argument("--sclora_calib_size", type=int, default=256)
    # LoRA-Null (null-space init; lora_null_init.py). residual_save preserves any alpha/r (alpha=2r OK).
    ap.add_argument("--lora_null", type=int, default=0, help="LoRA-Null: init adapter in the null space of knowledge-input activations (calib=nq_open). lora_null_init.py.")
    ap.add_argument("--lora_null_calib_size", type=int, default=256)
    ap.add_argument("--lora_null_dim", type=int, default=0, help="Null-space dimensionality; 0 -> rank-matched (=lora_r). See fidelity flag in lora_null_init.py.")
    ap.add_argument("--lora_null_freeze_a", type=int, default=0, help="Freeze A during training (paper's best-preservation variant); default 0 trains both (head-to-head fairness).")
    # B4 eval-matched calibration arm (PI-approved): which corpus feeds the data-aware calibration
    # loaders (sclora D- / lora_null / cordapp). eval_matched = windowed MMLU auxiliary_train
    # (disjoint from all test splits) instead of nq_open train; same windowed-2048 mechanism.
    ap.add_argument("--calib_source", choices=["nq_open", "eval_matched"], default="nq_open",
                    help="Calibration corpus for the sclora/lora_null/cordapp loaders. eval_matched "
                         "= MMLU auxiliary_train windows (B4 registry-artifact test).")
    # CorDA++ (dynamic Context-oriented Decomposition; cordapp_init.py). OPTIONAL-APPLY: Tier-B, gated
    # (PI decision P6 = controlled-only, contingent on validate_cordapp_cpu.py + a 1-GPU 0-step gate PASS).
    ap.add_argument("--cordapp", type=int, default=0, help="CorDA++ dynamic cov-selection + per-layer rank "
                    "allocation KPM init (reuses corda_init decomposition). Residual method -> "
                    "residual_save + finalize_dynamic_rank_config. cordapp_init.py.")
    ap.add_argument("--cordapp_n", type=int, default=5, help="CorDA++ candidate covariance pool size N "
                    "(paper value unresolved; consortium default 5). One covariance per calib round.")
    ap.add_argument("--cordapp_calib_size", type=int, default=0, help="Total calib prompts for CorDA++ "
                    "(0 -> cordapp_n*256, i.e. ~256 windows per round).")
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
    # [train_cs] idempotency guard: if this adapter was already fully trained (run_config.json is
    # written last in this script, so it marks a completed run), skip retraining and let the
    # downstream `&& eval_one_gpu.py` run. Turns a dispatcher retry of a banked adapter into
    # seconds of eval instead of hours of retrain (added 2026-07-15 for fleet eval recovery).
    if os.path.exists(os.path.join(out_dir, 'run_config.json')) and \
       os.path.exists(os.path.join(out_dir, 'adapter_model.safetensors')):
        print(f'[train_cs] {run_name}: complete adapter already at {out_dir} — SKIP train, proceed to eval', flush=True)
        return
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
    if args.replay_frac > 0:
        from datasets import Dataset, concatenate_datasets
        import random as _rnd
        nq = load_dataset("google-research-datasets/nq_open", split="train")
        k = int(args.replay_frac * len(data))
        _rnd.seed(args.seed)
        idx = _rnd.sample(range(len(nq)), min(k, len(nq)))
        rep_rows = [{"instruction": nq[i]["question"] + "?",
                     "input": "",
                     "output": (nq[i]["answer"][0] if nq[i]["answer"] else ""),
                     "answer": (nq[i]["answer"][0] if nq[i]["answer"] else "")}
                    for i in idx]
        data = concatenate_datasets([data, Dataset.from_list(rep_rows)])
        print(f"[replay] mixed {len(rep_rows)} nq_open QA examples "
              f"({args.replay_frac:.0%} of adapt set) into training", flush=True)
    train_data = data.shuffle(seed=args.seed).map(gen_and_tok, remove_columns=data.column_names)

    _attn = os.environ.get("DIAG_ATTN", "eager")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16,
                                                 device_map="cuda:0",
                                                 attn_implementation=_attn)
    print(f"[diag] attn_implementation requested={_attn} "
          f"actual={getattr(model.config, '_attn_implementation', '?')}", flush=True)
    cordapp_res = None
    if getattr(args, "cordapp", 0):
        import cordapp_init as Cpp
        _targets = args.target_modules.split(",")
        _csz = args.cordapp_calib_size or (args.cordapp_n * 256)
        # Precompute on the RAW model (BEFORE the PEFT wrap): stream N calib rounds -> per layer pick the
        # best covariance (Eq 8) -> allocate per-layer ranks to a tau param-matched to the fixed-rank arm
        # (Eq 9-10). Windowed nq_open calib (repo-faithful, full-rank C) as for corda/lora_null.
        _cpp_prompts = _nq_train_windows(_csz, source=args.calib_source)
        cordapp_res = Cpp.precompute_cordapp(model, _cpp_prompts, tokenizer, _targets,
                                             fixed_rank=args.lora_r, N=args.cordapp_n,
                                             calib_size=_csz, max_len=2048, bs=1,
                                             scaling=args.lora_alpha / args.lora_r)
        args._cordapp_patterns = (cordapp_res["rank_pattern"], cordapp_res["alpha_pattern"])
        print(f"[cordapp] N={args.cordapp_n} calib={_csz} tau(nominal)={cordapp_res['nominal_tau']} "
              f"realized={cordapp_res['realized_tau']} over {len(cordapp_res['ranks'])} layers", flush=True)
    fullft_snap = None
    if args.full_ft:
        _tg = set(args.target_modules.split(","))
        fullft_snap = {n: m.weight.detach().to("cpu", torch.bfloat16).clone()
                       for n, m in model.named_modules()
                       if n.split(".")[-1] in _tg and hasattr(m, "weight")}
        for p in model.parameters():
            p.requires_grad_(True)
        adapter_desc = f"full_ft (dense dW snapshot: {len(fullft_snap)} target mats)"
    else:
        model, adapter_desc = build_adapter(args.method, model, args)
    trainable, total = run_lib.count_trainable(model)
    model.config.use_cache = False
    print(f"[run {run_name}] method={args.method} trainable={trainable:,} ({100*trainable/total:.3f}%) "
          f"adapter={adapter_desc}", flush=True)

    if getattr(args, "corda", 0):
        import corda_init as Ci
        # scaling!=1 OK: residual_save preserves any alpha/r (validated); loss-preservation checked by err.
        # CorDA-KPA freezes the directions most responsive to the calibration data (the knowledge to
        # preserve). Paper/repo default = QA knowledge (nq_open), NOT general LM text.
        # OPTIONAL-BUT-RECOMMENDED calibration fidelity fix (handoff/23 sec 4): repo-faithful WINDOWED
        # nq_open calib (concat train questions -> 256 x 2048-token windows, ~524k tok) so C is full-rank
        # and the KPA "context" directions are properly sampled. The old loader fed 256 short validation
        # questions at max_len=256 -> rank-deficient C. Only runs when --corda 1 (CorDA is NOT in the
        # frepro4 queue unless a PI re-queues it -- see Q3 in the runbook).
        cprompts = _nq_train_windows(args.corda_calib_size)
        cov = Ci.collect_corda_cov(model, cprompts, tokenizer, calib_size=args.corda_calib_size,
                                   max_len=2048, bs=1)
        err = Ci.apply_corda(model, cov, r=args.lora_r)
        print(f"[corda] KPA init (windowed nq_open calib, 2048 tok/window) applied to {len(cov)} "
              f"layers; loss-preserving err={err:.2e}", flush=True)

    if getattr(args, "milora", 0):
        import milora_init as Mi
        # residual_save preserves any scaling (alpha!=r OK, e.g. faithful CLoRA r64/alpha128);
        # loss-preservation is checked by the returned err + validate_residual_zero_step.py.
        err = Mi.apply_milora(model, r=args.lora_r)
        print(f"[milora] minor-SVD init applied (alpha={args.lora_alpha},r={args.lora_r}); loss-preserving err={err:.2e}", flush=True)

    if getattr(args, "pissa", 0):
        import data_aware_init as Di
        # PiSSA = major (top-r) SVD init; residual method (residual_save preserves scaling).
        err = Di.inject_lora_init(model, Di.pissa_BAR(args.lora_r))
        print(f"[pissa] major-SVD init applied (alpha={args.lora_alpha},r={args.lora_r}); loss-preserving err={err:.2e}", flush=True)

    if getattr(args, "sclora", 0):
        import sclora_init as Si
        # scaling!=1 OK: residual_save preserves any alpha/r (validated); loss-preservation checked by err.
        # D+ = the finetuning task (with answer); D- = world knowledge to preserve (repo: nq_open).
        # OPTIONAL-BUT-RECOMMENDED calibration fidelity fix (handoff/23 sec 3): repo-faithful WINDOWED
        # nq_open D- (concat train questions -> 256 x 2048-token windows, ~524k tok) instead of 256 short
        # validation questions (~2.5k tok, ~200x under-sampled); D+ truncation raised to 2048 to match the
        # repo "MetaMATH" branch. [CHANGES QUEUED: alters the sclora cells' calibration -> requires the
        # kill+relaunch to take effect on frm_sclora_* / frc_sclora_*.]
        dplus = [run_lib.train_prompt(dp) for dp in data.select(range(min(args.sclora_calib_size, len(data))))]
        dminus = _nq_train_windows(args.sclora_calib_size, source=args.calib_source)
        # output-side balanced 2nd-moment with beta+sign folded in (repo-faithful); then eigh top-r.
        M = Si.collect_sclora_M(model, dplus, dminus, tokenizer, beta=args.sclora_beta,
                                max_len=2048)
        err = Si.apply_sclora(model, M, r=args.lora_r)
        print(f"[sclora] beta={args.sclora_beta} (output-side, windowed {args.calib_source} D-) init applied; "
              f"loss-preserving err={err:.2e}", flush=True)

    if getattr(args, "lora_null", 0):
        import lora_null_init as Ni
        # scaling!=1 OK: residual_save preserves any alpha/r (validated); loss-preservation checked by err.
        # MANDATORY repo-faithful calib (handoff/23 sec 1.5, HungerPWAY/LoRA-Null get_calib_data "nqopen"):
        # concat ALL nq_open TRAIN questions, 256 windows of 2048 TOKENS each (~524k tok => rank(C)=d_in),
        # forwarded bs=1 (per-sample max-norm, zero padding). The old loader (256 short validation
        # questions, ~2.5k tok, bs=4 padded) left C rank-deficient and the "null space" an arbitrary basis
        # of the unsampled input subspace -- so the pre-restart LoRA-Null cells did NOT test LoRA-Null.
        # This REPLACES the 12 degenerate lora_null cells (math lean 50-55 / CS 98-103).
        kprompts = _nq_train_windows(args.lora_null_calib_size,
                                     source=args.calib_source)   # seed 233, 2048-token windows
        cov = Ni.collect_lora_null_cov(model, kprompts, tokenizer,
                                       calib_size=args.lora_null_calib_size,
                                       max_len=2048, bs=1)   # bs=1: per-sample max-norm, zero padding
        # rank(C) diagnostic (repo prints (S_>0.1).sum() per layer): a [lora_null] rankdiag line in the
        # log CONFIRMS the full-rank calibration is live (its absence => the old degenerate path).
        for _n in list(cov)[:3]:
            _S = torch.linalg.svdvals(0.5 * (cov[_n] + cov[_n].transpose(-1, -2)).float())
            print(f"[lora_null] rankdiag {_n}: d={cov[_n].shape[0]} "
                  f"S>0.1:{int((_S > 0.1).sum())} S>1e-6:{int((_S > 1e-6).sum())}", flush=True)
        nd = args.lora_null_dim if args.lora_null_dim > 0 else None
        err = Ni.apply_lora_null(model, cov, r=args.lora_r, null_dim=nd)
        if args.lora_null_freeze_a:
            for _n, _m in model.named_modules():
                if "default" in getattr(_m, "lora_A", {}):
                    _m.lora_A["default"].weight.requires_grad_(False)
        print(f"[lora_null] null-space init (calib={args.calib_source}) applied to {len(cov)} layers "
              f"(null_dim={nd or args.lora_r}, freeze_a={bool(args.lora_null_freeze_a)}); "
              f"loss-preserving err={err:.2e}", flush=True)

    if getattr(args, "cordapp", 0):
        import cordapp_init as Cpp
        # Inject KPM init at each layer's ALLOCATED rank (PEFT resolved r^l from rank_pattern); folds
        # scaling into B and overwrites base.weight=W_res -> residual_save compatible (like corda).
        err = Cpp.apply_cordapp(model, cordapp_res["chosen_covs"], cordapp_res["ranks"])
        INIT_ERR_TOL = 1e-2   # bf16 loss-preserving round-trip (A1 init-error gate; abort loudly if not)
        assert err < INIT_ERR_TOL, f"[cordapp] init not loss-preserving: err={err:.2e} >= {INIT_ERR_TOL}"
        print(f"[cordapp] dynamic-rank KPM init applied to {len(cordapp_res['ranks'])} layers; "
              f"loss-preserving err={err:.2e}", flush=True)

    # CorDA/MiLoRA/SC-LoRA/LoRA-Null/PiSSA/CorDA++ overwrite base.weight=W_res but PEFT saves only the
    # adapter -> snapshot the init adapter now so we can persist a W0-relative (rank-2r) adapter at save.
    residual_method = bool(getattr(args, "corda", 0) or getattr(args, "milora", 0)
                           or getattr(args, "sclora", 0) or getattr(args, "lora_null", 0)
                           or getattr(args, "pissa", 0) or getattr(args, "cordapp", 0))
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
    # Peak-memory instrumentation (handoff/35): cumulative peak so far = init phase
    # (calibration/SVD passes included); reset, then measure the training phase alone.
    peak_init_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tr_out = trainer.train()
    dt = time.time() - t0
    peak_train_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    print(f"[mem] peak GPU alloc: init-phase {peak_init_gb} GB, train-phase {peak_train_gb} GB", flush=True)
    if args.full_ft and fullft_snap is not None:
        # dense-dW F_Delta on the adapt distribution, same hook math as the adapter runs
        model.eval()
        _mods = dict(model.named_modules())
        _deltas = {n: (_mods[n].weight.detach()
                       - snap.to(_mods[n].weight.device, _mods[n].weight.dtype))
                   for n, snap in fullft_snap.items()}
        from uio_inprocess import fdelta_inprocess
        try:
            _fd = fdelta_inprocess(model, tokenizer, deltas=_deltas)
        except Exception as _e:
            print(f"[full_ft] fdelta failed: {_e}", flush=True); _fd = {}
        del _deltas
        run_lib.write_json(os.path.join(out_dir, "fdelta_fullft.json"), _fd)
        print(f"[full_ft] dense-dW fdelta: {_fd}", flush=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    if residual_method:
        import residual_save as Rs
        n, r0 = Rs.convert_saved_to_w0_relative(out_dir, init_adapter)
        print(f"[residual_save] converted {n} layers to W0-relative rank-{2*r0} adapter "
              f"(was rank-{r0}); eval now uses correct W0 base", flush=True)
        if getattr(args, "cordapp", 0):
            # MANDATORY for CorDA++: residual_save only doubled the GLOBAL r/alpha; the per-layer
            # rank_pattern/alpha_pattern in adapter_config.json still hold the ORIGINAL r^l. Double them
            # so a reload rebuilds each layer at 2r^l (scaling preserved). Without this the reload rebuilds
            # each layer at the wrong per-layer rank and eval explodes. MUST run AFTER the conversion.
            import cordapp_init as Cpp
            nrp = Cpp.finalize_dynamic_rank_config(out_dir)
            print(f"[cordapp] finalized {nrp} per-layer rank_pattern/alpha_pattern entries "
                  f"(doubled to rank-2r^l to match the residual stacking)", flush=True)

    cfg = {
        "run_name": run_name, "method": args.method, "task": "commonsense_170k",
        "base_model": args.base_model, "args": vars(args), "adapter": adapter_desc,
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "grad_accum": grad_accum, "effective_batch": args.micro_batch_size * grad_accum,
        "train_runtime_s": round(dt, 1), "final_train_loss": tr_out.training_loss,
        "peak_mem_init_gb": peak_init_gb, "peak_mem_train_gb": peak_train_gb,
        "norm_trace": _ntc.trace,
        "git_commit": run_lib.git_commit(), "finished_at": run_lib.now_iso(),
    }
    run_lib.write_json(os.path.join(out_dir, "run_config.json"), cfg)
    run_lib.append_registry("train_registry.jsonl", cfg)
    print(f"[run {run_name}] done in {dt:.0f}s | loss {tr_out.training_loss:.4f} | "
          f"trainable {trainable:,} | saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
