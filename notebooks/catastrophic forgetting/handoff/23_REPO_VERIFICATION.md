# 23 — Reference-Repo Verification (P3)

Date: 2026-07-06. Verifier: reference-implementation verification agent.
Repos cloned (shallow) into `repro/`. No project .py files were edited; all fixes below are **patch text only**, to be applied at pool restart.

## 0. Clone inventory

| Repo | URL | Commit | License | Notes |
|---|---|---|---|---|
| LoRA-Null | github.com/HungerPWAY/LoRA-Null | `1e6808abb81fe10e50b8172c40ac9a8ab4f11e83` (2026-01-17) | **NONE** (no LICENSE file) | AAAI 2026. "Modified from iboing/CorDA". All-rights-reserved by default: do NOT ship derived code; our clean-room port is fine, keep clone out of release artifacts. |
| MiLoRA | github.com/sufenlp/MiLoRA | `c3c94693b26c800a96dba84a1fe92d7384b7c28d` (2025-05-31) | **NONE** at repo level; `train.py` carries Apache-2.0 header (inherited from PiSSA) | NAACL 2025. Math code forked from PiSSA; CS code = LLM-Adapters with only init changed. |
| SC-LoRA | github.com/CoffeePot1206/SC-LoRA | `b524394adf3be64340151fb6b7bb33fe7d25737c` (2026-04-20) | Apache-2.0 | arXiv:2505.23724. NOTE: mission URL guess `luomr22/SC-LoRA` 404s; the real repo is CoffeePot1206/SC-LoRA (already cited in our sclora_init.py header — the June audit did see the real repo). |
| CorDA | github.com/iboing/CorDA | `baffb03ac090f23305e5fb586a2d3c16df7f12db` (2025-01-13) | Apache-2.0 | NeurIPS 2024. **Contains NO CorDA++ code** (no rank allocation, no π/importance score anywhere in repo). |

---

## 1. LoRA-Null — GATES POOL RESTART

### 1.1 Their calibration, exactly
Files: `adapterlib/datautils.py::get_calib_data` ("nqopen" branch, lines 144-146 + 230-240), `adapterlib/act_aware_utils.py::calib_cov_distribution` (lines 100-166), `build_adapter.py`, `step1.sh`.

- Dataset/split/fields: `load_dataset("nq_open", split="train")` — **TRAIN split** (~87.9k rows), **questions only** (no answers): `tot_text = "\n\n".join(traindata["question"])`.
- Sample construction: **NOT individual questions.** All questions are concatenated into one text blob; then `nsamples=256` windows are drawn at random char offsets (`random.seed(seed)`, default seed **233**), each window is `seqlen*10` chars, tokenized, and truncated to the first **`seqlen=2048` tokens** (`get_calib_data` default; `build_adapter.py` does not override it).
- Batch/padding: **bs=1**, every sample exactly 2048 tokens, **zero padding tokens ever enter the covariance**.
- Hook (`calib_cov_distribution`): forward hook on **every `nn.Linear`**; per forward: `input = input[0].squeeze(0)` (2048, d); `input = input / torch.max(input).abs()` (abs-of-max, per-sample); `C += (inputᵀ @ input) / 256` — per-token outer products, `/256` **hardcoded** regardless of calib size. fp16 activations (model loaded fp16), cov accumulated in the activation dtype, `.float()` at decompose.
- **Total token count: 256 × 2048 = 524,288 tokens per layer.** d_in = 4096 (q/k/v/o/gate/up) and 11008 (down). 524k ≫ 11008 ⇒ **C is generically FULL-RANK at every layer.**

### 1.2 ANSWER TO THE KEY QUESTION — is the repo degenerate too?
**NO.** The repo's covariance is full-rank; its "null space" = the r **least-activated** (smallest-singular-value) directions of a well-sampled second moment. The repo even prints a rank diagnostic per layer: `print((S_ > 0.1).sum())` in `decomposition.py::decompose_to_adapter2` (singular_aware branch).

Our port (256 individual NQ **validation** questions, ~10 tokens each, ≈2.5–3k total tokens, batched 4 with padding) gives rank(C) ≤ ~3k:
- at d_in=4096 layers: an exact zero-eigenvalue subspace of dim ≥ ~1000; at down_proj (11008): dim ≥ ~8000;
- `eigh` returns an **arbitrary orthobasis** of that exact null space, so our V_null is essentially arbitrary directions of the unsampled subspace — not the repo's least-activated directions;
- padding rows additionally contaminate C, and the max-norm is per-batch-of-4 instead of per-sample.

**VERDICT: our LoRA-Null port is NOT repo-faithful on calibration. Doctrine = MATCH REPO** (spec + patch in §1.5). The published LoRA-Null numbers were produced with a full-rank C; running ours as-is tests a different (random-null-space) method.

### 1.3 Their null-space extraction, exactly
File: `decomposition.py::decompose_to_adapter2`, `singular_aware=True` branch (their LoRA-Null flag; `--singular_aware` in step1.sh):
- `U_, S_, V_ = torch.linalg.svd(C)` — **SVD of C directly** (no symmetrization, no damping in this branch), then `U_min_K = U_[:, -r:]` — **fixed r smallest** (no threshold). Default r in released scripts: **128**.
- `temp = W @ (U_min_K @ U_min_Kᵀ)`; `weight_residual = W − temp`; `U,S,V = torch.svd(temp)`, keep top-r; adapter built with `sigma_fuse='UV'`: **B = U·√S, A = √S·Vᵀ** (CorDA_adapter: ALinear=B out-side, BLinear=A in-side).
- Scaling: custom adapter module applies **no α/r scaling at all** (forward = ALinear(BLinear(x)) + W_res·x) ⇒ effective scaling **s=1 ≡ α=r**. (`--lora_alpha 128` in step2.sh is parsed but unused in Null mode.)
- Freeze: two released variants. **v1** (`train_model.py`, Null_mode): trains **both** ALinear and BLinear (all else frozen). **v2** (`train_model_freeze_a.py`): freezes everything not matching `"ALinear"` ⇒ **only B trains, A (the null-space map) frozen**. (Their name "freeze_a" refers to freezing the input-side matrix = LoRA-A.) Both match our `--lora_null_freeze_a` toggle semantics.
- Layer coverage: repo decomposes **every Linear except lm_head** (incl. o_proj/gate_proj). Ours: PEFT target modules only — accepted head-to-head design difference, keep.
- Their MetaMathQA training config (context): train[:100000], 1 epoch, lr 2e-5, cosine, warmup 0.03, wd 0, eff. batch 128 (1×128), dropout 0, **model_max_length 512** (default; scripts don't override).

### 1.4 lora_null_init.py line-by-line vs repo — deviation list
| Item | Repo | Our port | Verdict |
|---|---|---|---|
| Calib text | nq_open TRAIN, concatenated questions, 256 × 2048-token windows (~524k tok) | nq_open VALIDATION, 256 individual questions (~2.5k tok) | **FIX (blocking)** |
| Batch/padding | bs=1, no pads | bs=4, padded, pads enter C | **FIX** (bs=1) |
| max-norm | per-sample `x/max(x).abs()` | per-batch-of-4 | **FIX** (bs=1 fixes) |
| abs-of-max semantics | `torch.max(x).abs()` | `x.max().abs()` | ✓ identical |
| /256 hardcoded | yes | yes (`/256.0`) | ✓ |
| Null basis | svd(C), last r cols of U | eigh(sym(C)), first nd cols | ✓ equivalent (C symmetric PSD) |
| null_dim | = r, fixed (no threshold) | default nd=r | ✓ |
| Factorization | top-r svd of W·P_null, B=U√S, A=√S·Vᵀ, W_res=W−WP | same | ✓ |
| Scaling | s=1 (α≡r) | inject with B/sc, α=r ⇒ s=1 | ✓ |
| Freeze variants | v1 both / v2 freeze-A | default both / `--lora_null_freeze_a` | ✓ |
| rank(C) diagnostic | prints `(S_>0.1).sum()` | none | add (in patch) |

### 1.5 FINAL CALIBRATION DOCTRINE + PATCH (apply at restart)
**Doctrine — "match repo":** nq_open **train** split, questions concatenated with `"\n\n"`, `random.seed(233)`, 256 random windows of `seqlen*10` chars each tokenized and truncated to **2048 tokens**, forwarded **bs=1** with no padding; per-sample abs-of-max normalization; `C += XᵀX/256`; null basis = r smallest singular directions of C; print a rank(C) diagnostic.

**Patch (text) — replace the calib-prompt construction inside the `if getattr(args, "lora_null", 0):` block of train_cs.py (currently lines 262-284):**

```python
    if getattr(args, "lora_null", 0):
        import lora_null_init as Ni
        from datasets import load_dataset as _ld
        import random as _rnd
        # REPO-FAITHFUL calib (HungerPWAY/LoRA-Null get_calib_data "nqopen"):
        # concat ALL nq_open TRAIN questions, sample 256 windows of 2048 TOKENS each
        # (~524k tokens => rank(C)=d_in). 256 short questions leaves C rank-deficient
        # and the "null space" arbitrary -- verified against repo commit 1e6808a.
        _rnd.seed(233)  # repo default calib seed
        _seqlen = 2048  # repo get_calib_data default
        try:
            nq = _ld("google-research-datasets/nq_open", split="train")
            tot_text = "\n\n".join(nq["question"])
        except Exception as e:
            print(f"[lora_null] nq_open load failed ({e}); falling back to wikitext calib", flush=True)
            wt = _ld("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            tot_text = "\n\n".join(wt["text"])
        kprompts = []
        for _ in range(args.lora_null_calib_size):
            i = _rnd.randint(0, len(tot_text) - _seqlen - 1)
            kprompts.append(tot_text[i:i + _seqlen * 10])  # char window; tokenizer truncates to 2048 tokens
        cov = Ni.collect_lora_null_cov(model, kprompts, tokenizer,
                                       calib_size=args.lora_null_calib_size,
                                       max_len=_seqlen, bs=1)  # bs=1: per-sample max-norm, zero padding
        # rank(C) diagnostic (repo prints (S_>0.1).sum() per layer)
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
        print(f"[lora_null] null-space init applied to {len(cov)} layers "
              f"(null_dim={nd or args.lora_r}, freeze_a={bool(args.lora_null_freeze_a)}); "
              f"loss-preserving err={err:.2e}", flush=True)
```

No change needed in `lora_null_init.py` itself (its hook math is repo-correct once fed 2048-token samples at bs=1). Paper footnote if any pre-restart LoRA-Null numbers are kept: "produced with a rank-deficient calibration covariance (256 short questions); the null space is then an arbitrary basis of the unsampled input subspace, unlike the reference implementation whose covariance is full-rank (256×2048-token windows)."

---

## 2. MiLoRA — α convention CONFIRMED

Files: `svd_init.py`, `scripts/run_svd_init.sh`, `scripts/run_train.sh`, `train.py`, README.
- Init (`initialize_lora_layer`): `lora_alpha = rank` hardcoded (line 24); mode "min" takes `U[:,-r:], S[-r:], V[-r:,:]`; `scaling = lora_alpha/rank = 1`; `S /= scaling` (no-op); **B = U√S, A = √S·V**; `base.weight -= scaling·B@A`. Both saved-adapter configs set **alpha = r** (LLM-Adapters type: α=r, dropout 0.05, targets q,k,v,up,down; QLoRA type: α=r, dropout 0.1, 7 targets).
- Math training (`run_train.sh`, LLM-Adapters setting): rank **64**, `LORA_ALPHA=2*RANK` for vanilla LoRA **but overridden to `LORA_ALPHA=RANK` when method is pissa or milora** — explicit in the script. LR **3e-4**, total batch **16** (2/dev ×8 GPU), **3 epochs**, MetaMathQA full (`train[:]`), `model_max_length 2048`, linear scheduler, warmup_steps 100, wd 0, dropout 0.05, deepspeed stage2.
- **Conclusion: MiLoRA runs at s=1 (α=r) in the official code — at init AND during training. CLoRA's published MiLoRA 63.53 can only correspond to s=1.** Consortium α=2r ruling stands for vanilla LoRA; MiLoRA must stay α=r.
- Our `milora_init.py` vs repo: factorization identical (minor split, √S both sides, residual subtraction). No deviations found.

## 3. SC-LoRA — port verified faithful; D- composition deviates

Files: `scloralib/act_aware_utils_output.py`, `scloralib/decomposition_two.py`, `build_sclora.py`, `tools/run_sclora_MATH_worldknowledge.sh`, `cordalib/datautils.py`.
- Covariance: **output-side forward hooks on every nn.Linear**, bs=1, per-sample `output/torch.max(output).abs()` (abs-of-max), **per-token outer products `outputᵀ@output`** (NOT sum-then-outer), single accumulator with sign+β folded in: `+ (1-β)·cov/c_size` for D+, `− β·cov/b_size` for D-. **All confirmed == our sclora_init.py.**
- Decompose: `eigh(C)` (symmetric assumed, no explicit symmetrization — ours symmetrizes, equivalent), top-r eigvecs Q; **B=Q, A=Qᵀ·W, W_res=W−BA**; custom adapter with no α scaling ⇒ **s=1**. == ours.
- D+ = **prompt WITH answer**: "MetaMATH" branch of `get_calib_data` formats `llama_chat_format.format(instruction=query, response=response)`, truncates to 2048 tokens. Our D+ uses `run_lib.train_prompt(dp)` (instruction+output of OUR task) — with-answer ✓, correct adaptation of "D+ = fine-tuning task".
- World-knowledge experiment config: r=128, **β=0.8**, lr 2e-5, calib 256 / backg 256, seed 233, D-="nqopen".
- **Deviation found:** repo's D- ("nqopen") uses the same concatenated-2048-token-window loader as LoRA-Null (§1.1) → 524k tokens; our D- = 256 individual validation questions (~2.5k tokens), max_len=cutoff. Less catastrophic than LoRA-Null (D- only subtracts inside M; no null-space extraction from it) but the "knowledge" term is under-sampled by ~200×. **Recommended: apply the same windowed-nqopen construction to the sclora `dminus` block of train_cs.py (lines ~242-260) at restart** — same recipe as §1.5 (`split="train"`, seed 233, 256 windows, bs=1 forwarding is already the case in collect_sclora_M).

## 4. CorDA — confirmed with two minor deviations; NO CorDA++ code

Files: `cordalib/act_aware_utils.py`, `cordalib/decomposition.py`, `build_corda.py`, `tools/build_KPA.sh`, `tools/train_CorDA.sh`.
- Calib: **single pass, no rounds** — 256 samples × 2048 tokens (identical nqopen windowed loader, §1.1), bs=1, `C += XᵀX/256` hardcoded, per-sample abs-of-max input normalization. KPA calib = nqopen (or triviaqa); IPA = MetaMATH + `--first_eigen`.
- Decompose (cov_aware): damp=0.01, `compensate = mean(diag(C))·damp·I`, **doubling** until `inv_error < 0.05` — confirmed. **Deviation (minor): repo's inv_error is `torch.dist(C_fix@C_inv, I)` = FROBENIUS norm; our corda_init.py uses spectral norm (`matrix_norm ord=2`).** Frobenius ≥ spectral ⇒ ours accepts smaller damp. Textual fix (corda_init.py line 51): replace `torch.linalg.matrix_norm(C_fix @ C_inv - I, ord=2) < 0.05` with `torch.dist(C_fix @ C_inv, I) < 0.05`.
- `Wc = W@C_fix`; svd; `V=(Vᵀ@C_inv)ᵀ`; KPA takes LAST r; B=U√S, A=√S·Vᵀ; W_res = W − U·diag(S)·Vᵀ; s=1. == ours.
- Layers: hooks on all Linear; decompose **all except lm_head**; ours = LoRA targets only (design choice, keep).
- **Calibration deviation (same as §1.1): our corda block feeds 256 individual validation questions and `collect_corda_cov` defaults to max_len=256** ⇒ rank-deficient C here too. For CorDA the damped inverse hides the exact-zero eigenvalues, but the KPA "context" directions are still under-sampled. If CorDA is re-queued, apply the §1.5 windowed loader to the corda block (train_cs.py lines ~211-227) and pass `max_len=2048`.
- **CorDA++ items: NOT RESOLVABLE from code.** The iboing/CorDA repo contains no CorDA++ implementation (no π operand, no rank allocation). Both open interpretation items (π = σ of C vs of WC; rank-allocation argmin direction) remain paper-interpretation-only; our cordapp_init.py stands as paper-faithful with flags.

## 5. Findings that touch earlier consortium verdicts

1. **LoRA-Null pre-restart results are not LoRA-Null.** They test "random basis of the unsampled input subspace" (degenerate C), not the repo's least-activated-direction method. Restart with §1.5 patch before drawing any LoRA-Null conclusion.
2. **CorDA/SC-LoRA "clean negatives" carry an asterisk.** Both were run with the same under-sampled short-question calibration (CorDA also max_len=256). The math of both ports is verified exact, but the calibration inputs differ from the reference by ~200× token count. If those negatives are load-bearing in the paper, one repo-faithful confirmation run each is cheap insurance; otherwise footnote it.
3. **MiLoRA α=r is now code-confirmed** (init and training), closing the s=1-vs-s=2 question for the 63.53 comparison; the matrix-campaign MiLoRA cells (α=r) are the correct convention.
4. **Licensing:** CorDA and SC-LoRA are Apache-2.0 (derived code shippable with attribution). LoRA-Null and MiLoRA have NO license (MiLoRA's train.py inherits PiSSA's Apache header only) — do not redistribute their code; our reimplementations are unaffected but the `repro/` clones must stay out of any released artifact.
5. **Repo default seeds/params worth recording:** calib seed 233 (LoRA-Null/CorDA/SC-LoRA), all use `/256` hardcoded + abs-of-max per-sample normalization + bs=1, all decompose every Linear except lm_head, all effective scaling s=1 (α≡r). LoRA-Null/CorDA/SC-LoRA released headline rank is 128.
