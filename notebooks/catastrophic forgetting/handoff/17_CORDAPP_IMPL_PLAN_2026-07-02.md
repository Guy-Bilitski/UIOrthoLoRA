# 17 — CorDA++ IMPLEMENTATION PLAN + FIX VERIFICATION (2026-07-02)

Plan + verify only. NO code changes made. Supersedes the "how" in `handoff/14_CORDA_PP_PLAN.md` §2/§3
where they conflict; §14 §8 (the algorithms + the three FIXes) is still authoritative on the *math* and
*policy*. This file adds the code-grounded verification (FIX 3), the exact PEFT API spec, and the crisp
per-arm PAUSE/KEEP verdict for the live `camp5` campaign.

Environment verified this session:
- Importable PEFT = **vendored** `/home/guy/UIOrthoLoRA/src/peft` (v **0.19.1**). The `.venv` resolves
  `import peft` to this tree (`python -c "import peft; print(peft.__file__)"` → `src/peft`). All file:line
  citations below are against `src/peft/...`.
- Live campaign: `gpu_pool.py --gpus 8 --tag camp5 --jobs jobs/combined_nocorda.txt` (PID 167660), currently
  training Qwen-CS **CLoRA** across the 7-LR grid. Job file `jobs/combined_nocorda.txt` (mtime Jul 2 16:14)
  contains arms: clora, lora_null, milora, sclora, dora, lorawd, lora — **CorDA is excluded** ("nocorda").

---

## 1. FIX-3 VERIFICATION (DONE — the gate for Path C feasibility)

### 1(a) Does `preprocess_corda` slice bottom-r using per-layer `rank_pattern`, or only global `LoraConfig.r`?
**VERDICT: it uses per-layer `rank_pattern`. Dynamic rank is feasible on PEFT as-is.** Full chain verified:

- `preprocess_corda` sets a per-module `.rank` from `rank_pattern`, falling back to global `r`:
  `src/peft/tuners/lora/corda.py:117-119`
  ```python
  for name, module in target_modules(model, lora_config):
      r_key = get_pattern_key(lora_config.rank_pattern.keys(), name)
      module.rank = lora_config.rank_pattern.get(r_key, lora_config.r)
  ```
- `crop_corda_eigens` slices the **bottom-`module.rank`** for KPM (top-rank for IPM) using that per-module
  rank, NOT `config.r`: `corda.py:333-336`
  ```python
  elif module.corda_method == "kpm":
      module.eigens.S_WC = module.eigens.S_WC[-module.rank:].clone()
      module.eigens.U_WC = module.eigens.U_WC[:, -module.rank:].clone()...
      module.eigens.V_WC = module.eigens.V_WC[:, -module.rank:].clone()...
  ```
- The LoRA layer's per-adapter rank is also `rank_pattern`-resolved before the module is built:
  `src/peft/tuners/lora/model.py:205-207` (`get_pattern_key(...)` → `rank_pattern.get(r_key, config.r)`)
  → `layer.py:187` (`self.r[adapter_name] = r`). `LoraModel._init_lora_weights`/`update_layer` are called
  with the same resolved `r`, so `corda_init` (see below) reads the matching per-layer rank.
- `corda_init` consumes `r = self.r[adapter_name]` (the per-layer value) and asserts `U/S/V` shapes equal
  that r: `layer.py:423,440-457`. So the eigens (cropped to `module.rank`) and the layer rank agree per layer.

**Conclusion:** feed `rank_pattern={module_name: r_l}` into `LoraConfig`, and both the covariance-SVD crop
(corda.py) and the adapter injection (layer.py) use `r_l` consistently. No fork needed for dynamic rank.

⚠️ One coupling to respect: `get_pattern_key` matches on suffix patterns. Because our target modules repeat
across 32 layers with identical leaf names (e.g. `...layers.7.self_attn.q_proj`), the `rank_pattern` keys
must be the **fully-qualified module names** (which `target_modules(model, config)` yields at corda.py:45),
not the bare `q_proj`. Use the exact names emitted by `named_modules()` as keys → each layer gets its own r.

### 1(b) Is `covariance_file` load-if-exists (can we inject per-layer covariances) or write-only?
**VERDICT: load-if-exists. We CAN inject per-layer-selected covariances. Path C dynamic-covariance is
feasible without forking PEFT.** `calib_cov_distribution` short-circuits and loads when the file exists:
`corda.py:162-166`
```python
if covariance_file is not None and os.path.exists(covariance_file) and os.path.getsize(covariance_file) > 0:
    all_covariance_matrix = torch.load(covariance_file, map_location=get_model_device(model))
    for name, module in target_modules(model, config):
        module.covariance_matrix = all_covariance_matrix[name]
    return                                  # <-- early return: run_model is NOT called
```
Format = a single `dict[str, Tensor]` keyed by the **fully-qualified module name** (matches
`target_modules` yield). It is only WRITTEN when it does not already exist (corda.py:228-233), so a
pre-written file is authoritative. `run_model` may then be `None` (corda.py:81 docstring confirms).

**PATH-C FEASIBILITY = YES on PEFT 0.19.1 as-is.** Both hooks needed by §14 (per-layer `rank_pattern` +
injectable `covariance_file`) exist and behave as the plan assumed. The custom work is entirely OUTSIDE
PEFT: a covariance-collection pass, the compactness/selection/allocation math, and writing the two files.

---

## 2. Path A — STATIC CorDA via PEFT (foundation + ablation anchor)

Exact spec for a new dispatch branch in `train_cs.py` (NOT to be written now). PEFT's static CorDA replaces
our custom `corda_init.py` path. Build order (differs from our current build-then-inject flow):

```python
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import CordaConfig
from peft.tuners.lora.corda import preprocess_corda

targets = args.target_modules.split(",")            # SHARED across all arms
corda_cfg = CordaConfig(
    corda_method="kpm",                             # retention mode (config.py:296; note DEFAULT is "ipm")
    covariance_file=<cov_path or None>,             # None for static; a path for Path C injection
    cache_file=None,                                # optional SVD cache (config.py:275)
    use_float16_for_covariance=False,               # fp32 covariance for the reconstruction identity check
    prune_temporary_fields=True,
    verbose=False,
)
lcfg = LoraConfig(r=16, lora_alpha=16, target_modules=targets,   # alpha==r → scaling==1 (matches our arms)
                  init_lora_weights="corda", corda_config=corda_cfg,
                  bias="none", task_type="CAUSAL_LM")

def run_model():                                    # forward calib prompts through the RAW base model
    for i in range(0, len(cprompts), bs):
        enc = tok(cprompts[i:i+bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=args.cutoff_len).to(model.device)
        model(**enc)

preprocess_corda(model, lcfg, run_model=run_model)  # BEFORE get_peft_model — sets module.eigens
model = get_peft_model(model, lcfg)                 # corda_init runs here (layer.py:395) → mutates base
```

Key correctness facts verified from source:
- `preprocess_corda` runs the covariance hook on the **raw** model, builds `eigens` (S_WC/U_WC/V_WC), and
  crops to per-layer rank (corda.py:56-151). It must run BEFORE `get_peft_model`.
- The residual is applied by `corda_init` at wrap time: `layer.py:459-474`. It divides S by scaling
  (`S /= self.scaling[adapter]`, layer.py:460 → since alpha==r, scaling==1, no-op) and subtracts
  `scaling * lora_B @ lora_A` from the base weight (layer.py:470-474). So **PEFT mutates the base weight
  natively IN MEMORY** — no in-memory injection code needed (drops the `apply_corda` call, not the residual
  concern; see FIX 2).
- KPM tail direction is guaranteed by the `[-module.rank:]` crop (corda.py:333-336) — adapter is built from
  the SMALLEST singular directions of `W·C_fix`; the large (knowledge) directions stay in the frozen
  residual. This matches our `corda_init.py` KPA semantics (`corda_init.py:57` `U[:, -r:]`).
- The CorDA covariance is uncentered 2nd moment with per-batch `input/torch.max(input).abs()` scaling
  (corda.py:178-193) — identical normalization to our `collect_corda_cov` (`corda_init.py:24-26`,
  `x / x.max().abs()`), so the PEFT static arm is numerically comparable to our archived custom CorDA.

Dispatch note: `train_cs.py` currently hard-codes `--method choices=["lora","uiortholora","clora"]`
(train_cs.py:121) and reaches CorDA via `getattr(args,"corda",0)` after `build_adapter`. Path A needs the
covariance/preprocess/get_peft_model to happen in ONE place BEFORE the generic `build_adapter`+inject flow.
Add a dedicated `--corda_engine {custom,peft}` branch in `main()` that, when `peft`, runs the block above
and SKIPS both `build_adapter`'s lora path and the `apply_corda`/`residual_method` custom path. (Spec only.)

---

## 3. Path C — CorDA++ (dynamic covariance + dynamic rank), concrete pseudocode

All custom work lives in a new module (spec name `cordapp_precompute.py`) that produces two artifacts, then
Path A consumes them. **No PEFT edits.** Notation from §14 §8 (arXiv:2506.13187 §IV).

### 3.0 Candidate covariance collection (custom pass, mirrors corda.py hook math)
For N sampling rounds, forward one calib batch per round through the raw base model with a forward-hook on
every target Linear. Per layer `l`, per round `i`, accumulate the CorDA-normalized covariance:
```
Ci_l = 0
for x in batch_i (input activations of layer l):        # x: (tokens, in_dim)
    x = x / x.max().abs()                                # corda.py:178 normalization
    Ci_l += x.T @ x                                      # uncentered 2nd moment (corda.py:185)
Ci_l /= sample_count_l                                   # corda.py:224-225 (divide by count)
```
Store `{l: [C1_l, ..., CN_l]}` in fp32. This is the candidate pool D = {I_1..I_N} (confirmed by web snippet:
one covariance per sampling round per layer). **N is a required-fetch blocker — see §8.**

### 3.1 Compactness metric π(C)  (§14 §8; paper §IV-A)
For each candidate C (in_dim × in_dim, SPD), compute the covariance-oriented SVD of `W·C_fix` exactly as
corda.py does (damped inverse loop, corda.py:274-293), yielding singular values σ_1 ≥ ... ≥ σ_R of `W@C_fix`.
```
pi(C) = sqrt(d_out * sigma_max(C)) / sigma_min(C)          # lower = more compact
```
IMPORTANT ambiguity to resolve at fetch time: whether σ_max/σ_min are singular values of the covariance C
itself, or of the covariance-oriented product `W·C_fix` (the SVD CorDA actually takes). §14 writes π in
terms of σ(C); the selection/allocation scores below use σ_{-r} of the *decomposition*. Implement π from the
**same SVD used for selection** for internal consistency, and flag the exact operand as a fetch item.

### 3.2 Dynamic covariance selection (Eq 7-8) — per layer, independent argmin
```
for each layer l:
    for each candidate Ci_l in pool[l]:
        S = singular_values(W_l @ fix(Ci_l))              # descending σ_1..σ_R
        score_i = log(pi(Ci_l)) * sum_{r=1..R}( S[-r] / S[0] )    # Eq 8; σ_-r = r-th SMALLEST
    C*_l = Ci_l with the MINIMUM score_i                  # argmin, per-layer, independent
covariance_file[l] = C*_l
```
Write `covariance_file` = `{fully_qualified_module_name: C*_l}` (fp32). This is exactly the dict format
`calib_cov_distribution` loads (corda.py:163-165). PEFT then does its own SVD on the injected C*_l.
`sum_{r=1..R}(σ_-r/σ_max)` uses the FULL spectrum (R = min(in,out)); it favors covariances whose energy is
concentrated (small tail) AND compact (small π).

### 3.3 Dynamic rank allocation (Eq 9-10) — greedy fill to param budget τ (KPM)
Runs AFTER selection, on the chosen C*_l per layer (their SVD spectra):
```
r[l] = 1 for all target layers l                         # start rank 1 everywhere
tau_realized = sum_l (in_l + out_l) * r[l]
while True:
    for each layer l:
        # KPM score: uses the r[l]-th smallest σ (the next tail component that would become an adapter dir)
        s[l] = log(pi(C*_l)) * S_l[-r[l]] / sum_{k=1..(R_l - r[l])} S_l[k]      # Eq 10 (KPM)
    l* = argmin_l s[l]                                    # increment the LOWEST-scoring layer
    r[l*] += 1
    tau_next = sum_l (in_l + out_l) * r[l]
    if tau_next > tau:                                    # overshoot by one step
        break                                             # (report the REALIZED count, see §6)
rank_pattern = {fully_qualified_module_name_l: r[l]}
```
- KPM adapter params per layer = (in+out)·r (bottom-r comps become the adapter). Budget τ = our
  r16-equivalent = **28,049,408** params (§14 §8; = avg rank 16 over 160 target matrices), NOT 320M.
- Because we overshoot-then-stop, realized τ′ ≠ nominal 16·(...). **Report τ′ exactly** (§6).
- Feed `rank_pattern` into `LoraConfig`. corda.py:117-119 + crop (333-336) + layer.py:423 all honor it
  (verified §1a). IPM variant (not our headline) would use `(R_l - r[l])` remaining comps and stop when
  τ′ < τ — spec'd in §14 §8 but out of scope for the KPM retention arm.

### 3.4 Mapping onto PEFT (summary)
| CorDA++ component | artifact | consumed by | verified at |
|---|---|---|---|
| dynamic covariance | `covariance_file` dict {name: C*_l} | `calib_cov_distribution` load-branch | corda.py:162-166 |
| dynamic rank | `rank_pattern` dict {name: r_l} in LoraConfig | preprocess crop + corda_init | corda.py:117-119,333-336; layer.py:423 |
| everything else | unchanged Path A | `preprocess_corda`+`get_peft_model` | §2 |

Ablations = flip one artifact to the static default:
- `corda_ppCov` (dyn-cov only): inject `covariance_file`, leave `rank_pattern={}` (uniform r=16).
- `corda_ppRank` (dyn-rank only): `rank_pattern` from allocation, but built on the **mean** covariance
  (single round, no selection) so per-layer C is not cherry-picked.
- `corda_pp` (headline): both. `corda_static` (Path A): neither.

---

## 4. FIX 2 — residual correctness (save→reload round-trip)

CorDA mutates the base weight (`W′ = W − scaling·BA`, layer.py:470-474). `save_pretrained` persists ONLY the
adapter; reloading it onto the ORIGINAL base double-counts / is silently wrong — the exact **residual_save
bug class** we already hit (see `residual_save.py:1-16`, `handoff/13 §6/§10`).

**PEFT has a NATIVE fix, and it does exactly what our `residual_save.py` does** (rank-2r stacking):
- `PeftModel.save_pretrained(..., path_initial_model_for_weight_conversion=<dir>)`
  (`peft_model.py:190-197,220-222,325-331`) triggers `save_mutated_as_lora` (peft_model.py:247-285) →
  `LoraModel.subtract_mutated_init` (model.py:909-945), which computes
  `ΔW = A·B − A_0·B_0 = [A | A_0]·[B | −B_0]ᵀ` (model.py:936-943). That is IDENTICAL to
  `residual_save.convert_saved_to_w0_relative` (residual_save.py:49-52). Result: a rank-2r LoRA adapter that
  is correct against the ORIGINAL W0, so the eval/forensics harness (which reloads W0) is unchanged.

Required round-trip (spec):
1. After `preprocess_corda` + `get_peft_model` (base now mutated, adapter = init CorDA), **save the INITIAL
   adapter to disk BEFORE training** to a dir `initial_dir` (this is the `A_0,B_0` reference).
2. Train.
3. `model.save_pretrained(out_dir, path_initial_model_for_weight_conversion=initial_dir)`. PEFT reloads the
   initial adapter, subtracts, and writes the rank-2r W0-relative adapter + a config with `init_lora_weights
   = True` (peft_model.py:326-328). GOTCHA: the initial adapter's config must be saved with
   `init_lora_weights = True`; PEFT does this via the deepcopy at peft_model.py:326-327, and REJECTS a
   still-"corda" initial adapter (peft_model.py:270-279) — so save the initial dir through this path, not by
   hand.
4. **CONSTRAINT for dynamic rank:** `save_mutated_as_lora` raises if `use_rslora` AND (`rank_pattern` or
   `alpha_pattern`) are both set (peft_model.py:248-253). We do NOT use rslora, so `rank_pattern` +
   conversion is allowed. But VERIFY the round-trip on a dynamic-rank config specifically (each layer's
   rank-2r stack must match its own r_l).

**Init-output-invariance check must run AFTER reload, NOT in-memory.** In-memory the mutated base + init
adapter reconstructs W0 trivially (that is how the residual is defined); the bug only manifests on the
reloaded artifact. So: load the SAVED adapter onto a FRESH base model and assert `forward(base) ≈
forward(reloaded_peft_model)` within tol on a batch of eval prompts. A 0-step training run must convert to
ΔW = 0 (B_init cancels itself) → reloaded model == base (the self-check in residual_save.py:16 and the gate
`validate_residual_zero_step.py`, handoff/13 §6). Reuse that gate against the PEFT-native path.

Alternative if the native path is fragile with dynamic rank: persist the MUTATED base itself (save the whole
residual model) and eval by loading that base + the rank-r adapter. Heavier on disk (~15GB/run) but avoids
the 2r conversion entirely. Prefer the native conversion; keep this as fallback.

---

## 5. FIX 1 — CALIBRATION DISTRIBUTION (HIGHEST PRIORITY) + live-campaign verdict

### The problem, confirmed from source
Our retention eval = `bbh_fewshot, mmlu_pro, mmlu, arc_challenge, truthfulqa_mc2`
(`eval_one_gpu.py:124-126`) — academic/reasoning MC, **zero factoid QA**. Every calibration-using arm
currently calibrates on **nq_open** (factoid QA): CorDA (train_cs.py:218-219), SC-LoRA D− (train_cs.py:241-
242), LoRA-Null (train_cs.py:260-261), with a **wikitext-2 fallback** if nq_open fails to load
(train_cs.py:221-223, 244-246, 263-265). KPM/CorDA-style methods only protect directions the covariance
exercises (paper §III-C: eval-distribution covariance works best). nq_open → protects the WRONG subspace for
our eval → handicaps exactly the data-aware arms → risks a FALSE "data-aware inits don't retain" (off-curve)
conclusion. This is a strawman and MUST be fixed before any off-curve claim.

### The fix (eval-matched calibration set)
- Build a SHARED calibration prompt set from **MMLU/ARC auxiliary_train splits** (`cais/mmlu` auxiliary_train
  and `allenai/ai2_arc` ARC-Challenge/Easy `train`), **256 samples**, formatted as the same question prompts
  the eval uses, **DISJOINT from the test splits** (auxiliary_train/train ≠ test — assert no overlap by
  question-hash). Use it as: CorDA calib, SC-LoRA D− (knowledge to preserve), LoRA-Null calib — one shared
  set so all data-aware arms see identical retention-direction targets.
- SC-LoRA D+ stays the fine-tuning task (commonsense/math) — unchanged (train_cs.py:239).
- Add a **sensitivity arm**: the SAME method (start with CorDA-static or SC-LoRA) calibrated on nq_open vs
  eval-matched, to quantify the calibration-distribution effect (this becomes a paper ablation: "data-aware
  retention is calibration-distribution-dependent").
- Covariance cache key MUST include `calib_hash` so nq_open and eval-matched caches never collide (§14 §4).

### PER-ARM PAUSE / KEEP VERDICT for the live `camp5` campaign (jobs/combined_nocorda.txt)
Rule: an arm is CONFOUNDED iff it builds its init from a calibration/knowledge distribution (→ off-curve
verdict depends on that distribution). Calibration-free arms are SAFE regardless.

| arm | calib source (verified) | data-aware? | VERDICT |
|---|---|---|---|
| **corda** | nq_open (train_cs.py:218) — already purged from camp5 (`nocorda`) | YES | already excluded — re-run under eval-matched calib (Path A/C) |
| **sclora** | D− = nq_open (train_cs.py:241-242) | YES | **PAUSE + RE-RUN** under eval-matched calib. Its off-curve deviation (−3.3pp, handoff/13 §2) is CONFOUNDED. |
| **lora_null** | nq_open (train_cs.py:260-261) | YES | **PAUSE + RE-RUN** under eval-matched calib. Null-space of the WRONG (factoid) activations. |
| **milora** | none — bottom-r SVD of W only (milora_init.py:27, no calib) | NO | **KEEP** — calibration-free, distribution-invariant. |
| **dora** | none (LoRA + magnitude decouple, train_cs.py:94) | NO | **KEEP** — calibration-free. |
| **lorawd** | none (LoRA + weight_decay, train_cs.py:22 arm) | NO | **KEEP** — calibration-free. |
| **lora** | none | NO | **KEEP** — calibration-free (the law's anchor). |
| **clora** | none — FROZEN RANDOM orthonormal P pairs, seed 42, NOT data (train_cs.py:37-60) | NO | **KEEP** — currently training; calibration-free, safe. |

**Bottom line: of the 8 arms, only `sclora` and `lora_null` are confounded by FIX 1 and should be paused +
re-run under eval-matched calibration; `corda` is already out. The other 5 (milora, dora, lorawd, lora,
clora) are calibration-free and safe to keep running.** Since the running job file is `combined_nocorda.txt`,
the operator action is: (a) let the current calibration-free arms finish, (b) do NOT trust any `sclora_*` /
`lora_null_*` rows from this campaign for the off-curve claim, (c) queue eval-matched re-runs of sclora +
lora_null (+ corda_static/corda_pp) as a SEPARATE later pool (single-scheduler rule, §9). NOTE: this is a
plan recommendation — per the hard constraints, do not touch the running pool; the operator decides.

---

## 6. FAIRNESS / PARITY (dynamic rank breaks nominal parity)

- Dynamic rank ⇒ the nominal "r16" label is meaningless for corda_pp. **Report the REALIZED trainable-param
  count** τ′ (from §3.3), computed as `sum_l (in_l+out_l)·r_l` over target modules, AND cross-check against
  `run_lib.count_trainable(model)` after wrap (train_cs.py:205) — they must agree. Log both in
  `run_config.json` (there is already a `trainable_params` field, train_cs.py:320-322).
- Constrain corda_pp to the **r16-equivalent budget τ = 28,049,408** (§14 §8: avg rank 16 over 160 target
  matrices), NOT the paper's 320M (paper used r=128). The greedy allocator (§3.3) stops at the first
  overshoot; report the realized count (will be within one (in+out) step of τ).
- Target-module set: our default `q_proj,k_proj,v_proj,up_proj,down_proj` (train_cs.py:134) = 5 mats × 32
  layers = 160 matrices. Confirm 160 == "160 target matrices" in §14 (it does). SHARE this exact set +
  the 7-LR grid + seed 42 + linear schedule + adamw_torch (train_cs.py:291-297) across all arms.
- The rank-2r save conversion (FIX 2) means the SAVED adapter is rank-2r_l, but the TRAINABLE params during
  training are rank-r_l — report the TRAINING count for parity, note the saved artifact is 2r_l for eval.

---

## 7. VALIDATION CHECKLIST (handoff/14 §6 turned into runnable checks)

Run each BEFORE trusting any CorDA/CorDA++ result. (⚠) = must be post-reload, not in-memory.
1. **PEFT version + signatures recorded** — DONE: 0.19.1 vendored; `CordaConfig(cache_file, covariance_file,
   corda_method∈{ipm,kpm}, verbose, use_float16_for_covariance, prune_temporary_fields)` (config.py:247-319);
   `preprocess_corda(model, lora_config, run_model=None, hooked_model=None)` (corda.py:56-62).
2. **Init-output invariance (⚠ AFTER RELOAD)** — load saved adapter on fresh base; assert forward ≈ base
   within tol on eval prompts; 0-step run ⇒ ΔW=0 ⇒ reloaded == base. Reuse `validate_residual_zero_step.py`.
3. **KPM tail-direction sanity** — adapter built from SMALLEST singular dirs of W·C_fix (crop `[-rank:]`,
   corda.py:333-336); assert the frozen residual retains the LARGE (top) directions (not inverted). Check:
   `‖top-r component of (W − W_res)‖ ≈ 0` and `‖bottom-r component‖ ≈ ‖W − W_res‖`.
4. **Reconstruction identity (fp32)** — with r = full min_dim and 0 comps discarded, `W_res + scaling·B@A`
   reproduces W within fp32 tol (matches our `corda_init.py` self-test, corda_init.py:83-92). Use
   `use_float16_for_covariance=False` so this is meaningful.
5. **Calib/eval disjointness** — assert the eval-matched calib questions (MMLU/ARC aux-train) have ∅ hash
   overlap with the test splits used by `eval_one_gpu.py` (bbh/mmlu/mmlu_pro/arc_challenge/truthfulqa).
6. **Parity** — realized τ′ reported and matched to 28,049,408 ± one step; `count_trainable` == analytic
   `sum_l (in_l+out_l)·r_l`; target set/LR grid/seed/schedule identical across arms.
7. **Covariance-file round-trip** — write `covariance_file`, run `preprocess_corda(run_model=None)`, assert
   the loaded per-layer C equals what was written (corda.py:162-166 load branch), and that per-layer eigen
   shapes == rank_pattern[l].

---

## 8. OPEN ITEM — candidate-pool size N (fetch blocker)

- WebFetch of arXiv:2506.13187 (abs + /html) FAILED this session (firewalled — returned no output). WebSearch
  partially resolved the DEFINITION: N = number of sampling rounds; pool D = {I_1,...,I_N}, one covariance
  `C_i^(l)` per batch per layer (confirms §3.0). **The EXPERIMENTAL value of N (appendix) is NOT resolved.**
- **REQUIRED-FETCH BLOCKER before finalizing `dynamic_covariance`:** obtain, verbatim from the paper, (i) the
  experimental N; (ii) whether π's σ_max/σ_min are of C or of W·C_fix (see §3.1 ambiguity); (iii) the exact
  Eq 7/8/9/10 forms to confirm §14 §8's transcription; (iv) confirm KPM calib = NQ-open, 256 samples in the
  paper (we OVERRIDE this to eval-matched per FIX 1, but need it for faithful reproduction of their number).
- Until N is fetched: default to a small pool (e.g. N=8) ONLY for a pipeline smoke test, clearly flagged as
  non-faithful; do not report corda_pp numbers built on a guessed N.

---

## 9. RUN PLAN (ordering vs the live campaign)

Single-scheduler rule (handoff/13 §7): NEVER launch a 2nd 8-GPU pool while camp5 is live. New arms run AFTER
camp5 drains.

New arms (add to `ARMS` in `make_campaign_jobs.py` — spec only, do not edit now):
- `corda_static` — Path A, PEFT static CorDA, KPM, eval-matched calib, r16. (`--corda_engine peft`
  + eval-matched calib flag.)
- `corda_pp` — Path C, dyn-cov + dyn-rank, τ = 28,049,408. (needs precompute of covariance_file +
  rank_pattern; pass their paths.)
- `corda_ppCov` — dyn-cov only (uniform r16).
- `corda_ppRank` — dyn-rank only (mean covariance).
- `sclora_evalcalib`, `lora_null_evalcalib` — the FIX-1 re-runs of the two confounded arms.
- (optional) `corda_static_nqopen` — the nq_open-vs-eval-matched sensitivity arm.

`make_campaign_jobs.py` already emits `train_cs.py <flags> && eval_one_gpu.py ...` per (arm × LR × seed) and
skips cells with an existing `results/<run>/summary.json` (make_campaign_jobs.py:46,54-55) → resumable.
Precompute step (covariance_file + rank_pattern for corda_pp/ablations) is a ONE-OFF per (model, calib_hash,
N) that must run before the pool; cache and reuse across LRs/seeds (§14 §4 cache key).

Ordering: (1) camp5 drains. (2) run the CorDA++ precompute (single GPU, cheap). (3) launch ONE new pool
(`corda_static`, `corda_pp`, ablations, sclora/lora_null eval-matched re-runs) × 7 LRs × seed 42 via the
orchestrator (`SKIP_VALIDATION=1 bash run_all_experiments.sh` pattern, handoff/13 §3). (4) figures via
`paper_figs_v2.py`; dedup old vs new run_names by latest `evaluated_at` (handoff/13 §2).

---

## 10. SUMMARY OF VERDICTS
- **FIX-3 (Path C feasibility): YES on PEFT 0.19.1 as-is.** rank_pattern is honored per-layer end-to-end
  (corda.py:117-119,333-336; model.py:205-207; layer.py:187,423); covariance_file is load-if-exists
  (corda.py:162-166). No PEFT fork needed; all custom work is external (precompute + math + two files).
- **FIX-2: use PEFT-native `path_initial_model_for_weight_conversion`** (peft_model.py:325-331 →
  model.py:936-943) = same rank-2r stacking as residual_save.py; verify AFTER reload; not blocked by
  rank_pattern (only blocked when combined with rslora, which we don't use).
- **FIX-1 (most urgent): PAUSE + re-run `sclora` and `lora_null` under eval-matched calibration; `corda`
  already excluded; KEEP `milora, dora, lorawd, lora, clora` (all calibration-free).**
- **Top risks:** (1) N unresolved (fetch blocker) — corda_pp numbers not trustworthy until fetched;
  (2) rank_pattern key mismatch — must use fully-qualified module names or all layers collapse to global r;
  (3) residual round-trip under dynamic rank — verify the 2r conversion per-layer post-reload, else the
  known ‖ΔW‖-explosion / ~0-retention bug returns silently.
