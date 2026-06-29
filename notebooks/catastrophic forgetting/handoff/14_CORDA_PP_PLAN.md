# CorDA++ IMPLEMENTATION PLAN — execute AFTER the current 2×2 campaign

Decision (2026-06-29): run **CorDA++** (the advanced, dynamic variant), built as **Path C** on top of
PEFT's static CorDA. Static CorDA = foundation + ablation point; CorDA++ = headline. Run after the
campaign so there's no GPU contention (single-scheduler rule).

## 0. Ground truth (verified §2, 2026-06-29)
- PEFT 0.19.1 (vendored at `src/peft/`) has **static CorDA only**: `CordaConfig(cache_file,
  covariance_file, corda_method∈{ipm,kpm}, verbose, use_float16_for_covariance, prune_temporary_fields)`,
  `preprocess_corda(model, lora_config, run_model=, hooked_model=)`, `init_lora_weights="corda"`. NO
  dynamic-covariance / dynamic-rank / compactness fields. → CorDA++ must be implemented by us.
- We use **KPM** (retention). Calib = **nq_open** (knowledge to retain), 256 samples, DISJOINT from our
  retention eval (BBH/MMLU/MMLU-Pro/ARC/TruthfulQA — nq_open not in it). Shared budget across data-driven inits.

## 1. PREREQUISITE / BLOCKER — fetch the CorDA++ paper (no internet here)
Cannot implement faithfully without the exact algorithms (no-strawman rule — same discipline that caught
the wikitext bug). Hand this to an internet agent; need verbatim from **arXiv:2506.13187**:
> Fetch arXiv:2506.13187 (CorDA++). Return the EXACT definitions of: (a) §IV-A the "compactness metric"
> (the precise formula — energy concentration in the leading singular values of the covariance-oriented
> SVD; give the equation and what's normalized); (b) the **dynamic covariance selection** algorithm — how
> candidate covariance matrices are sampled and the per-layer selection rule; (c) the **dynamic rank
> allocation** algorithm — the exact rule mapping the metric to a per-layer rank under a fixed global
> parameter budget; (d) the KPM calibration dataset + sample count used in their experiments; (e) any
> hyperparameters/defaults (pool size, budget). Quote the equations, don't paraphrase.

## 2. Foundation step — swap to PEFT static CorDA (Path A) in train_cs.py
Replaces our custom `corda_init.py` path (more defensible: maintainers' code). Refactor the CorDA branch:
- Build `LoraConfig(r, lora_alpha, target_modules=<shared>, init_lora_weights="corda",
  corda_config=CordaConfig(corda_method="kpm"))`.
- Define `run_model()` that forwards ~256 nq_open questions through the RAW base model (covariance hook).
- Call `preprocess_corda(model, lora_config, run_model=calib)` **BEFORE** `get_peft_model` (different from
  our current build-then-apply flow — needs a dedicated dispatch path in build_adapter/main).
- Then `get_peft_model(model, lora_config)`. **PEFT manages the residual natively → DROP `residual_save`
  for CorDA** (remove corda from `residual_method`; `save_pretrained` works normally). Cleaner.
- Archive the custom-corda runs like the wikitext ones; this becomes the `corda_static` arm.

## 3. CorDA++ layer (Path C) — feasible on PEFT machinery
PEFT hooks that make this implementable without forking: **`covariance_file`** (feed precomputed/selected
covariances into preprocess_corda) + **`rank_pattern`** in LoraConfig (per-module ranks) + a custom
covariance-collection pass. Components (fill exact math from §1 fetch), each an ablatable flag:
- `compactness` metric per target layer (paper §IV-A).
- `dynamic_covariance=True`: collect a POOL of candidate covariances (multiple nq_open batches); per layer
  pick the covariance maximizing compactness → write to `covariance_file`.
- `dynamic_rank=True`: map metric → per-layer rank under a FIXED GLOBAL trainable-param budget → pass as
  `rank_pattern`. (More rank where context is less compact.)
- Flags isolate each contribution and reproduce the CorDA→CorDA++ delta.

## 4. Fairness / parity (critical — dynamic rank breaks nominal parity)
- **Report EFFECTIVE trainable-param count**, not nominal rank, for CorDA++ (dynamic rank drifts params).
- Constrain CorDA++'s total trainable params to the SAME budget as the r-fixed arms (e.g. match LoRA r16's
  param count) so the retention-vs-capacity axis stays clean.
- Shared calib (256 nq_open) + identical target_modules/LR-grid(7)/seed/schedule/optimizer as all arms.
- Covariance cache key = (base_model, calib_hash, N, dtype); reusable across corda_static/corda_pp only.

## 5. Run plan (after the 2×2)
- New arms via `make_campaign_jobs.py`: `corda_static` (Path A), `corda_pp` (dyn-cov + dyn-rank), plus
  ablations `corda_ppCov` (dyn-cov only) / `corda_ppRank` (dyn-rank only) to show each contribution.
- LR sweep (7 LRs) × seed 42 first, same harness. Launch via `run_all_experiments.sh` (single scheduler)
  once the current 2×2 + lock-the-off-curve seeds are done.
- Covariance fp32 (~43GB/7B) for the reconstruction identity check; fp16 only if VRAM-bound AND §10 passes.

## 6. Validation checklist (§10 of the brief — run before trusting any CorDA/CorDA++ result)
- [ ] PEFT version + CordaConfig signature recorded (done: 0.19.1, see §0).
- [ ] Init-output invariance: pre-train forward of CorDA-init model == base within tol.
- [ ] KPM direction sanity: adapter from TAIL components, principal (knowledge) frozen (not inverted).
- [ ] Reconstruction identity (r→discard 0 comps) reproduces W within fp32 tol.
- [ ] Disjointness: covariance(nq_open train) ∩ retention eval = ∅.
- [ ] Parity: effective trainable params reported; CorDA++ matched to the shared budget.

## 7. Open confirmations (from §1 fetch)
exact compactness formula · candidate-pool size · dynamic-rank budget rule · their KPM calib set/N ·
whether an official CorDA++ code release exists (separate from iboing/CorDA static).
