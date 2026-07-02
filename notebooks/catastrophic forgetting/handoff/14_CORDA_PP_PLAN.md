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
candidate-pool size N (experiment appendix) · whether an official CorDA++ code release exists.

---

## 8. REVISIONS before building (expert review 2026-06-29) — these OVERRIDE §0/§2/§5 where they conflict

### §1 BLOCKER CLEARED — CorDA++ algorithms (arXiv:2506.13187 §IV), implement exactly:
- **Compactness metric:** π(C) = √(d_out · σ_max(C)) / σ_min(C). Lower = more compact. Bound ‖ΔY‖₁ ≤ π(C)·σ₋ᵣ.
- **Dynamic covariance selection (Eq 7–8):** sample N batches → per-layer candidates C_i^(l). Score
  s(C_i^(l)) = log(π(C_i^(l))) · Σ_{r=1}^{R} (σ₋ᵣ/σ_max). Per layer pick **argmin** (lowest wins), independently per layer.
- **Dynamic rank allocation (Eq 9–10):** all layers start r^(l)=1; score s^(l) = log(π(C^(l))) · σ₋_{r^(l)} /
  Σ_{k=1}^{R^(l)−r^(l)} σ_k; each step **increment the lowest-scoring layer**. KPM: filtered (bottom) comps =
  adapters; accumulate τ′ = Σ(d_in+d_out)·r^(l); stop when τ′ > τ (overshoots by one → **report REALIZED
  param count**). IPM: remaining comps = adapters; τ′ = Σ(d_in+d_out)·(R^(l)−r^(l)); stop when τ′ < τ.
- **Paper KPM setup:** NQ-open, 256 samples; LoRA/CorDA **r=128**; CorDA++ matched to equiv budget → all three
  **320M trainable**. Confirms param-budget parity is correct. **OUR budget = r16-equivalent = 28,049,408
  (avg rank 16 over 160 target matrices), NOT 320M** — match CorDA++ to our arms, report realized count.
- STILL OPEN: candidate-pool size **N** (experiment appendix, not method §) — fetch before finalizing dynamic_covariance.

### FIX 1 (HIGH) — calibration distribution must MATCH our retention eval (supersedes the nq_open guidance in §0/§5)
KPM only protects directions the covariance exercises; paper §III-C: covariance from the eval distribution
works best. Our eval = BBH/MMLU/MMLU-Pro/ARC/TruthfulQA (academic/reasoning), NOT factoid QA → **nq_open is a
poor proxy**, handicapping the calibration-using arms (CorDA/CorDA++/SC-LoRA/CLoRA/LoRA-Null) vs the
calibration-free arms (LoRA/MiLoRA/PiSSA) → risks a false "data-aware inits don't retain" conclusion.
→ Switch KPM calibration to **eval-distribution-matched** questions (MMLU/ARC auxiliary-train, DISJOINT from
  test), 256, **shared across all calibration-using arms**. Add a **sensitivity arm** (nq_open vs eval-matched).
→ ⚠️ **IMPLICATION:** this likely confounds the current **off-curve finding for BOTH CorDA AND SC-LoRA** —
  "forget more than budget" may just be "preserved the wrong (nq_open) knowledge for our eval." The off-curve
  claim is now pending the eval-matched calibration for ALL data-aware arms, not just CorDA's nq_open recal.

### FIX 2 (HIGH) — do NOT drop residual handling for PEFT CorDA (corrects §2's "drops residual_save")
CorDA mutates the base (W′ = W − BA). `save_pretrained` → reload onto the ORIGINAL base is silently wrong.
Use PEFT's **`path_initial_model_for_weight_conversion`** at save (or persist the mutated base). VERIFY the
save→reload round-trip; run the init-output-invariance check **AFTER reload, not in-memory** (in-memory passes
even when the reloaded model is corrupt — the residual_save bug class we already hit).

### FIX 3 — verify two PEFT behaviors BEFORE building Path C
(a) `preprocess_corda` must slice bottom-r using the per-layer rank from **`rank_pattern`**, NOT global
`LoraConfig.r` — confirm in `src/peft/tuners/lora/corda.py`, else dynamic_rank gives inconsistent inits.
(b) confirm **`covariance_file` is load-if-exists** (so we can inject per-layer-selected covariances), not write-only.
