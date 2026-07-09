# FIX-1 Fairness & Calibration Study — PLAN (DRAFT for expert review)

**Status:** draft to be reviewed by an expert panel (fairness/calibration, experimental design/stats,
adapter-implementation expert, adversarial peer-reviewer), then finalized.

## 0. Why this study exists
The per-adapter audit (handoff/18) confirmed all 8 ports are mathematically FAITHFUL. The only
publication blocker is a single experimental confound (FIX-1): the three data-aware arms — **CorDA,
SC-LoRA, LoRA-Null** — calibrate on **nq_open** (factoid QA) while retention is measured on
**academic reasoning** (BBH / MMLU-Pro / MMLU / ARC / TruthfulQA). A data-aware init only protects the
subspace its calibration exercises, so nq_open may make these methods preserve the WRONG knowledge →
their "off-curve, forgets-more-than-budget" result (CorDA −3.0pp, SC-LoRA −3.3pp) could be a
**calibration artifact, not a method property**. Publishing the off-curve claim without a fair-calibration
control invites a fatal "you strawmanned SOTA" rejection.

**Goal:** measure these methods under a *genuinely fair* usage+calibration protocol so that whatever we
conclude is unimpeachable. Either outcome publishes:
- eval-matched calibration moves them ONTO the curve → the law is universal + calibration-sensitivity is a finding;
- they stay OFF the curve under fair calibration → a REAL, defensible method limitation.

## 1. Arms in scope
- **Data-aware (the subjects):** CorDA-static (PEFT-native), CorDA++ (dynamic, handoff/17), SC-LoRA, LoRA-Null.
- **Calibration-free backdrop (already clean, reuse existing/running results):** LoRA, LoRA+wd, DoRA, MiLoRA, CLoRA — provide the on-curve reference the subjects are measured against.

## 2. Calibration protocol — THE CRUX (fairness axis #1)
- **Eval-matched calibration set:** pooled from **MMLU `auxiliary_train`** + **ARC (challenge+easy) `train`**
  splits — same academic-reasoning distribution as the retention eval, provably **DISJOINT from every eval
  test split**. 256 samples (matches the CorDA/CorDA++ paper budget), stratified across MMLU subjects + ARC.
- **Formatted as the model sees them at eval** (question + answer-choice framing), so the collected
  covariance reflects the true eval input distribution (CorDA++ paper §III-C).
- **Shared IDENTICALLY across all data-aware arms:** same 256 samples, same order, same tokenizer, same
  `max_len`, same covariance-collection budget. One cache keyed by `(base_model, calib_hash, N, dtype)`.
- **Sensitivity axis (the actual experiment):** each data-aware arm run under TWO calibrations —
  `{nq_open (existing baseline), eval-matched (new)}`. The existing VALID nq_open LR-sweep runs
  (`lrsw_sclora_*`, `lrsw_lora_null_*`, and CorDA once re-added) are reused as the nq_open arm — **no redo**.
- **Disjointness gate:** assert `calib ∩ eval-test == ∅` before any run; log the hashes.
- **Run-name tag:** `_calibEM_` (eval-matched) vs `_calibNQ_` (nq_open) so the registry never mixes them.

## 3. Usage fairness (fairness axis #2) — identical knobs, best-config per method
- **Identical across ALL arms:** target_modules, LR grid (7: 2e-5..1e-3), optimizer, schedule, precision,
  max_len, seed set, eval harness (already audited fair — no per-method branches).
- **Each method at its BEST/recommended config (no strawman):** SC-LoRA β=0.5 (+ report the |max(Y)| vs
  max(|Y|) normalization sensitivity we flagged); LoRA-Null default AND its `freeze_a` best-preservation
  ablation; CorDA KPA; CorDA++ dynamic-rank at matched budget.
- **The wd-knob symmetry question (OPEN — needs panel ruling):** the paper is framed around THE LAW to
  sidestep method-ranking fairness (only LoRA got the wd knob). Proposal: keep the law as headline, but add
  a **"wd-for-everyone" mini-ablation** (apply weight_decay=0.3 to the best data-aware arm) so we can
  honestly state we tested giving every method the magnitude knob. Panel: is this sufficient, or must every
  arm get a full wd sweep?

## 4. Parameter parity (fairness axis #3)
- **Report REALIZED trainable-param counts** for every arm (CorDA++ dynamic rank drifts params; DoRA adds a
  magnitude vector; r16 vs r32 differ).
- **Rank-matched head-to-head:** include one rank where ALL arms share r (r16) for a clean fixed-rank
  comparison, in addition to each method's native rank.
- **Param-matched controls:** constrain CorDA++ to the r16-equivalent budget (28,049,408 params); add the
  param-matched **LoRA+wd control (B5)**.

## 5. Statistics
- **Seeds 42/43/44** for all data-aware arms + the key controls, to put error bars on the off-curve claim
  (the ANCOVA "forgets more than budget" test needs replication to survive review).

## 6. Models × domains — depth-first
- **Primary testbed: Llama-2 commonsense** (mature nq_open baseline already exists → cheapest path to the
  fair-vs-unfair contrast). Answer the fairness question DEFINITIVELY here first.
- **Then replicate the headline** on **Qwen-CS**; extend to **math** (both models) as breadth.
- Rationale: don't spend 4× compute across the full 2×2 before the core fairness result is established on one cell.

## 7. Implementation prerequisites (from handoff/17 + handoff/18)
- Wire an eval-matched calibration loader + a `--calib_source {nq_open,eval_matched}` flag into train_cs.py
  for corda / sclora / lora_null (and CorDA++).
- CorDA-static via PEFT `preprocess_corda`; CorDA++ Path C (dynamic covariance + `rank_pattern`) — **resolve
  the open candidate-pool size N** (arXiv:2506.13187 appendix; WebFetch was firewalled — fetch before finalizing CorDA++).
- **residual_save bf16 caveat:** store the init-cancellation adapter in **fp32** for the new runs (removes the
  ~3e-3 init error the audit flagged) — cheap, cleaner. (Decision: switch vs keep+disclose.)
- **Registry hygiene:** new pool tag; single post-fix eval pass; figures dedup by latest `evaluated_at`;
  `_calibEM_/_calibNQ_` in run-name.

## 8. Ops / execution
- **Single scheduler** (hard rule — multi-pool collisions caused ~45h loss before). Run after / interleaved
  with camp5's clean-arm completion; do NOT launch a second concurrent pool.
- Generate jobs via a make_campaign_jobs.py extension (+ `--calib_source`, + CorDA++ arms, + rank-matched control).
- Compute estimate + wall-clock to be filled by the design reviewer.

## 9. Decision rules (pre-registered)
- Off-curve deviation shrinks to within CI of the calibration-free curve under eval-matched calib → **artifact**;
  report law universality + calibration sensitivity.
- Deviation persists under fair calibration + fair usage + seeds → **real limitation**; publish as such.
- Pre-register these thresholds BEFORE running to avoid post-hoc rationalization.

## 10. Open decisions for the panel
1. Exact eval-matched calib composition (MMLU-aux + ARC vs adding others; stratification; 256 enough?).
2. wd-knob symmetry — mini-ablation vs full wd sweep for all arms.
3. Depth-first (Llama-CS) vs full-2×2 breadth from the start.
4. Rank-matched control design (r16 for all vs multiple ranks).
5. residual_save fp32 switch vs keep+disclose.
6. CorDA++ inclusion now vs after N is fetched (static CorDA may suffice for the fairness claim).
7. Is the calibration truly symmetric, or does any method still get an unfair peek (e.g., SC-LoRA also uses D+ = task data)?
