# FIX-1 Fairness & Calibration Study — FINALIZED PLAN

**Status:** FINAL. Integrates a 4-lens expert panel (fairness/calibration [primary], experimental
design/stats, adapter-implementation, adversarial peer-reviewer). Supersedes
`19_FAIRNESS_STUDY_PLAN_DRAFT.md`. Panel verdict: **needs_revision (unanimous)** — the fairness
*instinct* was right, but the draft reintroduced fairness confounds through the back door
(non-identical collectors, pre-fix run reuse, mixed rank/scaling, single-seed reference, unfalsifiable
decision rules). This document closes every blocking gap with **locked values**, prioritizing
**fairness of usage + calibration above all**.

**North star (unchanged):** measure CorDA-static / SC-LoRA / LoRA-Null under a *provably symmetric*
usage+calibration protocol so the off-curve verdict is unimpeachable. Both outcomes publish:
- eval-matched calib moves them ONTO the curve → law is universal + calibration-sensitivity finding;
- they stay OFF the curve under fair calib + fair usage + seeds → an EARNED method limitation.

---

## 0. WHAT CHANGED FROM THE DRAFT (panel-driven, one line each)

1. **No reuse of any existing `lrsw_*` row.** They are single-seed (all s42) and pre-fix (commits
   `f32145557` / `f68f78a8` / `21517195`, i.e. BEFORE `fe0f9be3`+`2602f57d` gen_cap/BBH fixes). The
   headline nq-vs-EM contrast would otherwise confound calibration with an eval-harness change.
   **Everything is re-run in ONE post-fix pass at HEAD.** (blocking, all four reviewers)
2. **One shared calibration-forward routine** for CorDA / SC-LoRA / LoRA-Null. Today the three
   collectors diverge on `max_len` (256 / 1024 / 2048 defaults), batch (bs=4 padded vs bs=1 unpadded),
   and padding (PAD=`<unk>` enters CorDA/LoRA-Null covariance). "Shared 256 samples" is necessary but
   NOT sufficient — the *forward pass* must be unified. (blocking)
3. **Single locked operating point: r=16, α=16 (scaling=1) for EVERY arm**, including fresh re-runs of
   the calibration-free reference arms (lora, lorawd, dora, milora, clora). The draft compared
   scaling=1 subjects against a mostly-scaling=2 / mixed-rank reference — "off-curve" partly measured
   rank+scaling. (blocking)
4. **Off-curve statistic is now pre-registered and support-bounded** with numeric CI thresholds, to
   defeat the CorDA runaway-‖ΔW‖ extrapolation artifact (corda ‖ΔW‖ 36→54741 vs clean 2.5→200). (blocking)
5. **Reference curve gets the SAME 3 seeds** as the subjects, so "within CI of the curve" is even defined. (major)
6. **CorDA++ (dynamic cov + dynamic rank) is DEFERRED** — unresolved N (fetch-blocked), unresolved
   π-operand, and two extra data-driven DoF that break the rank-matched control. Static CorDA fully
   answers the calibration-strawman charge. (major, all four)
7. **wd knob given symmetrically** to every arm ({0, 0.3} at best-LR/seed42), not a one-arm mini-ablation. (major)
8. **fp32 residual-save for ALL residual arms in the new pool** (uniform, since we re-run everything). (minor→locked)
9. **Disjointness gate hashes the EXACT lm-eval-rendered test docs**, not raw HF splits, per task; excludes
   ARC-Challenge test ids; dedups the calib pool. (blocking)
10. **Figure loader (`paper_figs_v2.py`) fixed**: PREF extended to the fair tag, dedup by
    `argmax(evaluated_at)` not keep-last-line, corda un-hidden by tag, dry-run gate. (blocking)

---

## 1. ARMS IN SCOPE

**Data-aware subjects (3):** `corda_static` (PEFT-native KPM), `sclora`, `lora_null`.
- `corda_pp` / `corda_ppCov` / `corda_ppRank` are **OUT of FIX-1** (deferred; §11 user-decision D3).

**Calibration-free reference (5):** `lora`, `lorawd`, `dora`, `milora`, `clora` — provide the on-curve
reference. **Re-run fresh at r16/α16 in this pool** (NOT reused from camp5, which is mixed-rank/scaling
and single-seed).

**LoRA-Null estimand:** two labeled variants — `lora_null` (freeze_a=0, head-to-head, trains A+B like
every other arm) and `lora_null_freezeA` (freeze_a=1, the authors' best-preservation config). Both are
plotted on the law; the head-to-head verdict uses freeze_a=0, and freeze_a=1 is reported as the method's
own strongest point. Never silently mixed.

---

## 2. LOCKED EXPERIMENT MATRIX + CELL COUNT + WALL-CLOCK

### 2.1 Axes (locked)
| Axis | Locked values | Count |
|---|---|---|
| **Base LR grid** | 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3 | 7 |
| **Seeds** | 42, 43, 44 | 3 |
| **Model** | Llama-2-7b (CS) primary; Qwen2.5-7B (CS) headline replication | 1 + replication |
| **Domain** | commonsense_170k (depth-first); math as later breadth | 1 |
| **Operating point** | r=16, α=16, scaling=1 (ALL arms) | 1 |

### 2.2 CELL COUNT — Llama-2 commonsense (the definitive cell)

| Block | Arms | Calib | LR × seed | Cells |
|---|---|---|---|---|
| **A. EM subjects** | corda_static, sclora, lora_null | eval-matched | 7 × 3 | **63** |
| **B. NQ subjects** (single-variable contrast) | corda_static, sclora, lora_null | nq_open | 7 × 3 | **63** |
| **C. LoRA-Null freeze_a=1** | lora_null_freezeA | EM + NQ | 7 × 3 × 2 | **42** |
| **D. Reference curve** | lora, lorawd, dora, milora, clora | n/a (calib-free) | 7 × 3 | **105** |
| **E. wd symmetry** | all 8 arms (3 subj@EM + 5 ref) | wd=0.3, best-LR only | 1 LR × 3 | **24** |
| **F. SC-LoRA β sweep** | sclora @ EM | β∈{0.3,0.7} (0.5 already in A) | 2 × 1 | **2** |
| **G. SC-LoRA β→1 diagnostic** | sclora @ EM (isolates D+ peek) | β=0.95 | 1 × 1 | **1** |
| **H. SC-LoRA norm sensitivity** | sclora @ EM, max(\|Y\|) variant | best-LR | 1 × 1 | **1** |
| **Llama-2 CS TOTAL** | | | | **301** |

### 2.3 CELL COUNT — Qwen2.5-7B CS headline replication (pre-submission, NOT full grid)
Only the headline arms at the full LR×seed grid; no ablation blocks C/E/F/G/H.

| Block | Arms | Calib | LR × seed | Cells |
|---|---|---|---|---|
| Subjects EM | corda_static, sclora, lora_null | EM | 7 × 3 | 63 |
| Subjects NQ | corda_static, sclora, lora_null | nq_open | 7 × 3 | 63 |
| Reference | lora, lorawd, dora, milora, clora | n/a | 7 × 3 | 105 |
| **Qwen CS TOTAL** | | | | **231** |

### 2.4 GRAND TOTAL + 8-wide B200 WALL-CLOCK
- **Llama-2 CS: 301 cells. Qwen CS: 231 cells. GRAND TOTAL = 532 cells.**
- **Per-cell wall-clock (train+broad eval), measured basis:** train median **2.0 h/cell** (from
  `train_registry.jsonl`, n=93 Llama-2 lrsw runs, 7089–8600 s typical); broad eval ~1.5–2 h/cell
  (BBH 5-shot + MMLU-Pro + MMLU/ARC/TruthfulQA at gen_cap=1024). **Lock ~4 h/cell** conservative.
- **8-wide throughput:** 4 h / 8 GPU = **0.5 GPU-cell-hours of wall per cell**.
  - Llama-2 CS: 301 × 4 h / 8 = **~150 h (~6.3 days)**.
  - Qwen CS: 231 × 4 h / 8 = **~116 h (~4.8 days)**.
  - **GRAND TOTAL ≈ 266 h wall-clock ≈ 11.1 days at 8-wide.**
- **Depth-first budget to the FIX-1 verdict (Llama-2 CS Blocks A+B+D only = 231 cells):**
  231 × 4 h / 8 = **~116 h (~4.8 days)** — this is the critical path to the falsifiable off-curve
  answer; C/E/F/G/H (70 cells, ~35 h) and the Qwen replication follow.
- One-off CorDA-static covariance precompute (single GPU, both calibrations): ~1 GPU-hour, off critical path.

---

## 3. FAIRNESS GUARANTEES CHECKLIST (usage + calibration symmetry, with verification)

Every row is enforced identically across the 3 data-aware subjects, and where applicable across all 8
arms. "Verified by" = the concrete gate/assert that proves it before any figure is trusted.

### 3.1 CALIBRATION symmetry (axis #1)
| # | Guarantee | Enforcement | Verified by |
|---|---|---|---|
| C1 | **Same 256 calib prompts, same order** across corda/sclora-D−/lora_null | one shared, seeded sampler; log sample ids | assert identical `calib_hash` in the 3 arms' `run_config.json` |
| C2 | **Same forward routine**: bs=1 UNPADDED for all three (removes PAD=`<unk>` contamination of CorDA/LoRA-Null covariance) | single `calib_forward()` used by all collectors | assert `(bs, pad_masked)` triple identical in run_config; PAD tokens never enter any covariance/2nd-moment |
| C3 | **Same `calib_max_len`** passed EXPLICITLY (never per-collector default) | pass `--calib_max_len` to all three; fix `train_cs.py:224` to stop hard-defaulting CorDA to 256 | assert the three collectors receive identical max_len at runtime; logged |
| C4 | **Same N=256, same stratification** (proportional over 57 MMLU subjects + ARC-C/E) | shared pool builder | log per-subject counts; sum==256 |
| C5 | **calib ∩ eval-TEST = ∅ (provable)** — hash calib vs the EXACT items lm-eval renders per task, not raw HF splits | materialize test docs via the same task objects `eval_one_gpu.py:120-131` builds; hash normalized strings | per-task overlap count == 0 asserted; ARC-Challenge test ids excluded from pool; MMLU-aux hashed vs arc_challenge test (MMLU-aux aggregates ARC/OBQA/RACE) |
| C6 | **calib pool internally deduped** (MMLU-aux vs ARC-train near-dups) | unique normalized-question hash before use | assert 256 unique hashes |
| C7 | **nq-vs-EM differ ONLY in `--calib_source`** — same engine, same collector, same eval pass | one covariance engine per method across BOTH calibrations (§6) | diff the two run configs: only `calib_source`/`calib_hash` differ |
| C8 | **Statistic asymmetry disclosed, not hidden**: SC-LoRA hooks OUTPUT 2nd-moment; CorDA/LoRA-Null hook INPUT. "Identical calibration" = identical DATA + identical FORWARD, never identical statistic | documented in plan + paper | stated explicitly (this row); no code change — it is method-intrinsic |

### 3.2 USAGE symmetry (axis #2)
| # | Guarantee | Enforcement | Verified by |
|---|---|---|---|
| U1 | **Same operating point r=16, α=16, scaling=1** for EVERY arm | ARMS table rewritten (§5) | assert `alpha==r==16` in every adapter_config; residual arms already require it (`residual_save.py:59`) |
| U2 | **Same target_modules** `q,k,v,up,down` (160 matrices) | shared `--target_modules` | logged; 160 matrices asserted |
| U3 | **Same LR grid, optimizer (adamw_torch), linear schedule, warmup, precision (bf16 train), max_len, epochs** | one job template | no per-method training branch (audited fair, `18 §3d`) |
| U4 | **Same eval harness, bit-for-bit** (fresh W0 + PeftModel, same suite, gen_cap, max_len, BBH-fix) | `eval_one_gpu.py`, no per-method branch (`18 §3e`) | single post-fix eval pass |
| U5 | **Same seeds 42/43/44** for subjects AND reference | §5 job gen | seed distribution check |
| U6 | **Same wd knob offered to all** ({0, 0.3} at best-LR/seed42) | Block E for all 8 arms | wd is uniformly wired at `train_cs.py:295`; each arm reported at its best (LR, wd) |
| U7 | **Realized trainable params matched <2%** across arms at r16/α16 | report `count_trainable` (`train_cs.py:205,321`) | table of realized params; DoRA's +magnitude-vec disclosed exactly |
| U8 | **Uniform init fidelity**: fp32 residual-save cancellation term for ALL residual arms (corda, sclora, lora_null, milora) | switch `residual_save.py:51-52` to fp32 for this pool | 0-step gate (`validate_residual_zero_step.py`) per residual arm: post-reload ‖ΔW‖ < 1e-4 |
| U9 | **Each method at its BEST config, no strawman**: SC-LoRA β swept {0.3,0.5,0.7}; LoRA-Null both freeze_a variants; CorDA KPM; repo-canonical SC-LoRA \|max(Y)\| normalization (+ one sensitivity pair) | Blocks A/C/F/H | per-arm best (LR, knob) reported |

### 3.3 DISCLOSED (faithful) asymmetries — kept, NOT equalized away
| # | Asymmetry | Why kept | How neutralized |
|---|---|---|---|
| D1 | **SC-LoRA uses D+ = fine-tuning task** at init (extra task-peek CorDA/LoRA-Null lack) | intrinsic to SC-LoRA (a task-AND-retention-aware method); removing it = unfaithful port | disclosed in paper; report SC-LoRA on the **law axis**, not head-to-head "retains less than CorDA"; add **β→1 diagnostic** (Block G) to isolate D+ vs D− contribution |
| D2 | **INPUT vs OUTPUT statistic** (C8) | method-defining | disclosed; comparison is on the law, not on "same covariance" |
| D3 | **EM calib is eval-distribution-adjacent** — only data-aware arms use it | it is DISJOINT auxiliary data (MMLU-aux/ARC-train), legitimately available to ANY method; calib-free arms *forgo* it by design | §9 states "onto the curve under EM" does not privilege data-aware arms; the law holds for calib-free arms with NO peek (the stronger result). Off-curve verdict must ALSO hold on the **held-out** eval subset (§4.3) that the calib never saw |

---

## 4. EVAL-MATCHED CALIBRATION SPEC (exact)

### 4.1 Composition (locked)
- **Sources:** `cais/mmlu` split `auxiliary_train` + `allenai/ai2_arc` (ARC-Challenge and ARC-Easy)
  split `train`. **Do NOT add BBH/TruthfulQA material** — BBH has no clean train split; TruthfulQA-mc2
  has no disjoint train. Keeping them out makes them a clean held-out transfer test (§4.3).
- **N = 256.** Matches the CorDA/CorDA++ paper budget and the existing `*_calib_size` defaults
  (`train_cs.py:142,146,149`) so nq_open and EM share a sample budget.
- **Stratification:** proportional round-robin across the 57 MMLU subjects + ARC-C/E → 256. Fixed
  sampling seed; log the chosen ids.
- **Formatting:** render prompts through the **same MC template the eval uses** (question + lettered
  answer choices), so the collected covariance reflects the eval input distribution — not raw questions.

### 4.2 Disjointness + adequacy (gates, must pass before any run)
- **Disjointness (C5/C6):** hash the 256 calib prompts (normalized) against the EXACT test docs lm-eval
  renders for ALL five eval tasks (`bbh_fewshot`, `mmlu_pro`, `mmlu`, `arc_challenge`, `truthfulqa_mc2`),
  per task; assert overlap==0. Exclude ARC-Challenge **test ids** from the pool. Hash MMLU-aux vs
  arc_challenge test (MMLU-aux aggregates ARC). Dedup the 256 internally. Log all hashes.
- **Adequacy check (one-off, appendix):** recompute the top-r (r=16) input-covariance subspace at N=256
  vs N=512 on ONE layer; report the principal-angle drift. If the r16 subspace is stable (small angles),
  256 is defensible; if unstable, **bump to 512 for ALL arms identically** (both calibrations). Reported
  empirically, not by citation.

### 4.3 Calib-aligned vs held-out reporting (design reviewer, blocking-adjacent)
`mmlu` and `arc_challenge` share their SOURCE with the calib; `bbh_fewshot`, `mmlu_pro`, `truthfulqa_mc2`
do not. Report retention split two ways and **judge the off-curve verdict on the HELD-OUT subset**
{bbh_fewshot, mmlu_pro, truthfulqa_mc2} — the genuine test of "does data-aware retention transfer beyond
its own calibration distribution." The calib-aligned subset {mmlu, arc_challenge} is reported separately
and must NOT drive the headline off-curve number alone.

### 4.4 Locked calib settings
| Setting | Value | Rationale |
|---|---|---|
| `--calib_source` | `{nq_open, eval_matched}` (new flag) | the ONLY variable between the two arms |
| N | 256 (bump to 512 uniformly iff adequacy check fails) | paper budget + existing defaults |
| `calib_max_len` | **512** (passed to all three collectors) — see §11 D2 | fits MMLU/ARC question+all choices; 256 (CorDA default) drops choice framing, 4096 blows up covariance-collection cost. **Flagged for user** because "true eval-match" argues for the 5-shot context length. |
| forward | bs=1, unpadded, identical routine | removes PAD contamination + engine asymmetry |
| cache key | `(base_model, calib_hash, N, calib_max_len, bs, dtype)` | nq/EM caches never collide; identical consumption provable |
| stratification | proportional MMLU-subject + ARC, seeded, ids logged | reproducible, disjointness-checkable |

---

## 5. PER-METHOD USAGE SETTINGS (best-config, adapter-expert corrections)

### 5.1 The rewritten ARMS table (all r16/α16, scaling=1)
Replaces `make_campaign_jobs.py:17-26` for this pool. **Every arm is r16/α16** (the current camp5 table
mixes r16/r32 and scaling 1/2 — that mixing is the reason the draft reference curve was unfair).

| arm | flags (locked for FIX-1 pool) | scaling | residual? |
|---|---|---|---|
| `lora` | `--method lora --lora_r 16 --lora_alpha 16` | 1 | no |
| `lorawd` | `--method lora --lora_r 16 --lora_alpha 16 --weight_decay 0.3` | 1 | no |
| `dora` | `--method lora --use_dora 1 --lora_r 16 --lora_alpha 16` | 1 | no |
| `milora` | `--method lora --milora 1 --lora_r 16 --lora_alpha 16` | 1 | yes (fp32) |
| `clora` | `--method clora --clora_k 1024 --lora_r 16 --lora_alpha 16` | 1 | no |
| `corda_static` | `--method lora --corda 1 --corda_engine peft --lora_r 16 --lora_alpha 16 --calib_source <>` | 1 | yes (fp32) |
| `sclora` | `--method lora --sclora 1 --sclora_beta 0.5 --lora_r 16 --lora_alpha 16 --calib_source <>` | 1 | yes (fp32) |
| `lora_null` | `--method lora --lora_null 1 --lora_r 16 --lora_alpha 16 --lora_null_freeze_a 0 --calib_source <>` | 1 | yes (fp32) |
| `lora_null_freezeA` | `... --lora_null_freeze_a 1 ...` | 1 | yes (fp32) |

> Note the deliberate change: `lora`, `dora`, `clora`, `lorawd` move from α=2·r (scaling=2) to α=r
> (scaling=1) so **scaling is not a second hidden variable** in the on/off-curve read. Native-rank /
> native-scaling rows (camp5) are kept ONLY as a secondary "best-config" appendix view, never as the
> reference the off-curve verdict is read against.

### 5.2 Adapter-expert corrections (locked)
1. **CorDA engine consistency (blocking):** build BOTH the nq_open and eval-matched CorDA arms with the
   SAME engine. Custom `corda_init.py` (`/256`, bs=4) and PEFT-native `preprocess_corda`
   (`/sample_count`, squeeze bs=1) are different estimators — mixing them confounds calibration with
   the engine. **Lock: one engine per method for both calibrations.** Recommended = keep the custom
   collectors + `--calib_source` swap (cheapest correct path); if PEFT-native CorDA is used, its
   nq_open baseline is re-run through PEFT-native too. Do NOT reuse archived custom nq_open CorDA rows
   against a PEFT-native EM arm.
2. **SC-LoRA eval-matched target = D− ONLY** (`train_cs.py:242`). D+ = fine-tuning task
   (`train_cs.py:239`) stays unchanged. The draft's "one shared set for all" wording is wrong for
   SC-LoRA and is corrected: the shared EM set is CorDA calib / SC-LoRA **D−** / LoRA-Null calib.
3. **SC-LoRA β:** sweep {0.3, 0.5, 0.7} at seed 42 under EM; report best (keep 0.5 primary if not
   dominated). β is SC-LoRA's core preserve/adapt knob — a single point invites "under-tuned SOTA."
   Plus a **β→1 diagnostic** (Block G) isolating the D+ peek from the D− retention term.
4. **SC-LoRA normalization:** `Y.max().abs()` (`sclora_init.py:44`) is repo-canonical (confirmed by the
   2026-06-22 correction note). Run ONE `max(|Y|)` sensitivity pair at best-LR/s42 (Block H) as an
   appendix robustness point.
5. **LoRA-Null:** `null_dim=r` (rank-matched, faithful default); run freeze_a=0 (head-to-head) AND
   freeze_a=1 (paper best) as two labeled arms, both on the law.
6. **CorDA-static** is KPM (retention mode; PEFT default is IPM — must set `corda_method="kpm"`),
   fp32 covariance for the reconstruction identity, `alpha==r`.

---

## 6. STATISTICS + PRE-REGISTERED DECISION RULES

### 6.1 Off-curve statistic (LOCKED, falsifiable)
The paper's headline. Pre-registered exactly:
1. Fit the reference curve `retention ≈ f̂(log‖ΔW‖_F)` on the **5 calibration-free arms** at r16/α16
   (monotone/piecewise-linear or spline). Use `dw_sv_max` / Frobenius ‖ΔW‖ (input-independent,
   `uio_inprocess.py:56`) as the x-axis — NOT the CS-input token-weighted F-delta (that is measured on
   the wrong distribution; keep it only as a labeled proxy or recompute on retention inputs).
2. Score each data-aware run's residual `ret(arm) − f̂(log‖ΔW‖_arm)` **only inside the ‖ΔW‖ window
   covered by ≥3 clean arms** (report `[dw_lo, dw_hi]`). Data-aware points outside the window are
   "out of support" and are **NOT extrapolated** — this is what defeats the CorDA ‖ΔW‖=36→54741
   runaway artifact (the clean arms live at 2.5→200; comparing corda at 36 against a fit dominated by
   <40 is leverage, not method).
3. Require **≥3 in-support data-aware points per arm** or the off-curve claim is declared **untestable**
   for that arm.
4. Report the residual with a **bootstrap CI over seeds AND over the curve fit**.
5. The verdict is read on the **held-out eval subset** (§4.3).

### 6.2 Power / MDE (pre-registered)
Run-to-run retention SD ≈ 1 pp (adjacent-LR spread in the real data). With **3 seeds**: SE of a mean
≈ 0.6 pp, two-group Δ SE ≈ 0.8 pp → **minimum detectable off-curve shift ≈ 2.5 pp at 80% power**. The
observed deviations (CorDA −3.0, SC-LoRA −3.3) are detectable; a **post-calibration residual between the
MDE and 0 is reported as INCONCLUSIVE, not "artifact."**

### 6.3 Decision rules (LOCKED numeric thresholds — register with the git commit BEFORE running)
- **ARTIFACT** iff the EM off-curve residual **95% CI lies entirely within ±1.5 pp of 0** (TOST
  equivalence) → report law universality + calibration-sensitivity finding.
- **REAL LIMITATION** iff the **CI upper bound < −1.5 pp** (still off-curve by more than the MDE) →
  publish as an earned method limitation.
- **INCONCLUSIVE** otherwise → report as such, state the power. (Prevents a low-power null being
  mislabeled "artifact.")
- All three verdicts are read on the **held-out subset** first; the calib-aligned subset is a secondary
  robustness view.

---

## 7. IMPLEMENTATION PREREQUISITES + OPS

### 7.1 Code prerequisites (spec only — do NOT edit any pipeline file as part of finalizing this plan)
1. **Shared calib module:** a single `calib_forward()` (bs=1, unpadded, explicit `calib_max_len`) + a
   seeded EM pool builder (MMLU-aux + ARC-train, stratified, deduped, disjointness-gated). Wire a
   `--calib_source {nq_open, eval_matched}` and `--calib_max_len` flag into `train_cs.py` for corda /
   sclora / lora_null. Fix `train_cs.py:224` to pass `calib_max_len` to `collect_corda_cov` (currently
   hard-defaults 256).
2. **Disjointness gate** (`§4.2`): materialize lm-eval-rendered test docs per task; hash; assert; log.
   Reuse the validation-checklist item `17 §7.5`.
3. **residual_save fp32:** for this pool, write the init-cancellation term in fp32
   (`residual_save.py:51-52` drop the `.to(bf16)`), applied to ALL residual arms (corda, sclora,
   lora_null, milora). Re-run `validate_residual_zero_step.py` per residual arm: post-reload
   ‖ΔW‖ < 1e-4.
4. **CorDA-static via PEFT** `preprocess_corda` with `corda_method="kpm"`, or the custom engine — but
   ONE engine for both calibrations (§5.2.1). Save the initial adapter pre-train and convert with
   `path_initial_model_for_weight_conversion` (or the existing `residual_save` path). Verify the
   round-trip AFTER reload, not in-memory (`17 §4`).
5. **Figure loader (`paper_figs_v2.py`) — blocking fixes:**
   - extend `PREF` (line 102/113) to include the fair-pool tag so `_calibEM_`/`_calibNQ_` rows load;
   - replace keep-last-line dedup (line 110) with **`argmax(evaluated_at)`** (parse `now_iso`, keep max);
   - un-hide `corda` by tag (lines 118-119) so fair corda_static rows plot;
   - use the SAME tag scheme for nq and EM (do NOT reuse `lrsw_` for nq and a new tag for EM);
   - **dry-run** to confirm exactly the intended run_names load and NO pre-fix row (commit < `2602f57d`)
     survives.

### 7.2 Ops (locked)
- **Single scheduler (hard rule):** never launch a 2nd 8-GPU pool while another is live (multi-pool
  collisions cost ~45 h before). Run this pool AFTER camp5 drains; do NOT touch the running pool.
- **Registry hygiene:** new pool tag; `_calibEM_` / `_calibNQ_` in every run-name; ONE post-fix eval
  pass; **assert every `summary.json` `git_commit` ≥ `2602f57d`** for any row used in a figure/table.
- **Calib flag:** `--calib_source` + `--calib_max_len` logged in `run_config.json`; assert identical
  `calib_hash` across the 3 subjects per calibration.
- **Resumability:** `make_campaign_jobs.py` already skips cells with an existing `summary.json`;
  extend `ARMS`/seeds/`--calib_source` per §5 (spec only).
- **CorDA precompute** (if PEFT-native or CorDA++ later): one-off per (model, calib_hash, N), cached.

---

## 8. SUMMARY OF PANEL-CONFLICT RESOLUTIONS (explicit reasoning)

| Conflict | Positions | RESOLUTION (fairness-first) |
|---|---|---|
| **residual_save fp32 vs bf16** | fairness+impl: fp32-for-all-residual; design+adversarial: bf16-for-all (to keep nq/EM bit-identical) | **fp32 for ALL residual arms.** Both camps' true requirement is *symmetry*. The bf16-for-all argument was contingent on *reusing* pre-fix nq rows; since we RE-RUN everything fresh in one pass (§0.1), fp32-for-all is bit-identical across nq AND EM AND removes the method-varying ~3e-3 floor on the exact axis (‖ΔW‖) the thesis regresses. Fairness wins with no cost. Gate: 0-step ‖ΔW‖<1e-4 per residual arm. |
| **calib_max_len: 256 vs 512 vs ≥2560/4096** | fairness: 256 (CorDA default/faithful); impl: 512; adversarial: match eval 5-shot context | **Lock 512**, flag context-length as a **user decision** (§11 D2). 256 silently drops MMLU/ARC choice framing the "eval-matched" claim needs; 4096 is faithful to the eval context but ~8× covariance-collection cost. 512 fits question+all choices — the substantive content — at modest cost. The residual tension (adversarial wants full eval context) is real → user calls it. |
| **wd knob: mini-ablation vs full sweep** | fairness+adversarial: symmetric grid for all; design+impl: 2–3 point at best-LR | **{0, 0.3} for ALL 8 arms at each arm's best-LR, seed 42** (Block E, 24 cells). Symmetric (defeats "you tuned the magnitude knob only for LoRA") without a full 2× LR×wd blow-up. wd is uniformly wired (`train_cs.py:295`) so this is nearly free. Report each arm at its best (LR, wd). |
| **CorDA++ now vs later** | UNANIMOUS: defer | **DEFER.** Static CorDA answers the calibration-strawman charge; corda_pp has unresolved N (fetch-blocked), unresolved π-operand, and 2 extra data-driven DoF that break the rank-matched control → a NEW unfaithful-reproduction strawman risk. Removes 63–70 cells + the fetch-blocker from the critical path. (User may override → §11 D3.) |
| **reuse old nq runs** | UNANIMOUS: no | **Re-run all, one post-fix pass.** Reuse conflates calibration with collector+eval-harness fixes. |
| **rank/scaling** | UNANIMOUS (design+adversarial+impl): one operating point | **r16/α16 scaling=1 for ALL arms**, reference re-run at that point. Native-rank rows demoted to appendix. |
| **SC-LoRA D+** | UNANIMOUS: keep, disclose | **Keep D+ = task; disclose; report SC-LoRA on the law axis; add β→1 diagnostic.** |
| **off-curve statistic** | design reviewer only (others silent) | **Adopt the support-bounded, pre-registered statistic (§6.1)** — no other reviewer objected and it is the single change that makes the headline falsifiable. |

---

## 9. FRAMING GUARANTEES FOR THE PAPER (so a hostile reviewer cannot flip the charge)

- **"Onto the curve under EM calib" does NOT privilege the data-aware arms:** the EM calib is DISJOINT
  auxiliary data (MMLU-aux/ARC-train) legitimately available to ANY method; the calibration-free arms
  *choose* to forgo it. The law holding for the calib-free arms **with no peek at all** is the stronger
  result. The off-curve verdict is judged on the **held-out** eval subset the calib never saw (§4.3).
- **SC-LoRA's D+ is method-intrinsic** (a task-AND-retention-aware method); it is disclosed, and
  SC-LoRA is compared on the **law axis**, never head-to-head "retains less than CorDA."
- **INPUT vs OUTPUT statistic** is method-defining and disclosed; "identical calibration" means
  identical DATA + identical FORWARD, never identical statistic.
- **Native-rank best-config rows are reported** alongside the r16/α16 head-to-head, so no reviewer can
  say a method was shown only at a handicapped rank.

---

## 10. VALIDATION CHECKLIST (must all pass before any row is trusted)
1. `alpha==r==16` in every adapter_config (U1).
2. Identical `calib_hash` across the 3 subjects, per calibration (C1); nq/EM run configs differ ONLY in
   `calib_source`/`calib_hash` (C7).
3. Identical `(calib_max_len, bs, pad_masked)` across the 3 collectors (C2/C3).
4. calib ∩ lm-eval-rendered test == ∅ per task; ARC-Challenge test ids excluded; MMLU-aux vs ARC-test
   clean; 256 unique calib hashes (C5/C6).
5. 0-step post-reload ‖ΔW‖ < 1e-4 for corda/sclora/lora_null/milora (U8).
6. Realized `count_trainable` matched <2% across arms at r16/α16 (U7).
7. Reference curve and subjects both at seeds 42/43/44 (U5).
8. Every figure/table row `git_commit ≥ 2602f57d` (post gen_cap+BBH fix).
9. `paper_figs_v2` dry-run loads exactly the intended run_names, dedups by `argmax(evaluated_at)`,
   corda un-hidden (§7.1.5).
10. N=256-vs-512 subspace-angle adequacy check reported (§4.2).
11. Decision-rule thresholds (§6.3) committed to git BEFORE launch.

---

## 11. DECISIONS THAT NEED THE USER (panel could NOT settle)

**D1 — Depth-first vs full breadth.**
Panel consensus is depth-first on Llama-2 CS to the falsifiable verdict (231 cells, ~4.8 days), then the
Qwen-CS headline replication (231 cells, ~4.8 days) BEFORE submission (a single-model off-curve headline
invites "holds only on Llama" rejection). Math is later breadth. **User confirms:** run the full
Llama-2 CS matrix (301 cells) + Qwen CS replication (231) = **532 cells / ~11 days** now, OR stop at the
Llama-2 depth-first verdict (231 cells / ~4.8 days) and gate Qwen on a positive/interesting result?

**D2 — Calibration context length (`calib_max_len`).**
Locked provisionally at **512** (fits MMLU/ARC question+all choices). The adversarial reviewer argues a
truly "eval-matched" claim should collect covariance at the eval 5-shot context length (≥2560, up to
4096), at ~5–8× covariance-collection cost (one-off per model, off the critical path). **User picks:**
512 (cheap, captures choice framing) or full eval context (strictly defensible "eval-matched" wording).

**D3 — CorDA++ now vs later.**
All four reviewers say DEFER (static CorDA answers the fairness charge; corda_pp is fetch-blocked on N,
has an unresolved π-operand, and its dynamic rank breaks the rank-matched control). This is the panel
recommendation, but including CorDA++ is a scientific-scope call. **User decides:** exclude CorDA++ from
FIX-1 (recommended), or fetch N verbatim from arXiv:2506.13187 first and add corda_pp + the
corda_ppCov/corda_ppRank ablations (+~70 cells, ~35 h) as a stretch block?

**D4 — Extra seeds for a sub-2.5 pp null.**
3 seeds give MDE ~2.5 pp. If the study must prove a **<2.5 pp residual is a genuine null** (not just
inconclusive), raise sclora + corda_static (EM only) to 5 seeds (2 arms × 7 LR × 2 extra seeds = 28
cells, ~14 h) to push MDE to ~1.5 pp. **User decides** whether a sub-MDE null needs to be provable or
"inconclusive" is an acceptable verdict there.
