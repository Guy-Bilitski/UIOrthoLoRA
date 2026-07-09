# 18 — Consolidated Per-Adapter Faithfulness Audit (Publication Readiness)

Editor: publication-readiness consolidation of the 8-arm per-adapter audit+verify pass, plus
first-hand audit of the two shared machinery files (`residual_save.py`, `eval_one_gpu.py`).
Date: 2026-07-02. Thesis under test: *magnitude law* — retention is governed by ‖ΔW‖, and plain
LoRA+wd matches the fancy data-aware adapters.

**Headline:** every port is code-faithful. The blocker is NOT porting — it is a
**calibration↔eval distribution confound (FIX-1)** that taints the *empirical claim* for the three
calibration-using arms (CorDA, SC-LoRA, LoRA-Null). The per-adapter audits for SC-LoRA and LoRA-Null
marked them "faithful / no blocking issues"; that verdict is correct about the CODE but **under-scoped
the experiment** — handoff/17 (today) and handoff/13 §2 explicitly classify both as CONFOUNDED and
mandate pause + re-run, exactly like CorDA. This report supersedes those two "clean" verdicts on the
experimental (not code) axis.

---

## 1. Per-adapter verdict table

| Arm | Code-port verdict | Experiment verdict | One-line |
|---|---|---|---|
| **lora** (baseline) | faithful | GO | Textbook PEFT LoRA, α/r=2, B=0 init; the law's anchor. No calib, no residual. Nothing to attack. |
| **lorawd** (wd=0.3, HERO) | faithful | GO | Plain LoRA + `weight_decay=0.3` in TrainingArguments; wd applied to A/B (not in no_decay filter). The paper's control; simplest possible, unattackable. |
| **dora** | faithful | GO | Canonical PEFT DoRA; magnitude vec = ‖W‖_c at init, +876,544 params (disclosed, exact). Calib-free. |
| **milora** | faithful | GO | Bottom-r SVD of W only; **calibration-free → distribution-invariant → NOT confounded**. α==r enforced, residual-save wired. Safe. |
| **corda** (KPA) | faithful (math) | **NO-GO (as-is) / FIX-FIRST** | Init math correct; residual-save validated. But **nq_open calib ≠ eval → off-curve claim CONFOUNDED**. Already omitted from tables. |
| **sclora** | faithful (port) | **FIX-FIRST** | Port correct (corrected 2026-06-22). But **D− = nq_open → same FIX-1 confound**; −3.3pp off-curve is CONFOUNDED (handoff/17 §5). Audit's "no blocking" is code-only. |
| **lora_null** | faithful (port) | **FIX-FIRST** | Port correct. But **calib = nq_open → null-space of the WRONG (factoid) activations → same FIX-1 confound** (handoff/17 §5). Audit's "no blocking" is code-only. |
| **clora** | faithful | GO | Frozen random-orthonormal P (seed 42, `detach()`), Frobenius penalty wired into `CLoRATrainer.compute_loss` every step; calib-free, param-parity exact. P is train-only (correctly absent at eval). |

---

## 2. CONFIRMED blocking issues (most-severe first)

### BLOCK-1 — Calibration↔eval distribution confound (FIX-1) on all three data-aware arms
- **Arms:** corda, sclora, lora_null.
- **File:line:**
  - corda: `train_cs.py:218-219` (`nq_open` question calib)
  - sclora: `train_cs.py:241-242` (D− = `nq_open`)
  - lora_null: `train_cs.py:260-261` (null-space calib = `nq_open`)
  - eval distribution: `eval_one_gpu.py:124-126` (`bbh_fewshot`, `mmlu_pro` [+ broad: mmlu, arc_challenge, truthfulqa_mc2]) — academic/reasoning MC, **zero factoid QA**.
- **Why it blocks:** CorDA/SC-LoRA/LoRA-Null all freeze or preserve the subspace *most responsive to
  the calibration distribution*. Calibrating on factoid QA (nq_open) but evaluating retention on
  academic reasoning means they preserve the WRONG knowledge for our eval. Their observed retention
  deficits (CorDA −3.0pp, SC-LoRA −3.3pp; handoff/13 §2) may be **artifacts of the calibration
  mismatch, not properties of the methods**. Publishing "data-aware inits don't retain / are
  off-curve" from these runs is a strawman and will be caught by reviewers. This is the project's own
  HIGHEST-priority fix (handoff/17 §5, handoff/14 §8 Fix 1).
- **Fix (per handoff/17 §5):** Build a SHARED eval-matched calibration set from MMLU/ARC
  `auxiliary_train` splits (256, disjoint-by-question-hash from eval), and use it as CorDA calib /
  SC-LoRA D− / LoRA-Null calib. Re-run all three. Add an nq_open-vs-eval-matched **sensitivity arm**
  (this becomes the paper's "data-aware retention is calibration-distribution-dependent" ablation).
  Covariance cache key MUST include `calib_hash` so caches don't collide.
- **Status note:** The per-adapter verify passes for sclora/lora_null returned `net_verdict:
  faithful, blocking_issues: []`. That is CORRECT for the code port but WRONG for publication of the
  empirical claim. Treat this BLOCK-1 as overriding those two clean verdicts.

### BLOCK-2 — Off-curve / retention-deficit claims must not be published from the current (nq_open) runs
- **Arms:** corda (already excluded), sclora, lora_null.
- **Why it blocks:** Downstream of BLOCK-1. CorDA is already OMITTED from tables/figures (handoff/13
  §2, `nocorda` campaign). SC-LoRA and LoRA-Null rows from the current campaign are equally
  confounded and must NOT be trusted for the off-curve verdict. `paper_figs_v2.PREF` also loads
  `mtx_`/`mtxm_` rows (handoff/13:102) — those are the OLD wikitext-calibrated CorDA runs and are
  doubly invalid; verify they are excluded before any figure regen.
- **Fix:** Gate all three arms out of the headline retention-vs-‖ΔW‖ curve until the eval-matched
  re-runs land. Report them only in the FIX-1 sensitivity ablation, with the mismatch disclosed.

*(No CONFIRMED code-correctness blockers exist. The init math, scaling==1 enforcement,
loss-preserving reconstruction, and rank-2r residual conversion all verified correct across arms.)*

---

## 3. Cross-cutting issues

### (a) Calibration fairness — FIX-1 (see BLOCK-1)
The only fairness defect that reaches "blocking." Calibration-free arms (lora, lorawd, dora, milora,
clora) are distribution-invariant and **SAFE regardless** — they are the trustworthy backbone of the
law. Only the three calibration-using arms are confounded. This cleanly partitions the paper:
5 safe arms carry the magnitude-law claim; 3 confounded arms are quarantined to a re-run + ablation.

### (b) Residual save/reload correctness (`residual_save.py`) — CORRECT, one non-blocking precision note
- **Math verified.** `A'' = [A_tr ; A_init]` (2r,in), `B'' = [B_tr , −B_init]` (out,2r) ⇒
  `1·B''A'' = B_tr A_tr − B_init A_init = ΔW` relative to the ORIGINAL W0 (lines 49-50). This exactly
  cancels the in-memory `base.weight = W_res` overwrite that PEFT does not persist — the documented
  "reload explodes" bug class is correctly neutralized.
- **scaling==1 guard present** (line 59 assert `alpha==r`; α,r both → 2r keeps scaling=1, lines 61-62).
- **Non-blocking precision note (my audit):** `A_init`/`B_init` are captured in fp32 (`capture_init_adapter`
  lines 32-33, good), the stack is done in fp32 (lines 49-50), but the result is written back in the
  adapter's native **bf16** (lines 51-52 `.to(A_tr.dtype)`). So the `−B_init` cancellation term is
  bf16-quantized: a 0-step run's ΔW is not bit-exactly zero, only bounded by bf16 rounding. This is
  exactly why logged loss-preserving errs sit at ~2e-3…6e-3 (well under the 1e-2 validation gate). No
  action needed for publication, but if a reviewer asks "is init truly loss-preserving," the honest
  answer is "to bf16 precision (~3e-3 max abs), validated by the 0-step gate," not "exactly."
- **Coverage caveat (unverified):** the end-to-end 0-step gate (`validate_residual_zero_step.py`) is
  reported PASSED in handoff/13 but was not independently re-run in this audit; recommend one 0-step
  smoke per residual arm before the camera-ready to confirm no per-arm regression.

### (c) Parameter parity across arms — OK for the LAW, minor rank mismatch disclosed
- lora/lorawd/sclora/clora/milora r=32 ≈ 56,098,816 trainable; dora r16 = 28,925,952
  (+876,544 magnitude vec, exact); corda main arm r=128 (≈224M, rank-matched to its own baseline).
- **Ranks are NOT matched across the whole matrix** (r16 vs r32 vs r128 vs CLoRA k) — acknowledged in
  handoff/13:46 as "Fine for the LAW" (the claim is about magnitude, not method/rank). Residual
  conversion to rank-2r does NOT change trainable capacity (it only changes the saved representation).
  **Publication caveat:** any *head-to-head at fixed rank* claim (as opposed to the magnitude-law
  regression) needs a rank-matched control — handoff/13:46,92 flags this as still owed.

### (d) Eval-harness fairness (`eval_one_gpu.py`) — FAIR, one comparability caveat
- **Identical treatment of every arm.** Every method: fresh base W0 (`from_pretrained`, line 69) +
  `PeftModel.from_pretrained` (line 79), then the SAME adapt legs (8-task CS or gsm8k), SAME F-delta,
  SAME retention suite (BBH+MMLU-Pro core), SAME `gen_cap`, SAME `max_len`, SAME BBH metric fix
  (`bbh_metric_fix`, lines 57-58). **No per-method branch anywhere** — calibration-using and
  calibration-free arms are evaluated bit-for-bit identically. The eval harness itself introduces no
  unfairness; the confound (BLOCK-1) is entirely upstream at init/calibration.
- **Comparability caveat (cross-cutting):** the gen_cap and BBH-normalization fixes (handoff/15, /16,
  commits fe0f9be3/2602f57d) changed retention numbers for the affected models. Rows in
  `campaign_summary.jsonl` written BEFORE those fixes are not directly comparable to post-fix rows
  (matters mainly for the 2nd model / Qwen; Llama-2 BBH fix was a no-op per line 55-56). Ensure the
  final table is built from a single, post-fix evaluation pass — do not mix pre/post-fix rows.

---

## 4. GO / FIX-FIRST / NO-GO — per arm and overall

| Arm | Verdict | Rationale |
|---|---|---|
| lora | **GO** | Calib-free, textbook, the anchor. |
| lorawd | **GO** | Calib-free, the HERO control; unattackable. |
| dora | **GO** | Calib-free, canonical PEFT, params disclosed. |
| milora | **GO** | Calib-free (SVD of W only) ⇒ FIX-1 cannot apply; residual-save correct. |
| clora | **GO** | Calib-free (random P), penalty verified active in the loss. |
| corda | **NO-GO as-is / FIX-FIRST** | Code faithful, but off-curve claim CONFOUNDED (BLOCK-1); already excluded. Re-run under eval-matched calib to promote. |
| sclora | **FIX-FIRST** | Code faithful; −3.3pp off-curve CONFOUNDED (BLOCK-1). Pause + re-run under eval-matched calib. Overrides the "no blocking" per-adapter verdict on the empirical axis. |
| lora_null | **FIX-FIRST** | Code faithful; null-space of wrong distribution CONFOUNDED (BLOCK-1). Pause + re-run. Overrides the "no blocking" per-adapter verdict on the empirical axis. |

### Overall: **FIX-FIRST (conditional GO)**
The **magnitude-law thesis itself is publishable now** on the 5 calibration-free arms
(lora, lorawd, dora, milora, clora) — all code-faithful, fairly evaluated, and immune to the FIX-1
confound. The paper can ship its central claim (magnitude governs retention; LoRA+wd matches milora,
dora, clora) on this backbone.

Any claim that *involves the three calibration-using arms* — especially "data-aware inits are
off-curve / retain worse" — is **NO-GO until the eval-matched calibration re-runs land** (BLOCK-1/2).
The clean path: publish the law on the 5 safe arms, quarantine corda/sclora/lora_null to the FIX-1
sensitivity ablation, and let the eval-matched re-run either put them back on the curve (strengthens
the law: geometry doesn't help even when calibration is fair) or keep them off it (then, and only
then, the off-curve claim is earned).
