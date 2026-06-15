# 08 — FORWARD PLAN: three threads (2026-06-15)

Grounding finding (controlled D1 set, n=7, corrected k1024_v128, use_de×LR): retention is a
near-perfect function of the **Frobenius norm** of ΔW.
  corr(ret, ‖ΔW‖_F)=−0.984 | spectral ‖ΔW‖₂=−0.857 | σ-weighted(≈F∆)=−0.459 | weight-dir μ_E=−0.088
Plain Frobenius BEATS the σ-weighted/F∆ proxy and crushes weight-basis direction. (Within ONE
structure so far — Thread 2 tests whether it's universal.)

Provenance discipline (per 07): tag claims [OURS]/[HYP]/[VERIFY]; do not assert hypotheses as fact.

---

## THREAD 1 — The Frobenius-norm law: magnitude governs CF, weight-direction does not  [OURS]
**Claim:** ‖ΔW‖_F is the dominant (near-sufficient, within fixed structure) predictor of retention;
weight-SVD-basis directional confinement is irrelevant (μ_E r≈−0.09). CLoRA's random-subspace
penalty "works" only by shrinking ‖ΔW‖_F, not by protecting directions (random subspace ⇒ spectrally
neutral, confirmed: out_top≈0.5).

**What we have:** D1 controlled grid (above); CLoRA forensics (spectrally neutral); LoRA anchor.
**Experiments:**
- E1a [now, no GPU]: predictor bake-off across ALL runs + per-method — retention vs {‖ΔW‖_F, ‖ΔW‖₂,
  ACTUAL F∆ (fdelta_token_weighted), μ_E, data-basis leakage}. Which single scalar wins? (extend
  analyze_magnitude_law.py). Headline if Frobenius beats actual-F∆ across the board.
- E1b [running]: λ_E/λ_D sweep = controlled directional intervention at ~fixed structure → confirm
  forcing weight-direction down (at ~fixed magnitude) does NOT improve retention.
- E1c [running]: LoRA+weight_decay sweep reproduces CLoRA's (‖ΔW‖→retention) curve → CLoRA = magnitude
  regularizer (D2-killer).
**Novelty/risk:** "magnitude predicts CF" is known (CLoRA). The CONTRIBUTION is the sharp + controlled
parts: (i) plain Frobenius BEATS F∆/spectral (contradicts the data-weighted-disruption framing);
(ii) weight-direction is causally useless; (iii) CLoRA works by accident. Scope as a controlled/
negative result, not "we discovered magnitude matters." [HYP] until E1a confirms across structures.

---

## THREAD 2 — Does rank/structure matter BEYOND ‖ΔW‖_F? ("rank surprisingly mitigates CF")  [OURS/HYP]
**Observation:** corrected UIO retention rises with k_val (rank-like): k512→k1024→k2048 retains better;
conventional wisdom says more capacity ⇒ MORE forgetting. Surprising → worth nailing.
**The pivotal question:** is this an INDEPENDENT rank effect, or just rank↑ ⇒ ‖ΔW‖_F↓ ⇒ CF↓ (i.e.
Thread 1 in disguise)? Current data is confounded (higher k_val also lowers magnitude).
- [HYP-A, boring]: retention = f(‖ΔW‖_F) regardless of rank ⇒ all ranks fall on ONE curve. Rank only
  helps by lowering ‖ΔW‖_F. (Would COLLAPSE thread 2 into thread 1 — still a clean universal law.)
- [HYP-B, novel]: at MATCHED ‖ΔW‖_F (and/or matched task CS), higher rank STILL retains better ⇒
  spreading/structure matters beyond total energy. (Genuinely interesting, less-claimed.)
**Experiments (NEW GPU runs):**
- E2a: LoRA rank sweep r∈{4,8,16,32,64,128,256}, fixed everything; log CS, ret, ‖ΔW‖_F, ‖ΔW‖₂, F∆.
- E2b (the control): tune LR per rank to MATCH task CS (e.g. CS≈70) across ranks, then compare
  retention. Independent rank effect ⇔ retention differs at matched CS.
- E2c: re-plot retention vs ‖ΔW‖_F colored by rank — ONE curve ⇒ HYP-A; rank-separated ⇒ HYP-B.
- E2d: replicate on UIO k_val sweep (corrected, fixed LR AND fixed-magnitude variants).
- E2e [lit]: confirm/locate the conventional "rank↑⇒forget↑" claim to know if we contradict it.
**Novelty/risk:** ~50/50. If HYP-A, folds into the Frobenius law (still publishable as the unifying
statement). If HYP-B, it's the most novel thread. EITHER outcome is informative; the controlled
matched-CS / matched-‖ΔW‖_F design is what makes it credible.

---

## THREAD 3 — Where the Frobenius law BREAKS: data-basis leakage (the frontier)  [HYP, needs design]
**Question:** at FIXED ‖ΔW‖_F, does aligning the update with the DATA/activation-covariance subspace
increase CF? I.e. is data-basis direction the SECOND-order correction to the first-order Frobenius law?
This is "where the field is at" (CorDA/SC-LoRA say data-basis is THE basis) — our distinct angle is
(a) the relationship to the Frobenius law (first vs second order), and (b) CONTROLLED/continuous
dissociation, which no method-paper provides.
**Build/measure:**
- E3a [built, queued]: forensics_databasis on all checkpoints — retention vs DATA-basis leakage vs
  WEIGHT-basis leakage vs CorDA-basis leakage. Does data-basis leakage predict retention beyond ‖ΔW‖_F?
- E3b: data-vs-weight basis MISALIGNMENT — principal angles between top-C_X subspace and top-W0-SVD
  subspace per layer. Quantify how different "important per data" is from "important per weights".
- E3c [next-level, NEEDS DESIGN]: re-base UIOrthoLoRA on the CorDA decomposition W0·C_X^{1/2} so its
  knobs control DATA-basis direction; then a controlled magnitude×data-direction dissociation — the
  one experiment no one can currently run. Spec the layer change before building.
**Novelty/risk:** "data basis matters" is CorDA/SC-LoRA's (NOT ours). Our defensible angle: is it
first- or second-order vs ‖ΔW‖_F, and the controlled/continuous instrument. Honest: could just
confirm CorDA. Pursue only if E3a shows data-basis adds predictive power BEYOND Frobenius.

---

## Immediate (no-GPU) next action
Run E1a predictor bake-off (retention vs Frobenius / spectral / actual F∆ / μ_E / data-basis) across
ALL runs and per-method. This decides whether the Frobenius law is universal (Thread 1 headline) or
structure-dependent (motivates Thread 2). When phase-2 lands it auto-includes the grid+λ+mag_control.

## The cleanest potential discovery (falsifiable, to defend not assert)  [HYP]
"In PEFT, catastrophic forgetting is governed to first order by a single structure/direction-agnostic
scalar — the Frobenius norm of the adapter update — which dominates spectral concentration, the
data-weighted disruption metric F∆, and weight-basis directional confinement; alignment with the
data-activation subspace is at most a second-order correction." Contrarian (field builds basis/
direction machinery), simple, testable. MUST verify against lit (esp. that F∆/spectral are the
accepted proxies we're beating) before claiming.
