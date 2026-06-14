# 05 — DISCOVERY PLAN (re-plan, 2026-06-14)

Supersedes the "beat CLoRA" framing in `02_EXPERIMENT_PLAN.md`. We are NOT trying to win a
benchmark. We are using **corrected UIOrthoLoRA as a controllable instrument** + thermometers
+ method-agnostic forensics to answer a scientific question about PEFT & catastrophic forgetting.

## The pivot (why this works)
- UIOrthoLoRA loses the head-to-head with CLoRA and that won't change (ceiling ~74 CS, ties at best).
- BUT it is the ONLY adapter with explicit knobs to **independently set** the two candidate drivers
  of forgetting: **directional preserved-subspace leakage** (`use_de`, `k_vec`, + a continuous
  λ_E/λ_D penalty to build) and **update magnitude** (`learning_rate`, `initial_sigma`).
- That makes it an apparatus for **controlled interventions**, not just correlational observation.
- The major-term bug fix (`drop_major`) is what makes the instrument *valid*: pre-fix the
  "preserved" subspace was 99.8% perturbed, so the thermometers lied; post-fix `use_de=0` is a true
  zero-directional-leakage baseline. (See finding #4, `test_a5_drop_major.py`, `leakage.py` full-ΔW.)

## The central question
**What actually governs the adaptation↔forgetting tradeoff in PEFT — magnitude or spectral
direction — and do existing methods' stated mechanisms (CLoRA orthogonality, MiLoRA minor-band,
PiSSA principal-band) work the way they claim?**

## Why correlation isn't enough (the confound — proven 2026-06-14)
CLoRA forensics (k128→k2048): retention correlates with EVERY metric at r≈−0.97
(out_top_0.5, out_top_0.05, sigma_resp, dw_F, sv_max all collinear — as k rises, magnitude
shrinks AND spectral location drifts together). **The k-sweep cannot separate them.** Only the
instrument's independent knobs (hold magnitude fixed, vary direction) — and the λ penalty — break it.

## Candidate discoveries (ranked)

### D1 — FLAGSHIP (causal): "Magnitude, not spectral direction, drives forgetting."
Intervention with corrected UIOrthoLoRA: hold ‖ΔW‖ fixed, vary directional leakage (use_de off→on,
then continuously via λ). Measure retention.
- Prelim signal: clean arm (μ_E≡0) still forgets at high magnitude (ret→4); leaky arm (μ_E~1.8)
  retains when gates brake magnitude (ret~25).
- PROVES IT: at matched magnitude, retention ≈ invariant to directional leakage.
- KILLS IT: leaky arm forgets more at matched magnitude → direction has independent effect.
- Payoff: refutes the directional-orthogonality design principle; says "suppress effective update
  magnitude in high-σ directions." Contrarian + actionable.

### D2 — forensic audit: "CLoRA reduces forgetting by magnitude, not by protecting directions."
KEY FACT (verified 2026-06-14, repo `train_cs.py:CLoRARegularizer` + CLoRA paper one-stage setting +
Gemini): in the ONE-STAGE setting we run, CLoRA's anchor P is a **frozen RANDOM orthonormal** matrix
(penalty λ(‖AᵀP_A‖²+‖BᵀP_B‖²)). A random subspace is uncorrelated with knowledge-bearing directions
(overlap with W's top-r subspace ≈ k/d, tiny), so it cannot be "protecting" them. What blocking a
random k-subspace DOES is remove k dims of update freedom → shrink ΔW's effective magnitude as k grows.
Forensics confirm: CLoRA ΔW spectrally neutral (out_top≈0.5 ∀k); as k↑, ‖ΔW‖↓ (30.9→18.9), CS↓, ret↑.
- SCOPE: one-stage forgetting only — NOT CLoRA's sequential-CL mode (there P = prior-task params).
- We are re-attributing CLoRA's mechanism, NOT disputing that it works (it does).
- TODO before publishing: confirm exact penalty eq against the CLoRA paper (Gemini = secondary src).

### D2-KILLER — the money experiment: "magnitude alone reproduces CLoRA."
If D2 holds, plain LoRA + a SUBSPACE-FREE magnitude knob (AdamW weight decay on the adapter) should
reproduce CLoRA's (‖ΔW‖ → retention) curve. Overlay LoRA-wd sweep vs CLoRA k-sweep on the
(magnitude, retention) plane (same FAST retention scale). SAME curve ⇒ CLoRA's random-subspace
apparatus adds nothing beyond magnitude control. Queued: jobs/mag_control_lora.txt (wd∈{.01,.05,.1,.3,1})
+ jobs/reeval_fast_baselines.txt (CLoRA re-eval at matched scale). Added `--weight_decay` to train_cs.py.

### D3 — second study: "Where in the spectrum you adapt sets the tradeoff."
Sweep `k_val` (which singular band is trainable) on one backbone with the thermometers reading where
ΔW lands. Adjudicates MiLoRA (minor) vs PiSSA (principal) vs CLoRA (orthogonal) — a live debate —
under controlled conditions. Interesting regardless of D1's outcome.

### D4 — supporting: unifying weight-space predictor of retention across methods.
A single scalar (e.g. σ²-weighted preserved-subspace response) predicting retention across
LoRA/CLoRA/UIO/MiLoRA on one curve. RISK: collinear / may re-derive ‖ΔW‖ & F∆. Use as a figure, not
the headline.

## Experimental program

**Phase 1 (now) — establish D1 + D2 core.**
1. LoRA r32 forensics (baseline "neutral" anchor). [LoRA retraining on GPU5]
2. Add in-process method-agnostic forensics to `uio_inprocess` (call `forensics.module_forensics`
   on the live model) so UIO runs are directly comparable to CLoRA/LoRA on the same spectral axes.
3. Controlled D1 grid, ALL corrected (`drop_major=1`): clean (use_de=0) vs leaky (use_de=1) ×
   magnitude ladder (LR ∈ {1e-3,3e-3,1e-2,2e-2}), k1024_v128. Read CS, retention, thermometers,
   forensics, ‖ΔW‖. The matched-magnitude clean-vs-leaky pairs are the causal contrast.
4. n=3 seeds (42/43/44) on 3 anchor points for error bars.

**Phase 2 — sharpen + breadth.**
5. Wire continuous λ_E/λ_D leakage penalty (B2) into training → continuous directional-leakage knob
   at fixed magnitude = the cleanest D1 intervention.
6. `k_val` spectral-band sweep (D3).
7. Port + forensics MiLoRA / PiSSA / LoRA-Null / SC-LoRA (recipes in memory `method-port-recipes`).

**Phase 3 — robustness + write.**
8. Generalize: Llama-3-8B and/or RoBERTa-GLUE (the paper's own diagnostic setting; the "watch both
   sides" leakage-penalty result, exp B3).
9. Figures, error bars, paper draft.

## Phase-1 facts established (2026-06-14 eve)
- BASELINE / retention ceiling = **26.0** (base Llama-2-7B, answer-only BBH 33.1 + MMLU-Pro 19.0).
  Good configs (CLoRA-k2048 25.7, UIO-A5 ~26.9) essentially TIE it → forgetting <1 pt for strong
  configs (noise). Retention CANNOT go higher; the attractive lever is ADAPTATION at the ceiling.
- HYPERPARAM AUDIT: LR (3e-4→5e-2) and k_val (256→4096) thoroughly swept. **k_vec (rotation rank)
  UNDER-explored** (mostly ratio 0.125). Full-rotation `uioT_k410` (k_vec=k_val=410, LEGACY) =
  **CS 72.7 @ ret 25.0 @ ‖ΔW‖ₘₐₓ 9** — best adaptation-per-magnitude in the whole set.
  => jobs/kval_kvec_grid.txt: JOINT 2-D (k_val × k_vec) map, drop_major=1. k_val = which spectral
  BAND is adapted (= D3) ; k_vec = rotation freedom within it. They INTERACT (small band + full
  rotation beat large band + low rotation on CS-per-magnitude). Hypothesis: rotation freedom buys
  high CS at low magnitude → high CS at the retention ceiling; band size trades adapt vs preserve.
- High LR raises CS to ~74 but only at retention-destroying magnitude (dwSVmax 200+). So magnitude,
  not LR per se, is the binding constraint — consistent with D1.

## Honest risk register
- D1 might come back "direction also matters" → still publishable (a quantified decomposition).
- D2's "neutral" claim needs the LoRA anchor; CLoRA's target is an activation context subspace, not
  W's top singular subspace — state precisely, don't conflate.
- If everything just reduces to "‖ΔW‖ predicts forgetting" (already ≈ CLoRA's F∆), there's no A*
  discovery — the novelty MUST be the causal direction-vs-magnitude dissociation (D1) and the
  mechanism audit (D2). Judge honestly after Phase 1.
