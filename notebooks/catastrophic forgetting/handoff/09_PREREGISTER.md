# 09 — PRE-REGISTRATION: what's running, predicted outcome, and what EACH outcome means

Purpose: before results land, commit to (a) the question, (b) the prediction, (c) the interpretation
of BOTH outcomes. Every row must be **bidirectionally informative** — if an experiment is only
meaningful one way, it's a weak experiment. Status as of 2026-06-15 ~13:40.

| # | Experiment (status) | Thread | Question | Metric / comparison | PREDICTED | If CONFIRMED | If REFUTED | grade |
|---|---|---|---|---|---|---|---|---|
| 1 | CLoRA fast re-eval ×5 (**2/5 done**, ~2.5h) | T1 | Does ret~‖ΔW‖_F hold ACROSS LoRA/CLoRA/UIO on ONE curve at matched eval scale? | pooled corr(ret,‖ΔW‖_F), fast scale; do arches interleave? | pooled r tightens from −0.79(conf.) → ~−0.9; interleave | Frobenius law is a CROSS-ARCH first-order constraint (anchor) | arches sit on SEPARATE curves → structure matters beyond magnitude (bigger T2 story) | ANCHOR (near-circular; necessary, not the headline) |
| 2 | **Data-basis forensic ×6** (queued, ~3h) | T1/T3 | Does the DIRECTIONAL norm ‖ΔW·C_X^{1/2}‖_F predict forgetting BETTER than raw ‖ΔW‖_F? | corr(ret, data_resp) vs corr(ret, ‖ΔW‖_F), pooled | directional ≥ raw (direction modulates slope) | **THE non-circular headline**: direction-in-data-basis is the 2nd-order term — carry-paper plot | forgetting is direction-AGNOSTIC magnitude; CorDA/SC-LoRA's basis is NOT the operative variable (contrarian, also publishable) | ★ HEADLINE |
| 3 | **LoRA+weight_decay ×5** (running, ~this PM) | T1 | Does magnitude-matched L2 retain AS WELL as CLoRA at the same ‖ΔW‖_F? | LoRA-wd vs CLoRA on (‖ΔW‖_F, ret) plane | same curve (L2 reproduces CLoRA) | CLoRA's random-subspace adds NOTHING beyond magnitude (the reattribution) | CLoRA retains better at matched norm → structure matters independently; "illusion" framing COLLAPSES (sharper, different paper) | ★ FALSIFIER |
| 4 | grid k410 lr2e2 + clean cells (running, ~this PM) | T2/instr | Does drop_major help or HURT CS, and does it depend on rotation k_vec? | corrected vs legacy CS across k_vec (grid_k410 lr1e2 already: corrected 48 << legacy 72.7) | interaction: helps at low k_vec, hurts at full k_vec — OR lr1e2 artifact (lr2e2 recovers CS) | "bug fix improves both axes" is CONFIG-DEPENDENT (honesty correction) | lr2e2 recovers CS→70 ⇒ it was an LR artifact, drop_major is fine | DISAMBIGUATE |
| 5 | LoRA rank sweep r∈{4..256} (queued, overnight) | T2 | At fixed LR, does higher rank retain better, and is it explained by ‖ΔW‖_F? | ret vs ‖ΔW‖_F colored by rank — ONE curve or rank-separated? | likely HYP-A (folds into Frobenius law) | rank only helps via magnitude ⇒ unifies field under Frobenius law | HYP-B: rank-separated ⇒ genuine structural effect of the spectral tail (novel) | KINGMAKER-prereq (feeds E2b matched-CS) |
| 6 | norm_trace in rank sweep + UIO (overnight) | cliff | Does ‖ΔW‖_F keep INFLATING after task loss plateaus? | d‖ΔW‖_F/dt vs loss trajectory | uncertain (AdamW+wd may NOT inflate) | motivates early-stop / "magnitude tax after task learned" story | norm plateaus with loss ⇒ no free stopping lunch; kills that sub-idea | ASSUMPTION-CHECK |
| 7 | λ_E/λ_D sweep ×5 (running, ~this PM) | T1 | At ~fixed structure, does forcing weight-basis direction (μ_E) DOWN improve retention? | ret vs λ, controlling resulting μ_E & ‖ΔW‖_F | weak/none at fixed magnitude (weight-dir irrelevant) | weight-basis direction is CAUSALLY irrelevant (controlled D1) | λ improves ret at fixed magnitude ⇒ weight-direction matters causally | CONTROLLED-D1 |

## Methodology fixes from Gemini review (2026-06-15) — applied before runs land
- **#2 covariance BUG FIXED (critical).** `forensics_databasis.py` was computing C from the COMMONSENSE
  (fine-tuning TASK) prompts = C_task; forgetting is out-domain, so the predictor must use C_retain.
  Now `--cov_source retain` (DEFAULT) loads **MMLU-Pro** (our retention benchmark) — validated it loads.
  Output now `databasis_<run>_<cov_source>.json`. The queued GPU5 runs exec the file at runtime ⇒ they
  use the fixed retain covariance. ZERO-SUM FOLLOW-UP (Gemini): also run `--cov_source task` on the same
  checkpoints to show C_X-alignment↓ ⇔ C_task-alignment↑ (capacity trade-off) — queue when a GPU frees.
- **#3 L2-trap verified clean.** AdamW weight_decay touches ONLY lora_A/lora_B (base+layernorm frozen,
  not in optimizer) — confirmed empirically. So the magnitude-matched L2-vs-CLoRA comparison is valid.
- **#5 "diffusion" test (Gemini).** Track σ₁=‖ΔW‖₂ (we log dw_sv_max) ALONGSIDE ‖ΔW‖_F across the rank
  sweep: if higher rank retains better by spreading the update (lower σ₁ at matched ‖ΔW‖_F), that's the
  mechanism. The analysis must plot σ₁ vs rank at matched Frobenius/CS.
- Nomenclature: keep generic protocol-based run/checkpoint names (lora_rN, clora_kN, grid_…) for clean
  cross-architecture aggregation — already followed.

## RESULTS vs PREDICTIONS (first wave, 2026-06-16 ~00:30)
- **#2 directional-vs-raw norm → CONFIRMED but MARGINAL.** corr(ret,directional ‖ΔW·C_retain^½‖)
  beats corr(ret,raw ‖ΔW‖_F): −0.809 vs −0.735 (n=13), −0.793 vs −0.768 (n=8 deduped fast). Direction
  DOES modulate the slope, but the margin is small → headline is present, not yet convincing. Firming up
  with C_task zero-sum (GPU0, running) + rank-sweep databasis. RISK: if it stays Δ~0.02 it's a weak headline.
- **#1 scale-unified ONE curve → REFUTED (and that's GOOD).** Pooled cross-arch/structure
  corr(ret,‖ΔW‖_F) is only −0.465 (within fixed structure −0.93..−0.98). So raw Frobenius is NOT a clean
  cross-structure constraint → structure/direction matters beyond raw magnitude. This AVOIDS the
  "it's all just ‖ΔW‖_F = known F∆ folklore" collapse (the pre-reg red flag). Consistent with #2.
- **#4 drop_major × rotation → NOT an LR artifact; rotation has an OPTIMUM.** lr2e2 made CS WORSE (39<48),
  so corrected full-rotation-dE1's low CS is real, not LR. But the rotation ladder at k410 (dE1,lr1e2):
  v51→CS23, v205→**CS71/ret25.7**, v410→CS48; and clean full-rot **v410 dE0→CS69/ret26.3 @ spec8.2**.
  ⇒ the CORRECTED instrument DOES reach strong Pareto (CS~70 @ ret~26, BETTER retention than legacy
  uioT_k410 72.7/25) — at the right rotation/use_de. The first CS48 point was a bad cell, not a limit.
- **#5 rank → PARTIAL, deflating the surprise for LoRA.** r4 ret25.36, r8 25.32, r32 23.4 ⇒ higher LoRA
  rank → MORE forgetting (CONVENTIONAL), OPPOSITE to "rank mitigates CF". The surprise was likely
  UIO-k_val-specific (more low-σ directions at lower magnitude), NOT general. Await full sweep + E2b matched-CS.

## Self-audit: is each row meaningful either way?
ALL 7 are bidirectionally informative ✓ — every CONFIRMED and REFUTED cell is a publishable statement,
not a null. That is the check the user asked for ("are we on the right track to MEANINGFUL results").

## Which rows actually carry the paper
- **#2 (directional vs raw norm)** is THE headline — the non-circular claim. Everything else supports it.
- **#3 (L2 falsifier)** decides whether the story is "structure=magnitude" (reattribution) or "direction
  matters independently" (different, sharper paper). Either is a paper; we just need to KNOW which.
- **#1** is an ANCHOR, NOT a discovery (r~−0.9 is near-true by construction per Claude-web critique).
  Do not headline it. It only earns its place if it interleaves cleanly (rules out a confound).
- **#5/#7** are the controlled/causal backbone (rank-beyond-magnitude; direction-beyond-magnitude).

## Red flags to watch (would mean we are NOT on track)
- If #2 shows directional ≈ raw AND #3 shows L2 ≈ CLoRA AND #5 is HYP-A AND #7 is null ⇒ the WHOLE
  story collapses to "forgetting = ‖ΔW‖_F, full stop" = CLoRA's F∆ folklore = NOT novel. In that case
  the only contribution left is the controlled/causal METHODOLOGY + the CLoRA-random-subspace audit
  (a modest measurement paper). We should pre-commit to that honest downgrade if the data says so.
- The data-driven REGULARIZER (idea 1: soft data-covariance directional penalty) is the constructive
  follow-up IF #2 confirms — it is NOT yet run; it's the "so what do we DO about it" of the finding.
