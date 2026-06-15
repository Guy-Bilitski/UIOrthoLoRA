# 06 — INSIGHTS (live, 2026-06-14 night)

## THESIS (sharpened by the D1 verdict, 2026-06-15)
Retention is governed by **σ-weighted update magnitude** (each perturbed direction weighted by its
singular-value / importance) — NOT by directional orthogonality. Equivalently: perturbing low-σ
(tail) directions forgets MILDLY even at large magnitude; perturbing high-σ (preserved) directions
forgets SEVERELY. The orthogonal-subspace methods reduce forgetting by cutting σ-weighted magnitude,
not via their advertised directional mechanism.

  retention ≈ f( σ-weighted magnitude ) ≈ f( sigma_resp · ‖ΔW‖ ) ≈ f( F-delta )

### D1 VERDICT (7 pts, corrected k1024_v128, use_de×LR ladder, same FAST scale):
  corr(ret, MAGNITUDE dw_sv_max) = -0.857   |   corr(ret, DIRECTION μ_E) = -0.088
  => magnitude FIRST-order, direction SECOND-order (near-zero standalone).
  Matched-magnitude pairs: mostly flat across μ_E (mag~2.5: clean 26.4 vs leaky 27.6) with a small
  hint of a direction effect at mid magnitude (mag~12: clean 25.7 vs leaky 23.1) -> direction is
  second-order, NOT zero. λ-penalty sweep (running) pins this down.
  NUANCE that refines the earlier "preserved-only" idea: clean runs have out_top=0 (perfectly
  tail-confined) YET retention still falls with magnitude (26.4 @mag2.5 -> 22.6 @mag16.8). So tail
  magnitude ALSO forgets — mildly. Hence σ-WEIGHTED magnitude (both bands, importance-weighted),
  not preserved-band-only.

### Novelty honesty
σ-weighted magnitude ≈ CLoRA's F-delta (known to predict forgetting). The NOVEL contribution is the
NEGATIVE result + reattribution: directional/orthogonality control (the design principle behind
CLoRA/O-LoRA/OPLoRA/UIOrthoLoRA) is near-irrelevant (μ_E r=-0.09); these methods work via magnitude.

Evidence (suggestive, confounded by mixed scale + old runs lacking preserved_F):
- legacy major-term (preserved Frob ~45–66): retention CATASTROPHIC (3.9–18.5)
- corrected tail-confined (preserved ~0): retention ROBUST (23–27) even at ‖ΔW‖ up to ~14
- CLoRA/LoRA (spectrally neutral → ~½ magnitude in top): retention scales with magnitude (22.5↔25.6)
- pooled corr(ret, dw_sv_max) only −0.52 BECAUSE tail-confined high-magnitude updates DON'T forget;
  the strong forgetting was preserved-subspace perturbation.

Why this is better than "magnitude not direction":
- It's a SINGLE measurable scalar that should collapse ALL methods (LoRA/CLoRA/UIO/MiLoRA) onto ONE
  curve — a method-independent law.
- It explains CLoRA (random-subspace → spectrally neutral → ½ magnitude hits top → forgets ∝ mag),
  the major-term catastrophe, and corrected-UIO's robustness — all at once.
- It reframes the field's "be orthogonal" intuition as "minimize magnitude in the important subspace"
  (which orthogonality is one — not the only, and for CLoRA not even an effective — way to achieve).

## THE CLEAN TEST (when Phase-2 lands — all runs log preserved_F + forensics at one scale)
Plot retention vs preserved_F (and vs sigma_resp·‖ΔW‖) across LoRA/CLoRA/corrected-UIO/legacy-UIO.
- ONE curve, method-independent  -> the law holds (meaningful).
- retention residual after preserved_F still depends on direction -> direction has independent effect.
- preserved_F no better than total ‖ΔW‖  -> collapses to known magnitude result (modest).
TODO: extend analyze_d1_d2.py with corr(ret, preserved_F) and corr(ret, sigma_resp*dwF) pooled + per-method.

## PARETO (CS vs retention; fast+full mixed -> indicative)
Frontier: CLoRA-k1024 (79.9,25.6) — CLoRA-k2048 (65.4,26.8) — UIO-corr a5_k2048 (64.4,26.9).
UIO sits ON the frontier ONLY at the high-retention corner; CLoRA owns the high-CS corner (~80).
Adaptation lever: full-rotation (k_vec→k_val) buys CS at LOW magnitude (uioT_k410: CS72.7@mag9) →
stays under the preserved-magnitude budget → should push UIO's frontier up. kval_kvec_grid tests it.

## OPEN QUESTIONS / NEXT INSIGHT HUNTS
- Does direction add ANYTHING beyond preserved_F? (λ-penalty grid = clean intervention.)
- Is the preserved-magnitude→retention curve LINEAR / THRESHOLD / LOG? (need same-scale sweep.)
- Do MiLoRA/PiSSA (minor/principal-band adapters) sit on the same curve? (port + forensics.)
- Full-scale unified retention re-eval to kill the scale confound before any headline number.
