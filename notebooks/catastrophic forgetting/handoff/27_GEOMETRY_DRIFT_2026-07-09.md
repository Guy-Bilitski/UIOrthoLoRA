# Geometry-drift + magnitude-battery findings (2026-07-09)

Computed from the 320 saved adapters (Llama-2-7B; validated ΔW=(α/r)B@A reconstruction — recon
spec_max == recorded dw_sv_max to 4 decimals). Pipelines: geo_drift_phase1.py (base-W SVD ×160),
geo_drift_phase2.py (per-adapter+per-layer battery). Labeled master: results/geo_drift/master_labeled.jsonl
(303 with outcomes; labels from train_registry.jsonl args, NOT the broken run.split/summary "method"=LORA).
Metrics per adapter×layer: fro (true Frobenius ‖ΔW‖_F), spec, stable_rank, eff_rank, e_top/e_bot (ΔW
energy in base-W top/bottom-256 LEFT subspace), ein_top/ein_bot (right/input side), amp_top (‖U_topᵀΔW
V_top‖/‖ΔW‖, MiLoRA-Table-7 style).

## VERDICT (stress-tested to the PI's bar)

**1st-order dominant lever = MAGNITUDE.** retention vs log F_Δ: pooled r=−0.81 (Spearman −0.91,
p≈9e-73), CS −0.79, math −0.94, and WITHIN EVERY METHOD r∈[−0.75 (CorDA), −0.94 (DoRA)], all p<0.02;
survives controlling for rank. Rock-solid, systematic, reproducible.

**2nd-order lever = RANK (honest, modest).** partial r(retention, log r | log F_Δ) = −0.56 (p=5e-19,
CS): at matched magnitude, higher-rank/more-spread updates forget more. Caveat: stable_rank is 0.84
collinear with log r; residual effect marginal once BOTH r and method are controlled → state as
"magnitude first, rank second," not a mystery axis.

**REJECTED claim (do NOT put in paper): "principal-direction concentration is a 2nd-order forgetting
axis."** The preliminary partial r (amp_top/e_top −0.58/−0.51) was carried ENTIRELY by two principal-init
outliers (SC-LoRA, PiSSA). Drop them → amp_top flips to +0.25; on-curve methods → −0.08 to −0.14 (n.s.);
ANCOVA amp_top beyond log F_Δ+method ΔR²=0.0002 (p=0.53). Independent CE-to-base metric agrees: MiLoRA
(minor) ≈ LoRA (mild-top) at matched magnitude. Among on-curve methods, alignment does not predict
retention.

**GEOMETRY'S REAL CONTRIBUTION = a MEASUREMENT/FINGERPRINT TOOL** (constructive framing, no "geometry
doesn't matter"): the SVD-alignment metrics recover each method's init design from the *trained* adapter,
and the signature persists through 3 epochs —
- MiLoRA: only method with e_bot>e_top (minor-singular init), both CS and math.
- SC-LoRA: input-side principal spike ein_top 0.41 (q/k, early layers 0.61–0.75); ERODES with LR
  (ein_top 0.70→0.21, r=−0.96) — quantitatively confirms the SC-LoRA paper's own "constraint erodes with
  steps" limitation.
- CorDA: input-side minor spike ein_bot 0.49 (MLP) + magnitude blow-up (F_Δ 27.5, 92% energy in down_proj).
- PiSSA (n=1): principal (e_top 0.188) — the worst forgetter.
- LoRA / LoRA+wd / CLoRA / DoRA: near the random-alignment baseline.
This explains the two law-outliers (PiSSA worst; SC-LoRA −5.7pp below the CS magnitude law) mechanistically
WITHOUT positing a universal geometric axis.

## Per-method CS signature table (means; lawres = residual vs CS law ret=16.96−6.79·logF_Δ)
| method | n | ret | F_Δ | e_top | e_bot | ein_top | ein_bot | amp_top | stable_rank | lawres |
|---|---|---|---|---|---|---|---|---|---|---|
| CLoRA | 31 | 24.1 | 0.47 | 0.060 | 0.047 | 0.066 | 0.050 | 0.058 | 7.5 | +1.46 |
| CorDA | 20 | 10.3 | 27.5 | 0.078 | 0.048 | 0.041 | 0.494 | 0.049 | 13.0 | −3.03 |
| DoRA | 25 | 22.8 | 0.88 | 0.079 | 0.046 | 0.082 | 0.050 | 0.075 | 5.6 | +3.57 |
| LoRA | 29 | 21.7 | 0.79 | 0.071 | 0.047 | 0.076 | 0.051 | 0.068 | 8.8 | +2.34 |
| LoRA+wd | 41 | 25.3 | 0.43 | 0.072 | 0.048 | 0.086 | 0.050 | 0.072 | 6.5 | +1.73 |
| LoRA-Null | 7 | 24.6 | 0.45 | 0.126 | 0.035 | 0.080 | 0.054 | 0.099 | 6.7 | +0.55 |
| MiLoRA | 20 | 23.6 | 0.59 | 0.067 | 0.115 | 0.077 | 0.115 | 0.066 | 7.7 | +1.62 |
| SC-LoRA | 44 | 9.7 | 1.38 | 0.104 | 0.041 | 0.410 | 0.021 | 0.177 | 19.4 | −5.66 |

## Figure spec (4 panels): A magnitude law scatter; B stress-test bars (amp_top/ein_top collapse+flip
when PiSSA/SC-LoRA dropped, stable_rank stays ≈−0.6); C method-fingerprint heatmap (z-scored metrics ×
methods); D per-layer concentration + SC-LoRA erosion inset.

## Paper actions
- Headline: magnitude 1st-order (within-method), rank 2nd-order (honest, modest).
- Geometry section = measurement tool + outlier explanation, NOT a competing axis.
- Cite our metric recovering MiLoRA-minor / SC-LoRA-principal / CorDA signatures from trained adapters.
- Cross-reference the CE-to-base result (MiLoRA≈LoRA at matched magnitude) as independent confirmation.
