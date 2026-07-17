# 06 — POOLED ΔR² LADDER + GEOMETRY SEED-STABILITY (post-freeze addendum)

`[WRITTEN 2026-07-17 — addendum analyst. Sources: results/*/summary.json,
results/forgetting_merged.jsonl, results/geo_drift/adapter_metrics_merged.jsonl,
results/quarantine_diverged.txt. Scripts: ladder_2026-07-17.py,
seed_stability_2026-07-17.py; outputs: ladder_output_2026-07-17.txt,
seed_stability_output_2026-07-17.txt. Preflight reproduces §18.1 exactly
(n=1035, pooled r=−0.847, all six family cells to 3 decimals) before any new
number is emitted. §18 is unchanged; these numbers live in §19.]`

## Purpose

The freeze supports "magnitude first-order, geometry second-order" with scattered
evidence: pooled partials (§18.4), per-family R² comparisons (§18.5–18.6), and the
mediation battery (05). This doc assembles the claim into ONE nested-regression
table over the frozen pool, and tests whether the geometry effect is seed-stable.

## Post-freeze data note (ledger correction, additive)

Seven runs synced 2026-07-17 12:49, after §18 was cut (1 lrsw, 2 qwsw, 4 qwswm;
names in ladder_2026-07-17.py STRAGGLERS). They are excluded from the primary pool
so all numbers below are §18-commensurable; the current-pool variant (n=1042)
moves nothing (V2: every ΔR² within 0.002 of primary).

## 1. THE LADDER (primary: run-level, family FE, quarantine-included)

**Ladder A — frozen pool ∩ geometry, n=1034** (geometry join 99.9%):

| step | R² | adj R² | ΔR² | F(step) |
|---|---|---|---|---|
| M0 family FE | 0.390 | 0.387 | 0.390 | — |
| M1 + log10 F_Δ | 0.785 | 0.784 | **+0.395** | **1890** |
| M2 + geometry (e_top, log spec_max, stable_rank) | 0.802 | 0.801 | **+0.017** | 29.5 |
| M3 + method dummies (10) | 0.808 | 0.805 | +0.006 | 3.5 |

**Ladder B — ∩ CE, n=911** (CE join: 100% Llama families, 61% Qwen — replicate-seed
CE only, §18.6 disclosure):

| step | R² | ΔR² | F(step) |
|---|---|---|---|
| M0 family FE | 0.375 | 0.375 | — |
| M1 + log10 F_Δ | 0.795 | +0.420 | 1849 |
| M2 + KL (CE drift) | 0.800 | +0.005 | 22.0 |
| M3 + geometry | 0.824 | +0.024 | 41.5 |
| M4 + method dummies | 0.828 | +0.004 | 2.2 |
| [alt] M1 + geometry, no CE | 0.815 | +0.020 | 33.1 |
| [alt] M0 + KL only, no magnitude | 0.715 | +0.340 | 1077 |

Standardized betas (Ladder A final pre-method model): log F_Δ **−0.744** vs
e_top −0.081, log spec_max +0.087, stable_rank −0.138. The magnitude coefficient
is 5–9× any geometry coefficient in standard-deviation units.

**Reading.** (i) Magnitude explains ~23× the variance geometry does at the margin
(ΔR² 0.395 vs 0.017) and ~66× method identity. (ii) Geometry and method identity
are *jointly significant but bounded* — consistent with §18.4's "real, bounded,
second-order", now as one number. (iii) KL adds only +0.005 after magnitude at
run level: CE drift and F_Δ carry largely shared signal here. This does NOT
contradict 05's family-level result that KL alone beats log F_Δ within 5/6
families — as a single pooled predictor magnitude wins (0.420 vs 0.340 same-sample);
05's mediation (KL as the *channel*) is about structure, not marginal variance.
Quote both, at their own granularities.

## 2. ROBUSTNESS (variants, ladder_output_2026-07-17.txt)

| variant | n | ΔR²(F_Δ) | ΔR²(geo) | ΔR²(method) |
|---|---|---|---|---|
| V1 quarantine-excluded | 1002 | +0.328 | +0.010 | +0.007 |
| V2 current pool (stragglers in) | 1041 | +0.392 | +0.017 | +0.006 |
| V3 seed-averaged cells | 343 | +0.370 | +0.023 | +0.004 (F=0.9, n.s.) |
| V4 no family FE | 1034 | +0.717 | +0.010 | +0.012 |

V4's M1 R² = 0.717 independently reproduces §18.6's frozen X-axis number (0.72,
n≈1034) — the ladder is anchored to the freeze. The ordering
magnitude ≫ geometry ≥ method holds in every variant; at cell level (V3) method
identity is not significant at all once magnitude+geometry are in.

## 3. GEOMETRY SEED-STABILITY (seed_stability_output_2026-07-17.txt)

Partial r(x, ret | family FE + log F_Δ), per seed (s42/43/44 full, s45 partial
n=78, s46 skipped n=9):

| term | s42 | s43 | s44 | s45 | mean±sd | sign-consistent |
|---|---|---|---|---|---|---|
| log F_Δ (ref) | −0.814 | −0.825 | −0.781 | −0.890 | −0.828±0.046 | 4/4 |
| e_top | −0.214 | −0.129 | −0.129 | −0.401 | −0.218±0.128 | 4/4 |
| log spec_max | +0.119 | +0.027 | −0.030 | +0.304 | +0.105±0.146 | 3/4 |
| stable_rank | −0.189 | −0.195 | −0.286 | −0.581 | −0.313±0.184 | 4/4 |

**Reading.** The magnitude partial is seed-invariant (sd 0.046). The two geometry
terms with a consistent story (e_top: touching the base top-subspace costs
retention; stable_rank: broader updates cost retention) are sign-stable 4/4 but
3–8× smaller and noisier. log spec_max — the term behind §18.4's pooled +0.117 —
is the least stable (3/4, crosses zero at s44): quote spec-based direction effects
with a seed-variability caveat; prefer e_top/stable_rank as the geometry exhibits.
(The +0.117 of §18.4 vs +0.047 here reflects control sets: §18.4/A3 n=1018 without
the other geometry covariates; not a conflict, different regression.)

## 4. WHAT THIS ADDS TO THE PAPER

- One table (Ladder A/B) replaces three scattered exhibits as the quantitative
  backbone of the title claim. Suggested placement: main text, after the relation
  figure; robustness variants to appendix.
- The honest headline sentence: "Conditional on model×task family, effective update
  magnitude explains an additional 39.5% of retention variance (F≈1.9k); update
  geometry adds 1.7% (F≈30); method identity 0.6% (F≈3.5)."
- Seed-stability closes a reviewer hole: the second-order geometry effect is not a
  seed artifact, and we now know which geometry metric NOT to headline (spec_max).
