# Commonality decomposition — magnitude / geometry-shape / CE drift

Pool: frozen(deduped) ∩ geometry ∩ CE, n=911. All components are dR2 over
family FE. Blocks: M=[log10 F_delta]; G=[stable_rank, eff_rank, e_top, e_bot,
amp_top]; C=[KL drift].

## FRAMING (binding caveat)
CE/KL drift is measured on base-model text; it is quasi-tautologically close to
retention (both quantify change to base behavior) and is DOWNSTREAM — a
consequence of the update, not a knob. Read CE as the proximal channel /
early-warning signal, magnitude as the upstream controllable variable. Its
'unique share' is diagnostic value, not causal leverage.

## Three-block decomposition (sums to the full-model dR2)

| component | primary | extended (M incl. log spec_max) |
|---|---|---|
| unique: magnitude | +0.033 | +0.033 |
| unique: geometry-shape | +0.031 | +0.032 |
| unique: CE drift | +0.009 | +0.008 |
| shared: M∩G only | +0.052 | +0.052 |
| shared: M∩C only | +0.181 | +0.181 |
| shared: G∩C only | -0.004 | -0.004 |
| shared: M∩G∩C | +0.154 | +0.154 |
| TOTAL (full model dR2) | +0.456 | +0.456 |

Single-block dR2 same-sample: M +0.420, G +0.234, C +0.340.

## Cell-level cluster bootstrap (B=2000, resample recipe cells)

| quantity | point | 95% CI |
|---|---|---|
| unique magnitude | +0.033 | [+0.019, +0.051] |
| unique geometry-shape | +0.031 | [+0.015, +0.051] |
| unique CE drift | +0.009 | [+0.001, +0.019] |
| shared magnitude∩CE | +0.181 | [+0.118, +0.234] |

Ordering unique_M > max(unique_G, unique_C): 1078/2000 bootstrap replicates.

## What does CE capture that magnitude doesn't (and vice versa)? Two-block M vs C.

Pooled (family FE, n=911): R2(M)=+0.420, R2(C)=+0.340,
R2(M+C)=+0.425 -> unique(M)=+0.085,
unique(C)=+0.005, shared=+0.335.

| family | n | R2(M) | R2(C=KL) | R2(M+C) | unique M | unique C | shared |
|---|---|---|---|---|---|---|---|
| lrsw | 180 | 0.785 | 0.739 | 0.816 | +0.077 | +0.030 | +0.708 |
| lrswm | 120 | 0.747 | 0.856 | 0.860 | +0.004 | +0.112 | +0.743 |
| qwsw | 93 | 0.696 | 0.406 | 0.701 | +0.295 | +0.005 | +0.401 |
| qwswm | 99 | 0.679 | 0.627 | 0.717 | +0.090 | +0.038 | +0.589 |
| frc | 275 | 0.861 | 0.737 | 0.882 | +0.145 | +0.021 | +0.716 |
| frm | 144 | 0.863 | 0.814 | 0.876 | +0.063 | +0.013 | +0.800 |

Reading: pooled, magnitude keeps a large unique share beyond CE while CE adds
little beyond magnitude; WITHIN families the balance shifts (05's mediation:
KL is the proximal channel in the math/Qwen arms) — quote both granularities.
