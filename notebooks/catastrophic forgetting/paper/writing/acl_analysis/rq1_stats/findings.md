# RQ1 statistics pass — findings (2026-07-30)

RQ1: when retention-aware adapters are compared under one protocol, swept over
tasks, learning rates, and seeds, are any significantly different?

All numbers below verified by `verify_rq1_stats.py` (independent recompute:
ALL OK, see `verify_log.md`). Pool conventions identical to the frozen layer;
every script preflights key_numbers.md section 18.1 (n=1035, r=-0.847).

## 1. Head-to-head vs LoRA+wd, exact p + Holm (`head2head_corrected.*`)

26 method x family comparisons at best-adaptation operating points; 25
testable (PiSSA/llama_math is 1-seed vs 3, delta reported without a test:
dRet -26.3, dAdapt -17.1).

- **Retention: after Holm across all 25, NO method is significantly better
  than LoRA+wd anywhere. Two are significantly worse** (CLoRA llama_cs
  -4.25 pp, p_holm=0.035; CLoRA qwen_math -7.56 pp, p_holm=0.024).
- **Adaptation: after Holm, no method beats LoRA+wd; five are significantly
  worse** (llama_cs: LoRA -2.58, LoRA-Null -2.89, CLoRA -3.46, MiLoRA -4.56;
  llama_math: CLoRA -6.14).
- **The one frozen-layer exception dissolves under correction:** SC-LoRA on
  Qwen-math GSM8K +8.26 pp has raw p=0.035 (3 seed pairs) but Holm within
  family p=0.21, Holm across all p=0.46. Report as suggestive, not
  significant. Its retention delta there is +0.17 (p=0.63).

## 2. TOST equivalence at matched magnitude (`tost_offsets.*`)

Model: ret ~ log10 F_delta + method dummies (ref LoRA+wd), CR1 cluster-robust
SE at recipe-cell level, deduped n=1034 pool, G=343 cells (pooled model with
family FE). Equivalence at margin m iff 90% CI inside (-m, +m).

- Pooled: **7/9 methods statistically equivalent to LoRA+wd within +/-3 pp**
  at matched magnitude; 3/9 already at +/-2 pp (LoRA+wd r16, MiLoRA,
  MiLoRA+wd). No method reaches +/-1 pp (power, not evidence of difference).
- The two non-equivalences are one-sided and BELOW: **PiSSA -7.1 pp
  [90% CI -9.0, -5.3] and SC-LoRA -4.1 pp [-5.6, -2.6]** retain less than
  LoRA+wd even at matched magnitude (pooled; per-family: pissa frc -5.9 /
  frm -11.2, sclora frc -3.6 / frm -2.9, consistent with the frozen
  section 18.4 offsets).
- Per-family at +/-2 pp: 14/39 offsets equivalent; every Qwen offset has a
  CI too wide to bound (see power notes), Llama grids give the tight ones.

## 3. Power / MDE (`power_notes.*`)

Exact MDE (two-sided paired t, alpha=.05, power=.8) at the observed common
seeds and empirical SD of paired deltas:

| family | median cell SD (ret) | median MDE | max MDE |
|---|---|---|---|
| llama_cs | 0.37 | 2.7 pp | 5.0 pp |
| llama_math | 0.75 | 3.4 pp | 5.1 pp |
| qwen_cs | 0.85 | 1.7 pp | 49.9 pp |
| qwen_math | 0.67 | 4.5 pp | 33.7 pp |

The head-to-head battery can only detect ~2-5 pp differences at typical
cells (worse where a method's best cell is seed-unstable, e.g. SC-LoRA).
So: (a) "n.s." never certifies equality — the TOST bounds above are the
positive statement; (b) sub-2pp method differences are undetectable at
3-5 seeds under this protocol.

## The RQ1 sentence for the paper

Under a common protocol with per-method learning-rate sweeps, no adapter is
significantly better than LoRA+wd in retention or adaptation at its best
operating point (25 Holm-corrected paired comparisons; 2 retention and 5
adaptation comparisons significantly worse); at matched update magnitude,
seven of nine methods are statistically equivalent to LoRA+wd within +/-3 pp
(TOST, cluster-robust), the exceptions (PiSSA, SC-LoRA) falling below; the
design can detect ~2-5 pp effects, so differences smaller than that remain
open, and method identity in any case explains only DR^2 = +0.006 of
retention variance once magnitude is controlled (frozen ladder).
