# Independent numerical verification — ladder / seed_stability / ds284b (2026-07-18)

Verifier: fresh loader written from the documented filter spec; OLS via
`numpy.linalg.lstsq` (QR/SVD path — NOT normal equations); correlations via
`scipy.stats.pearsonr` / `spearmanr` (tie-aware). numpy 2.3.5, scipy 1.16.3.
Scripts: `verify_part1.py`, `verify_part2.py`; raw outputs `part1_out.txt`, `part2_out.txt`
(all in this scratchpad). No repo files modified.

## 1. Verdict table (committed value vs independent recomputation)

### Preflight (ladder_output lines 5–12)
| quantity | committed | recomputed | verdict |
|---|---|---|---|
| pool n (primary) | 1035 | 1035 | CONFIRMED |
| pooled r | −0.847 | −0.8466 | CONFIRMED |
| lrsw (n, r) | 180, −0.886 | 180, −0.8862 | CONFIRMED |
| lrswm | 120, −0.865 | 120, −0.8646 | CONFIRMED |
| qwsw | 151, −0.840 | 151, −0.8399 | CONFIRMED |
| qwswm | 164, −0.830 | 164, −0.8303 | CONFIRMED |
| frc | 276, −0.928 | 276, −0.9279 | CONFIRMED |
| frm | 144, −0.929 | 144, −0.9290 | CONFIRMED |
| geo_pool n | 1034 | 1034 | CONFIRMED |
| ce_geo_pool n | 911 | 911 | CONFIRMED |

### Ladder A (n=1034)
| step | committed R² (dR², F) | recomputed | verdict |
|---|---|---|---|
| M0 fam | 0.390 | 0.3898 | CONFIRMED |
| M1 +logfd | 0.785 (+0.395, F=1890.3) | 0.7852 (+0.3954, F=1890.30) | CONFIRMED |
| M2 +geo | 0.802 (+0.017, F=29.5) | 0.8023 (+0.0171, F=29.5) | CONFIRMED |
| M3 +method | 0.808 (+0.006, F=3.5) | 0.8083 (+0.0060, F=3.5) | CONFIRMED |
| std betas | logfd −0.744, e_top −0.081, lspec +0.087, srank −0.138 | −0.7436, −0.0809, +0.0868, −0.1382 | CONFIRMED |

### Ladder B (n=911)
| step | committed | recomputed | verdict |
|---|---|---|---|
| M0 | 0.375 | 0.3750 | CONFIRMED |
| M1 | 0.795 (+0.420, F=1849.1) | 0.7948 (+0.4198, F=1849.1) | CONFIRMED |
| M2 +KL | 0.800 (+0.005, F=22.0) | 0.7997 (+0.0049, F=22.0) | CONFIRMED |
| M3 +geo | 0.824 (+0.024, F=41.5) | 0.8240 (+0.0243, F=41.5) | CONFIRMED |
| M4 +method | 0.828 (+0.004, F=2.2) | 0.8278 (+0.0038, F=2.2) | CONFIRMED |
| [alt] M1+geo | 0.815 (+0.020, F=33.1) | 0.8151 (+0.0204, F=33.1) | CONFIRMED |
| [alt] M0+KL | 0.715 (+0.340, F=1076.5) | 0.7147 (+0.3397, F=1076.5) | CONFIRMED |
| std betas | −0.531 / −0.192 / −0.054 / +0.055 / −0.201 | −0.5311 / −0.1916 / −0.0542 / +0.0548 / −0.2009 | CONFIRMED |

### Variants
| variant | committed key line | recomputed | verdict |
|---|---|---|---|
| V1 (n=1002) | M0 0.463, M1 0.791 (+0.328), M2 0.801, M3 0.808 | 0.4633 / 0.7910 (+0.3277) / 0.8011 / 0.8084 | CONFIRMED |
| V2 (n=1041) | 0.394 / 0.787 / 0.804 / 0.810 | 0.3945 / 0.7868 / 0.8037 / 0.8098 | CONFIRMED |
| V3 cells (n=343) | 0.444 / 0.813 (+0.370, F=664.5) / 0.836 (+0.023) / 0.840 | 0.4436 / 0.8131 (+0.3695, F=664.5) / 0.8358 (+0.0227) / 0.8399 | CONFIRMED |
| V4 no-FE | 0.717 / 0.727 / 0.739 | 0.7168 / 0.7271 / 0.7390 | CONFIRMED |

### Partials, addenda
| quantity | committed | recomputed | verdict |
|---|---|---|---|
| partial r(e_top) | −0.173 | −0.1727 | CONFIRMED |
| partial r(lspec) | +0.047 | +0.0469 | CONFIRMED |
| partial r(srank) | −0.225 | −0.2247 | CONFIRMED |
| Add.1 fam+geo R² | 0.741 (+0.351) | 0.7406 (+0.3508) | CONFIRMED |
| Add.1 commonality | uniq(mag) +0.062, uniq(geo) +0.017, shared +0.334 | +0.0617 / +0.0171 / +0.3337 | CONFIRMED |
| Add.2 collinearity | lspec +0.931, e_top −0.134, srank +0.316 | +0.9307 / −0.1340 / +0.3164 | CONFIRMED |
| Add.2 shape-first | 0.505 (+0.116) → 0.801 (+0.296) | 0.5054 (+0.1155) → 0.8014 (+0.2961) | CONFIRMED (note: "+0.116" is 0.1155, prints as +0.116 under their rounding path; trivial) |
| Add.2 commonality | uniq(mag) +0.296, uniq(shape) +0.016, shared +0.099 | +0.2961 / +0.0162 / +0.0993 | CONFIRMED |

### Seed stability
| seed | committed (pr logfd/e_top/lspec/srank) | recomputed | verdict |
|---|---|---|---|
| s42 n=328 | −0.814 / −0.214 / +0.119 / −0.189 | −0.8143 / −0.2136 / +0.1189 / −0.1893 | CONFIRMED |
| s43 n=313 | −0.825 / −0.129 / +0.027 / −0.195 | −0.8251 / −0.1289 / +0.0271 / −0.1947 | CONFIRMED |
| s44 n=306 | −0.781 / −0.129 / −0.030 / −0.286 | −0.7812 / −0.1290 / −0.0301 / −0.2864 | CONFIRMED |
| s45 n=78 | −0.890 / −0.401 / +0.304 / −0.581 | −0.8900 / −0.4011 / +0.3039 / −0.5810 | CONFIRMED |
| s46 | skipped n=9 | n=9 skipped | CONFIRMED |
| OLS coef rows | all four seeds | match to 4 dp | CONFIRMED |

### DS-284B
| quantity | committed | recomputed | verdict |
|---|---|---|---|
| rows 21/21, adapt 20/21, missing lorawd_s42 | — | reproduced | CONFIRMED |
| per-method means (all 28 cells) | table | match to printed precision | CONFIRMED |
| pooled Spearman stable_rank | +0.86 | +0.857 (scipy) | CONFIRMED |
| pooled eff_rank/lspec/lfro | +0.75 / −0.43 / −0.79 | +0.750 / −0.429 / −0.786 | CONFIRMED |
| per-family Spearman rows | 6×4 table | all match (scipy, tie-aware) | CONFIRMED |
| adapt primary | n=19, r=+0.203, ρ=−0.102 | n=19, +0.2028, −0.1019 | CONFIRMED |
| adapt sensitivity | n=20, r=−0.176, ρ=−0.230 | −0.1760 / −0.2304 | CONFIRMED |
| adapt range / diverged | 53.1–80.0; 25.7 | 53.1–80.0; 25.7 | CONFIRMED |
| 7B pooled fingerprints | table | match | CONFIRMED |

## 2. Pitfall audit

1. **Duplicate run_names across results/*/ dirs — PASS (with one semantic caveat).**
   1511 summary.json globbed → 1042 rows after filter; 1042 unique run_names, 0
   duplicate keys, 0 run_name-field-vs-dirname mismatches, 0 double-counted rows.
   CAVEAT (data hygiene, not loader bug): `frc_lorawdr16_wd0p3_lr3e4_c256_s42_reeval`
   carries fdelta=0.3275, retention_mean=26.84 — byte-identical to
   `..._s42` — so the frozen n=1035 pool contains the same data point twice.
   It has NO geometry row, so it drops out of every regression pool (this is
   exactly why geo_pool=1034); it only inflates the preflight n and (negligibly)
   the pooled/frc r, and those values are themselves the §18.1 freeze, so
   internal consistency holds. Impact: cosmetic (~1/1035).
2. **Method-token extraction — PASS.** 10 tokens: clora 122, dora 76, lora 134,
   lora_null 96, lorawd 321, lorawdr16 10, milora 157, milorawd 2, pissa 5,
   sclora 112 (primary pool). Zero lora_null runs parsed as 'lora'
   (`body.startswith("lora_null")` guard verified against all 96). lorawdr16
   rows are the frc `frc_lorawdr16_wd0p3_*` sweep (genuinely distinct token);
   milorawd = 2 lrsw runs; pissa = 5 frc runs. Note: milorawd(2)/pissa(5) are
   tiny dummy groups — harmless for the R² ladder but their method-dummy
   coefficients individually mean little (not reported anyway).
3. **Design rank / dummy trap — PASS.** Intercept + fam[1:] + method[1:]
   throughout. numpy matrix_rank = p for every model (largest: 19/19 in A-M3,
   20/20 in B-M4). cond(X'X) = 3.6e5 (A-M3) / 4.2e5 (B-M4) — benign for
   Gaussian elimination with partial pivoting; normal-equations path loses ~6
   digits of ~16, so their stdlib solver is numerically safe here.
4. **F-test formula — PASS.** ((SSR_r−SSR_f)/q)/(SSR_f/(n−p_f)) with p = number
   of columns including intercept: verified analytically against the code and
   numerically (my QR-based recomputation reproduces every printed F to the
   printed digit: 1890.3, 29.5, 3.5, 1849.1, 22.0, 41.5, 2.2, 33.1, 1076.5,
   1560.1, 1902.8, 664.5, 15.4, 0.9, 2611.9, 13.0, 5.2).
5. **Standardized betas, population vs sample sd — PASS (provably immaterial).**
   beta_std = b·sd(x)/sd(y); the √(n/(n−1)) factor cancels exactly between
   numerator and denominator. Verified numerically: ddof=0 and ddof=1 give
   identical values to 4 dp.
6. **Seed regex — PASS with one known miss.** Zero pool runs with trailing seeds
   outside 42–49. Exactly one run has the seed non-terminal:
   `frc_lorawdr16_wd0p3_lr3e4_c256_s42_reeval` (seed=None; would form its own
   singleton V3 cell and be excluded from all per-seed subsets) — but it has no
   geometry row so it never reaches V3 or seed-stability. No wrong cell-grouping
   occurs in any pool actually used.
7. **CE join — PASS.** forgetting_merged.jsonl: 1354 lines, 1313 with finite
   forgetting_kl, 1313 unique run_names → 0 multi-row keys, so 'last wins' never
   fires. forgetting_kl ≥ 0 for all rows (0 negatives) — consistent with a KL.
8. **Geometry join — PASS.** adapter_metrics_merged.jsonl: 1470 valid rows,
   1470 unique 'run' keys, 0 duplicates/conflicts; join is exact-name so no
   cross-family collision is possible without duplicate keys, and there are none.

## 3. Within-cell dependence (the big check)

Cells = recipe (run name minus trailing seed): K=343 cells, N=1034, mean 3.01
seeds/cell (1–5).

- **ICC of M1 residuals within cell: 0.782** (one-way ANOVA, MSB=59.34,
  MSW=5.02, k0=3.014). Residual dependence is large, as suspected.
- **Design effect ≈ 2.58 → effective n ≈ 401.** The OLS F=1890.3 for the
  magnitude step overstates evidence by roughly this factor (naive deflation
  → F≈730; still astronomically significant).
- **CR1 cluster-robust SEs (model fam+logfd+geo, 343 clusters):**
  - b(logfd) = −14.87: t_OLS = −17.9 → t_cluster = −7.9. Survives decisively.
  - b(e_top) = −24.10: t −5.1 → −3.5. Survives.
  - b(srank) = −0.233: t −7.4 → −3.5. Survives.
  - b(lspec) = +1.48: t +2.1 → **+1.4 — lspec is NOT significant once
    clustering is respected** (it was already the weakest and is the
    magnitude-contaminated regressor the Addendum-2 argument removes).
- **Cell-level cluster bootstrap, B=2000 (resample 343 cells with replacement),
  95% percentile CIs:**
  | quantity | point | 95% CI |
  |---|---|---|
  | magnitude ΔR² (M1−M0) | +0.395 | [+0.311, +0.482] |
  | geometry unique ΔR² (M2−M1) | +0.017 | [+0.007, +0.032] |
  | shape-geometry unique ΔR² | +0.016 | [+0.006, +0.032] |
  | magnitude unique vs shape (Add.2) | +0.296 | [+0.203, +0.386] |
  | std beta logfd | −0.744 | [−0.894, −0.615] |
  - P(magnitude ΔR² > geometry unique ΔR²) = 1.0000 (2000/2000 reps)
  - P(magnitude-unique > shape-unique, Add.2 decomposition) = 1.0000
- **Conclusion:** the magnitude-vs-geometry ordering survives clustering with
  no overlap whatsoever — the CIs are separated by an order of magnitude
  ([0.31,0.48] vs [0.007,0.032]). Consistent with V3 cell-level (0.370 vs 0.023).
  However, any wording that leans on the raw F=1890 (or the KL step F=22 /
  geometry F=29.5 as literal p-values) should be softened or cluster-adjusted:
  the honest run-level t for logfd is ≈ −8, not −43.

## 4. Errors found, ranked by impact

1. (Low) Duplicate data point in the frozen pool: `..._s42_reeval` ==
   `..._s42` exactly; inflates n=1035 by one and is invisible to all regressions
   (no geo row). Worth a footnote if n=1035 is quoted as "distinct runs".
2. (Low, inferential not numerical) OLS F/t statistics in the committed outputs
   ignore within-cell correlation (ICC=0.78, design effect ≈2.6). No conclusion
   flips — magnitude survives at t≈−8 cluster-robust; e_top/srank survive at
   ≈−3.5 — but lspec's positive partial (+0.047, t_cluster≈1.4) should not be
   described as a real effect.
3. (Cosmetic) Addendum-2 "+0.116" for the shape-first step is 0.1155 (their
   printed dR2s don't sum: 0.390+0.116+0.296=0.802 vs printed 0.801); pure
   rounding-path artifact, values themselves correct.

No arithmetic, join, parsing, or rank errors found. Every one of the ~90
committed numbers checked reproduces to the printed precision under an
independent QR-based/scipy implementation.
