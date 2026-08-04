# Correlation cartography — which metric predicts retention? (2026-07-18)

`[Correlation-cartographer pass. Pool: frozen §18.1 pool, preflight-reproduced
exactly, then deduped (n=1034; frc_lorawdr16_wd0p3_lr3e4_c256_s42_reeval dropped
per 09 Q4). Joins: geometry 100%, CE/KL 88% (Qwen 60-62%, seed-block missingness,
ignorable for regressions per 09 Q4; per-seed Qwen CE analyses barred and not
performed). All inference cluster-robust (cluster = recipe cell) or cell-level
bootstrap per 09 Q1 — no naive F/p is quoted anywhere in this directory.
Scripts: 01_reproduce_anchors.py … 06_ce_proxy.py; every number below traces to
one of them.]`

## 0. Reproduction of frozen anchors (01_reproduce_anchors.py)

Everything reproduces exactly before anything new is claimed:

- §18.1: pooled r(ret, log F_Δ) = −0.847 (n=1035) and all six family (n, r) cells
  to 3 decimals — OK.
- §19.1 ladder A (n=1034): R² 0.390 (family FE) → 0.785 (+0.395 magnitude) →
  0.802 (+0.017 geometry) → 0.808 (+0.006 method) — all four steps match to 3
  decimals.
- 06 §5 commonality split (shape geometry): unique(M)=+0.296 / unique(G)=+0.016 /
  shared=+0.099 — match.

Full detail: anchors_reproduction.md.

## 1. Ranked top findings

1. **F_Δ is the best single retention predictor in every view of the data**
   (03_league_table). Same-sample (n=911) ΔR² over family FE: log F_Δ **+0.420**
   > log spec_max +0.349 ≈ log‖ΔW‖_F +0.348 > KL drift +0.340 > log LR +0.207 >
   stable rank +0.116 > every other geometry metric (≤ +0.032). Cluster-robust
   t = −12.0. The top three are all magnitude measures — the magnitude *block*
   is the story, and F_Δ (data-dependent effective magnitude) is its best member.

2. **CE drift is a near-substitute correlationally, but adds almost nothing
   beyond magnitude** (04_commonality): pooled two-block split R²(M)=0.420,
   R²(C)=0.340, unique(C beyond M)=**+0.005**, unique(M beyond C)=**+0.085**,
   shared=+0.335. CE is downstream (a consequence, not a knob): the huge shared
   component is magnitude's causal signal read out in behavior space.

3. **Cheap-proxy result (new, paper-worthy)** (06_ce_proxy): a per-family
   knee-calibrated KL→retention mapping predicts held-out (leave-cells-out)
   retention with **RMSE 1.3–2.0 pp / MAE 0.8–1.2 pp on the four Llama families**
   — about 2× the single-seed noise floor of the retention eval itself
   (seed SD 0.4–0.9 pp) — and detects "damaged" runs (>5 pp below family
   ceiling) with **AUC ≥ 0.976 in all six families**. On Qwen it is a tripwire,
   not a ruler (RMSE 5–6 pp, MAE 2.5–3.7, tail-driven). KL-knee beats the best
   log F_Δ calibration within family in **6/6 families**.

4. **A common drift budget: the KL knee sits at ≈0.26–0.29 nats in 4/6 families
   spanning both base models and both task types** (ce_proxy_knees.csv; frc 0.40,
   frm has no flat region — its below-knee slope is already −8.3 pp/decade).
   Candidate headline for the monitor story: "keep KL-to-base under ~0.3 nats."

5. **Combining metrics beats magnitude alone only in-distribution, and modestly**
   (05_cv): leave-cells-out OOS R² 0.783 (magnitude) → 0.807 (+geometry) → 0.815
   (all); RMSE 4.38 → 4.04 pp. Under leave-one-family-out the additions stop
   helping (raw 0.628 vs 0.659 vs 0.639; intercept-oracle all ≈ 0.56–0.58) —
   geometry/CE gains do not transfer across families.

6. **LR confirmed as a proxy, mid-table** (03, extends §18.5): ΔR² +0.207 —
   rank 6 of 11, below every magnitude measure and CE, above every geometry
   metric. Per-family R² 0.22–0.52 vs F_Δ's 0.69–0.86.

7. **Geometry stays second-order — with one honest nuance**: in the 3-block
   commonality (M/G/C), unique(G, 5-metric shape block)=+0.031 [+0.015,+0.051]
   is statistically indistinguishable from unique(M)=+0.033 [+0.019,+0.051]
   (ordering holds in only 1078/2000 cell-bootstrap replicates). This does NOT
   overturn "magnitude first": M's unique share collapses only because its
   downstream readout (CE) was admitted as a competitor and absorbs the shared
   dose signal (M∩C +0.181, M∩G∩C +0.154). Against exogenous blocks only
   (M vs G, the §19/06 comparison), the split remains 0.296 / 0.016 / 0.099 —
   reproduced here. Framing rule: commonality with a mediator measures
   *diagnostic* value, not causal leverage; keep the M-vs-G split as the
   causal-knob exhibit and the 3-block as the monitoring exhibit.

## 2. League table (03_league_table.py; full table in league_table.md/csv)

Ranked by same-sample ΔR² over family FE (n=911, CE∩geometry pool):

| # | metric | block | ΔR² | cluster t | per-family R² range |
|---|---|---|---|---|---|
| 1 | log10 F_Δ | magnitude | +0.420 | −12.0 | 0.69–0.86 |
| 2 | log10 spec_max | magnitude | +0.349 | −15.8 | 0.57–0.75 |
| 3 | log10 ‖ΔW‖_F | magnitude | +0.348 | −15.5 | 0.50–0.83 |
| 4/5 | CE / KL drift | CE drift | +0.340 | −12.1 | 0.41–0.86 |
| 6 | log10 LR | training knob | +0.207 | −11.5 | 0.22–0.52 |
| 7 | stable rank | geometry | +0.116 | −7.3 | 0.08–0.67 |
| 8 | amp_top | geometry | +0.032 | −4.7 | 0.03–0.24 |
| 9 | effective rank | geometry | +0.025 | −2.9 | 0.00–0.33 |
| 10 | e_top | geometry | +0.015 | −2.7 | 0.00–0.22 |
| 11 | e_bot | geometry | +0.008 | +4.7 | 0.00–0.07 |

Cross-checks against frozen numbers: per-family R²(F_Δ) and R²(LR) reproduce
§18.5's six pairs; KL-alone +0.340 reproduces ladder B's alt row. spec_max is
listed under magnitude (r = +0.92 with log F_Δ family-demeaned; 06 §5, 09 Q1c).
Only 1/6 families puts KL above F_Δ within family on this pool (lrswm 0.856 vs
0.747; qwsw is reversed hard, 0.41 vs 0.70) — 05's family-level "KL beats F in
5/6" was the pre-freeze n=857 quarantine-EXCLUDED pool; on the frozen
quarantine-included pool it is 1/6 (commonality_mc_per_family.csv). Quote both
with their pool conventions stated.

## 3. Commonality (04_commonality.py; commonality.md, *_3block.csv)

Three blocks on n=911, all ΔR² over family FE, total model +0.456:

| component | value | 95% cell-bootstrap CI |
|---|---|---|
| unique magnitude | +0.033 | [+0.019, +0.051] |
| unique geometry-shape (5 metrics) | +0.031 | [+0.015, +0.051] |
| unique CE drift | +0.009 | [+0.001, +0.019] |
| shared M∩C only | +0.181 | [+0.118, +0.234] |
| shared M∩G∩C | +0.154 | — |
| shared M∩G only | +0.052 | — |
| shared G∩C only | −0.004 | — |

CE captures almost nothing magnitude doesn't (unique C = +0.005 two-block,
+0.009 three-block); magnitude retains +0.085 beyond CE pooled — plausibly the
channel-B/format component CE cannot see (05 §2). Per family, unique(C) is
meaningful only in lrswm (+0.112); unique(M) peaks in qwsw (+0.295, where KL is
weakest, R²=0.41). Extended magnitude block (adding log spec_max) changes
nothing (±0.001).

## 4. Cross-validation (05_cv.py; cv_results.md/csv, fig_cv_pred_vs_actual)

| model | leave-cells-out R² (RMSE pp) | LOFO raw R² | LOFO intercept-oracle R² |
|---|---|---|---|
| magnitude | 0.783 (4.38) | 0.628 | 0.578 |
| magnitude+geometry | 0.807 (4.13) | 0.659 | 0.580 |
| magnitude+CE | 0.785 (4.36) | 0.612 | 0.558 |
| all | 0.815 (4.04) | 0.639 | 0.567 |
| CE only | 0.700 (5.15) | 0.279 | 0.483 |

Verdict: in-distribution, geometry buys ~0.3 pp RMSE and CE ~0.1 pp on top of
magnitude — real but marginal. Across families nothing beats magnitude alone.
Per-held-out-family slope transfer is heterogeneous (cv_results.md): magnitude
transfers poorly INTO lrsw (0.054) where CE transfers well (0.716), and the
reverse for frm (CE 0.189) — neither mapping is family-universal at level+slope;
this is exactly why the proxy analysis (below) calibrates per family.

## 5. CE-as-cheap-proxy verdict (06_ce_proxy.py; ce_proxy.md, fig_ce_proxy)

**Verdict: YES as a per-family-calibrated monitor and damage tripwire; NO as a
drop-in replacement for the retention eval, and never as a control knob.**

- Cost: one forward pass over ≤40 WikiText blocks (~3.4 s wall in the store) vs
  a full BBH+MMLU-Pro eval.
- Quantitative substitution (leave-cells-out CV, knee calibration): Llama
  families RMSE 1.31–1.95 pp, MAE 0.83–1.19 pp — ~2× the seed noise floor of the
  retention measurement itself. Qwen: RMSE 5.0–6.0 pp (MAE 2.5–3.7; error is
  tail-driven — the flat region calibrates fine, the post-knee cliff is steep,
  −35…−40 pp/decade, so small KL error = large retention error).
- Screening: AUC ≥ 0.976 (all six families) for flagging runs >5 pp below the
  family ceiling. As a go/no-go tripwire it is essentially perfect everywhere.
- The knee: ~0.26–0.29 nats in lrsw/lrswm/qwsw/qwswm, 0.40 frc, no flat region
  in frm. "Stay under ~0.3 nats of KL drift" is a defensible cross-model
  operating rule with the frm caveat stated.
- Calibration transfer within base model (task→sibling task): KL transfers
  RAW worse than log F_Δ in 5/6 pairs (ce_proxy_transfer.csv) — the mapping
  needs per-family (or at least per-task-type) calibration; it is not free
  across tasks.
- Mandatory caveats (stated in ce_proxy.md): CE is downstream; quasi-tautology
  with retention; blind to channel B (format damage — consistent with the
  +0.085 that magnitude keeps beyond CE); Qwen coverage 60–62% with barred
  per-seed use (09 Q4); two CE protocols mixed — benign here too (per-protocol
  r(KL,ret) within 0.04 in every family that mixes them, matching 09 Q2).

## 6. LR as predictor (task 7; 03_league_table.py)

League position: **6 of 11** — ΔR² +0.207 pooled (cluster t −11.5), roughly half
of magnitude's +0.420 and below CE's +0.340; above all shape-geometry metrics.
Cell-level ΔR² +0.213 (granularity-robust). Consistent with §18.5's per-family
R² battery (reproduced exactly here) and §2's framing: LR is how you *reach* a
magnitude; F_Δ is the variable.

## 7. Surprises

- **stable rank is by far the best geometry metric pooled (+0.116) but wildly
  family-heterogeneous** (R² 0.67 in lrswm vs 0.08 in qwswm) — a pooled-exhibit
  hazard; per-family panels required if it is shown at all.
- **No pooled adaptation–retention tradeoff**: family-partialed r(adapt, ret) =
  **+0.39** (runs that adapt well also retain well — over-dosed runs lose both).
  Per family it spans −0.22 (lrswm) to +0.86 (frm). Kills any "you must trade
  retention for adaptation" framing at pool level; the tradeoff exists only
  along the dose axis within a recipe.
- **qwswm's KL–retention rank correlation is weak (Spearman −0.555)** despite
  Pearson −0.792 — heavy-tail leverage; another reason Qwen is tripwire-only.
- **e_bot's raw pooled +0.33 collapses to +0.11 family-partialed** — a family-
  composition artifact; never quote raw pooled geometry correlations.
- **LOFO asymmetry**: the F_Δ slope does not transfer into lrsw (0.05) while the
  KL slope does (0.72); reversed for frm. Neither weight-space nor behavior-
  space mapping is universal — supports per-family calibration as a design
  point, not a limitation footnote.

## 8. Candidate paper exhibits

MAIN:
- League table (league_table.md, compacted to ~6 rows) — one-look version of
  "magnitude first, geometry second, LR a proxy".
- fig_ce_proxy (6-panel KL→retention calibration) + the knee table — the new
  cheap-monitor contribution ("~0.3 nats drift budget"); pairs naturally with 05
  §5's saturation table.
- Commonality M-vs-G split (already §19/06) with the 3-block M/G/C version as
  its monitoring-flavored extension (quote with the mediator caveat verbatim).

APPENDIX:
- fig_corr_heatmap (raw + family-partialed panels; the raw-vs-partialed contrast
  itself teaches why family FE is mandatory).
- cv_results.md + fig_cv_pred_vs_actual (honest "combining barely helps, and not
  across families").
- ce_proxy_transfer.csv (why per-family calibration), corr_vs_retention.md,
  per-family correlation CSVs.

## 9. File manifest

Scripts: corr_common.py, 01_reproduce_anchors.py, 02_correlation_matrix.py,
03_league_table.py, 04_commonality.py, 05_cv.py, 06_ce_proxy.py.
Tables: anchors_reproduction.md; corr_vs_retention.{md,csv}; corr_pooled_*.csv;
corr_family_<fam>_pearson.csv; league_table.{md,csv}; commonality.md,
commonality_3block.csv, commonality_mc_per_family.csv; cv_results.{md,csv};
ce_proxy.md, ce_proxy_calibration.csv, ce_proxy_knees.csv, ce_proxy_transfer.csv.
Figures: fig_corr_heatmap.{png,pdf}, fig_cv_pred_vs_actual.{png,pdf},
fig_ce_proxy.{png,pdf}.
