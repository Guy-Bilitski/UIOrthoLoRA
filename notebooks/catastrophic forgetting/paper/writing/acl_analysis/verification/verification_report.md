# ADVERSARIAL VERIFICATION OF THE 2026-07-18 ACL-ANALYSIS LAYER

`[Verifier pass, 2026-07-18. Every headline claim of observatory/, correlations/,
adjudication/, insights/ was re-derived from raw canonical data
(results/*/summary.json ⋈ geo_drift/adapter_metrics_merged.jsonl ⋈
forgetting_merged.jsonl, quarantine_diverged.txt) with INDEPENDENT code —
own loader, own OLS/partial/cluster-SE/hinge/AUC implementations, different CV
fold RNGs and knee grids. Scripts: verify_common.py, verify_pools_signs.py,
verify_league_commonality.py, verify_ceproxy.py, verify_adjudication.py,
verify_insights.py (+ inline checks logged in verify_mvg_anchor.log,
verify_lrband_spot.log, verify_misc*.log). Python: repo venv
/home/guyb/UIOrthoLoRA/.venv/bin/python (same as the analysis scripts).
Nothing outside verification/ was modified.]`

**Bottom line: the four analyses are numerically solid — every load-bearing
number I re-derived reproduced exactly or within stated noise. I found NO
fabricated or wrong-pool number. I found 8 corrections (wrong wording or wrong
secondary number), 2 overclaims that must be weakened before the manuscript
(both around SC-LoRA / Qwen-math and the adaptation-retention sign), and a
handful of numbers I could not fully re-derive (listed).**

Verdict counts: **CONFIRMED 34 · CORRECTED 8 · OVERCLAIMED 2 · UNVERIFIABLE/not-re-derived 5.**

---

## A. CROSS-AGENT CONSISTENCY

### A1. Pool conventions — CONFIRMED (with one required disclosure)
- My independent loader reproduces §18.1 exactly: n=1035, pooled r=−0.847, all
  six family (n, r) cells to 3 decimals (`verify_pools_signs.log`).
- Observatory master 1097 = 1035 on-pool + 39 withheld (CorDA/++) + 7 stragglers
  + 16 non-finite rows; on_pool sum = 1035. Consistent with their description.
- The `_reeval` duplicate is present in the frozen 1035 and byte-identical to its
  parent (confirmed); correlations' dedupe to 1034 and the geometry-join n=1034
  coincide (the duplicate has no geometry row). Adjudication's working pool is
  *different* (stragglers IN, quarantine OUT, qwswm `_ep6_` OUT, frm c256 only,
  math retention = BBH) — its §18.1 assertion is preflight-only. **All three
  conventions are legitimate, but the paper must state, once, which pool each
  exhibit uses.**
- Robustness: the magnitude headline is convention-robust (pooled r −0.847
  frozen / −0.847 +stragglers / −0.864 quarantine-excluded; per-family worst
  drift qwswm −0.830→−0.743, already disclosed in §18.6 as the clean-subset
  −0.70 rule). **The adaptation-retention sign and the KL-vs-F_Δ family count
  are NOT convention-robust — see A2/A3.**
- One benign cross-agent inconsistency to fix editorially: observatory's qwswm
  LoRA+wd op point is an `_ep6_` cell (F_Δ 0.114) while adjudication excludes
  `_ep6_` and lands on the 3e-4 non-ep6 cell (adapt means nearly identical,
  68.97 both) — pick one convention for the paper table.

### A2. Adaptation-retention sign — CONFIRMED, but the per-family sign census is convention-dependent → qualifier mandatory
My recompute (frozen pool, run level): frc +0.24, frm +0.86, lrsw +0.15,
lrswm −0.22, qwsw +0.49, qwswm +0.71 → positive 5/6; family-partialed pooled
r = +0.38 (n=1035) / **+0.39 on the deduped n=1034** — both agents exact.
BUT quarantine-excluded: lrsw flips to −0.21, census becomes 4/6, partialed
drops to +0.16. The positive sign is driven by above-knee joint-collapse runs
(many of them quarantined-but-finite).

**The one safe sentence for the paper:**
> "Across the pool there is no adaptation-retention trade-off: run-level
> adaptation and retention are positively correlated (family-partialed
> r = +0.39 on the frozen pool; sign driven by above-knee runs that lose both,
> and attenuated to +0.16 when quarantined runs are excluded). A trade-off
> exists only locally, along the dose axis near each recipe's frontier — so
> cross-method comparisons must be made at operating points, not by pooled
> scatter."
Do NOT quote "positive in 5/6 families" without the quarantine-convention
qualifier (it is 4/6 on the quarantine-excluded pool, and lrsw's sign flips).

### A3. KL-vs-F_Δ within family — BOTH NUMBERS CONFIRMED; frozen-pool convention is binding for the paper
Independently reproduced both sides of the flip (`verify_pools_signs.log`):
- Frozen pool (quarantine INCLUDED, CE join n=911): KL (best of raw/log) beats
  F_Δ within family in **1/6** (lrswm 0.856 vs 0.747; qwsw reversed hard,
  0.41 vs 0.70) — matches correlations exactly.
- Quarantine-EXCLUDED pool: **5/6** — matches doc-05's old claim.
The flip is real and is caused by the 32 quarantined-but-finite far-collapse
runs (extreme-KL leverage), i.e. by pool convention, not by a bug on either side.

**Which is right for the paper: the frozen quarantine-included pool.** Every
frozen §18/§19 number lives on it; quoting the 5/6 would silently switch pools.
**Exact safe wording:**
> "Within family, log F_Δ is the stronger single predictor of retention in 5 of
> 6 families on the frozen pool (KL wins only Llama-math-sweep, 0.86 vs 0.75);
> an earlier analysis reporting the reverse (5/6 for KL) used the pre-freeze
> quarantine-excluded pool, where removing far-collapse runs favors the
> behavioral metric. Same-sample pooled ΔR² is 0.420 (F_Δ) vs 0.340 (KL) under
> either framing."

---

## B. HEADLINE SPOT-CHECKS

### B4. Correlations league table + commonality — CONFIRMED (exact)
Same-sample n=911 re-derived independently (`verify_league_commonality.log`):
ΔR² over family FE: log F_Δ **+0.420** (cluster-robust t −12.0, G=330 cells),
spec_max +0.349 (−15.8), fro_total +0.348 (−15.5), CE/KL +0.340 (−12.1),
LR **+0.207** (−11.5), stable rank +0.116 (−7.3), amp_top +0.032, eff rank
+0.025, e_top +0.015, e_bot +0.008 — every value and every cluster-t matches.
Commonality: unique(C beyond M) = **+0.005**, unique(M beyond C) = **+0.085**,
shared +0.335; 3-block uM +0.033 / uG +0.031 / uC +0.009, shared M∩C +0.181,
M∩G∩C +0.154, M∩G +0.052, G∩C −0.004 — all exact. Ladder (n=1034): 0.390 →
0.785 → 0.802 exact. M-vs-G anchor 0.296/0.016/0.099 exact **with the
shape-only G block (e_top + stable_rank)**.
**Required disclosure:** two different "geometry" blocks are now in circulation
(2-metric shape → unique(G)=+0.016; 5-metric shape → unique(G)=+0.031). Each
quote must name its block, or a reviewer will read a contradiction.

### B5. CE-proxy (new paper exhibit — verified hardest) — CONFIRMED with one CORRECTION and two framing notes
Independent implementation (own hinge fit with a different knee grid, own
leave-cells-out folds over 3 RNG seeds × {5,10}-fold, own Mann-Whitney AUC;
`verify_ceproxy.log`):
- **Llama RMSE 1.30–1.98 pp / MAE 0.83–1.21 pp — CONFIRMED** (claim 1.31–1.95 /
  0.83–1.19; differences are fold noise).
- **AUC ≥ 0.976 in 6/6 — CONFIRMED exactly** (min = frc 0.976; others
  0.989–0.996).
- **Knee — CONFIRMED with widened band:** my fits give 0.26–0.31 nats for
  lrsw/lrswm/qwsw/qwswm (their table 0.259–0.290), frc 0.38–0.40, frm 1.69 with
  below-knee slope −8.3 pp/decade (exact match). Knee location is grid-sensitive
  by ±0.02–0.05 nats → quote "≈0.26–0.30 nats" or "~0.3 nats", not "0.26–0.29".
- **KL-calibration beats the best log-F_Δ calibration in 6/6 families —
  CONFIRMED** under my independent CV.
- **CORRECTION — "Qwen RMSE 5–6 pp":** qwsw ≈ 6.0–6.2 (confirmed), but qwswm is
  **3.6–3.9 pp** under my folds/finer knee grid vs their 5.05 — the number is
  fold- and grid-sensitive because the error is tail-driven. Safe wording:
  "on Qwen the mapping degrades several-fold (RMSE ≈4–6 pp, tail-driven and
  fold-sensitive) — a tripwire, not a ruler." Do not print "5–6" as if stable.
- Framing note 1: as a *damage tripwire*, log F_Δ is equally good (AUC
  0.98–1.00, and better than KL in 3 families). KL's real advantages are the
  quantitative calibration and needing only a forward pass (no adapter-weight
  access); phrase the exhibit that way, not as "only KL can screen".
- Framing note 2 (already in ce_proxy.md, must survive into the paper): CE is
  downstream/quasi-tautological, blind to channel B, Qwen coverage 60–62% with
  per-seed use barred, two CE protocols mixed (benign, Q2).

### B6. Adjudication — core CONFIRMED; three CORRECTIONS; one OVERCLAIM (SC-LoRA/E4)
Re-derived on the adjudication-convention pool built independently
(`verify_adjudication.log`):
- **Operating points — CONFIRMED exactly**: lrsw LoRA+wd 81.75±0.17 / 25.86±0.37
  (F_Δ 0.399±0.012); frm LoRA+wd 66.79±0.79 GSM8K at BBH 33.57±1.04 ≥ base 33.1;
  qwsw LoRA+wd 87.43±0.23 / 40.07±0.68; qwswm SC-LoRA 77.23±0.79 GSM8K at BBH
  47.71±0.23 (F_Δ 0.107), LoRA+wd 68.97±3.33 at 47.54±0.43. Δ = **+8.26 pp,
  paired t = 5.18 (p=0.035, 3 seed pairs)** — matches "+8.3, t=5.2". CLoRA qwsw
  87.02±0.19/39.52±1.15 ✓; SC-LoRA qwsw seed lottery 27.85±15.96 ✓.
- **Deterministic Pareto frontier — CONFIRMED for llama_math (100% LoRA+wd,
  incl. the wd0.5 cell), qwen_cs (100% LoRA+wd incl. the flagged n=1 7e-5 cell),
  qwen_math (100% SC-LoRA)**. Bootstrap P values taken from their table
  (LoRA+wd 1.00/1.00/1.00; SC-LoRA qwen_math 1.00, LoRA+wd 0.53; MiLoRA
  llama_cs 0.498) — consistent with the deterministic frontier; the bootstrap
  itself was not re-run.
- **CORRECTION 1 — llama_cs frontier**: "every non-dominated cell belongs to
  LoRA+wd" is only true with the E6 MiLoRA+wd arm excluded from ranking. My
  frontier recompute finds MiLoRA+wd (n=1 cell, 80.22/26.66) non-dominated.
  The findings do handle MiLoRA+wd separately, but the frontier sentence needs
  the footnote "E6 arms (MiLoRA+wd, 1–2 runs) excluded from ranking".
- **CORRECTION 2 — head-to-head tally**: 26 comparisons and **0 retention
  losses (0/26) — CONFIRMED** from their own head2head table (all ret W/L = 0
  wins for opponents) and my paired recompute of the key cells. But the
  decomposition "17 dominated / 8 ties-with-losses" is wrong against their own
  CSV: it is **14 dominated** (outside-noise loss on both axes), 8
  tie-one-axis-lose-other, 2 pure ties, 1 win-adaptation-lose-retention
  (LoRA-Null qwen_math), 1 reverse win (SC-LoRA qwen_math). Use 14, not 17.
- **CORRECTION 3 — "its winning cell is its LOWEST-magnitude cell (F_Δ 0.107)"
  is factually false**: SC-LoRA's qwswm lr2e-5 cell has F_Δ 0.055 < 0.107
  (and observatory's parallel claim "smallest op-point F_Δ of the family" is
  also false — LoRA+wd's op point sits at F_Δ 0.102–0.114, and the n=1
  LoRA@7e-5 op point at 0.063). Safe replacement:
  > "its winning cell is a below-knee, low-magnitude cell (F_Δ = 0.107, second-
  > smallest of its own sweep and essentially equal to LoRA+wd's op-point
  > magnitude) — the counterexample wins at small magnitude, not by tolerating
  > a large one."
- **Seed SD / LR band / divergence — CONFIRMED**: LoRA+wd 0.43 and SC-LoRA 3.06
  reproduce as the median over the four families of per-family median
  within-cell SDs (state that aggregation); SC-LoRA qwsw median 7.08 exact.
  Safe-LR band totals decompose exactly from lr_band.csv (LoRA+wd 26/29 with
  qwen_cs on the family-relative band; SC-LoRA 6/25; LoRA 10, CLoRA 12,
  MiLoRA/LoRA-Null 13, DoRA 8); frm LoRA+wd per-LR means independently
  reproduced (5/5 landed LRs within 2 pp of 33.1). Divergence: **CLoRA 0/121
  exact**, lora 7/179, milora 9/162, lora_null 3/97, sclora 4/113, dora 3/73
  all exact; LoRA+wd numerator 6 exact but my denominator is 156 vs their 146
  (bookkeeping; rate 3.8% vs 4.1% — footnote-level).
- Qwen-CS "no method within 2 pp of base at any LR" — CONFIRMED (abs-band
  safe_2pp = 0 for every method).

**OVERCLAIM 1 — "genuine geometry win" for SC-LoRA on Qwen-math (the sentence
reviewers will attack).** The win itself is real and reproduced (+8.26 pp,
paired t=5.2, tied BBH, P(frontier)=1.00). But:
1. E4 (key_numbers §18.3) is a **Llama-CS-only, retention-side** control:
   eval-matched calibration moved SC-LoRA from −3.39 pp below curve to +0.92 pp
   above. It licenses "don't blame SC-LoRA's Llama retention deficit on
   geometry"; it does NOT license "the Qwen-math adaptation win is geometry".
2. The qwswm run used the standard **nq_open calibration** (verified in
   train_cs.py: all SC-LoRA/LoRA-Null/CorDA inits share the nq_open loader;
   the eval-matched b4_* E4 arm exists only for Llama-CS). There is **no
   eval-matched control on Qwen-math**, and E4 proves SC-LoRA outcomes can move
   ~4 pp with calibration-corpus choice.
3. n = 3 seed pairs; p = 0.035.
**Safe wording:**
> "The one reversal is SC-LoRA on Qwen-math: +8.3 pp GSM8K at statistically
> tied BBH (paired over 3 seeds, t=5.2), P(on frontier)=1.00 — a genuine
> within-harness win for its data-aware initialization as configured
> (nq_open calibration). Consistent with E4 — the win occurs at small update
> magnitude, and SC-LoRA's retention deficits elsewhere were shown to be
> calibration-set artifacts — but E4 also shows SC-LoRA's results are sensitive
> to the calibration corpus, and no calibration-sensitivity control exists for
> this family; we therefore attribute the win to the method-as-configured, not
> to subspace geometry per se."
Do not print "genuine geometry win" unqualified anywhere.

### B7. Insights — CONFIRMED on all five checked headlines; two denominator CORRECTIONS
(`verify_insights.log`, `verify_insights_part2.log`, `verify_misc.log`)
- **Fragility ordering — CONFIRMED exactly**: normalized slopes reproduce
  (MMLU-Pro −0.36…−0.73, BBH −0.29…−0.64, MMLU −0.15…−0.49, TQ +0.13…−0.21);
  ordering MMLU-Pro > BBH > MMLU > TQ identical in 6/6; **Kendall W = 1.000**
  (own implementation). Keep the chance-floor caveat. (The χ² p=4.4e-4 is a
  family-level test, m=6 — fine, but W itself is the quotable quantity.)
- **TruthfulQA — CONFIRMED exactly**: cell-level r = +0.40/+0.65/+0.76/+0.79 on
  the four Llama families (p ≤ 8e-3), −0.87/−0.90 on Qwen; slopes +2.2…+5.0
  pp/decade; broad-slope attenuation **24/26/27/30 % Llama, 12/14 % Qwen** —
  matches "24–30 %" and "12–14 %".
- **Three knobs — CONFIRMED**: wd stage-1 partial r(logF, wd | logLR) = −0.766
  (t=−6.9, 36 cells) vs claimed −0.762 (t=−6.4, n=33) — tiny wd-cell-set
  difference, same conclusion; residual ns (−0.23); ΔR²(wd beyond F) = +0.005
  on 0.902→0.907 (claim 0.006, 0.902→0.908). CLoRA k: **exact**
  (F_Δ 0.615→0.347, ret 22.68→25.26, ρ = −1.00/+1.00). Rank knob: exact
  (F 0.516→0.747, ret 24.70→22.13). **k-tax exact: 76.8→69.4 at lr3e-4, peak at
  k256–512** — and the wording correctly protects CLoRA's published numbers.
  On-curve residuals: substance confirmed (all three knob-series lie within
  ~±1.8 pp of the frc curve; frc cell r = −0.951 exact) but the exact mean
  residuals are hinge-fit-dependent (mine +0.31/−0.29/+1.01 vs their
  +0.26/+0.53/+1.73) — quote "all residuals ≤ ~1.8 pp, ns", not the means.
- **Free lunch — CONFIRMED exactly**: peak-below-knee vs global = 81.8/81.8,
  81.4/81.9, 87.8/87.8, 58.5/59.1, 68.5/68.5, 77.2/77.2 → **99.0–100 % in 6/6**
  (§18.2 knees). Reachability: plain LoRA @3e-4 **0/4** below knee ✓, CLoRA
  @3e-4 1/5 ✓.
  **CORRECTION (denominators)**: "sclora 1/7" is SC-LoRA's *all-LR* frc cells,
  not lr3e-4 (it has one 3e-4 cell) — the sentence mixes conventions; and
  "LoRA+wd 12/31" — my recompute gives **12/26** (wd>0 grid cells, r16-variant
  excluded; 15/29 including it). Numerator 12 confirmed, denominator does not
  reproduce; state "roughly 40–46 % of LoRA+wd grid cells sit below the knee
  (vs 0 for plain LoRA at lr3e-4), reaching 81.4 adapt below the knee" —
  the 81.4 is exact.
- Insights 6 (share_q), 7 (fingerprints), 8 (adaptation-side ordering): NOT
  re-derived here (appendix-tier, per-matrix store); no red flags in their
  scripts' conventions; keep at the confidence labels they already carry.

### B8. Observatory — CONFIRMED with two wording CORRECTIONS
- spec_max ≡ dw_sv_max: **r = 0.99998 (n=1034)** — quote "> 0.9999" or
  "r ≈ 1.000", not "1.0000" (it is not literally 1).
- Matched-F_Δ spread — CONFIRMED exactly (lrswm@−1.0 0.63, lrsw@−1.0 0.99,
  qwswm@−1.5 0.91, qwsw@−1.0 1.78; also frc@−1.0 1.81 — include frc or say
  "0.6–1.8 pp"; frm has no below-knee kept bin, disclose). qwswm@−0.5 30.6 pp
  flagged correctly.
- Stable-rank partials — CONFIRMED exactly (frc −0.333, frm −0.323, lrsw −0.595,
  lrswm −0.666, qwsw −0.004, qwswm +0.073), incl. the lrsw suppression effect
  (raw −0.42 → partial −0.60).
- Seed-noise — CONFIRMED exactly (mean within-cell ret SD 0.43–0.94 Llama,
  1.92/2.89 Qwen; adaptation 8.67 lrsw / 8.18 frc).
- CE-drift block — CONFIRMED (r(logKL, logF) +0.84…+0.96; r(logKL, ret)
  −0.78…−0.92; KL ≡ CE − H₀ exact; base entropy family-sd ≤ 0.021,
  Llama 1.82–1.83 / Qwen 1.93). Op-point KL: frm 20.7× ≈ "21×", 0.216 vs 4.455
  exact; LoRA+wd lowest in 4/6 ✓ (frc's "15×" relies on splitting lorawd-r16
  as its own method — footnote it).
- **CORRECTION**: finding 3's "smallest op-point F_Δ of the family
  (0.399±0.012)" for lrsw LoRA+wd is contradicted by their own
  m2_op_points.csv (SC-LoRA 0.376, MiLoRA+wd 0.296). Say "lower op-point F_Δ
  than every non-wd method except SC-LoRA". Same issue for the qwswm
  "smallest op-point F_Δ (0.107)" bullet (LoRA@7e-5 n=1 sits at 0.063; LoRA+wd
  at 0.102–0.114) — restrict to "smallest among multi-seed op points" only if
  the ep6 convention is also stated, or drop "smallest".

---

## C. QUALIFIER COMPLIANCE — PASS
- **No naive-OLS F/p headlined.** Correlations quotes cluster-robust t only
  (verified against my own CR1 implementation — all match). Insights' p-values
  are cell-level (sanctioned by 09 Q1) or family-level (Kendall). Adjudication
  uses paired/within-cell tests only.
- **No per-seed Qwen CE anywhere** (checked all four findings; observatory and
  correlations carry the bar explicitly).
- **CLoRA framing faithful everywhere**: the k-knob adaptation-tax paragraph
  explicitly states "does NOT contradict CLoRA's published high-k wins …
  their numbers are faithful"; adjudication's verdict states "CLoRA's published
  numbers are faithful … answered YES with margin" at matched capacity. No
  "suspect/strawman/artifact" framing of published CLoRA numbers found. The
  string "60.8" appears in no findings file.
- **No "geometry doesn't matter" phrasing**; all four use magnitude-first /
  geometry-second with bounded second-order effects, and none quotes the dead
  positive-spec_max effect as geometry evidence.
- 284B is referenced only as design-family recurrence (observatory finding 6),
  consistent with the 09 Q3 sign-test bar.

## Not independently re-derived (flag, don't quote as verified)
1. correlations 05_cv table (leave-cells-out 0.783/0.807/0.815, LOFO rows) —
   plausible given the verified league/commonality numbers, but not recomputed.
2. adjudication verdict-table "ret@matched-adapt" column (−2.2…−26.3 pp) —
   depends on an LR-re-matching interpolation rule I did not re-implement;
   directionally consistent with the verified op-point tables.
3. adjudication overheads (2.15× DoRA, +3.34 GB CLoRA, init FLOPs) — [EXTERNAL]
   constants from INTERESTING_INSIGHTS.md, correctly labeled as such.
4. Pareto bootstrap P values (table-checked + deterministic frontier verified;
   resampling not re-run).
5. insights 6/7/8 (share_q, per-matrix fingerprints, adaptation-side ordering).

---

# SAFE-TO-QUOTE LEDGER (numbers + wording the paper may use from the new analyses)

**Correlations**
- League table (n=911, family FE, same-sample): ΔR² log F_Δ **+0.420**
  (cluster t −12.0) > spec_max +0.349 ≈ ‖ΔW‖_F +0.348 > CE/KL +0.340 > LR
  **+0.207** > stable rank +0.116 > all other geometry ≤ +0.032.
- unique(CE beyond magnitude) = **+0.005**; unique(magnitude beyond CE) =
  **+0.085**; shared +0.335. 3-block (5-metric shape block, mediator caveat
  verbatim): uM +0.033, uG +0.031, uC +0.009, M∩C +0.181, M∩G∩C +0.154.
- M-vs-G causal-knob split **0.296 / 0.016 / 0.099** — always with "shape-only
  geometry block (e_top, stable_rank)" stated.
- KL-vs-F within family: F_Δ better in **5/6 on the frozen pool** (wording in
  A3 above; always name the pool).
- CE-proxy: Llama RMSE **1.3–2.0 pp** (MAE 0.8–1.2), ≈2× the retention eval's
  own seed noise; damage-detection **AUC ≥ 0.976 in 6/6**; drift budget
  "**keep KL-to-base under ~0.3 nats**" (knees ≈0.26–0.30 in 4 families,
  frc ≈0.4, frm none — below-knee slope already −8.3 pp/decade); Qwen =
  tripwire not ruler (RMSE ≈4–6 pp, tail-driven, fold-sensitive).

**Adjudication**
- "LoRA+wd loses none of 26 head-to-heads; **14** opponents dominated outside
  noise; **no method beats it on retention (0/26)**; the single reversal is
  SC-LoRA on Qwen-math."
- P(on frontier) = 1.00 for LoRA+wd on Llama-CS/Llama-math/Qwen-CS (with the
  E6-MiLoRA+wd-excluded footnote on Llama-CS); 1.00 for SC-LoRA on Qwen-math.
- Op points as in B6 (all exact). Safe-LR band LoRA+wd **26/29** vs SC-LoRA
  6/25 (state qwen_cs uses the family-relative band). Seed SD 0.43 vs 3.06
  (median-of-family-medians; 7.08 on Qwen-CS). CLoRA **0/121** in-band
  divergences. Qwen-CS: no method within 2 pp of base at any LR.
- SC-LoRA exception with the B6 safe wording (calibration-scoped; "F_Δ 0.107,
  second-smallest of its sweep, ≈ LoRA+wd's op-point magnitude").

**Insights**
- Fragility: "MMLU-Pro > BBH > MMLU > TruthfulQA in all six families, Kendall
  W = 1.000" + chance-floor caveat.
- TruthfulQA rises with forgetting on all 4 Llama families (r +0.40…+0.79,
  +2.2…+5.0 pp/decade) and falls on Qwen (−0.87/−0.90); including TQ attenuates
  the broad slope **24–30 % (Llama)** / 12–14 % (Qwen).
- Three knobs: wd partial −0.76 (cells), CLoRA-k ρ = −1.00/+1.00
  (F_Δ 0.615→0.347, ret 22.7→25.3), rank r8→r32 F 0.52→0.75 / ret 24.7→22.1;
  all knob series within ~±1.8 pp of the frc curve; ΔR²(wd|F) ≈ 0.005.
- k-tax: "at fixed lr3e-4 in our harness, k 128→2048 costs adaptation
  76.8→69.4 (peak k256–512) while buying retention along the curve" + the
  CLoRA-faithful disclaimer verbatim.
- Free lunch: "**99–100 % of peak adaptation lies below the knee in all six
  families**"; plain LoRA @lr3e-4 has 0/4 frc cells below the knee vs ~40–46 %
  of LoRA+wd grid cells (12/26), reaching 81.4 adaptation below the knee.

**Observatory**
- Pooled r(log F_Δ, ret) −0.847; per-family −0.830…−0.929 (frozen §18.1).
- Matched-F_Δ method spread **0.6–1.8 pp** in below-knee bins (5 families with
  such bins; frm has none), rising to 4–10 pp in collapse territory.
- spec_max and dw_sv_max are the same measurement (r > 0.9999) — cite one.
- Stable-rank fingerprint (DoRA 4.5 … PiSSA 18.1) + partials −0.32…−0.67 Llama,
  ≈0 Qwen — "magnitude first, geometry second, second-order part
  model-dependent" (descriptive, run-level).
- Op-point KL: LoRA+wd lowest in 4/6; frm spans ~21× (0.216 vs 4.455).
- Seed noise: retention SD 0.43–0.94 pp Llama vs 1.9–2.9 Qwen; adaptation SD
  ~8–9 pp on CS (format collapse) — single-seed CS adaptation rankings
  unreliable.

# DO NOT QUOTE
1. "SC-LoRA's winning cell is its lowest-magnitude cell" / "smallest op-point
   F_Δ of the family" (both false: lr2e-5 cell F_Δ 0.055; LoRA+wd op 0.102,
   LoRA@7e-5 0.063).
2. "Genuine geometry win" for SC-LoRA Qwen-math without the E4
   calibration-scope qualifier (no eval-matched control exists on Qwen-math).
3. "17/26 dominated" (it is 14/26 by the head2head table).
4. "KL beats F_Δ in 5/6 families" without naming the pre-freeze
   quarantine-excluded pool (frozen pool: 1/6).
5. "Positive adaptation-retention correlation in 5/6 families" without the
   quarantine-convention caveat (4/6 quarantine-excluded; lrsw flips sign).
6. "Qwen CE-proxy RMSE 5–6 pp" as a stable number (qwswm is 3.6–5.1 depending
   on folds/grid; say ≈4–6, tail-driven).
7. "LoRA+wd 12/31 cells below knee" (denominator does not reproduce; use 12/26
   or a percentage) and "sclora 1/7 at lr3e-4" (that is the all-LR count).
8. r = 1.0000 for spec_max≡dw_sv_max (it is 0.99998); knee band "0.26–0.29"
   (use 0.26–0.30 / "~0.3 nats").
9. Exact on-curve residual means (+0.26/+0.53/+1.73) — hinge-fit-dependent;
   quote the ≤~1.8 pp bound.
10. LoRA+wd divergence "6/146" (denominator unverified; 6/156 in my count —
    if quoted, re-derive the denominator or give the rate as ≈4 %).
11. The 09-verification bars remain in force: no naive OLS F/p as evidence
    strength, no per-seed Qwen CE, spec_max is magnitude, 284B sign-test
    framing only, CLoRA published numbers are never challenged.
