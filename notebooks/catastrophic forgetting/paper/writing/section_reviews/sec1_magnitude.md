# Section 1 review — "Magnitude analysis" (artifact_status_report.html)

**Reviewer:** section-validator (magnitude section) · **Date:** 2026-07-10
**Ground truth:** `results/campaign_summary.jsonl` (latest-`evaluated_at` dedup), `paper/writing/data/key_numbers.md`, `fig_cross_literature.py`, live `results/*/summary.json`.
**Verdict: SOLID with two required edits** (Qwen understatement now factually stale; CLoRA-callout "~2× lower F_Δ" violates the baked-in PI-critic guard and is wrong for the LoRA baseline).

---

## (a) Claim-by-claim verdict table

| # | Claim in §1 | Recomputed | Verdict |
|---|---|---|---|
| 1 | LR panel: R² = 0.32 (retention vs log LR) | R² = 0.3211, r = −0.567, n=49 | **CONFIRMED** (minor: R² is computed on the 49-cell pool *excluding CorDA*, but the panel draws 8 adapters incl. CorDA — one caption clause fixes it) |
| 2 | DW panel: r = −0.86, R² = 0.74 | r = −0.8579, R² = 0.7359, p = 3.4e-15, n=49 | **CONFIRMED** |
| 3 | Fit-line constants in JS (`DWFIT` slope −14.78, inter 17.85, xmin −0.87, xmax 0.57) | slope −14.78, intercept 17.85, x∈[−0.871, 0.571] | **CONFIRMED (exact)** |
| 4 | "holds within every one of the seven assessed adapters (r −0.86 to −0.97)" | LoRA −0.954, LoRA-Null −0.864, LoRA+wd −0.887, DoRA −0.969, MiLoRA −0.938, CLoRA −0.903, SC-LoRA −0.972 (each n=7). Range = [−0.972, −0.864] | **CONFIRMED** (CorDA, excluded, is also −0.901) |
| 5 | Second architecture: "Qwen2.5-7B commonsense, LoRA arm, r = −0.88; other adapters … not part of the assessed set on this model" | LoRA-only r = −0.883 — number CONFIRMED. **But the hedge is STALE:** the full Qwen CS multi-adapter sweep is complete — 7 adapters × 7 LRs = 49 cells, pooled r = −0.857, R² = 0.735, slope −31.98 (matches key_numbers §11, verified from raw registry) | **CORRECTED / UNDERSOLD** — upgrade to the pooled full replication; the current sentence actively denies data that now exists |
| 6 | Spearman ρ = −0.90 | ρ = −0.896, p = 3.5e-18 | **CONFIRMED** |
| 7 | "saturating fit beats both a line and a parabola on cross-validation" | LOOCV RMSE: hinge 3.24 < linear 3.33 < quadratic 3.73 | **CONFIRMED**, but the hinge-vs-linear margin is only ~3% — keep the verb soft ("beats", not "decisively beats") |
| 8 | knee at update size ≈ 0.37 | hinge knee log10 F = −0.437 → F_Δ = 0.366; hinge R² = 0.81 | **CONFIRMED** (and it genuinely sits at LoRA+wd's operating point 0.39) |
| 9 | below-ceiling slope ≈ −21 pp/decade | hinge descending-arm slope = −20.5; identical to a plain linear fit on the 27 points right of the knee (−20.5) | **CONFIRMED** |
| 10 | partial r = −0.87 controlling for adapter identity | within-method-demeaned partial r = −0.868 | **CONFIRMED** |
| 11 | permutation p < 5×10⁻⁵ | 0 of 100,000 permutations reach \|r\| ≥ 0.858 → p < 1e-5 | **CONFIRMED** |
| 12 | Embedded scatter `S{}` matches raw summaries | all **55/55** points match registry to the printed decimals (log-LR, log10 F_Δ, retention). CorDA lr1e-3 correctly omitted (residual-save explosion, retention null) | **CONFIRMED** |
| 13 | CLoRA Table 4: r = −0.98, slope −14.7 on their ten rows | r = −0.980, slope −14.65 | **CONFIRMED numerically** — but see logic issues below |
| 14 | "against our −14.8. …the same slope" | −14.8 is our **core (BBH+MMLU-Pro)** slope; CLoRA T4 is **BBH-only**. The like-for-like comparison (per `fig_cross_literature.py`) is our BBH-only slope **−14.34** (r −0.79) vs their −14.65 | **CORRECTED** — conclusion survives (still same slope, arguably cleaner), but the printed pairing is a metric mismatch a reviewer will catch |
| 15 | "their F_Δ levels run ~2× lower, so it is a parallel law" | **WRONG for the baselines**: CLoRA's plain-LoRA rows sit at F_Δ 0.79–1.03 vs our plain-LoRA 0.23–1.42 — **overlapping**, not 2× lower. Only their constrained k-series (0.14–0.36) runs lower. `fig_cross_literature.py` carries an explicit PI-critic guard: *"do NOT claim '~2x lower F_Delta'"* | **UNSUPPORTED — must fix** (violates a baked-in guard) |

## (b) Logic check — is the "three ways" argument airtight?

The three legs (within-method, second architecture, R² 0.74 vs 0.32) are each individually verified. Remaining holes a reviewer can exploit:

1. **Reverse-causality / common-cause (LR drives both).** F_Δ is measured post-hoc; both F_Δ and forgetting are outcomes of LR. The R²-doubling argument only partially closes this. **Closed by a new computation (below, improvement #1):** at every *fixed* LR, F_Δ still predicts retention across methods (per-LR r = −0.50…−0.98, mean −0.83; LR-demeaned pooled r = −0.82, slope −20.6), and the reverse partial correlation r(retention, logLR | logF_Δ) = **+0.46** — once update size is controlled, higher LR is not harmful at all. This is the strongest available "F_Δ mediates LR" statement and is currently absent from §1.
2. **"They collapse onto one descending curve" (lead paragraph) quietly absorbs SC-LoRA**, which is the one significant below-curve deviator (−4.15 pp, p=0.006; disclosed in §2/§3 but not in §1). One clause ("with a single disclosed exception, §2") keeps the honest boundary inside the section that makes the claim.
3. **CLoRA T4 fit is two families, not one cloud.** The pooled −14.7 blends a baseline family (slope −12.7) and the k-series (−18.9); with n=10 the k-series has high leverage. The saving grace already computed in `fig_cross_literature.py`: worst drop-one \|r\| ≥ 0.95. Quote that robustness rather than the bare −14.7, and compare BBH-to-BBH (−14.3 vs −14.7).
4. **Rank/param-count confound** (LoRA/DoRA/LoRA-Null r16 vs MiLoRA/SC-LoRA r32 vs CLoRA k1024) is acknowledged in key_numbers §13 but nowhere in §1. The within-method leg is immune to it (rank fixed within a series); say so in one sentence — it turns a confound into a strength.
5. **Ceiling/censoring** is handled well by the callout (the hinge treatment is the right analysis; Spearman + below-ceiling slope + knee all reproduce).

## (c) Statistics quality

- Pearson on log F_Δ with a ceiling is the right *headline* only because the callout supplies Spearman, the hinge fit, and the below-ceiling slope — keep them travelling together.
- The permutation test as stated ("p < 5×10⁻⁵") is consistent with 0/100k exceedances (p < 1e-5); fine.
- LOOCV hinge-vs-linear margin is small (3.24 vs 3.33); the *qualitative* case for the hinge (residual structure, below) is stronger than the CV case — consider citing both.

## (d) Improvements, prioritized (concrete, with cost)

1. **Add the fixed-LR / mediation statistic (zero new runs, ~30 min).** "At each of the seven learning rates, update size still predicts retention across adapters (mean within-LR r = −0.83; LR-demeaned pooled r = −0.82, slope −20.6 ≈ the below-ceiling slope); conversely, controlling for F_Δ, LR's partial correlation with retention flips to +0.46 — the learning rate harms retention only through the update size it produces." This closes the single biggest causal-direction hole and directly strengthens the fig7 message. Verified numbers above.
2. **Upgrade the Qwen sentence + add a Qwen mini-panel (zero new runs, ~1–2 h).** The full 49-cell Qwen CS replication exists and reproduces the law at pooled r = −0.86 / R² = 0.735 — numerically indistinguishable from Llama's −0.858. Replace the "LoRA arm only" hedge with the pooled result and (ideally) add a small third panel or inset scatter of the 49 Qwen points with their own fit. Caveats to carry: slope is steeper (−32, Qwen retention scale — do not merge fits), and Qwen LoRA+wd within-method r is −0.17 (flat by construction, F_Δ range compressed by wd). This converts "one arm on a second model" into "full second-model replication" — the largest single credibility gain available today.
3. **Fix the CLoRA callout (text-only, ~15 min).** (i) Compare BBH-to-BBH: their −14.7 vs our −14.3 (not −14.8). (ii) Delete "~2× lower F_Δ" — their LoRA baseline (0.79–1.03) overlaps ours; only the k-series runs lower. Replace with: "their unconstrained baselines sit at the same F_Δ levels as ours; their k-series extends the law's low-magnitude end." (iii) Add the drop-one robustness (every 9-point subset keeps r ≤ −0.95) and note the pooled slope blends −12.7 (baselines) / −18.9 (k-series) — quoting the range preempts the leverage objection.
4. **Per-seed law overlay from the existing 3-seed matrix (zero new runs, ~1 h).** No `lrsw_` s43/s44 cells have landed (LR sweep is still single-seed s42 — checked today), but the 3-seed `mtx_` rank/wd matrix is complete and gives a seed-stability figure now: law refit per seed gives r = −0.90 / −0.91 / −0.89 (s42/s43/s44, n=34 each; excl. CorDA −0.84/−0.86/−0.85), slopes −23.5/−25.0/−25.2. One small 3-line overlay (or a caption sentence) turns "single seed" from a limitation into a demonstrated non-issue *for the law itself*. When `lrsw_` s43/s44 cells do land, redo on the sweep proper.
5. **Surface the heteroscedasticity insight (new, zero cost).** Around the hinge fit, residual SD is **0.60 pp left of the knee vs 3.46 pp right of it** (linear fit: 1.75 vs 3.77). Large updates don't just forget more — they forget **less predictably** (~6× the spread). This is a genuinely new, practitioner-relevant sentence ("below the knee, retention is both high and *reliable*") and further motivates the small-update operating point. It also justifies the hinge over the line better than the thin LOOCV margin does.
6. **One-clause caption fixes (5 min each):** LR-panel R² excludes CorDA while the panel draws it — say "R² over the 49 assessed cells; CorDA line shown for context, excluded from all fits"; add "single disclosed exception (SC-LoRA, §2)" to the lead's "one descending curve"; note the within-method leg is rank-confound-free.

## (e) New insights found in the magnitude data (not yet surfaced anywhere)

- **F_Δ fully mediates LR** (improvement #1): partial r(ret, LR | F_Δ) = +0.46. Strongest causal-ordering evidence in the dataset.
- **Heteroscedasticity around the knee** (improvement #5): 6× residual-spread increase past F_Δ ≈ 0.37.
- **The LR-demeaned within-LR slope (−20.6) coincides with the below-ceiling hinge slope (−20.5)** — two independent estimators of the "true" cost of magnitude agreeing at ≈ −21 pp/decade; a nice one-liner for the paper.
- The hinge knee (0.366) sits essentially on LoRA+wd's best-adapt F_Δ (0.394) — already hinted in the artifact; worth making the "wd parks you at the knee" framing explicit in §3.

## Files
- Artifact under review: `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/paper/writing/artifact_status_report.html` (§1, lines 126–148; JS data lines 331–332)
- Guard violated by claim 15: `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/paper/writing/fig_cross_literature.py` (header, "do NOT claim '~2x lower F_Delta'"; BBH↔BBH slope −14.3)
- Ground truth: `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/results/campaign_summary.jsonl`, `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/paper/writing/data/key_numbers.md` (§0–2, §11)
