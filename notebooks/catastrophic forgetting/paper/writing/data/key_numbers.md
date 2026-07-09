# Key numbers (load-bearing) — SINGLE SOURCE OF TRUTH

**Authoritative recompute date:** 2026-07-02 (B6 reconciliation pass).
**Source:** `writing/data/campaign_summary.jsonl` (identical to the live
`results/campaign_summary.jsonl`), computed with `/home/guy/UIOrthoLoRA/.venv/bin/python`.
**Dedup rule:** keep the row with the **latest `evaluated_at`** per `run_name` (359 raw rows →
343 unique). This is the rule used for every number below. Model = **Llama-2-7B, seed 42** unless
marked Qwen.

**Provenance tag on each block:** `[RECOMPUTED 2026-07-02]` = re-derived from the registry this pass
and is the ground truth; `[EXTERNAL]` = from handoff docs, not recomputable from this registry (cite
source, do not silently "correct").

Every number here overrides any conflicting value in `01`–`03` / `CONCLUSIONS_AND_IDEAS.md`.
Contradictions found and fixed are logged in `06_reconciliation.md`.

---

## 0. Definitions & units

- **Magnitude axis** = `fdelta` field. **CORRECTION (2026-07-09, CLoRA-paper-expert audit against the
  actual PDF): this is NOT the Frobenius norm ‖ΔW‖_F.** It is CLoRA's **F_Δ metric (their Eq 3)**:
  mean over tokens of `‖ΔW·x‖/‖x‖` on 100 real eval inputs, averaged over all updated matrices — a
  data-dependent effective-output-change measure. `fdelta.py`/`uio_inprocess.fdelta_inprocess`
  implement exactly this; the old "token-weighted Frobenius" label here and in `analyze_matrix.py:9`
  was WRONG (a CLoRA-reading reviewer would catch it). GOOD NEWS: our axis is therefore **directly
  comparable to CLoRA Table 4's F_Δ column** (their LoRA 0.79, LoRA-L2 0.29, k2048 0.14), and our
  `dw_sv_max`/`dw_sv_mean` correspond to their ‖ΔW‖ column (spectral norm). PAPER ACTION: relabel
  every "‖ΔW‖_F / Frobenius" mention of this axis to "F_Δ (effective update magnitude, CLoRA Eq 3)".
  Predictive robustness (validator, 2026-07-09): F_Δ R²=0.74 for retention vs dw_sv_mean 0.36 /
  dw_sv_max 0.33 / log-LR 0.32 — F_Δ is uniquely predictive among available axes. Range ~0.05–3.7 on
  the clean sweep. The "72–1395" numbers in older handoffs are an older unnormalized scale — do not
  use them.
- **Retention (core)** = mean(BBH answer-only 3-shot `bbh_fewshot`, MMLU-Pro 5-shot CoT) = field
  `retention_mean`.
- **Retention (broad)** = mean(BBH, MMLU-Pro, MMLU, ARC-c, TruthfulQA) = field `retention_broad`.
- **Base (no-FT) ceiling:** BBH-AO 33.10, MMLU-Pro 18.96, **core = 26.0** `[EXTERNAL: h00#6, h05]`.
- **Llama-2 sweep** = run_names prefixed `lrsw_` (CS) / `lrswm_` (math). **Adaptation** = `cs_avg`
  (CS-8 accuracy for CS; GSM8K exact-match for math, same column).

---

## 1. THE MAGNITUDE LAW — r(retention_core, log10 ‖ΔW‖_F) `[RECOMPUTED 2026-07-02]`

| Dataset | r | R² | slope (pp/decade) | n | p | status |
|---|---|---|---|---|---|---|
| **Llama-2 CS, pooled (6 active methods)** | **−0.86** | 0.74 | −14.8 | 49 | 3.4e-15 | mature |
| Llama-2 CS, on-curve (excl. SC-LoRA) | **−0.92** | 0.84 | −10.0 | 42 | 2.1e-17 | mature |
| **Llama-2 math, pooled** | **−0.97** | 0.93 | −10.1 | 14 | 2.4e-8 | mature (LoRA/LoRA+wd/DoRA only; sparse) |
| **Qwen-2.5 CS, pooled (7 adapters, core)** | **−0.86** | 0.735 | −32.0 | 49 | 3.7e-15 | **replication (2nd model), mature** `[07-10]` |
| Qwen-2.5 CS, pooled (7 adapters, broad) | **−0.94** | 0.878 | −26.1 | 49 | 4.2e-23 | replication (2nd model), mature `[07-10]` |
| Qwen-2.5 CS, LoRA only (core) | −0.88 | 0.78 | −34.8 | 7 | 8.5e-3 | (LoRA subset of the pooled sweep) |
| Qwen-2.5 math, pooled (BBH-only) | −0.05 | 0.00 | ~0 | 10 | ns | **does NOT yet replicate** (low-LR only; "+0.67" core = broken MMLU-Pro parser, §11) |

**Headline to quote:** r ≈ **−0.86 pooled, −0.92 on the 5 well-behaved methods**; slope ≈ −10 to
−15 pp/decade. **Both** slope conventions are correct depending on which fit: −14.8 = 6-method
pooled; −10.0 = 5-method on-curve. Use the matching pair (−0.86/−14.8 or −0.92/−10.0), never mix.

> The "6 active methods / n=49" pool contains LoRA (7) + LoRA+wd (7) + MiLoRA (7) + CLoRA (7) +
> DoRA (7) + SC-LoRA (7) + **LoRA-Null (7, silently labeled `lora` by the generator)** = 49. CorDA
> is **excluded** from every law/figure. So the pool is really **7 distinct adapters**, and the
> "6 methods" label is an artifact of the LoRA-Null labeling bug (§9). The law/r/slope are identical
> whether LoRA-Null is pooled into LoRA or kept separate (same 49 points).

### 1a. Within-method r(retention_core, log ‖ΔW‖_F), Llama-2 CS `[RECOMPUTED 2026-07-02]`

| method | r | n |
|---|---|---|
| LoRA (plain only, n=7) | −0.95 | 7 |
| LoRA (as plotted, incl. LoRA-Null, n=14) | −0.90 | 14 |
| LoRA-Null | −0.86 | 7 |
| LoRA+wd | −0.89 | 7 |
| DoRA | −0.97 | 7 |
| CLoRA | −0.90 | 7 |
| MiLoRA | −0.94 | 7 |
| SC-LoRA | −0.97 | 7 |
| CorDA (latest-dedup, incl. lr1e-3 explosion) | −0.90 | 7 |

Every adapter individually traces the same downward line (all r ≤ −0.86); points interleave across
methods. **NOTE:** these supersede the older within-method list (LoRA −0.97 / LoRA+wd −0.95 /
CLoRA −0.98 / DoRA −0.86 / CorDA −0.91 / MiLoRA −0.96 / SC-LoRA −0.88) that appears in `01`/
`CONCLUSIONS` — that list was from a stale eval/dedup and is **wrong**; use the table above.

## 2. LR IS A WEAKER PROXY THAN ‖ΔW‖ (Llama-2 CS, n=49) `[RECOMPUTED 2026-07-02]`

| Predictor of retention_core | R² | r |
|---|---|---|
| log10(learning rate) | **0.32** | −0.57 |
| log10(‖ΔW‖_F) | **0.74** | −0.86 |

R² more than doubles switching the x-axis from LR to the ‖ΔW‖ it produces (fig7 message). The brief
cites "0.75 vs 0.35"; the registry-verified values are **0.74 vs 0.32** — use these.

## 3. PER-METHOD BEST-ADAPT OPERATING POINTS — Llama-2 CS `[RECOMPUTED 2026-07-02]`

Each row = method's max-`cs_avg` LR. (= `table_main_cs`.)

| Method | best LR | CS-8 | Ret-core | Ret-broad | ‖ΔW‖_F | σ_max | robust (ret_core≥24, /7) |
|---|---|---|---|---|---|---|---|
| **LoRA+wd(0.3)** | 5e-4 | **81.6** | **25.6** | 33.2 | **0.394** | 34.3 | 6/7 |
| SC-LoRA | 5e-5 | 80.1 | 22.5 | 32.5 | 0.559 | 11.3 | 1/7 |
| MiLoRA | 3e-4 | 79.9 | 24.7 | 33.6 | 0.543 | 48.5 | 5/7 |
| LoRA | 3e-4 | 79.1 | 24.4 | 32.7 | 0.623 | 40.7 | 5/7 (plain LoRA subgroup) |
| CLoRA | 5e-4 | 78.4 | 21.9 | 30.1 | 0.643 | 29.8 | 5/7 |
| DoRA | 2e-4 | 78.3 | 24.8 | 32.9 | 0.445 | 38.8 | 4/7 |

**Robustness caveat:** the generator counts the plotted "LoRA" series (LoRA + LoRA-Null pooled) at
**10/14**; the plain-LoRA subgroup alone is **5/7**. Quote 5/7 for plain LoRA. CorDA excluded (§8).

**Winner:** LoRA+wd(0.3) is simultaneously highest-adaptation (81.6), highest core-retention among
the high-adapters (25.6, ≈ base ceiling 26.0), lowest ‖ΔW‖ (0.394), widest safe band (6/7).

## 4. PER-METHOD BEST-ADAPT — Llama-2 math / GSM8K `[RECOMPUTED 2026-07-02]`

| Method | best LR | GSM8K | Ret-core | Ret-broad | ‖ΔW‖_F | σ_max |
|---|---|---|---|---|---|---|
| **LoRA+wd(0.3)** | 5e-4 | **50.6** | **24.6** | 33.6 | 0.399 | 27.5 |
| LoRA | 3e-4 | 46.5 | 22.9 | 31.5 | 0.520 | 35.3 |
| DoRA | 2e-5 | 33.3 | 25.2 | 33.8 | 0.327 | 7.1 |

Math sweep is sparse (only LoRA n=7, LoRA+wd n=6, DoRA n=1). LoRA+wd wins both axes.
**NOTE:** LoRA+wd math Ret-broad = **33.6** and DoRA math Ret-broad = **33.8** (earlier docs printed
34.0 and 32.9 respectively — corrected here).

## 5. FAIRNESS / ANCOVA (fig2, Llama-2 CS, n=49) `[RECOMPUTED 2026-07-02]`

- Pooled **linear** log-fit R² = **0.74**; adding per-method intercepts → R² = **0.87**.
- ANCOVA F(5,42) = **8.34**, p < 0.001 — improvement real but driven by ONE method.
- Per-method residual from the **pooled spline** curve (this is what fig2 / the box-plot report;
  * = p<0.05 vs 0):

| Method | residual μ (pp) | p | verdict |
|---|---|---|---|
| LoRA (incl. LoRA-Null, n=14) | +0.79 | 0.079 | on the law |
| LoRA+wd | +0.06 | 0.83 | on the law |
| MiLoRA | +1.04 | 0.14 | on the law |
| CLoRA | +0.09 | 0.80 | on the law |
| DoRA | +1.37 | 0.23 | on the law |
| **SC-LoRA** | **−4.15** | **0.006 \*** | **below the law (only significant deviator)** |

5 of 6 plotted series straddle 0. SC-LoRA forgets ~4.15pp MORE than its ‖ΔW‖ budget predicts.
**PROVISIONAL** — pending calibration-distribution sensitivity (nq_open vs eval-matched) + seeds
43/44. **CorDA is EXCLUDED from the fit**, so it has no residual in this table (the older
"CorDA −3.0pp off-curve" number is not from the current 6-series fit — do not cite it; §8).

> Residuals are against the **spline** baseline, not the linear line. Against the linear pooled fit
> the per-method means differ (e.g. LoRA+wd +1.04, CLoRA −0.76, DoRA +1.92, SC-LoRA −4.65); the
> paper reports the **spline** residuals above (matches fig2). Do not mix the two baselines.

## 6. MAGNITUDE BUDGET (fig8, Llama-2 CS, n=49) `[RECOMPUTED 2026-07-02]`

- Adaptation (cs_avg) slope = **+20.3** pp/decade of ‖ΔW‖ (r=+0.39, p=6e-3) — magnitude BUYS adapt.
- Retention slope = **−14.8** pp/decade of ‖ΔW‖ — magnitude COSTS retention.
- Sweet-spot band `‖ΔW‖_F ∈ [0.31, 0.62]` `[EXTERNAL: fig8/h13 design choice; not a fitted CI]`:
  near-max adaptation AND retention ≈ base. LoRA+wd (0.394) sits inside; un-regularized high-adapters
  (LoRA 0.623, CLoRA 0.643, SC-LoRA 0.559) sit at/past the right edge.

## 7. PER-BENCHMARK DEGRADATION (fig5, pp per decade of ‖ΔW‖, Llama-2 CS, n=49) `[RECOMPUTED 2026-07-02]`

| Benchmark | slope | r | base ceiling |
|---|---|---|---|
| MMLU | **−23.4** | −0.93 | uncalibrated |
| MMLU-Pro | −15.2 | −0.89 | 18.96 |
| ARC-Challenge | −14.9 | −0.93 | uncalibrated |
| BBH | −14.3 | −0.79 | 33.10 |
| **TruthfulQA** | **−0.5** | −0.10 | uncalibrated |

MMLU dies fastest; TruthfulQA essentially flat/immune.

## 8. CorDA status (B1/B2) `[RECOMPUTED 2026-07-02]`

- CorDA is **excluded from every law, figure, table, and ANCOVA residual**. It is NOT one of the
  adapters "on the curve"; treat its count as **not-yet-assessed**, not off-curve.
- Under latest-`evaluated_at` dedup the 7 `lrsw_corda_*` CS rows are the **2026-06-29/30 re-eval**,
  and one is the residual-save **explosion** (`lrsw_corda_r16_lr1e3_s42`: `fdelta=515.77`,
  `cs=0.0`, `ret_core=0.0`). So the current-latest CorDA rows are contaminated; the earlier 06-25
  rows are cleaner but were superseded by timestamp. **Neither set is publishable** because CorDA
  was mis-calibrated on wikitext-2 (should be nq_open) and the calib↔eval mismatch (B4) is unresolved.
- Any CorDA number (incl. the old "77.9/19.9" best point and "CorDA −0.91 within-method /
  −3.0pp off-curve") is **PENDING** the nq_open re-run + fair-calibration pass — do not report.

## 9. Honest coverage framing (B1) — how many adapters are "on the curve"

**Distinct adapters in the mature Llama-2 CS sweep = 7:** LoRA, LoRA+wd, DoRA, MiLoRA, SC-LoRA,
CLoRA, LoRA-Null. (CorDA is the 8th in the study design but is **excluded / not assessed**.)

- **On the curve (residual indistinguishable from the law, p>0.05):** LoRA, LoRA+wd, MiLoRA, CLoRA,
  DoRA — **5 adapters**. LoRA-Null is pooled into the LoRA series (labeling bug) but its within-method
  r=−0.86 and it tracks the same curve, so honestly **6 of the 7 assessed adapters are on the curve.**
- **Off the curve (provisional):** SC-LoRA (−4.15pp, p=0.006) — **1 adapter**, and PROVISIONAL
  (calib↔eval confound + single seed).
- **Not assessed:** CorDA — excluded pending nq_open re-run and fair calibration.

**Approved coverage sentences (use verbatim):**
> "Across the **6 of 8** adapters we can currently assess fairly (LoRA, LoRA+wd, DoRA, MiLoRA,
> CLoRA, LoRA-Null), retention lies on a single ‖ΔW‖_F curve; SC-LoRA is the one provisional
> below-curve deviator, and CorDA is withheld pending a calibration-fairness fix."

Do **not** write "the law holds across 8 adapters" or "across 6 methods" without this qualifier: the
sweep contains **7** distinct adapters (one, LoRA-Null, mislabeled), CorDA is **excluded**, and the
"6 methods" label double-hides LoRA-Null. See §1 note and §10.

## 10. DATA-LABELING BUG (must fix before camera-ready) `[RECOMPUTED 2026-07-02]`

The generator sets `method = run_name.split("_")[1]`, so `lrsw_lora_null_*` (a **distinct** adapter)
is classified as `"lora"`. Consequences: the plotted "LoRA" series and its robustness count silently
pool **7 LoRA-Null points into plain LoRA** (n=14, robust 10/14; plain LoRA alone n=7, robust 5/7).
The pooled law (49 points) and the best-adapt LoRA point (79.1/24.4, a plain-LoRA run) are
**unaffected**. Fix the legend/robustness/n and the "6 methods" wording before publication.

## 11. Qwen-2.5 replication status `[RECOMPUTED 2026-07-10, data-verifier-confirmed — SUPERSEDES the 07-02 block]`

> **STALE (07-02, do NOT cite):** "CS = 7-LR **LoRA-only** sweep; other 5 adapters NOT yet run;
> ~13 of ~112 cells complete; math LoRA r=+0.67." Between 07-02 and 07-10 the Qwen **CS** arm
> completed as a **full multi-adapter sweep**, and the Qwen **math** multi-adapter sweep began
> draining on Node B. The numbers below are authoritative; they supersede the "Qwen … LoRA only"
> rows in the §1 table (rows updated to match).

- **CS — COMPLETE, full multi-adapter replication (NOT LoRA-only).** 7 adapters × 7 LRs = **49
  assessed cells** (LoRA, LoRA+wd0.3, LoRA-Null, DoRA, MiLoRA, SC-LoRA, CLoRA-k1024; 1 CorDA point
  excluded per §8). F_Δ = `headline.fdelta` (== `fdelta_token_weighted`); retention = `retention_mean`.
  - **Pooled core: r = −0.86, R² = 0.735, slope −31.98 pp/dec (n=49).**
  - Pooled broad: r = −0.94, R² = 0.878, slope −26.10 (n=49).
  - LoRA-only core: r = −0.88 (reproduces the prior published value exactly).
  - Within-method core r: DoRA −0.905, LoRA −0.883, MiLoRA −0.882, SC-LoRA −0.838, CLoRA −0.767,
    LoRA-Null −0.730, **LoRA+wd −0.165** (flat by construction — wd caps F_Δ so its 7 points do not
    span the axis; same shallow-slope signature LoRA+wd shows on Llama, not a failure of the law).
  - **The Qwen pooled core r (−0.86) is numerically indistinguishable from Llama-2's −0.858** — the
    law transfers across architecture at the same strength. Report r/direction as the headline; the
    steeper slope (−32 vs −15) is the Qwen retention scale, not a different law. Do NOT merge Qwen and
    Llama into one fit (different base ceilings).
- **math — IN PROGRESS, does NOT yet replicate (report on BBH-only).** Assessed union = **10 cells**
  (LoRA 5 LRs + LoRA+wd0.3 5 LRs), **all low/mid-LR** → F_Δ spans only **0.038–0.159** (~0.6 decade),
  no high-magnitude points. On the **campaign-correct BBH-only** metric the fit is **FLAT: pooled
  r = −0.05 (ns), LoRA-only r = −0.24 (ns)**. **The frequently-quoted "+0.67" (and pooled +0.60) is
  on CORE retention, which includes the KNOWN-BROKEN MMLU-Pro math parser — it is a parser artifact,
  NOT a positive math law; never report Qwen-math as positively correlated.** Math retention stays
  **BBH-only** (see §0). The 39 genuinely-new high-LR resolution cells (5e-4/1e-3 × 6 adapters) are
  in flight on Node B (`jobs/frepro4_qwen_B_keep.txt`); until they land, Qwen-math is "pending", not
  a result.
- **Coverage now ≈ 59/112 planned Qwen cells** (49 CS + 10 math). Present CS as a full second-model
  replication; present math as in-progress/pending.

## 12. Numbers that are [EXTERNAL] (not recomputable from this registry — cite, don't "fix")

- Base ceiling 26.0 / BBH-AO 33.10 / MMLU-Pro 18.96 (h00#6, h05).
- Rank sweep "r4→r256: ret 25.4→8.5" (h09 #5) — **not in this registry**; the `grid_*` rows are
  UIOrthoLoRA configs, not a plain-LoRA rank sweep. Cite h09; do not present as recomputed.
- Full-scale clean wd reference points "wd0.1 80.4/24.86 ties CLoRA-k1024 79.8/24.85; wd1.0
  76.7/26.87" (h12 ledger) — these are **fast-eval / earlier matrix** numbers, NOT the `lrsw_` sweep
  in this file; keep them tagged as h12-ledger, not as sweep results.
- Sweet-spot band [0.31, 0.62] is a design/annotation choice (fig8), not a fitted interval.
- Directional norm ‖ΔW·C_retain^½‖ −0.79 vs −0.77 (h09 #2) — external.

## 13. Known fairness caveats (state honestly)

- Single seed (s42) for all mature results; n=7 per method in CS.
- Ranks NOT matched: LoRA/DoRA/LoRA-Null r16; MiLoRA/SC-LoRA r32; CLoRA k1024. Frame the LAW, not a
  method ranking.
- Only LoRA has the wd knob; a param-matched LoRA+wd control is still needed.
- CorDA excluded entirely (wikitext-calib bug fixed→nq_open re-running; calib↔eval fairness open, B4).
- SC-LoRA off-curve deviation is provisional.
- LoRA-Null labeling bug (§10).

---

## 14. LoRA-Null SPLIT-CONVENTION RECOMPUTE `[RECOMPUTED 2026-07-02, second pass]`

**Supersedes** the pooled-convention parts of §1a/§3/§5/§9/§10 wherever they conflict. Scripts:
`writing/analysis_a1_a4.py` (stats) and `writing/make_figs_split_lora_null.py` (figures; a patched
copy of `paper_figs_v2.py` — the canonical generator is untouched). Registry:
`writing/data/campaign_summary_clean.jsonl`. Convention: `lrsw_lora_null_*` is its **own series**;
the sweep is **7 adapters × 7 LRs = 49 cells** (CorDA withheld). The labeling bug of §10 is thereby
resolved at the analysis level; figures fig0–fig8 + op_points regenerated 2026-07-02 under this
convention.

- **Pooled law (n=49): numerically IDENTICAL** to §1 (same 49 points, relabeled only):
  r=−0.858, R²=0.736, slope −14.8 pp/dec, p=3.4e-15.
- **On-curve law now = 6 series** (LoRA, LoRA-Null, LoRA+wd, DoRA, MiLoRA, CLoRA; n=42, same points
  as the old "excl. SC-LoRA" fit): r=−0.92, R²=0.84, slope −10.0. Identical to §1 row 2.
- **Within-method r (split):** LoRA (plain) −0.95, LoRA-Null −0.86, LoRA+wd −0.89, DoRA −0.97,
  MiLoRA −0.94, CLoRA −0.90, SC-LoRA −0.97. Range −0.86…−0.97. (Matches §1a.)
- **Spline residuals (7 series):** LoRA +0.99 (p=0.21), LoRA-Null +0.60 (p=0.27), LoRA+wd +0.06
  (p=0.83), MiLoRA +1.04 (p=0.14), CLoRA +0.09 (p=0.80), DoRA +1.37 (p=0.23), **SC-LoRA −4.15
  (p=0.006, only significant deviator)**. The old pooled "LoRA +0.79 (n=14)" splits into the first
  two rows. Six of seven assessed adapters are on the law.
- **Intercept ANCOVA (7 series):** R² 0.736 → 0.870, ΔR²=0.134, **F(6,41)=7.05, p=3.3e-5** —
  significant overall but driven almost entirely by SC-LoRA. (Replaces F(5,42)=8.34.)
- **Robustness (ret_core≥24, /7):** LoRA+wd **6/7**; LoRA (plain) 5/7; **LoRA-Null 5/7**; MiLoRA
  5/7; CLoRA 5/7; DoRA 4/7; SC-LoRA **1/7**. (The 10/14 pooled figure is retired.)
- **LoRA-Null best-adapt point (CS):** lr5e-4, CS-8 78.9 / ret-core 23.6 / ret-broad 31.0 /
  ‖ΔW‖_F 0.696 / σmax 54.6. Safe point (ret≥24): lr2e-5, 73.0/26.2.
- **Coverage sentence (updated, use this):** "Across the **seven of eight** adapters we can assess
  (LoRA, LoRA-Null, LoRA+wd, DoRA, MiLoRA, CLoRA, SC-LoRA), six lie on a single ‖ΔW‖_F curve;
  SC-LoRA is the one provisional below-curve deviator, and CorDA is withheld pending a
  calibration-fairness fix."

## 15. A1–A3 LAW-STRENGTHENING TESTS `[RECOMPUTED 2026-07-02, second pass]`

Script: `writing/analysis_a1_a4.py` (split convention, n=49). These close the old
"intercept-only, in-sample" hedge.

- **A1 slope-interaction ANCOVA.** Adding per-method **slopes** on top of per-method intercepts:
  full 7 series **F(6,35)=9.32, p<0.001** (significant) — but **restricted to the six on-curve
  series: F(5,30)=0.28, p=0.92** (slopes statistically indistinguishable). Per-method slopes
  (pp/dec): LoRA −11.7, LoRA-Null −9.1, LoRA+wd −8.1, DoRA −10.3, MiLoRA −10.4, CLoRA −10.2,
  **SC-LoRA −26.0** — SC-LoRA alone bends the curve; the six on-curve adapters share ONE slope.
- **A2 leave-one-method-out predictive check.** Pooled linear law fit on 6 series predicting the
  7th: held-out RMSE LoRA 2.25, LoRA-Null 2.52, LoRA+wd 1.74, DoRA 3.58, MiLoRA 2.24, CLoRA 2.41
  (mean **2.46 pp** over the six on-curve) vs in-sample RMSE 3.07 (pooled) / **2.16 (method-aware,
  7 intercepts)**. Held-out **SC-LoRA: RMSE 9.05, mean error −7.40 pp** (consistent with its
  provisional below-curve status).
- **A3 joint F-test on the six on-curve intercepts** (n=42, excl. SC-LoRA): R² 0.838 → 0.871,
  **F(5,35)=1.79, p=0.14 — not significant**; on-curve intercepts are statistically
  indistinguishable.

**Approved summary sentence:** "Among the six on-curve adapters, neither the intercepts
(F(5,35)=1.79, p=0.14) nor the slopes (F(5,30)=0.28, p=0.92) differ statistically, and the pooled
law fit with any one of them held out predicts that adapter's seven points to within 2.5 pp RMSE —
about the in-sample accuracy of a method-aware model (2.2 pp). The full-pool tests
(F(6,41)=7.05 intercepts; F(6,35)=9.32 slopes; both p<0.001) are significant, driven almost
entirely by SC-LoRA (provisional)."
