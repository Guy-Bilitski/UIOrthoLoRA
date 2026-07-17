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

## 2. LR IS A WEAKER PROXY THAN ‖ΔW‖ `[REWRITTEN 2026-07-16 — adversarial-review repair; supersedes the 07-02 framing]`

**Do NOT quote "R² doubles vs LR" against continuous log-LR alone** — a reviewer fitting LR **as
dummies** nearly ties F_Δ in the sweep families (lrsw 0.765 vs 0.785), because the LR grid is
coarse and F_Δ is LR-monotone within a sweep. The claim survives on three stronger legs (full
numbers: §17.2, `analyze_adversarial_2026-07-16.py`):

1. **Fixed-LR strata:** holding LR constant, F_Δ still predicts retention — r = −0.7…−0.97
   across strata with LR ≥ 1e-4 in every family (variation from rank/wd/method at the same LR).
2. **Partials:** r(F_Δ, ret | LR-dummies) = −0.58…−0.91 (|t| = 7.6…32.6) vs
   r(LR, ret | F_Δ) = −0.18…+0.29 (|t| ≤ 4): F_Δ retains nearly all its signal with LR
   controlled; LR retains almost none with F_Δ controlled.
3. **Decoupling grids (frc/frm):** where wd/rank/method vary at fixed-ish LR, r(log F_Δ, log LR)
   drops to 0.64/0.50 while R²(F_Δ) stays 0.86 and R²(LR-dummies) collapses to 0.39/0.37 —
   when the proxy decouples from the quantity, the quantity wins.

Legacy single-seed values (log-LR continuous 0.32 vs F_Δ 0.74, n=49) remain correct but are the
weakest form of the argument; cite the strata/partials instead.

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

**AUTHORITATIVE MATH HEADLINE `[ADDED 2026-07-10]`:** the table above is the OLDER `lrswm_` sweep
(50.6 etc.) — the **`frm_` faithful math-recipe block supersedes it as the math headline**:
LoRA+wd(0.3) lr2e-4 GSM8K **67.25** / Ret-core 25.14 (`frm_lorawd_wd0p3_lr2e4_c256_s42`; seed 43 =
65.88/26.01 `frm_lorawd_wd0p3_lr2e4_c256_s43`), plain LoRA(wd0) lr1e-4 **64.97** / 22.58
(`frm_lorawd_wd0_lr1e4_c256_s42`), MiLoRA lr1e-4 62.85/23.94 (`frm_milora_lr1e4_c256_s42`),
CLoRA-k256 lr3e-4 60.80/19.02 (`frm_clora_k256_lr3e4_c256_s42`), PiSSA lr3e-4 49.66/3.62
(`frm_pissa_lr3e4_c256_s42`). Quote the frm_ numbers, not the 50.6 row, for the math story.

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

## 16. REGISTRY REFRESH `[RECOMPUTED 2026-07-14, post-qwswm ingestion — SUPERSEDES §11 "math in progress", §4 stale rows, and the "provisional SC-LoRA" framing above]`

Full verification detail: `paper/writing/registry_refresh_2026-07-14.md`. Registry after
ingestion: **622 rows, 606 unique** (backup: `results/campaign_summary.jsonl.bak_pre_qwswm_ingest_2026-07-14`).
Ingestion note: 51 qwswm cells (evaluated 07-09–07-14, node B) had `results/*/summary.json` but
were never appended to the registry; appended 2026-07-14 with their original `evaluated_at`
(SMOKE run excluded). Lead approved 2026-07-14.

### Qwen-2.5 math — NOW REPLICATES QUALITATIVELY
- **Headline (s42 only, BBH vs log10 F_Δ): n=56, r=−0.70, slope −14.4 pp/decade**, F_Δ span
  0.029–0.90 (~1.5 decades). Two diverged cells excluded with disclosure
  (`qwswm_lora_r32_lr5e4_s42` F_Δ=20.9/all-zero; `qwswm_lorawd_wd0p3_lr1e3_s42`).
  All-seeds variant (incl. 7 s43/s44 cells): n=63, r=−0.66, −13.3 pp/dec.
- Slope −14.4 sits INSIDE the cross-domain band (−12.7…−18.9; Llama −14.3/−14.8).
- Per-method r (all seeds): lora_r32 −0.84, sclora −0.82, clora −0.79, dora −0.70, milora −0.65,
  lora_null −0.63; **lorawd +0.09 = flat by construction** (wd caps F_Δ; same signature as CS).
- Strength to quote: **qualitative/directional replication — single seed, BBH-only retention**.
  Do NOT quote as a quantitative law match (no calibrated ceiling; MMLU-Pro excluded by
  convention). The §11 "+0.67 broken parser" framing: parser output on the new r32 rows is sane
  (39–43); keep the BBH-only convention but describe MMLU-Pro as "excluded by convention
  (unreliable on early Qwen-math rows)", not "broken".

### W2/B5a param-matched control — CLOSED
- `frc_lorawdr16_wd0p3_lr5e4_c256_s42` (r16, 28.0M params = plain-LoRA-matched):
  **CS 81.04 / ret 26.27 / F_Δ 0.334** — reproduces the r32 operating point (81.6/25.6/0.394)
  at HALF the parameters. The capacity confound is dead.
- Plain-LoRA rank ladder @ lr3e-4 (frc, c256): r8 79.0/24.0/F_Δ 0.518 → r16 79.6/23.0/0.603 →
  r32 73.5/22.2/0.739. More capacity → larger F_Δ → lower retention (and lower adaptation at
  r32); no rank benefit once F_Δ is controlled.
- **Disclosed anomaly:** the lr3e-4 r16 sibling (`frc_lorawdr16_wd0p3_lr3e4`) lands in a
  deterministic answer-format collapse basin: CS 13.53 (below-chance per-dataset: piqa 3.1,
  winogrande 1.4) with INTACT retention 26.84 and healthy training (31,210 steps, final loss
  0.805 < working sibling). Reeval reproduces (13.54). Not a magnitude effect (F_Δ=0.328).
  Seed replicate queued. Report the 5e-4 cell; disclose the 3e-4 basin.

### Base ceilings (C5) — LANDED (Llama-2); broad retention now calibrated
- `base_llama2_noft` (07-12): BBH 32.96, MMLU-Pro 18.82 → core 25.89; MMLU 40.88, ARC-C 44.80,
  TruthfulQA 38.85 → **broad ceiling 35.26**.
- **CANONICAL core ceiling stays 26.0** (external h00/h05 snapshot: 33.10/18.96 → 26.03);
  in-registry 25.89 is the confirming replicate (snapshot difference; decision 2026-07-14).
- "Ret-broad is uncalibrated / texture only" hedges are RETIRED.
- **TruthfulQA immunity is real, not a floor artifact:** base 38.85; fine-tuned range 31.4–39.5
  (mean 35.7; 1/49 above base); slope −0.46 ns → a constant ≈−3 pp fine-tuning offset,
  magnitude-independent.

### Seeds & math (frm) — 3-seed complete
- Headline verified: `frm_lorawd_wd0p3_lr2e4` GSM8K 67.25/65.88/67.25 → **66.79±0.79**
  (retention SD ≤0.53 across all six 3-seed frm configs; full table in the refresh report).
- DoRA s44 landed → **all 7 CS operating points are now 3-seed**.
- SC-LoRA math (`frm_sclora_lr1e4`, 3-seed): GSM8K 60.9±0.4, ret 18.1±0.2 @ F_Δ≈0.86 — sits on
  the frm math law (n=48, r=−0.90), residual −1.8 pp: magnitude explains the SC-LoRA↔LoRA+wd
  math gap.

### Geometry sign-flip — convention decision 2026-07-14
- Print the retention_mean-consistent values: **amp_top +0.31, ein_top +0.41, e_top −0.17**
  (n=173, drop SC-LoRA; PiSSA absent from CS subset). The BBH-only variant gives
  +0.27/+0.29/−0.15 — same qualitative claim (2 of 3 flip), do not mix bases in one paragraph.

### Still genuinely pending (post-refresh)
CorDA clean nq_open CS re-run (zero post-07-11 corda cells); Qwen CS seeds 43/44; Qwen base
ceilings (no-FT); b4_lora_null lr1e4/3e4 + b4_cordapp full evals; r16-collapse seed replicate.

---

## 13. FULL-CAMPAIGN RECOMPUTE `[RECOMPUTED 2026-07-16, n>=3 seed campaign]`

**Source:** all `results/*/summary.json` on d001 after the 30-node seed campaign
(1418 summaries; 1299 usable = finite F_Δ & retention, corda/smoke excluded), plus the
fleet-merged geometry (`results/geo_drift/adapter_metrics_merged.jsonl`, 1398 adapters) and CE
(`results/forgetting_merged.jsonl`, 1308 adapters) aggregates. Recompute script:
`analyze_full_2026-07-16.py`. Seeds 42/43/44 everywhere (+45/46 partial via idle-GPU gap-fill).

### 13.1 The magnitude law, per family — r(retention_core, log10 F_Δ), ALL seeds pooled

| Family | r | rank-r | n | seeds |
|---|---|---|---|---|
| Llama-2 CS (lrsw) | **−0.886** | −0.911 | 178 | 42–45 |
| Llama-2 math (lrswm) | **−0.865** | −0.833 | 120 | 42–44 |
| Qwen-2.5 CS (qwsw) | **−0.837** | −0.779 | 134 | 42–45 |
| Qwen-2.5 math (qwswm) | **−0.830** | −0.602 | 156 | 42–44 |
| Llama CS grid (frc) | **−0.928** | −0.952 | 271 | 42–46 |
| Llama math-395k (frm) | **−0.929** | −0.969 | 142 | 42–44 |
| ALL pooled | −0.845 | **−0.922** | 1001 | |

Consistent with the 2026-07-02 single-seed values (−0.86/−0.97) at ~4× the n — the law
replicates across seeds, both models, and both task types.

### 13.2 Seed-averaged law + error bars (the campaign's goal)

- Seed-averaged per-cell r: lrsw −0.916 (54 cells), lrswm −0.871 (42), qwsw −0.800 (49),
  qwswm −0.832 (58), frc −0.928 (74), frm −0.896 (51).
- **287 cells now have n≥3 seeds** (49/36/40/47/72/43 per family).
- Within-cell seed noise is small: SD(retention) 0.33–1.0 pp (Llama), 2.1–2.6 pp (Qwen);
  SD(F_Δ) 0.01–2.0. The law's ~15–30 pp retention range is 10–50× the seed noise.

### 13.3 Geometry & CE corroboration at full n (was n=71 forensics)

- Spread vs retention (n=1222): spec_mean **−0.758**, fro_total −0.743, spec_max −0.738;
  stable_rank_w −0.36, eff_rank_w −0.30.
- **Partial correlation:** r(spec_max, ret | log F_Δ) = **+0.195** with r(F_Δ, spec_max)=0.92 —
  spread is largely collinear with magnitude; **F_Δ is the dominant single predictor.**
- CE (independent measure, n=1094): r(log F_Δ, forgetting_ce) = **+0.786**;
  r(forgetting_ce, retention) = −0.653.

### 13.4 In flight (not in this recompute)

- 7 remaining sweep dora evals (~2–5 h ETAs) + ~90 gap-fill s45/s46 cells still evaluating.
- **DeepSeek-V4-Flash 284B generalization run:** all 7 methods training (lora/milora/lorawd at
  3 seeds; dora/clora/sclora/lora_null seed-42 up, remaining seeds staged) — first summaries
  expected ~24 h; NOT yet part of any number above.

## 17. ADVERSARIAL-REVIEW RECOMPUTE `[2026-07-16, n=1018 rows — script: analyze_adversarial_2026-07-16.py]`

Repairs from `paper/writing/adversarial_review_2026-07-16.md` §3. Source data as §13-full-campaign
plus the dora tail / gap-fill cells that landed 07-16 (1018 usable rows vs 1001).

### 17.1 Functional form: it is a MAGNITUDE RELATION with a knee, not a single log-linear law

2-segment fits beat a single line in 5/6 families (F = 8.6–38.4; frm F=1.6 is the exception).
Shape = flat-then-falling: below the knee slopes are ≈0 (Qwen +0.9/+1.5 pp/dec, Llama −2.4…−13.8),
above it −7.4…−40.9 pp/dec. Qwen bottom-half (median split) even flips sign (+0.33 CS / +0.41 math).
Monotonicity above the knee is robust: healthy-only (ret≥15) r = −0.68…−0.94; drop-top-F_Δ-quartile
keeps r = −0.27…−0.94 (weakest qwswm — its signal is tail-anchored; disclose).
**Normalized slopes (pp/dec ÷ family retention range) do NOT converge: −0.34…−0.71** →
per the pre-registered decision rule, headline wording = **"magnitude relation"**, "law" only with
the knee caveat. Knees (log10 F_Δ): lrsw 0.00, lrswm −0.48, qwsw −0.68, qwswm −0.92, frc −0.45, frm −0.50.

### 17.2 LR rescue (backs the §2 rewrite)

| family | R²(F_Δ) | R²(LR-dummies) | partial r(F_Δ\|LR) | partial r(LR\|F_Δ) | fixed-LR strata r (lr≥1e-4) |
|---|---|---|---|---|---|
| lrsw | 0.785 | 0.765 | −0.61 (t=9.9) | −0.18 | −0.71…−0.97 |
| lrswm | 0.747 | 0.638 | −0.58 (t=7.6) | −0.16 | −0.55…−0.85 |
| qwsw | 0.707 | 0.658 | −0.68 (t=10.6) | +0.18 | −0.67…−0.87 |
| qwswm | 0.689 | 0.605 | −0.74 (t=13.6) | +0.29 | −0.67…−0.94 |
| frc | 0.861 | 0.391 | −0.89 (t=32.6) | +0.23 | −0.86…−0.94 |
| frm | 0.863 | 0.367 | −0.91 (t=25.5) | −0.01 | −0.83…−0.96 |

Low-LR strata (≤5e-5) are flat/noisy (r −0.16…+0.58) — consistent with the 17.1 knee: below the
knee there is nothing for ANY predictor to explain.

### 17.3 Direction is a real ~1–4 pp second-order effect (magnitude still dominates)

- Partial r(log spec_max, ret | log F_Δ) = **+0.117 adapter-level (n=1016, t=3.8)**, +0.115
  cell-level (n=328, t=2.1). (The 07-16 review quoted +0.195/+0.198 on the wider 1222-row
  geometry match incl. non-family runs; family-only is smaller but same sign/significance.)
  spec_mean carries no residual signal (+0.03, ns) — the residual direction signal lives in the
  top singular value.
- Method offsets at matched F_Δ (OLS ret ~ log F_Δ + method): **SC-LoRA −1.3…−3.7 pp below**
  (sig. in lrsw/lrswm/frc/frm), **PiSSA −5.9/−11.4 pp** (frc/frm), DoRA +1.7…+4.6 above.
  Qwen offsets ns (larger seed noise). → "magnitude, not direction" must be softened to
  "magnitude first-order (R² 0.69–0.86); direction/method a 1–4 pp second-order offset".

### 17.4 Adaptation-efficiency ANCOVA (the sharper slogan)

Retention cost: R²(ret ~ log F_Δ) = 0.69–0.86; adding method dummies gains only +0.01…+0.14
with offsets ≤ a few pp. Adaptation bought: R²(adapt ~ log F_Δ) ≈ 0.00–0.37 in sweeps (0.86 frm),
and method dummies move adapt by a 4.9–16.0 pp spread. **The retention cost of a unit of update
magnitude is near-universal; methods differ mainly in how much adaptation that unit buys.**

### 17.5 Within-cell micro-test (closest thing to a causal signal)

At FIXED recipe (cell), seed-level log-F_Δ fluctuations predict seed-level retention fluctuations:
pooled demeaned r = **−0.713, n=952 obs / 290 cells, t=−31.3**. (Raw-F_Δ demeaned r = −0.18 —
the effect is log-scale, as the relation predicts.) → main text.

### 17.6 ARC-c contamination control

retention_broad without ARC-c (ARC-c is both trained-on in CS and in the broad suite):
r changes by ≤0.09 everywhere (largest lrswm −0.906→−0.816); qwswm/frm unchanged. Non-issue,
one disclosure sentence.

### 17.7 Format-collapse control (proxy)

Per-item parse rates were not retained in 7B artifacts (per-subtask rows ARE saved for DeepSeek
runs from 07-16 on). Proxy = drop cells with adapt<25 or any zero retention task: r moves
≤0.03 in 4 families; lrsw −0.886→−0.866 (15 dropped), **qwswm −0.830→−0.696 (9 dropped)** —
Qwen-math's pooled r is partly carried by collapsed cells; quote the clean −0.70 alongside.

### 17.8 CE corroboration, WITHIN family (framing repair)

r(log F_Δ, CE) = +0.81…+0.92 (partly mechanical — same ΔW); the evidential link is
**r(CE, retention): lrsw −0.860, lrswm −0.923, qwsw −0.631, qwswm −0.792, frc −0.858, frm −0.896**.
Qwen coverage holes: 123 runs lack CE (`jobs/ce_backfill_qwen.txt`); fills as fleet finalize
(geo+CE battery) completes per node — recompute before freeze.

### 17.9 F_Δ decomposition

R²(ret ~ log‖ΔW‖_F) = 0.56 vs R²(ret ~ log F_Δ) = 0.71 — the gap IS the adapt-distribution
(direction) weighting; label the axis "effective update magnitude on the adaptation
distribution". Alignment (F_Δ/‖ΔW‖_F) is NOT method-invariant (dora 2.7e-3 vs clora 1.55e-3,
lora_null 1.61e-3) but within-method spread is as large as the between-method gap.
dw_sv_max R² = 0.58.

### 17.10 Bookkeeping

- `analyze_full_2026-07-16.py` geometry keys fixed (stable_rank_w/eff_rank_w/spec_mean —
  two rows were silently empty).
- All numbers above exclude corda/cordapp/smoke and non-finite rows; families lrsw/lrswm/qwsw/
  qwswm/frc/frm only.

## 18. FINAL FREEZE — POST-FLEET-KILL `[FROZEN 2026-07-17, final synced dataset; scripts: analyze_full/adversarial/ebatch re-run on results/*/summary.json]`

> The fleet was killed overnight 2026-07-16/17 before the scheduled 06:00–08:00 freeze pass.
> This section IS that freeze, run on everything that synced (last summary.json ≈ 06:34Z).
> Dataset: **1,661 result dirs, 1,500 with full evals**; quarantine regenerated
> (`results/quarantine_diverged.txt`, 71 runs). `campaign_summary.jsonl` (645 rows) and
> `results_book/` are STALE — do not source numbers from them.

### 18.1 The magnitude relation — final, multi-seed, all arms (supersedes §13/§17 headline rows)

Pooled r(retention, log10 F_Δ), all seeds, quarantine-filtered (n=1035 usable):
| family | r | rank-r | n | seeds |
|---|---|---|---|---|
| Llama-2 CS (lrsw) | −0.886 | −0.908 | 180 | 42–45 |
| Llama-2 math (lrswm) | −0.865 | −0.833 | 120 | 42–44 |
| Qwen-2.5 CS (qwsw) | −0.840 | −0.778 | 151 | 42–45 |
| **Qwen-2.5 math (qwswm)** | **−0.830** | −0.582 | **164** | 42–44 |
| Llama CS grid (frc) | −0.928 | −0.952 | 276 | 42–46 |
| Llama math-395k (frm) | −0.929 | −0.969 | 144 | 42–45 |
| ALL pooled | −0.847 | −0.923 | 1035 | |

Seed-averaged (cell-level) r: lrsw −0.916 (56 cells), lrswm −0.871 (42), qwsw −0.799 (58),
qwswm −0.832 (62), frc −0.928 (74), frm −0.896 (51). Within-cell seed SD(ret): lrsw 0.94pp,
lrswm 0.33, frc 0.75, frm 1.00 (qwsw 2.73 / qwswm 2.07, inflated by seed-unstable F_Δ cells).
**Qwen math is no longer qualitative-only**: 3 seeds, n=164, r=−0.830 (clean-subset −0.70 per
§17.7 convention — quote both).

### 18.2 Functional form (A1 final)

2-segment beats linear in every family (F2seg 1.6–40.0); knees (log10 F_Δ): lrsw −0.02,
lrswm −0.48, qwsw −0.69, qwswm −0.91, frc −0.45, frm −0.50. Below-knee slopes ≈ flat-to-mild
(−13.8…+2.0), above-knee steep (−7.5…−40.8). Normalized slopes do NOT converge
(−0.33…−0.70) → headline wording stays **"magnitude relation (flat-then-falling with a knee)"**,
"law" only with the knee caveat (pre-registered rule, §17.1).

### 18.3 E-batch — FINAL verdicts (supersedes assessment_2026-07-17.md counts)

- **E1 interventional — COMPLETE, 15/15 trained rescales** (dora f015/f040/f080 landed post-assessment)
  + 9/9 random-direction controls. Trained rescales: mean on-curve residual **+1.29 ± 2.07pp**,
  within-set r=−0.732. Random-direction at matched F_Δ: mean −1.76 ± 1.32pp ⇒ **direction penalty
  −3.05pp vs trained**. Upscaling asymmetry confirmed (clora 0.65→0.78: −3.86pp below curve;
  randdir f080 −1.8…−3.8 below). Rescale>retrain holds: e.g. e1_lora_f040 ret 26.93/adapt 75.4
  vs trained lr3e-4 twin 24.4/79.1.
- **E2 full-FT anchor — COMPLETE 3/3**: monotone (26.9→26.2→17.1 as F_Δ 0.023→0.395) but
  −4.1…−8.6pp BELOW the LoRA-family curve at matched F_Δ (dense ΔW: dw_sv_max ~4 vs 30–40;
  fft F_Δ under-counts dense mass — disclosed). Universal in form, family-specific in level.
- **E3 Qwen densification — PARTIAL 13/26** (9 qwsw + 4 qwswm mid-LR cells landed; 2nd wave lost).
  Bottom-half r: qwsw −0.04, qwswm −0.03 ⇒ **flat below the knee, not positive** — the old
  anti-replication is dead; remaining hole is only figure density.
- **E4 SC-LoRA eval-matched — 20/24 evaluated (lr1e3×2, s46×2 unevaluated)**: eval-matched mean
  residual **+0.92pp ABOVE curve (n=20)** vs nq_open-calibrated −3.39pp (n=24). SC-LoRA's
  deviation = calibration-set artifact, not method geometry. (b4_sclora_lr2e5_s42 adapt=13.2
  low-LR format fluke; retention unaffected — exclude from adapt comparisons.)
- **E5 replay — TRAINED 4/4, BENCHMARK-EVAL LOST (0/4).** CE salvage (forgetting.json):
  replay-5% CE vs matched plain-LoRA twins, all 4 cells lower —
  lr3e4: 2.248/2.204 vs 2.307/2.254; lr5e4: 2.465/2.462 vs 2.551/2.526 (Δ≈−0.05…−0.09 CE;
  KL likewise lower). Verdict: replay gives a small consistent CE-forgetting reduction;
  the benchmark-retention comparison died with the fleet — state as partially answered.
- **E6 wd-generalization — SPLIT VERDICT.** MiLoRA+wd0.3: +1.75/+2.36pp above curve (2/2,
  adapt 80.2 at lr5e4) — wd transfers. **DoRA+wd0.3: DEGENERATE** (benchmark evals lost, but
  salvage shows CE 20.8/10.4 vs DoRA twins 2.1/2.6, spec_max up to 1183) — naive AdamW wd on
  DoRA (incl. its magnitude vector) breaks training. Report as a boundary: wd is not a
  universally free knob; it transfers to MiLoRA, breaks DoRA as-implemented.
- **E7 bridging arms — 7/8**: brl (Llama, MedMCQA, attn-only) r=−0.878 (4/4);
  brq (Qwen) r=−0.995 (3/4, lr1e3 lost). Relation reproduces off-recipe on both models.
- **DSV4 284B — ABSENT (0/21 synced).** Trains + geometry completed on DeepSeek nodes; capped
  evals never synced before the kill. The generalization section has NO data in this repo —
  ledger as designed-but-lost (spec: handoff/DEEPSEEK_GEN_EXPERIMENT.md).

### 18.4 Direction/method offsets at matched F_Δ (A3 final, n=1018)

partial r(log spec_max, ret | log F_Δ) = +0.117 (t=3.7); cell-level +0.115 (t=2.1).
Method dummies (OLS, ref=clora): significant offsets are bounded ±1.2–4.6pp; frc grid:
sclora −3.7±0.5*, pissa −5.9±1.0*, lora_null −2.0±0.5*; frm: pissa −11.4±2.1* (collapse-driven),
dora +4.6±1.4*. Qwen arms: no significant offsets (±0.1–3.4, all n.s.).
⇒ direction/method identity = real, bounded, second-order; SC-LoRA's negative offset is the
E4 calibration artifact.

### 18.5 LR-artifact battery (A2 final)

R²(fd) vs R²(LR-continuous): 0.785/0.509 (lrsw), 0.747/0.516 (lrswm), 0.705/0.450 (qwsw),
0.689/0.308 (qwswm), 0.861/0.275 (frc), 0.863/0.223 (frm). Partials: r(fd|LR) −0.58…−0.91
(|t|≥7.6) vs r(LR|fd) −0.17…+0.29 (|t|≤4). Fixed-LR strata: r ≤ −0.7 at every LR ≥ 1e-4 in
every family. LR is a proxy; F_Δ is the variable.

### 18.6 Cross-checks that held (A5–A9 final)

Within-cell demeaned micro-test r=−0.713 (n=954/290 cells, t=−31.3); ARC-c exclusion moves r
≤0.09; format-collapse-clean r within 0.03 except qwswm −0.830→−0.695 (quote both); CE
corroboration r(CE,ret) −0.63…−0.92 per family (136 Qwen runs still lack CE —
jobs/ce_backfill_qwen.txt regenerated, unfillable without GPUs); F_Δ beats ‖ΔW‖_F
(R² 0.72 vs 0.56) and dw_sv_max (0.58).

### 18.7 Data-state ledger (for the paper's reproducibility + limitations sections)

COMPLETE: E1, E2, E4 (20/24), E6-MiLoRA, E7-Llama, Llama 3-seed spine (frc 284, frm 163,
lrsw 201, lrswm 120 evaluated), Qwen 3-seed sweeps (qwsw 155/182, qwswm 165/187), quarantine (71).
TRUNCATED/LOST: E3 13/26 · E5 0/4 benchmark (CE salvaged) · E6-DoRA 0/2 benchmark (degenerate
by CE) · brq 3/4 · qwsw 27 / qwswm 22 trained-not-evaluated · DSV4 0/21 · 284B ceiling absent ·
base-ceiling dirs: 4/22 evaluated (Llama 26.0 / broad 35.26; Qwen 44.35 landed 07-16).

---

## §19 POST-FREEZE ADDENDUM (2026-07-17, offline pass) — ADDITIVE ONLY

`[§18 is unchanged and remains the freeze. §19 records CPU-only analyses run after
the freeze plus two ledger corrections. Scripts + outputs: analysis_final/
ladder_2026-07-17.py, seed_stability_2026-07-17.py, ds284b_recurrence_2026-07-17.py.
Docs: analysis_final/06_pooled_ladder.md, 07_deepseek_284b_recurrence.md.
Every script preflight-reproduces §18.1 (n=1035, r=−0.847, all family cells to
3 decimals) before emitting anything.]`

### 19.1 Pooled nested ΔR² ladder (the one-table version of the title claim)

Frozen pool ∩ geometry, n=1034, run-level, family FE, quarantine-included:
R² 0.390 (family) → **0.785 (+0.395, F≈1890)** log10 F_Δ → 0.802 (**+0.017**, F=29.5)
geometry (e_top, log spec_max, stable_rank) → 0.808 (+0.006, F=3.5) method dummies.
CE-subset ladder (n=911): KL adds +0.005 after magnitude; KL alone +0.340 vs
magnitude alone +0.420 same-sample (05's family-level "KL beats log F_Δ" is
granularity-specific — quote both at their own levels). Standardized betas:
log F_Δ −0.744 vs geometry |β| ≤ 0.138. Robust in all variants (quarantine-excluded,
current pool, seed-averaged, no-FE); no-FE M1 R²=0.717 independently reproduces
§18.6's 0.72. Headline sentence: magnitude +39.5% variance, geometry +1.7%,
method +0.6%.

### 19.2 Geometry seed-stability

Partial r(·, ret | fam+log F_Δ) across seeds 42/43/44/45: log F_Δ −0.828±0.046 (4/4);
e_top −0.218±0.128 (4/4); stable_rank −0.313±0.184 (4/4); log spec_max +0.105±0.146
(**3/4, crosses zero at s44**). Second-order geometry is not a seed artifact, but
spec_max-based direction effects (§18.4's +0.117) need a seed-variability caveat —
prefer e_top/stable_rank as geometry exhibits.

### 19.3 Ledger correction #1 — DSV4 284B is not data-free

§18.3/§18.7 "DSV4 0/21, NO data" was true at freeze time for retention evals (still
lost, GPU-gated). Post-freeze salvage landed: **20/21 MedMCQA adapt scores**
(dsv4_adapt_n1000_logscores.jsonl; missing dsv4_lorawd_r16_lr5e4_s42),
**21/21 factor-only geometry** (adapter_metrics_deepseek.jsonl + permatrix),
**21/21 raw adapter tar sets integrity-verified** (SHA256SUMS, 63 parts).
Result: method update-shape fingerprint recurs at 284B/MLA — stable-rank method
ordering rank-r **+0.86 pooled**, positive in all 6 families; two 7B clusters
reproduce ({sclora,milora,lora_null} high-spread vs {lora,lorawd,dora,clora}
concentrated). Adapt (n=19, diverged dsv4_lora_null_r16_lr5e4_s44 excluded)
shows no fro-ordering (r=+0.20/rank −0.10) — descriptive only. spec/fro orderings
carry a per-method fixed-LR confound at 284B: do not quote. NO 284B retention/CE/
magnitude-relation claims (07 §3).

### 19.4 Ledger correction #2 — seven post-freeze stragglers

Seven runs synced 07-17 12:49 after the freeze (1 lrsw, 2 qwsw, 4 qwswm; names in
ladder_2026-07-17.py). Current pool = 1042. All §18 numbers remain quoted on the
frozen n=1035; the stragglers move ladder ΔR²s by ≤0.002 (V2). Any future re-freeze
should ingest them.
