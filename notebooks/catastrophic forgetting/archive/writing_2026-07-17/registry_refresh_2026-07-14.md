# Registry refresh verification — post-audit landed data

**Auditor role:** data-verifier / registry-vs-paper reconciliation (follow-up to `claims_coverage_audit_sat.md`, 2026-07-11).
**Date:** 2026-07-14. **Python:** `/usr/bin/python3` (scripts in session scratchpad `load.py` / `analyze.py`).
**Live registry:** `results/campaign_summary.jsonl` — 571 raw rows → **555 unique** by latest `evaluated_at` (dedup rule of `key_numbers.md` applied throughout). **66 cells** carry latest `evaluated_at` ≥ 2026-07-12, i.e. landed after the Saturday audit: `frc` 45, `frm` 10, `qwswm` 9, `base` 1, `b4` 1.
**Ground truth conventions:** `key_numbers.md` §0/§1/§4/§9/§11/§13. Diverged-cell convention (paper §Learning-rate sweep, L360): cells that diverge/collapse to all-zero evals are **dropped with disclosure** — applied here to `qwswm_lora_r32_lr5e4_s42` (F_Δ=20.9, all scores 0.0) and stated in every fit below.

> **Headline finding:** three of the audit's four "queued NOWHERE" gaps are now closed by
> landed data (r16 param-match control, base ceilings, Qwen math high-LR — the last only
> partially), and every recomputed headline the paper already carries still reproduces
> (lrsw pooled law r=−0.858/−14.78/n=49 re-verified this pass). The two new findings that
> need decisions are: **(1) the registry has an ingestion hole** — 51 evaluated Qwen-math
> cells (a near-complete 6-adapter × 7-LR sweep, s42, incl. 7 seed-replicate cells) sit as
> `results/qwswm_*/summary.json` on disk but were never appended to
> `campaign_summary.jsonl`; the in-registry Qwen-math verdict is therefore much weaker
> than the on-disk one. **(2)** one r16 control cell (`frc_lorawdr16_wd0p3_lr3e4`,
> cs_avg=13.53) is an adaptation-format collapse, not an eval bug — documented below,
> not papered over.

---

## 1. Qwen math law status (highest stakes)

**Filters:** run_name prefix `qwswm_`, dedup latest `evaluated_at`, all seed 42 (no other seeds in registry). Registry contains **14 cells**: 5 × `lora_r16` low-LR (07-01), 7 × `lora_r32` 7-LR sweep (07-13; lr {2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3}), 1 × `lora_r32_lr2e4_ep6` (6-epoch variant), 1 × `lorawd_wd0p3_lr3e4_ep6`. Diverged cell = `qwswm_lora_r32_lr5e4_s42` (F_Δ=20.89, cs/bbh/mmlu_pro/ret all 0.0) — excluded under the paper's existing NaN/divergence convention, with disclosure; fits also shown including it. BBH retention = `retention_bbh` where present, else `bbh` (identical field on these rows).

### Recomputed fits — BBH retention vs log10(F_Δ)

| set | n | r | R² | slope (pp/dec) | p | F_Δ span |
|---|---|---|---|---|---|---|
| pooled, all 14 minus diverged | 13 | **−0.795** | 0.63 | −10.65 | 1.2e-3 | 0.039–0.90 (**1.37 dec**) |
| pooled INCL diverged | 14 | −0.950 | 0.90 | −17.27 | 2.0e-7 | 0.039–20.9 (2.73 dec) |
| LoRA r16+r32 base recipe (excl ep6, excl div) | 11 | −0.790 | 0.62 | −10.55 | 3.8e-3 | 0.039–0.90 |
| LoRA r32 sweep only (excl div) | 6 | −0.840 | 0.71 | −12.09 | 3.6e-2 | 0.039–0.90 |
| LoRA r16 only (old low-LR) | 5 | −0.237 | 0.06 | −1.26 | 0.70 | 0.039–0.16 (0.61 dec) |

On `retention_mean` (core, MMLU-Pro-included): pooled excl div n=13 r=−0.738, slope −14.23, p=4.0e-3; r32-only n=6 r=−0.839, slope −18.0, p=0.037. The core metric now agrees in sign with BBH-only — the §11 "spurious +0.67" no longer reproduces on the current registry (see parser note below).

### Leverage caveat (do not omit)

The negative fit is **anchored by a single high-magnitude cell**, `qwswm_lora_r32_lr1e3_s42` (F_Δ=0.90, BBH 29.27). Sensitivity: drop lr1e3 → pooled r=−0.42 (p=0.17); drop lr1e3 + the two ep6 cells → r=−0.12 (p=0.74, the old "flat below the knee" regime). Leave-one-out r ranges −0.86…−0.42. Spearman on the r32 sweep alone is ρ=−0.60 (p=0.21, n=6). So **in-registry** Qwen math has moved from "flat, no high-LR points" to "negative and significant, but resting on one anchor cell plus the (excluded) diverged cell".

### The unregistered sweep (changes the verdict if ingested)

`results/qwswm_*/summary.json` holds **51 additional evaluated cells absent from campaign_summary.jsonl** (CLoRA-k1024 ×7, DoRA ×7, MiLoRA ×7, SC-LoRA ×7, LoRA-Null ×8, LoRA+wd0.3 ×9 incl. the five low-LR cells §11 counted as "assessed", LoRA r16 lr5e4/1e3, + 7 s43/s44 seed replicates; datestamps 07-09…07-14). Pooled disk-level fit (s42 only, all-zero cells excluded — `lorawd_wd0p3_lr1e3` F_Δ=15.8 also all-zero): **n=56, r=−0.703, R²=0.49, slope −14.40, p=1.5e-9, F_Δ 0.029–0.90 (1.49 decades)**. Within-method: LoRA-r32 −0.77, LoRA-r16 −0.77, MiLoRA −0.76, SC-LoRA −0.82, CLoRA −0.74, DoRA −0.70; LoRA+wd +0.60 ns over 0.51 dec (flat/capped by construction, same signature as both CS arms); LoRA-Null −0.54 ns. This is a genuine multi-adapter qualitative replication with many independent high-magnitude cells (dora lr1e3 BBH 35.7, milora lr1e3 14.0, sclora lr1e3 12.6, sclora 5e4 34.1, clora 1e3 39.9 …) — **but it is not in the registry**, so under the project's single-source-of-truth rule it cannot yet be quoted.

### MMLU-Pro parser caveat (key_numbers §11)

The new r32 rows' `mmlu_pro` values look **sane, not degenerate**: 39.45–42.29 across the healthy LRs (in line with the Qwen CS arm), dropping to 6.13 at lr1e3 coherently with BBH 29.27 (real collapse, not parser junk). The §11 "+0.67 core artifact" was computed on the old 10-cell union whose 5 `qwswm_lorawd` low-LR rows are **no longer (never were) in the registry**; on today's registry the core-metric fit is negative (−0.74). Recommendation: keep reporting Qwen math on BBH-only as the pre-registered convention, but soften "MMLU-Pro is broken on Qwen math" to "was unreliable on the early low-LR block; current r32 rows pass sanity" pending a spot-check of raw MMLU-Pro outputs.

### VERDICT to propose

Qwen math **now qualitatively replicates the law's direction in-registry (negative slope over 1.37 decades, pooled r=−0.80, p≈1e-3, n=13 excl. one disclosed diverged cell), but at honest strength this is "suggestive, single-anchor"** — the significance collapses if the one lr1e-3 cell is removed. The on-disk (unregistered) 56-cell sweep upgrades this to a solid qualitative replication (r=−0.70, p=1.5e-9, 6 adapters individually negative), comparable in character to the Llama math law though weaker than Qwen CS (−0.86).

- **Current paper text** (§law L497-506, Limits L1236-1245): "in progress and does not yet test the law … n=10 … flat (r=−0.05) … high-rate cells queued; until they land, Qwen math is pending, not a result."
- **Supported text (registry as-is):** "The Qwen math arm now spans 1.4 decades of F_Δ (n=13 assessed cells excl. one diverged at lr5e-4, disclosed) and the BBH-retention fit turns negative and significant (r=−0.80, slope ≈ −11 pp/dec, p=1e-3) — though the high-magnitude end currently rests on a single lr1e-3 cell, so we report this as a directional, not quantitative, replication."
- **Supported text (after ingesting the disk cells — recommended):** "a 6-adapter × 7-LR Qwen math sweep (n=56 assessed s42 cells, two diverged cells dropped with disclosure) gives pooled BBH-retention r=−0.70 (R²=0.49, slope −14.4, p=1.5e-9) over 1.5 decades, with every unconstrained adapter individually negative (−0.70…−0.82; LoRA+wd flat by construction) — a qualitative second-domain, second-architecture replication at moderate strength."
- **Severity: UPGRADE** (paper under-claims), but **gated on the ingestion decision**.
- **NEEDS HUMAN SIGN-OFF:** (a) why 51 evaluated qwswm summaries never entered `campaign_summary.jsonl` (collector not run for Node-B evacuated runs?) and whether to ingest them; (b) the lr1e-3-anchor disclosure wording; (c) whether the ep6 (6-epoch) variants belong in the pooled fit (excluding them changes pooled n=13 → 11, r −0.795 → −0.790; immaterial, but state the choice).

---

## 2. r16/r32 param-match 2×2 (closes audit W2/B5a)

**Filters:** the named `frc_*_c256_s42` cells, dedup latest. Note these are the **c256 faithful-CS recipe**, not the `lrsw_` sweep recipe — cross-recipe comparisons to the 81.6/25.6 op-point are qualitative. Law predictions below use the re-verified lrsw pooled fit (n=49, r=−0.858, slope −14.78, intercept from this pass); all frc cells share a common ≈ +2 pp recipe offset, so read the *ordering*, not the absolute residuals.

| cell | CS-8 | ret_core | F_Δ | law-pred | resid |
|---|---|---|---|---|---|
| frc_lorawdr16_wd0p3_lr3e4 (+_reeval) | **13.53 / 13.54** | 26.84 | 0.3275 | 25.01 | +1.83 |
| frc_lorawdr16_wd0p3_lr5e4 | **81.04** | 26.27 | 0.3339 | 24.89 | +1.38 |
| frc_lora_r8_lr3e4 | 78.99 | 23.96 | 0.5178 | 22.07 | +1.89 |
| frc_lora_r16_lr3e4 | 79.56 | 23.00 | 0.6031 | 21.09 | +1.91 |
| frc_lora_r32_lr3e4 | 73.46 | 22.19 | 0.7393 | 19.79 | +2.40 |
| frc_lorawd_wd0p3_lr3e4 (r32) | 76.17 | 27.28 | 0.3581 | 24.44 | +2.84 |
| frc_lorawd_wd0p3_lr5e4 (r32) | 82.05 | 25.04 | 0.4097 | 23.57 | +1.47 |

**Findings.**
- **LoRA+wd @ r16 (half the parameters, 28.0M) lands on the r32 operating point:** 81.04 CS-8 / 26.27 ret at F_Δ=0.334 vs the paper's r32 claim 81.6/25.6 @ 0.394 (lrsw) and the same-recipe r32 sibling 82.05/25.04 @ 0.410. The wd knob, not the extra rank, buys the operating point.
- **Plain LoRA slides down-curve as rank doubles at fixed LR 3e-4:** r8→r16→r32 raises F_Δ 0.518→0.603→0.739 and retention falls 23.96→23.00→22.19, tracking the law's slope (predicted fall r8→r32: −2.28 pp; observed −1.77). At r32 adaptation also drops (73.46) — extra rank buys magnitude, not accuracy, here.
- **Rank main-effect vanishes once F_Δ is controlled (qualitative, these cells only):** at matched F_Δ ≈ 0.33 the r16 and r32 wd0.3 cells sit within ~1 pp of retention of each other (26.3–26.8 vs 25.0–27.3), and the plain-LoRA rank series' residuals from the law are flat (+1.9/+1.9/+2.4) while raw retention spans 1.8 pp — rank acts through F_Δ.
- **Current paper text** (L156, Limits): "the matched-capacity control at rank 16 remains deferred (see Limitations)". → **Supported text:** "A rank-16 LoRA+wd control (28.0M params, matched to plain LoRA/DoRA) reproduces the r32 operating point (81.0 CS-8 / 26.3 ret-core at F_Δ=0.33), and a plain-LoRA rank series (r8/16/32 at fixed LR) moves along the same curve (F_Δ 0.52→0.74, retention 24.0→22.2) — the capacity objection resolves in the law's favor: rank shifts F_Δ, and F_Δ sets retention." **Severity: UPGRADE** (retire the "deferred" limitation; keep the single-seed, c256-recipe caveat).

### ANOMALY — `frc_lorawdr16_wd0p3_lr3e4` cs_avg=13.53 (investigated, not papered over)

- **Training is healthy:** `train_registry.jsonl` shows num_epochs=3, 31,210 steps (identical schedule to the lr5e4 sibling), runtime 7,235 s vs 7,224 s, final train loss **0.805 vs 0.824** (the "collapsed" cell has the *lower* loss); dwF trace smooth (peak ≈226, settling 188). **No ep (epoch) marker difference** — neither r16 sibling carries an `_ep` suffix; both are 3-epoch (the only ep6-suffixed runs in the registry are Qwen math).
- **The failure is adaptation output-format collapse, not knowledge loss or an eval bug:** per-dataset CS-8 (summary.json) = boolq 61.7, hellaswag 39.8, but piqa 3.1, winogrande 1.4, ARC-E 0.6, ARC-C 0.4, SIQA 0.3, OBQA 0.8 — **far below chance** on binary tasks, the signature of unparseable answer formatting. Retention is fully intact (BBH 34.03, MMLU-Pro 19.66, MMLU 47.18 → ret_core 26.84, *above* the no-FT base 25.89). The dedicated `_reeval` reproduces it to 0.01 (13.54), so it is deterministic, not an eval flake.
- **It is not unique:** the same frc wd-grid contains sibling low-cs cells at low effective update (wd0p3_lr1e4 cs 23.8, wd0p2_lr1e4 35.4, wd0p5_lr3e4 43.6, wd0p1_lr3e4 49.1, all with retention ≥24) — a "format-not-locked-in" basin analogous to MiLoRA's seed-collapse basin (audit W1). At r16 the wd0.3 effective update at lr3e4 is evidently below the format-locking threshold; lr5e4 clears it (81.0).
- **Recommendation:** treat the cell as a valid *retention/F_Δ* point but an adaptation-failure cell excluded from operating-point claims, disclosed the same way MiLoRA's 34.5 collapse seed is; do NOT delete it. **NEEDS HUMAN SIGN-OFF** (a seed replicate at r16/wd0.3/lr3e4 would settle whether the basin is deterministic or seed-luck).

---

## 3. Base ceilings (C5) — landed 07-12, closes an audit "queued NOWHERE" gap

**Cell:** `base_llama2_noft` (07-12, eval-only, no FT): BBH 32.96, MMLU-Pro 18.82 → **ret_core 25.89**; MMLU 40.88, ARC-C 44.80, TruthfulQA 38.85 → **ret_broad 35.26** (both field values verified = recomputed means).

**25.89 vs 26.0 reconciliation.** The paper's 26.0 is `[EXTERNAL: h00#6, h05]` (key_numbers §0/§12): BBH-AO **33.10** / MMLU-Pro **18.96** → 26.03, from an earlier handoff eval snapshot outside this registry. The new in-registry no-FT eval reads 0.14 pp lower on each component (32.96/18.82) → 25.89. This is eval-snapshot drift, not rounding (25.89 rounds to 25.9, not 26.0). Both are legitimate; they now differ in provenance: 25.89 is **same-harness, in-registry, recomputable**; 26.0 is external. **Proposed change:** paper §metrics L367 "for a core ceiling of **26.0**" → "for a core ceiling of **25.9** (in-registry no-FT eval; an earlier external snapshot gave 26.0)", and update the fig annotations ("base ceiling 26.0") accordingly — or keep 26.0 and tag the new cell as corroboration. Every "≈ ceiling" claim (LoRA+wd 25.6/25.9±0.4, hockey-stick asymptote 26.8, B4 SC-LoRA 26.5–27.0) survives under either choice; note the asymptote and several wd cells sit 0.9–1.9 pp *above* the measured ceiling, i.e. the "ceiling" is a soft reference, not a bound — worth one sentence. **Severity: FIX (small) — NEEDS HUMAN SIGN-OFF on which number becomes canonical.**

**Broad retention is no longer uncalibrated.** The hedges at L368-372 (%TODO), L1040, L1246-1250 ("broad … texture only, uncalibrated"; "%TODO(experiment): C5 base-ceiling calibration") are now stale: the broad ceiling is **35.26**, and the lrsw sweep's ret_broad spans 19.2–37.1 against it. **Proposed change:** delete both %TODOs, report broad retention as calibrated (ceiling 35.26), and re-express fig5's per-benchmark slopes against real ceilings (MMLU 40.88, ARC-C 44.80, TQA 38.85). **Severity: UPGRADE.**

**TruthfulQA immunity is real, not a floor artifact.** Base TQA = 38.85; the 49 lrsw s42 fine-tuned cells span 31.43–39.48 (mean 35.72, only 1/49 above base); slope vs log F_Δ = −0.46 pp/dec (r=−0.10, p=0.48) — re-verified. The values sit mid-scale, ~10-14 pp above the ~25 chance floor, with ample room to fall — and don't. Honest refinement: fine-tuning imposes a small **constant** offset (≈ −3.1 pp vs base) that is magnitude-independent. **Proposed change:** L580-584 "we read the TruthfulQA flatness cautiously … cannot rule out a near-chance-scoring floor artifact" + %TODO → "TruthfulQA's base ceiling (38.85) confirms the flatness is not a floor artifact: fine-tuned values (31.4–39.5) sit well above chance and show a small magnitude-independent offset (≈−3 pp) rather than a slope." **Severity: UPGRADE.**

---

## 4. frm_ 3-seed math

**Filters:** `frm_*_c256_s{42,43,44}` triplets, dedup latest; SD = sample SD (ddof=1). Headline **verified exactly**: `frm_lorawd_wd0p3_lr2e4_c256` GSM8K s42/s43/s44 = 67.25 / 65.88 / 67.25 → **66.79 ± 0.79 ("66.8±0.8" ✓)**, retention 25.40 ± 0.53.

All six 3-seed frm configs (GSM8K mean±SD | ret_core mean±SD | mean F_Δ):

| config | GSM8K | ret_core | F_Δ |
|---|---|---|---|
| lorawd_wd0p3_lr2e4 | **66.79 ± 0.79** | 25.40 ± 0.53 | 0.280 |
| lorawd_wd0p2_lr1e4 | 65.40 ± 1.73 | 25.12 ± 0.35 | 0.281 |
| lorawd_wd0_lr1e4 (plain-LoRA recipe) | 63.99 ± 0.87 | 23.02 ± 0.40 | 0.442 |
| milora_lr1e4 | 63.68 ± 0.80 | 23.97 ± 0.06 | 0.450 |
| sclora_lr1e4 | 60.47 ± 0.53 | 18.01 ± 0.37 | 0.856 |
| lora_lr3e4 | 59.59 ± 1.53 | 18.11 ± 0.31 | 1.288 |

(Seeds landed 07-11→07-14; s43/s44 for lora_lr3e4, milora, sclora are post-audit cells.) Retention SDs ≤ 0.53 across all six — the law's y-axis is seed-stable on math, matching the CS-arm finding.

**SC-LoRA vs LoRA+wd on the magnitude curve (qualitative check requested):** SC-LoRA math retention 18.0±0.4 at F_Δ=0.856 vs LoRA+wd 25.4±0.5 at F_Δ=0.280 — a 7.4 pp gap at 3.1× the magnitude. Against the pooled frm math law (s42 non-collapsed cells incl. c512 variants, n=48: r=−0.901, R²=0.81, slope −13.29, p=2.6e-18, F_Δ 0.20–10.8), SC-LoRA lr1e4's residual is **−1.77 pp** (essentially on-curve; its lr2e4 point is −4.09 below) and LoRA+wd's is −0.70. So **yes — the math points fall on the same magnitude curve to first order**: most of SC-LoRA's math forgetting is its magnitude, with a residual ~−2 pp below-curve tendency consistent with (and smaller than) its provisional CS-arm deviation. **Severity: UPGRADE** (paper can quote all six 3-seed math means; currently only the LoRA+wd headline is three-seed per L1239-40).

---

## 5. Sweep inventory refresh (coverage statements)

Unique cells by campaign prefix (555-cell dedup, this pass):

| prefix | unique | seed breakdown | notes |
|---|---|---|---|
| lrsw | 70 | s42 56 (49 sweep + 7 CorDA-contaminated), s43 7, s44 **7** | **DoRA s44 landed 07-11** — all 7 CS op-points now 3-seed |
| lrswm | 36 | s42 only | 5 adapters × 7 + sclora 1 |
| frm | 68 | s42 56, s43 6, s44 6 | six 3-seed triplets (item 4) |
| frc | 51 | s42 47, s43 2, s44 2 | wd×LR grid 33 + clora-k 9 + rank series 4 + lorawdr16 3 + pissa 1 + dora 1 |
| qwsw | 50 | s42 only | 49 CS + 1 corda (excluded) |
| qwswm | **14 in registry** | s42 only | **+51 evaluated on disk, unregistered (incl. 7 s43/s44)** |
| b4 | 5 | s42 | 4 sclora + **1 lora_null (lr2e5, 07-13)** |
| mtx / mtxm | 102 / 8 | 34×3 seeds / s42 | unchanged |
| base | 1 | — | `base_llama2_noft` (07-12) |

**Audit's four "queued NOWHERE" gaps — status:**
1. **Param-matched LoRA+wd @ r16 (W2/B5a): CLOSED.** `frc_lorawdr16_wd0p3_lr{3e4,5e4}` + plain-LoRA rank series `frc_lora_r{8,16,32}_lr3e4` all landed 07-12/13 (item 2). One cell is the disclosed format-collapse anomaly.
2. **CorDA clean nq_open CS re-run: STILL NOT RUN.** Zero corda cells evaluated after 07-11 (latest CorDA data: `frm_cordapp` math, 07-10; the 7 `lrsw_corda` rows remain the contaminated 06-29/30 re-eval). `b4_cordapp_r32_lr{1e4,3e4}` dirs exist with `forgetting.json` only (retention evaluated, no summary/registry row). CorDA stays withheld.
3. **Base-ceiling no-FT eval (C5): CLOSED for Llama-2** (item 3; one row covers all 5 benchmarks). A **Qwen** no-FT ceiling still does not exist (results_book 05 header: "no base reference evaluated"), which the Qwen slope-scale argument (−32 vs −15) leans on.
4. **Qwen math high-LR + Qwen seeds 43/44: PARTIALLY CLOSED, ingestion-gated.** In-registry: the r32 7-LR sweep landed (item 1), Qwen seeds still absent. On disk: the full 6-adapter math sweep + 7 s43/s44 math seed cells are evaluated but unregistered; **Qwen CS seeds 43/44 have not been run anywhere.**

**b4 LoRA-Null expansion:** partially landed — `b4_lora_null_r16_lr2e5_s42` in registry (ret 26.75 at F_Δ=0.149, eval-matched calibration ≈ ceiling, consistent with the B5 story); `b4_lora_null_r16_lr{1e4,3e4}` have `forgetting.json` only. 

**Still not run anywhere (unchanged honest limitations):** Qwen CS seeds 43/44; Qwen base ceiling; CorDA clean CS re-run; a seed replicate for the item-2 anomaly cell.

---

## Severity summary

| item | finding | severity |
|---|---|---|
| 1 | Qwen math negative in-registry (r=−0.80, 1.4 dec) but single-cell-anchored; 51-cell disk sweep (r=−0.70, p=1.5e-9) unregistered | **UPGRADE, gated — SIGN-OFF: ingest disk cells?** |
| 2 | r16 control closes W2: LoRA+wd@r16 = r32 op-point; plain LoRA r8→r32 slides down-curve; rank effect absorbed by F_Δ | **UPGRADE** (retire "deferred" limitation) |
| 2a | cs=13.53 cell = deterministic adaptation-format collapse (train healthy, retention intact, below-chance CS parsing) | **disclose, keep — SIGN-OFF: seed replicate** |
| 3 | Core ceiling 25.89 (in-registry) vs 26.0 (external snapshot) | **FIX — SIGN-OFF: canonical value** |
| 3 | Broad ceiling 35.26 calibrated; TQA immunity real (constant −3 pp offset, not floor) | **UPGRADE** (retire 2 %TODOs + hedges) |
| 4 | 66.8±0.8 verified; six 3-seed math configs; SC-LoRA math ≈ on-curve (resid −1.8) | **UPGRADE** |
| 5 | 3 of 4 audit gaps closed; CorDA + Qwen-CS-seeds + Qwen base ceiling remain | no-change (limitations stay, reworded) |

**Do-not-touch reminder:** this report proposes edits only; `paper.tex`, `key_numbers.md`, `NEXT_EXPERIMENTS.md` were not modified. The lrsw pooled law was re-verified this pass (r=−0.858, R²=0.736, slope −14.78, n=49, p=3.4e-15) — the paper's spine is still exact.
