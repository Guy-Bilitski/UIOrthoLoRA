# 06 — Reconciliation (B6): numeric contradictions vs ground truth

**Date:** 2026-07-02. **Method:** every headline number recomputed directly from
`writing/data/campaign_summary.jsonl` with `/home/guy/UIOrthoLoRA/.venv/bin/python`, dedup by
**latest `evaluated_at`** per `run_name` (359 raw → 343 unique). Ground truth now lives in
`writing/data/key_numbers.md`; this doc lists what disagreed and the corrected value.

Docs scanned: `01_project_narrative.md`, `02_figures_tables_explained.md`, `03_gaps_and_roadmap.md`,
`CONCLUSIONS_AND_IDEAS.md`.

Severity: **[MAJOR]** = wrong load-bearing number a referee would catch; **[MINOR]** = small drift or
loose rounding; **[UNITS]** = the old-scale artifact; **[FRAMING]** = a coverage/label overstatement.

---

## C1 [MAJOR] — Within-method correlations are stale/wrong in 01 and CONCLUSIONS

- **Where:** `01` §(e) line ~313; `CONCLUSIONS` §2.1 line ~76.
- **Claimed:** "Within-method r: LoRA −0.97, LoRA+wd −0.95, CLoRA −0.98, DoRA −0.86, CorDA −0.91,
  MiLoRA −0.96, SC-LoRA −0.88."
- **Ground truth (latest-dedup, retention_core vs log ‖ΔW‖_F):** LoRA(plain) **−0.95**, LoRA(as
  plotted, incl. LoRA-Null) **−0.90**, LoRA+wd **−0.89**, DoRA **−0.97**, CLoRA **−0.90**, MiLoRA
  **−0.94**, SC-LoRA **−0.97**, CorDA **−0.90**, LoRA-Null **−0.86**.
- **Nature:** nearly every value differs; DoRA and SC-LoRA are essentially swapped with what the docs
  claim (docs say DoRA is the loosest −0.86 and SC-LoRA −0.88; truth is DoRA −0.97 tight, SC-LoRA
  −0.97 tight, and the loosest is LoRA-Null −0.86). The old list predates the current eval/dedup.
- **Fix:** replace with `key_numbers.md §1a`. The qualitative point ("every adapter individually
  traces the same line, all r ≤ −0.86") still holds.

## C2 [MAJOR] — Qwen-CS correlation: 01 says −0.92, everyone else says −0.88

- **Where:** `01` §(e) line ~348 and Appendix line ~428 say Qwen-CS r = **−0.92**; `02` (line 124),
  `03` (lines 43/188), `CONCLUSIONS` (§2.3), and old `key_numbers` say **−0.88**.
- **Ground truth:** Qwen-CS LoRA 7-LR sweep, r(**core**, log‖ΔW‖) = **−0.88** (R²=0.78);
  r(**broad**, log‖ΔW‖) = **−0.92** (R²=0.85). Both are real — they differ by retention metric.
- **Fix:** the number is metric-dependent. Quote **−0.88 (core)** to match the Llama core-based
  headline; if −0.92 is used, label it explicitly as the **broad** metric. `01`'s −0.92 is not wrong,
  it is the broad number stated without the qualifier → make it consistent. See `key_numbers.md §11`.

## C3 [MAJOR] — Math law: n and r disagree across docs

- **Where:** `01` §(e) line ~314 and Appendix say math **r ≈ −0.93, n=8**; `03` (a) line ~17 says
  **r ≈ −0.93, n=8**. `02`/`CONCLUSIONS`/old `key_numbers` say **r = −0.97, n=14**.
- **Ground truth:** Llama-2 math pooled (LoRA n=7 + LoRA+wd n=6 + DoRA n=1) = **n=14, r=−0.97,
  R²=0.93, slope −10.1, p=2.4e-8.** The "n=8, r=−0.93" is stale (earlier partial math sweep).
- **Fix:** use **n=14, r=−0.97** everywhere. Correct `01` and `03`.

## C4 [MAJOR] — ANCOVA off-curve: "CorDA −3.0pp / SC-LoRA −3.3pp" is not the current fit

- **Where:** `01` §(e) PROVISIONAL line ~334; `03` C2 line ~83.
- **Claimed:** "SC-LoRA (−3.3pp) and CorDA (−3.0pp) forget MORE than their ‖ΔW‖ budget predicts."
- **Ground truth:** the current fig2 fit is on **6 series (CorDA excluded)**. The only significant
  deviator is **SC-LoRA at −4.15pp (p=0.006)**. CorDA has **no residual** in the current fit (it is
  not in the pool). The "−3.3 / −3.0" pair is from an older fit that still included CorDA.
- **Fix:** SC-LoRA residual = **−4.15pp**; drop the CorDA −3.0pp figure entirely (CorDA excluded).
  `CONCLUSIONS` §4/§5 already use −4.15 correctly; align `01` and `03`.

## C5 [MINOR] — Math Ret-broad values off by ~0.4–0.9pp

- **Where:** `02` line ~98 / `CONCLUSIONS` §3.2 table.
- **Claimed:** LoRA+wd math Ret-broad **34.0**; DoRA math Ret-broad **32.9**.
- **Ground truth:** LoRA+wd math Ret-broad = **33.6** (`lrswm_lorawd_wd0p3_lr5e4_s42`); DoRA math
  Ret-broad = **33.8** (`lrswm_dora_r16_lr2e5_s42`).
- **Fix:** 33.6 and 33.8. (Ret-core 24.6 / 25.2 and GSM8K 50.6 / 33.3 are correct.)

## C6 [MINOR] — "LoRA+wd retains ~34" at its best-adapt point

- **Where:** `01` §(e) line ~319 and `03` (a) line ~27 ("adapts ~81 CS AND retains ~34").
- **Ground truth:** at the best-adapt point (lr5e-4, CS 81.6) LoRA+wd Ret-**broad** = **33.2**,
  Ret-core = 25.6. "~34" is only reached at *lower* LRs (e.g. lr2e-5 broad 36.6). The "~34" is a
  loose gloss mixing operating points.
- **Fix:** say "retains ≈ 33 broad / ≈ 26 core at its best-adapt point" or drop the "~34".

## C7 [UNITS] — the "72–1395" and "‖ΔW‖ ≈ 72 at lr1e-3" magnitudes

- **Where:** `01` §(e) line ~317/319 and Appendix; `03` (a) line ~26 ("bounds ‖ΔW‖_F ≈ 72 at
  lr1e-3 vs 200–1395 for others").
- **Ground truth:** the registry `fdelta` (= ‖ΔW‖_F) ranges **~0.05–3.7**. LoRA+wd best-adapt
  fdelta = **0.394**; its lr1e-3 fdelta = **0.632**. There is no 72/200/1395 in the registry — that
  is the older **unnormalized** scale.
- **Fix:** replace all 72 / 200–1395 magnitudes with the `fdelta` scale. `CONCLUSIONS` §0 already
  flags this correctly; scrub the raw 72/1395 numbers from `01` and `03`.

## C8 [FRAMING] — "the law holds across 8 adapters" / "6 methods" overstates coverage (B1)

- **Where:** `CONCLUSIONS` §1 thesis line ~34 ("Across eight adapters … retention collapses onto one
  curve"); `01`/`02`/`03` "6 methods / n=49" throughout.
- **Ground truth:** the mature Llama-2 CS sweep contains **7 distinct adapters** (LoRA, LoRA+wd, DoRA,
  MiLoRA, SC-LoRA, CLoRA, **LoRA-Null**); **CorDA (the 8th) is excluded / not assessed**. The
  "6 methods" label hides LoRA-Null (labeling bug, §C9). Of the 6 fairly assessable adapters, **5 are
  on the curve** (LoRA, LoRA+wd, MiLoRA, CLoRA, DoRA), LoRA-Null tracks it too (within-r −0.86, pooled
  into LoRA), and **SC-LoRA is the 1 provisional off-curve deviator**.
- **Fix:** use the approved sentence in `key_numbers.md §9`: "Across the 6 of 8 adapters we can
  currently assess fairly … SC-LoRA is the one provisional below-curve deviator, and CorDA is
  withheld." Never claim 8-adapter coverage in the present tense.

## C9 [FRAMING] — LoRA-Null labeling bug inflates the LoRA series n and robustness

- **Where:** every doc's "n=49 = 6 methods × ~7 LRs" and the LoRA robustness count.
- **Ground truth:** `method = split("_")[1]` maps `lrsw_lora_null_*` → `"lora"`, so LoRA-Null's 7
  points pool into plain LoRA: plotted LoRA series **n=14, robust 10/14**; plain LoRA alone **n=7,
  robust 5/7**. The pooled law and the best-adapt LoRA point (79.1/24.4) are unaffected.
- **Fix:** quote **5/7** for plain-LoRA robustness; fix the legend/n before camera-ready. Already
  documented in `CONCLUSIONS` §6.10 and old `key_numbers` §9 — kept and sharpened.

## C10 [MINOR] — CorDA "best point 77.9/19.9" quoted as if usable

- **Where:** `01` Appendix line ~426; `CONCLUSIONS` §4 CorDA bullet ("Older wikitext point 77.9/19.9").
- **Ground truth:** under latest-dedup the CorDA rows are the 06-29/30 nq-partial re-eval, one of
  which is the residual-save **explosion** (lr1e-3: fdelta 515.77, cs 0.0). No CorDA number is
  publishable pending the nq_open re-run + fair-calibration pass.
- **Fix:** keep CorDA numbers only as "pending, do not report"; `CONCLUSIONS` already tags it PENDING —
  ensure `01` Appendix marks 77.9/19.9 as superseded/pending, not a live number.

---

## Numbers that RECONCILED (verified correct — no change)

- Pooled CS law **r=−0.86, R²=0.74, slope −14.8, n=49, p=3.4e-15** ✓
- On-curve CS law **r=−0.92, R²=0.84, slope −10.0, n=42** ✓
- LR-vs-‖ΔW‖ predictor **R² 0.32 vs 0.74** ✓ (correctly noted the brief's 0.35/0.75 is wrong)
- ANCOVA **F(5,42)=8.34, p<0.001**; pooled linear R²=0.74 → +intercepts R²=0.87 ✓
- Spline residuals **LoRA +0.79, LoRA+wd +0.06, MiLoRA +1.04, CLoRA +0.09, DoRA +1.37,
  SC-LoRA −4.15 (p=0.006)** ✓ (reproduced exactly from the generator's spline baseline)
- Per-benchmark slopes **MMLU −23.4, MMLU-Pro −15.2, ARC −14.9, BBH −14.3, TruthfulQA −0.5** ✓
- Budget slopes **adapt +20.3, retention −14.8 pp/decade**; sweet-spot [0.31,0.62] ✓
- CS best-adapt table (CS-8 / Ret-core / ‖ΔW‖_F): LoRA+wd 81.6/25.6/0.394, SC-LoRA 80.1/22.5/0.559,
  MiLoRA 79.9/24.7/0.543, LoRA 79.1/24.4/0.623, CLoRA 78.4/21.9/0.643, DoRA 78.3/24.8/0.445 ✓
- Math best-adapt CS-8/Ret-core/GSM8K: LoRA+wd 50.6/24.6, LoRA 46.5/22.9, DoRA 33.3/25.2 ✓
- Qwen-math LoRA **r=+0.67, p=0.21, n=5** (does not replicate) ✓
- Base ceiling **26.0 (BBH-AO 33.10 + MMLU-Pro 18.96)** ✓ (external, unchanged)

## One-line editing checklist for the four docs
1. Replace within-method r-list → `key_numbers §1a` (C1).
2. Qwen-CS: −0.88 (core) or label −0.92 as broad (C2).
3. Math law → n=14, r=−0.97 (C3).
4. Off-curve → SC-LoRA −4.15 only; delete CorDA −3.0pp (C4).
5. Math Ret-broad → 33.6 (LoRA+wd), 33.8 (DoRA) (C5).
6. Drop "retains ~34" gloss → 33.2 broad / 25.6 core (C6).
7. Purge 72 / 200–1395 magnitudes → fdelta scale (C7).
8. Coverage → "6 of 8 assessed, 5 on-curve + LoRA-Null, SC-LoRA off, CorDA withheld" (C8).
9. LoRA robustness → 5/7 plain; fix labeling (C9).
10. CorDA 77.9/19.9 → mark pending, not live (C10).
