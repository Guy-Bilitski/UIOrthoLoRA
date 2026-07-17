# 08 — Independent Verification of Blocker Remediation (B1–B6)

**Verifier date:** 2026-07-02. **Role:** independent verifier. **Method:** every claim below was
checked against the *actual* artifacts in `paper/writing/`, not against the remediation docs' own
self-report. Data numbers were re-derived from the registry with
`/home/guy/UIOrthoLoRA/.venv/bin/python`; figure/table generators were re-run to confirm they
reproduce the shipped files. Blockers split as the brief specified: **B1, B2, B3, B6 = solvable now
from existing data**; **B4, B5 = require new experiments → verify the PLAN, not results**.

---

## Per-blocker verdict table

| Blocker | One-line issue | Verdict | Evidence (verified) |
|---|---|---|---|
| **B1** | "law across 8 adapters" is really 6; CorDA dropped, LoRA-Null pooled | **RESOLVED** | `key_numbers.md §9` approved coverage sentence; `paper_draft.tex` abstract L46–50, intro item 1 L106–110, Limits item 4 L614–625 all say "six of eight … CorDA withheld, LoRA-Null pooled" and never place "eight" next to "one curve." |
| **B2** | shipped registry still had CorDA explosion + duplicate rows | **RESOLVED** | `campaign_summary_clean.jsonl` verified: **320 rows, 0 CorDA, 0 duplicate run_names, 0 fdelta>50, max dw_sv_max=1073.6**. `registry_cleaning_report.md` documents the 2-rule pipeline (359→343 dedup→320 CorDA-excl). Dirty source retained as frozen provenance copy (359 lines, == live results file). |
| **B3** | Claim 3 ("wins are an LR artifact") had NO figure | **RESOLVED (as *ingredients*), airtight form CORRECTLY-DEFERRED** | `fig9_lr_artifact.png` + `tables/table_lr_artifact.tex` exist and **reproduce exactly** from the generators. Draft §lr L391–405 shows the illusion from the study's own sweep; abstract L65–75 and intro item 3 L115 say "**ingredients** of a learning-rate artifact," not a proven published-win teardown. The published-LR-win teardown is deferred with a visible `%TODO(experiment)` L427–430. |
| **B4** | calibration↔eval mismatch makes "data-aware inits forget more" uninterpretable | **CORRECTLY-DEFERRED-TO-EXPERIMENT** | `07_experiment_plan.md` §"--calib_source" gives a costed P0 code change + eval-matched aux run. Draft Limits item 3 L602–613 embargoes ALL off-curve language and keeps SC-LoRA "provisional" until B4 lands. No fabricated B4 result exists. |
| **B5** | "LoRA+wd surpasses frontier" is single-seed noise; no param-matched control | **CORRECTLY-DEFERRED-TO-EXPERIMENT** | Verb softened to "**matches or edges**" everywhere (abstract L64, §pareto L490, honest-verb ¶ L490–499 retracts an earlier "dominates"). Limits items 1–2 L584–601 defer seeds 43/44 + param-matched r16/r32 control via `07_experiment_plan.md`. No fabricated seed/control result. |
| **B6** | numeric contradictions across four docs; need one source of truth | **RESOLVED** | `key_numbers.md` is the declared single source of truth; `06_reconciliation.md` logs every fix. All headline numbers in `paper_draft.tex` were spot-verified to match (see below). No "72/1395" scale survives in the draft body. |

---

## What I verified directly (not just read)

**Clean registry (B2).** Re-parsed `campaign_summary_clean.jsonl`: 320 rows, zero rows whose
`run_name` contains "corda", zero duplicate `run_name`s, zero `fdelta>50`, max `dw_sv_max`=1073.6
(the genuine `lrsw_dora_r16_lr1e3_s42` extreme-LR point, exactly as the cleaning report states). The
dirty `campaign_summary.jsonl` (359 rows) contains the single `fdelta=515.77, ret=0` explosion and
16 duplicate run_names (7 of them CorDA) — correctly removed in the clean file.

**fig9 + table_lr_artifact (B3).** Re-ran `make_table_lr_artifact.py`: the emitted `.tex` is
byte-consistent with the shipped `tables/table_lr_artifact.tex` (DoRA +5.9, MiLoRA +0.8, SC-LoRA
+26.0, LoRA-Null +19.5, CLoRA −14.2, CorDA +38.1; LoRA+wd swept frontier
5e5/1e4/3e4/5e4 → dominates every row). `make_fig9_lr_artifact.py` regenerates the PNG.

**Numbers in the draft vs `key_numbers.md` (B6).** Verified matches: pooled `r=−0.86, R²=0.74,
slope −14.8, n=49=7×7`; on-curve `r=−0.92, R²=0.84, slope −10.0, n=42`; LR-proxy `R²=0.32 vs 0.74`
(the brief's 0.35/0.75 correctly rejected); ANCOVA `F(5,42)=8.3`, SC-LoRA residual `−4.15 (p=0.006)`
as the sole significant deviator; CS main table (LoRA+wd 81.6/25.6/0.394 … DoRA 78.3/24.8/0.445);
math table now **3 rows** (LoRA+wd 50.6/24.6/33.6/0.399, LoRA 46.5/22.9/31.5/0.520, DoRA
33.3/25.2/33.8/0.327) with **n=14 consistent** — this closes the three-way math-table contradiction
that review-note Tier-1 item 1 flagged. Qwen-CS `−34.8`/Qwen-math `+0.67 (p=0.21, wrong sign)`
reported honestly. No `72`/`1395` magnitude appears anywhere in the draft body.

---

## Remaining flags (not blockers, but the verifier's honest caveats)

1. **[MINOR — disclosed tension, not a contradiction]** `key_numbers.md §8` says CorDA is excluded
   from "every law, figure, table," but CorDA **does** appear as a trajectory in `fig9` and a row in
   `table_lr_artifact`. This is *openly disclosed and correctly scoped* in the draft (fig9 caption
   L415–417 and Limits item 4 L616–619: "only as an illustration of the single-LR illusion … not a
   reportable operating point"). Recommendation: tighten `key_numbers.md §8` to say "excluded from
   every **headline law/Pareto/ANCOVA** figure and table (it appears only as a labeled diagnostic in
   the LR-artifact exhibit)" so the source-of-truth wording matches the draft's own careful scoping.

2. **[MINOR — generators read the dirty file]** `make_fig9`/`make_table_lr_artifact` read
   `campaign_summary.jsonl` (dirty) and re-implement dedup + a `fdelta>50` inline filter, rather than
   consuming `campaign_summary_clean.jsonl`. Output is correct (the inline filter drops the one
   explosion), but for release hygiene the exhibit generators should point at the clean registry so
   there is a single de-duplicated input of record.

3. **[MINOR — B3 scope]** Claim 3 is honestly demoted to "ingredients / the illusion reproduced from
   our own sweep." The strong published-win-teardown remains a hypothesis. The abstract/intro/
   conclusion wording is now consistent with the deferred `%TODO`; this is acceptable *as written*
   but the polemic "wake-up call" tone still rides on a Claim 3 that is ingredient-level, not
   airtight. Keep the measurement-methodology framing foregrounded until B3-airtight/B4/B5 land.

4. **[NICE]** arXiv IDs still carry `%TODO(data)` (SC-LoRA 2505.23724, CorDA 2406.05223, CorDA++
   2506.13187) and `references.bib` is placeholder. Mechanical, pre-submission.

---

## FINAL GO / NO-GO

**GO for the writing package as an honest representation of the evidence — with two one-line
tightenings recommended (flags 1 and 2 above).** The four now-solvable blockers (B1, B2, B3-as-
ingredients, B6) are genuinely RESOLVED in the shipped artifacts, and the two experiment-gated
blockers (B4, B5) are CORRECTLY DEFERRED with a costed plan and with all dependent claims embargoed
in the draft — no result is fabricated. The paper's numbers match the single source of truth, the
registry is clean, the "eight vs six" and "matches/edges vs surpasses" honesty gaps are closed, and
every remaining gap carries a visible `%TODO`. The one substantive nuance (CorDA appears as a labeled
diagnostic in the LR-artifact exhibit) is transparently disclosed in the draft; only the
`key_numbers.md §8` absolute wording should be aligned to it. This package does not overclaim.
