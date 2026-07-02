# FINALIZATION PLAN — master synthesis (lead sign-off)

**Date:** 2026-07-02. **Owner:** lead. **Purpose:** one authoritative plan that folds
`04_critique.md`, `05_review_notes.md`, `06_reconciliation.md`, `07_experiment_plan.md`,
`data/registry_cleaning_report.md`, `02_figures_tables_explained.md`, and `03_gaps_and_roadmap.md`
into (1) a per-blocker status table, (2) an ordered critical path, (3) a definition-of-done, (4) a
timeline pinned to the ~6–7-day Qwen 2×2 drain, and (5) the single highest-leverage next action.

**Ground truth for every number:** `data/key_numbers.md` (SINGLE SOURCE OF TRUTH). Any conflicting
figure in `01`–`03` / `CONCLUSIONS_AND_IDEAS.md` is wrong until edited to match.

---

## 1. Blocker status table (B1–B6)

Legend: **SOLVED NOW** = fully addressed from existing data/artifacts, only editing remains;
**PARTIAL** = the analysis/artifact exists but a claim-level edit or a small confirmation is still
open; **PLANNED-EXPERIMENT** = requires new compute (specced in `07`, nothing fabricated).

| # | Blocker | Status | What was done / what run is needed | Artifact / file that addresses it |
|---|---------|--------|-------------------------------------|-----------------------------------|
| **B1** | "Law across 8 adapters" is really 6; CorDA dropped for not fitting → coverage overstated | **PARTIAL (writing-solvable now)** | Reconciled to the honest count: **6 of 8 assessable, 5 on-curve + LoRA-Null tracking, SC-LoRA off, CorDA withheld**. Approved sentence exists. Remaining: purge the word "eight … one curve" from abstract/intro/conclusion and split LoRA-Null out (I6). No experiment needed for the *count*; CorDA only re-enters as an 8th series if the PLANNED B4 CorDA re-run succeeds. | `key_numbers.md §9`; `06_reconciliation.md C8`; edits to `paper_draft.tex` (abstract L46, intro L95, concl L582) + `01_project_narrative.md` |
| **B2** | Registry still contains CorDA residual-save explosion (fdelta=515) + duplicate rows → "faithful port" not clean | **SOLVED NOW** | Cleaning pass done: 359 raw → 343 dedup → **320 clean** (16 dup rows + all 23 CorDA rows removed). Dedup rule documented (latest `evaluated_at`). Max `dw_sv_max` in clean file now 1073 (a real extreme-LR DoRA point), the 54741/3300 CorDA blow-ups gone. **Faithful-port claim must be scoped to the 6 non-CorDA methods** — report says so explicitly. Remaining: point all generators/figures at `campaign_summary_clean.jsonl`, and either delete the raw file from the release or label it "raw, superseded." | `data/campaign_summary_clean.jsonl`; `data/registry_cleaning_report.md` |
| **B3** | Claim 3 ("wins are an LR artifact") had NO figure; also never reproduced any adapter's single-LR win | **PARTIAL (exhibit built; claim-strength edit open)** | The exhibit now EXISTS: **`fig9_lr_artifact.png`** (full 7-LR trajectory per method, each fancy method's best single-LR point ringed, LoRA+wd swept frontier dominating all six) + companion **`table_lr_artifact.tex`** (per-method Δadapt at matched LR: SC-LoRA +26.0, LoRA-Null +19.5, DoRA +5.9, MiLoRA +0.8, CLoRA −14.2, CorDA +38.1/−7.9ret). This directly answers "you never reproduced a win": the single-LR illusion is now shown numerically (strong for SC-LoRA/LoRA-Null/DoRA, weak for MiLoRA, absent for CLoRA, magnitude-collapse for CorDA). Remaining: (a) demote abstract/intro/conclusion phrasing from "is an artifact" to "we show the *ingredients* of the artifact" per single-seed caveat; (b) seed-average the load-bearing points (folds into B5c). | `figures/fig9_lr_artifact.png`; `tables/table_lr_artifact.tex`; generators `make_fig9_lr_artifact.py`, `make_table_lr_artifact.py`; `02_figures §fig9` |
| **B4** | Calibration↔eval mismatch makes every "data-aware inits forget more" statement uninterpretable | **PLANNED-EXPERIMENT** | Fully specced, not run. Prereq P0 (no GPU, few h): add `--calib_source {nq_open,eval_matched}` to `train_cs.py`, shared `load_calib_prompts`, eval-matched set = MMLU auxiliary_train + ARC-Challenge train (disjoint from test, hash-checked), emit `calib_source` to `summary.json`. Then **21 cells** (3 `_em` arms × 7 LR × s42, ≈7.5 h L2-CS). Until it runs, **all off-curve / "forget more than budget" language is EMBARGOED**, including the SC-LoRA −4.15pp ringing. Analysis = paired residual delta (nq_open − eval_matched) from the calibration-free pooled fit. | `07_experiment_plan.md §B4`; `train_cs.py` (P0 change); user must prefetch the two HF splits |
| **B5** | "LoRA+wd surpasses the frontier" is single-seed noise; arms mix r16/r32 and only LoRA has wd | **PLANNED-EXPERIMENT** | Specced. (a) **B5a param-match 2×2** {r16,r32}×{wd0,wd0.3}: add `lora_r32`, `lorawd_r16_wd0p3` → 14 cells ≈5 h. (b) **B5c seeds 43/44** on ~12 headline cells → 24 cells ≈7.5 h (run AFTER B3/B4 pin which points are load-bearing). (c) optional B5b wd-on-MiLoRA/DoRA. Until seeds land, the verb is **"matches/lands on"**, never "surpasses/wins." The +0.8–1.5pp margin is smaller than the documented 10–40pp seed-collapse basins. | `07_experiment_plan.md §B5`; `make_campaign_jobs.py` (ARMS/SEEDS edits) |
| **B6** | Numeric contradictions across four docs; need ONE source of truth | **SOLVED NOW (mechanical edits remain)** | Ground truth established and every headline number recomputed: `data/key_numbers.md` is the SINGLE SOURCE OF TRUTH; `06_reconciliation.md` logs all 10 contradictions (C1–C10) with corrected values (within-method r-list, Qwen-CS −0.88 core / −0.92 broad, math n=14 r=−0.97, off-curve = SC-LoRA −4.15 only, drop CorDA −3.0, purge the obsolete "72–1395" scale → fdelta 0.05–3.7, R² 0.32 vs 0.74). Remaining: apply the 10-line editing checklist to `01`, `03`, `CONCLUSIONS`, and the math table in `paper_draft.tex` / `table_main_math.tex`. | `data/key_numbers.md`; `06_reconciliation.md` (10-line checklist §end) |

**Net:** B2 and B6 are done (edits only). B1 and B3 are solvable-now writing fixes (the B3 exhibit
is built; the B1 count is reconciled). **Only B4 and B5 need new compute** — both cheap, both
specced, together ≈59 L2-CS cells / ≈20 h wall-clock (< 1 GPU-day).

---

## 2. Ordered critical path: now → submittable

### (a) Zero-cost writing/analysis fixes — mostly DONE, finish the edits (0 GPU, ~1–1.5 days of editing)
1. **[B6] Apply the 10-line reconciliation checklist** to `01`, `03`, `CONCLUSIONS`, and the math
   table. Every quoted number must trace to `key_numbers.md`. *(mechanical; DONE-analysis, edit-pending)*
2. **[B2] Repoint all generators to `campaign_summary_clean.jsonl`**; regenerate any figure/table
   that read the raw file; delete-or-relabel the raw file for release. *(clean file exists)*
3. **[B1] Strike "eight … one curve"** from abstract/intro/conclusion; use the `key_numbers §9`
   sentence ("6 of 8 assessed, 5 on-curve + LoRA-Null, SC-LoRA off, CorDA withheld").
4. **[I6] Split LoRA-Null out of the LoRA series**, recompute LoRA-only residual (+0.79 currently
   contaminated) and within-r, regenerate fig0/fig2, requote 5/7 plain-LoRA robustness.
5. **[B3] Align Claim-3 phrasing** to the built exhibit: "we show the ingredients of the artifact
   (best-LR not shared + dW≫LR as predictor + single-LR illusion table); a seed-averaged
   reproduce-and-dissolve is in progress." Add the honest caveats already in `02 §fig9`.
6. **[B4-embargo] Strike all off-curve language** (SC-LoRA "forgets more") until B4 runs; keep
   fig2 SC-LoRA ring only with "(provisional, calibration-confounded)" inline everywhere.
7. **[I1] Add the two deferred non-circularity tests** from existing data: slope-interaction ANCOVA
   + leave-one-method-out predictive RMSE. *(analysis-only, no GPU)*
8. **[I3] De-spin the sweet-spot band** to descriptive; **[N1] verify arXiv IDs; [N2] add compute
   appendix.** All mechanical.

### (b) Cheap experiments — the load-bearing new compute (< 1 GPU-day total, L2-CS first)
9. **[B4 P0]** Code: `--calib_source` flag + eval_matched aux set + reload-invariance/disjointness
   validation. *(few h, no GPU; user prefetches MMLU-aux + ARC-train)*
10. **[B4]** 21 `_em` cells (3 arms × 7 LR × s42) ≈ **7.5 h**. **HIGHEST LEVERAGE** — unembargoes
    all off-curve language and settles the top kill-shot either way.
11. **[B5a]** Param-match 2×2 (`lora_r32`, `lorawd_r16_wd0p3`) × 7 LR = 14 cells ≈ **5 h** →
    makes "LoRA+wd wins" a capacity-fair claim.
12. **[C5] Base-ceiling calibration** for MMLU/ARC/TruthfulQA (5 eval-only runs, ≈0.5 d) → makes
    "broad" retention and per-benchmark slopes interpretable; also settles "TruthfulQA immune" vs
    floor-artifact. *(no training)*
13. **[B5c]** Seeds 43/44 on the ~12 headline cells = 24 cells ≈ **7.5 h**, run AFTER 10/11 pin the
    load-bearing points → error bars; kills the n=1 desk-reject.

### (c) Expensive experiments — breadth for the STRONG paper (multi-GPU-day; behind the live pool)
14. **[Pool] Drain the combined 2×2 pool** — Qwen CS+math + L2-math structured arms. **~6–7 days,
    already running.** This is the wall-clock clock; unblocks every 2-model figure.
15. **[B4/B5 on Qwen-CS]** ~35 cells ≈ **45 h ≈ 2 GPU-days** — second-model generality of the
    fairness/param-match result. Defer until L2 verdicts land + pool drains.
16. **[B4/B5 on L2-math + Qwen-math]** ~50+ cells, several GPU-days — dual-domain generality.
    Note **Qwen-math currently ANTI-replicates (r=+0.67)**; must be reported, not buried (I2).
17. **[C6] CorDA++** advanced arm (algorithms transcribed) ≈ **3–4 d** — closes "you strawmanned
    SOTA." Only for the strong tier; requires realized-param-count matching to the r16 budget.
18. **[B1 CorDA re-run]** fresh CorDA with a PASSING 0-step self-check; only then may CorDA re-enter
    fig0/fig2/table as a true 8th series. If it stays off-curve after fair calibration, report it as
    a second off-curve method and retreat "geometry inert" → "inert for calibration-free inits."

**Sequencing rule (single 8-GPU scheduler):** (a) is free and runs now. Steps 10/11/13/15/16/17
contend for one scheduler — order **B4 → B5a → B5c → Qwen/math/CorDA++** because B4 can
retroactively invalidate off-curve claims the others would be built to explain. Never a second
concurrent pool (two pools each grab all 8 GPUs → OOM).

---

## 3. Definition of done (submission checklist)

**Data & hygiene**
- [ ] All figures/tables generated from `campaign_summary_clean.jsonl`; raw file removed or relabeled.
- [ ] Zero exploded (fdelta>10, ret=0) or duplicate `run_name` rows in the released data.
- [ ] Every headline number in the draft traces to `key_numbers.md`; `06` checklist fully applied.

**Claim 1 (the LAW) — defensible NOW**
- [ ] Pooled CS r=−0.86, on-curve −0.92; math n=14, r=−0.97 — quoted consistently.
- [ ] LoRA-Null split out; LoRA-only residual + within-r requoted; fig0/fig2 regenerated.
- [ ] Slope-interaction ANCOVA + leave-one-method-out predictive RMSE added (non-circularity shield).

**Claim 2 (LoRA+wd on the frontier)**
- [ ] Verb is "matches/lands on the frontier" unless B5c error bars clear zero → then "edges."
- [ ] B5a param-match 2×2 shows retention tracks wd/‖ΔW‖, not rank.
- [ ] Seeds 43/44 CIs on all headline Pareto/op-point cells.

**Claim 3 (LR artifact)**
- [ ] fig9 + table_lr_artifact seed-averaged on load-bearing points; phrasing = "ingredients of the
      artifact," honest per-method caveats (weak MiLoRA, absent CLoRA, magnitude-collapse CorDA) kept.

**Claim 4 (message / off-curve)**
- [ ] B4 run; off-curve embargo lifted per verdict (either "method-free once calibration fair" OR
      "small real residual, now defensible"). No off-curve sentence without the calibration caveat
      until then.

**Interpretability & breadth**
- [ ] Base ceilings for MMLU/ARC/TruthfulQA (C5); "broad" retention normalized; "TruthfulQA immune"
      confirmed not a floor artifact.
- [ ] Qwen framed as partial replication (~13/112); Qwen-math anti-replication reported explicitly,
      not omitted. "Two-model" language only after ≥5 Qwen adapters × both domains.
- [ ] arXiv IDs verified (fix impossible 2603.02224); `references.bib` built; compute/repro appendix.
- [ ] Tone matched to tier: measurement-methodology lead for minimum-defensible; polemic only if the
      full strong tier lands.

---

## 4. Rough timeline (pinned to the ~6–7-day Qwen 2×2 drain)

The combined pool (~6–7 d, running) is the clock. Everything cheap fits INSIDE it.

| Window | Track (parallel) | Deliverable |
|--------|------------------|-------------|
| **Days 0–2** | Writing (a): apply B6 checklist, B2 repoint, B1 count, I6 LoRA-Null split, B3 phrasing, I1 tests. In parallel, no GPU: **B4 P0 code** + user prefetches HF calib sets; **C5 base ceilings** (eval-only, doesn't need the training pool). | Internally-consistent draft; B2/B6 closed; base ceilings in. |
| **Days 2–5** | Pool still draining. First deliberate scheduler pause (or a freed slot) → **B4 (21 cells, ~7.5 h)** then **B5a (14 cells, ~5 h)**. | Off-curve embargo lifted; capacity-fair Claim 2. |
| **Days 5–7** | Pool finishes (Qwen CS+math + L2-math). Pin load-bearing points, run **B5c seeds (24 cells, ~7.5 h)**. Regenerate all 2-model figures. | Error bars on headline; 2×2 figures exist. |
| **End of week 1** | — | **MINIMUM-DEFENSIBLE "law" paper submittable** (Claims 1–2 solid, 3 as ingredients, 4 fair). |
| **Weeks 2–4** | Strong tier: B4/B5 on Qwen-CS (~2 GPU-d), L2-math + Qwen-math (~several GPU-d), **CorDA++** (~3–4 d), CorDA clean re-run for the true 8th series. | **STRONG "wake-up call" paper** with full breadth. |

Minimum-defensible tier: **~1 week from now** (cheap compute fits inside the pool drain).
Strong tier: **~3.5–4 weeks** (the expensive second-model/second-domain/CorDA++ tail).

---

## 5. Single highest-leverage next action

**Run B4: the eval-matched calibration re-run of CorDA / SC-LoRA / LoRA-Null (P0 code first, then
21 cells, ≈7.5 h, < 1 GPU-day).**

It is simultaneously the cheapest load-bearing experiment and the one that resolves the top referee
kill-shot. The entire thesis is "geometry is causally inert; forgetting is governed by ‖ΔW‖." The
only standing evidence *against* method-freeness is the SC-LoRA/CorDA off-curve deviation — and that
deviation is confounded by an unfair (nq_open vs academic-eval) calibration. B4 settles it either
way: if the `_em` arms snap onto the curve, the law becomes cleanly method-free (strongest possible
result, kill-shot removed); if they stay off, we have a real, now-defensible second-order effect.
Neither B5a nor B5c can be interpreted until B4 fixes what the pooled-fit residual even means — the
residual is the shared analysis object for all three. **B4 first.** (Prereq: the P0 `--calib_source`
code change and the user prefetching MMLU-aux + ARC-train, both doable today with no GPU.)

---

## Executive summary (12 lines)

1. Six blockers; **B2 and B6 are SOLVED now** (clean registry `campaign_summary_clean.jsonl` + single
   source of truth `key_numbers.md`), leaving only mechanical edits.
2. **B1 and B3 are writing-solvable now:** the missing Claim-3 exhibit (`fig9` + `table_lr_artifact`)
   is BUILT, and the honest adapter count (6 of 8, CorDA withheld) is reconciled.
3. **Only B4 and B5 need new compute**, both cheap and fully specced — together ≈59 L2-CS cells,
   ≈20 h wall-clock, under one GPU-day.
4. The LAW (Claim 1) is mature and defensible on Llama-2 today; the CONSEQUENCE (Claim 2) and
   DIAGNOSIS (Claim 3) are not yet referee-proof and need the fairness/seed/param-match work.
5. Critical path: finish zero-cost edits now → run B4 → B5a → C5 base ceilings → B5c seeds, all
   INSIDE the ~6–7-day Qwen pool drain.
6. **Highest-leverage next action: run B4** (eval-matched calibration re-run) — cheapest AND resolves
   the top kill-shot; it defines the residual object B5 depends on, so it must go first.
7. Until B4 lands, **all "data-aware inits forget more" language stays embargoed**; the SC-LoRA
   −4.15pp ring is provisional/confounded everywhere it appears.
8. Claim-2 verb is **"matches/lands on the frontier,"** never "surpasses," until seeds 43/44 error
   bars clear zero (the +0.8–1.5pp margin is inside documented 10–40pp seed-collapse basins).
9. Qwen is a partial replication (~13/112 cells); **Qwen-math anti-replicates (r=+0.67) and must be
   reported, not buried** — no "two-model" claim until ≥5 Qwen adapters × both domains.
10. Two tiers: **minimum-defensible "law" paper ≈1 week** (cheap compute fits inside the pool drain);
    **strong "wake-up call" paper ≈3.5–4 weeks** (Qwen/math breadth + CorDA++).
11. Guaranteed-defensible fallback if breadth slips: foreground the **measurement-methodology**
    contribution (LR-sweep-as-instrument + fair ‖ΔW‖_F axis), reserve the polemic for the strong tier.
12. Be honest and decisive: ship the tier the evidence supports; do not let the abstract outrun the
    figures — every headline number must trace to `key_numbers.md`.
