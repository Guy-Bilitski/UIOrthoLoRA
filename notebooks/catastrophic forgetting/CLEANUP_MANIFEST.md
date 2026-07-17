# CLEANUP MANIFEST — reconciled + EXECUTED (2026-07-09)

**Working dir:** `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/`
**Originally written:** 2026-07-06 (plan only) · **Executed + reconciled:** 2026-07-09 (repo-hygiene pass).

The 2026-07-06 version was a *plan* ("execute nothing"). This version records **what was actually archived**
(moved to `archive/`, NOT deleted), reconciles the inventory with current reality, and lists the remaining
candidates deliberately left in place. **A live multi-pool campaign is running** (`frepro4*` gpu_pool pools +
`auto_dispatch` on `jobs/master_dispatch.txt` — verify with `ps aux | grep -E 'gpu_pool|auto_dispatch'`), so
the same discipline applies: no `.py` imported by the live train/eval chain, no live `--jobs` file, and
nothing under `results/`/`repro/`/`/scratch` was touched. **Deletion was not used — archiving only.**

---

## Reconciliation with reality (what changed since the 2026-07-06 plan)

| Item | 2026-07-06 plan said | Reality 2026-07-09 |
|---|---|---|
| Live scheduler | one pool `frepro3` on `jobs/frepro_lean.txt` (pid 2932862) | **4 `gpu_pool` pools (`frepro4*`) + `auto_dispatch`** on `jobs/master_dispatch.txt` |
| CorDA++ wiring | "implemented, **wiring PENDING**, `DEFAULT_N` 8→5 at restart" | **WIRED into `train_cs.py`** (`--cordapp`, 29 refs) since the 2026-07-06 restart → `cordapp_init.py` is now a LIVE import |
| `repro/` | 2 vendored repos (CLoRA, LLM-Adapters) | **6 repos** (CLoRA, CorDA, LLM-Adapters, LoRA-Null, MiLoRA, SC-LoRA) — all KEEP (data + reference) |
| `papers/` | not present | **5 source PDFs** (CLoRA, CorDA++, MiLoRA, PiSSA, SC-LoRA) — KEEP (ground truth for the paper-expert fleet) |
| New analysis scripts | — | `geo_drift_phase{1,2}.py`, `forgetting_ce.py`, `retfix_*` (Jul 6–9) — all KEEP (current analysis) |
| `restart_staging/` | not covered | frepro4 restart patches — KEEP (may be reused for the Node-B bootstrap, `handoff/28`) |

---

## EXECUTED — archived to `archive/` on 2026-07-09 (87 files; `git mv`, reversible)

Each file was verified before moving to be (a) **not** imported by the live chain (`train_cs.py`,
`eval_one_gpu.py`, `run_lib.py`, `eval_cs.py`, `uio_inprocess.py`, `math_eval.py`, `bbh_metric_fix.py`,
`residual_save.py`, the 7 init modules), (b) **not** referenced by any live `--jobs` file or `auto_dispatch`
(checked against `ps`), and (c) superseded. See `archive/README.md`.

### `archive/scripts/` (24) — dead trainers / eval / orchestrators / job-generators
`train_lora_cs.py`, `eval_adapter.py`, `run_cs_eval.py`, `run_retention.py`, `eval_retention.py`,
`test_uio_roundtrip.py`, `diag_uio_save.py`, `status.py`, `run_all_experiments.sh`, `relaunch_clean.sh`,
`recover_and_resume.sh`, `run_campaign.sh`, `campaign_status.sh`, `chain_lrsweep_ext.sh`, `run_pipeline.sh`,
`run_pipeline_w3.sh`, `run_pipeline_w4b.sh`, `run_forensics.sh`, `make_campaign_jobs.py`,
`make_lr_ext_jobs.py`, `gen_lrsweep.py`, `gen_matrix.py`, `make_salvage_evals.py`, `make_report.py`.
(Self-contained legacy cluster: `run_cs_eval.py`/`run_retention.py` import `eval_adapter.py` — all three
moved together; nothing kept imports any of them.)

### `archive/analysis/` (10) — forensics/leakage/diagnostic one-offs (superseded by `paper_figs_v2.py` + `paper/writing/`)
`analyze_d1_d2.py`, `analyze_forensics.py`, `analyze_headline.py`, `forensics.py`, `forensics_databasis.py`,
`leakage.py`, `make_leakage_map.py`, `norm_trace.py`, `universal_curve.py`, `test_a5_drop_major.py`.

### `archive/jobs/` (53) — completed/superseded pre-frepro wave job lists
All `jobs/*.txt` **except** the 15 kept below: `_calib`, `_m_lrsw/lrswm/qwsw/qwswm`, `auto_l2cs/l2m/lrsw/`
`lrswm/qwcs/qwm/qwsw/qwswm`, `clora_lora_eval`, `clora_train`, `combined`, `combined_nocorda`, `corda_kpa`,
`d1_controlled`, `d1_lambda`, `databasis`, `databasis_task`, `dora_frontier`, `forensics_clora`,
`full_matrix`, `fullscale_wd`, `gate1_seeds`, `kval_kvec_grid`, `lr_sweep`, `lr_sweep_ext`, `lr_sweep_math`,
`mag_control_lora`, `matrix_cs`, `matrix_math`, `matrix_reuse`, `phase2_combined`, `rank_sweep_lora`,
`reeval_fast_baselines`, `salvage_evals`, `uio_*` (11), `validate_residual`, `wd_clean`, `wd_clean_lo`.
Provenance for these runs is preserved in `results/` + `logs/`.

---

## KEEP — LIVE (imported/executed by the running fleet; **do NOT touch**)

`gpu_pool.py`, `auto_dispatch.py`, `train_cs.py`, `eval_one_gpu.py`, `run_lib.py`, `eval_cs.py`,
`uio_inprocess.py`, `math_eval.py`, `bbh_metric_fix.py`, `residual_save.py`, and the seven init modules
`corda_init.py`, `cordapp_init.py`, `milora_init.py`, `data_aware_init.py`, `sclora_init.py`,
`lora_null_init.py`. Live `--jobs` files: `jobs/master_dispatch.txt`, `jobs/frepro4_{main5,b4,`
`headline_math2,inject}.txt`. Directories: `results/`, `logs/`, `/scratch/cf_models` (checkpoints).

## KEEP — ACTIVE-SUPPORT (current campaign; run out-of-band)
`make_frepro_jobs.py`, `build_lean.py` (reads `frepro4_{math,cs}.txt`), remaining `jobs/frepro*` lists,
`validate_frepro_residual.py`, `validate_residual_zero_step.py`, `validate_cordapp_cpu.py`,
`metamath_prep_395k.py`, `math_test_prep.py`, `base_retention_check.py`, `geo_drift_phase{1,2}.py`,
`forgetting_ce.py`, `retfix_retention_gate.py`, `retfix_bbh_only_report.py`, `paper_figs_v2.py`, the
`paper/` package, `handoff/`, `papers/`, `repro/`, `restart_staging/`, `models/`.

---

## Deliberately LEFT in place (candidates NOT moved — noted, not touched)

| File / dir | Why left |
|---|---|
| `analyze_matrix.py` | a **pending paper task** targets it (fdelta→F_Δ relabel, `handoff/26`) — still active |
| `analyze_magnitude_law.py`, `fdelta.py` | magnitude-axis reference code near the active law analysis; low value to move, keep visible |
| `paper_assets.py` | **deprecated** for the main table but retains useful LR-fairness/forensics analysis — archive later only with human sign-off |
| `metamath_prep.py` (100K) | superseded by `metamath_prep_395k.py`, but a small residual risk a stray job references `metamathqa_100k.json` → left; verify then archive |
| `models/lora_cs_l2-7b_r32/` | old small adapter checkpoint — confirm it is not a `results/` provenance reference before archiving |
| `restart_staging/` | frepro4 restart patches may be reused for the Node-B bootstrap (`handoff/28`) — do not archive yet |
| `__pycache__/` | live pools may be writing `.pyc`; safe to delete/regenerate only after the fleet drains |
| `frepro*` job lists + `master_dispatch.txt` | active reproduction-campaign machinery |

---

## Inconsistencies found between docs and reality (fixed in this pass)
1. `WORKDIR_ALIGNMENT.md` + this manifest described `frepro3`/`frepro_lean.txt` as the live pool → the live
   fleet is actually `frepro4*` pools + `auto_dispatch`/`master_dispatch.txt`. **Fixed** in both docs + `STATUS.md`.
2. CorDA++ was documented as "wiring PENDING" but is **wired** (29 refs in `train_cs.py`). **Fixed** in
   `WORKDIR_ALIGNMENT.md` (roster + LIVE-files) and this manifest.
3. `repro/` grew from 2 to 6 vendored repos and `papers/` (5 PDFs) is new — neither was in the old manifest.
   **Recorded** here.
4. `handoff/README.md` indexed only 00–20; docs 21–28 existed but were unlisted. **Fixed** (index extended).

---

# CLEANUP PASS 2 — post-freeze consolidation (2026-07-17)

Fleet is evacuated (handoff/41): no live pools, so the 07-09 "no deletion" constraint is lifted
for regenerable/superseded files. Canonical state: key_numbers.md §18 (+§19 addendum),
analysis_final/ 01–08.

## EXECUTED — DELETED on 2026-07-17 (git rm, recoverable from history)

| Files | Reason |
|---|---|
| `results_book/*.md` (10) | Declared STALE by §18 ("do not source numbers from them"); regenerable via `results_book.py` |
| `paper/writing/artifact_number_audit_final.md`, `artifact_review_round_final.md`, `artifact_feedback_round2.md` | Audits of the 07-14 artifact, fully superseded by `analysis_final/04_story_integrity.md` |
| `paper/writing/section_reviews/sec0–sec8*.md` (9) | Reviews of the round-4 paper.tex, which will be rewritten from PAPER_BLUEPRINT.md |
| `results/campaign_summary.jsonl.bak_pre_qwswm_ingest_2026-07-14` (untracked, plain rm) | Backup of a store §18 itself declares stale; current file retained for provenance |

## EXECUTED — ARCHIVED on 2026-07-17 (git mv → `archive/writing_2026-07-17/`, reversible)

Superseded-but-historical audit trail. Key supersessions: 07-02 writing suite + CONCLUSIONS_AND_IDEAS
+ FINALIZATION_PLAN → key_numbers §14+/analysis_final + PAPER_BLUEPRINT; claims_coverage_audit_sat +
registry_refresh_2026-07-14 → §16–§18 + analysis_final/04; NEXT_EXPERIMENTS + writing_readiness_2026-07-16
+ assessment_2026-07-17 → executed / explicitly superseded by §18.3; pi_review_figures_title +
author_recommendations + integration_notes → BLUEPRINT §5; STATUS.md → handoff/41 (fleet no longer live);
handoff/34–40 → handoff/41. `06_reconciliation.md` archived WITH pointer note (key_numbers cites it as its
B6 contradiction log — path updated in a stub).
