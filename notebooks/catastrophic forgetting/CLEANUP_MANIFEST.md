# CLEANUP MANIFEST — plan only (execute nothing now)

**Working dir:** `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/`  ·  **Written:** 2026-07-06.

This is a **PLAN**. Nothing here has been deleted, moved, or edited. A live pool
(`gpu_pool.py --tag frepro3`, pid 2932862) is running `train_cs.py`/`eval_one_gpu.py` **fresh per job**,
so **no `.py`, `jobs/*`, `results/*`, `logs/*`, or checkpoint may be touched until it drains.** Execute
this manifest only afterward, and confirm the ⚠ items with a human first.

**Conservatism rule applied:** anything referenced by `jobs/frepro_lean.txt`, imported by a live script,
or needed for results provenance = **KEEP**. Legacy job files, superseded one-off scripts, and dead
notebooks = **ARCHIVE** (move to a new `legacy/` dir — reversible, preserves git history). Only trivially
regenerable artifacts are flagged **DELETE**. Suggested target dirs: `legacy/scripts/`, `legacy/jobs/`,
`legacy/docs/`, `legacy/analysis/`.

---

## Inventory summary (counts per class)

| Class | Top-level `.py`/`.sh` | `jobs/*.txt` | Docs/other | Notes |
|---|---|---|---|---|
| **LIVE** | 14 | 1 (`frepro_lean.txt`) | `results/`, `logs/` (dirs) | imported by / executed by the frepro3 pool |
| **ACTIVE-SUPPORT** | 9 (+`paper_figs_v2.py`) | 3 (`frepro_{math,cs,all}.txt`) | `paper/`, `handoff/`, `repro/` | current-campaign generators, validators, prep, paper, reference repos |
| **LEGACY** | ~34 | ~52 | `README.md`, `agent_instructions.nd`, `models/` | superseded orchestrators, UIO-era, old analysis/forensics, pre-pivot jobs |
| **UNKNOWN** | 0 | 0 | 0 | all resolved via grep of imports + git/log context |

Directory footprint: 68 top-level files; `results/` = 504 subdirs + 68 loose json + 3 registries;
`logs/` = 673 files; `jobs/` = 56 files; `repro/` = 2 vendored git repos (CLoRA, LLM-Adapters).

---

## KEEP — LIVE (imported/executed by the running pool)

| File | Why KEEP | Risk if touched |
|---|---|---|
| `gpu_pool.py` | the running scheduler (frepro3) | **High** |
| `train_cs.py` | trainer, invoked by every job (fresh per cell) | **High** |
| `eval_one_gpu.py` | evaluator, invoked by every job | **High** |
| `run_lib.py` | imported by train_cs / eval_one_gpu / eval_cs | **High** |
| `eval_cs.py` | imported by eval_one_gpu | **High** |
| `uio_inprocess.py` | eval_one_gpu imports `fdelta_inprocess` + `CS_DATASETS` | **High** |
| `math_eval.py` | imported by eval_one_gpu for `math_faithful` | **High** |
| `bbh_metric_fix.py` | imported by eval_one_gpu (+ base_retention_check) | **High** |
| `residual_save.py` | imported by train_cs for residual-init methods | **High** |
| `corda_init.py` | imported by train_cs `--corda` + reused by cordapp_init | **High** |
| `milora_init.py` | imported by train_cs `--milora` | **High** |
| `data_aware_init.py` | imported by train_cs `--pissa` (`pissa_BAR`) | **High** |
| `sclora_init.py` | imported by train_cs `--sclora` | **High** |
| `lora_null_init.py` | imported by train_cs `--lora_null` | **High** |
| `jobs/frepro_lean.txt` | the 103-cell list the pool is executing | **High** |
| `results/` (all) | provenance; `campaign_summary.jsonl` + registries + per-run dirs | **High** |
| `logs/` (all) | pool is actively writing `logs/frepro3_*.log` | **High** |

## KEEP — ACTIVE-SUPPORT (current campaign; run out-of-band)

| File | Why KEEP | Risk |
|---|---|---|
| `make_frepro_jobs.py` | generates `frepro_math.txt` + `frepro_cs.txt` (handoff/20) | Med |
| `build_lean.py` | merges them into the running `frepro_lean.txt` | Med |
| `jobs/frepro_math.txt`, `jobs/frepro_cs.txt`, `jobs/frepro_all.txt` | source job files for the lean merge; regenerating the campaign reads them | Med |
| `validate_frepro_residual.py` | validates residual_save scaling≠1 for PiSSA/MiLoRA @ α≠r | Low |
| `validate_residual_zero_step.py` | 0-step residual round-trip gate (all residual arms) | Low |
| `validate_cordapp_cpu.py` | CorDA++ CPU validation (14/14 checks) | Low |
| `cordapp_init.py` | CorDA++ impl, CPU-validated, **wiring pending** (see ⚠ below) | Med |
| `metamath_prep_395k.py` | builds MetaMathQA-395K train file for Table 3 | Low |
| `math_test_prep.py` | builds Hendrycks MATH test.json for faithful eval | Low |
| `base_retention_check.py` | base-model retention baseline (BBH/MMLU-Pro) | Low |
| `paper_figs_v2.py` | **canonical** figure/table generator | Med |
| `paper/`, `paper/writing/` | figures, tables, LaTeX draft, `key_numbers.md`, clean registry | Med |
| `handoff/` (00–20 + README + data_snapshots) | project decision log / plans (docs) | Low |
| `repro/CLoRA`, `repro/LLM-Adapters` | vendored reference repos; **GSM8K/MATH test data + MetaMathQA live under `repro/LLM-Adapters/dataset` & `ft-training_set`** | **High** (data) |

---

## ARCHIVE candidates — LEGACY (move to `legacy/`, do not delete)

### Superseded orchestrators / job generators (pre-pivot camp5 2×2 campaign)
| File | Superseded by | Risk |
|---|---|---|
| `make_campaign_jobs.py` | `make_frepro_jobs.py` | Low |
| `run_all_experiments.sh` | frepro pipeline (make_frepro_jobs → build_lean → gpu_pool) | Low |
| `relaunch_clean.sh`, `recover_and_resume.sh`, `run_campaign.sh` | (chained the old orchestrator) | Low |
| `campaign_status.sh`, `chain_lrsweep_ext.sh` | ad-hoc status/chaining for old waves | Low |
| `make_salvage_evals.py`, `make_lr_ext_jobs.py` | old salvage/LR-ext generators | Low |
| `gen_lrsweep.py`, `gen_matrix.py` | old matrix/LR-sweep generators (`gen_matrix` still in `__pycache__`) | Low |
| `run_pipeline.sh`, `run_pipeline_w3.sh`, `run_pipeline_w4b.sh` | UIO wave auto-pipelines | Low |
| `run_forensics.sh` | invokes forensics/paper_assets (legacy analysis) | Low |

### UIO-era + old (reload-based) eval path
| File | Note | Risk |
|---|---|---|
| `train_lora_cs.py` | pre-`train_cs.py` trainer | Low |
| `eval_adapter.py`, `run_cs_eval.py`, `run_retention.py`, `eval_retention.py` | old sharded/reload eval; replaced by in-process `eval_one_gpu.py` | Low |
| `test_uio_roundtrip.py`, `diag_uio_save.py` | UIO save/reload diagnostics (UIO dead) | Low |
| `status.py` | old status printer | Low |
| `metamath_prep.py` | 100K MetaMathQA prep; superseded by `metamath_prep_395k.py` | ⚠ Med — verify no live job still points at `metamathqa_100k.json` |

### Analysis / forensics one-offs (superseded by paper_figs_v2 / paper package)
| File | Note | Risk |
|---|---|---|
| `paper_assets.py` | **DEPRECATED** (STATUS §5: pooled matrix+sweep → bogus main table); keep for its LR-fairness/forensics analysis | ⚠ Med |
| `analyze_d1_d2.py`, `analyze_forensics.py`, `analyze_headline.py`, `analyze_magnitude_law.py`, `analyze_matrix.py` | per-wave analysis scripts | Low |
| `forensics.py`, `forensics_databasis.py`, `leakage.py`, `make_leakage_map.py` | leakage/forensic thermometers (measurement-fallback era) | Low |
| `fdelta.py` | standalone reload F-delta (eval_one_gpu uses uio_inprocess.fdelta_inprocess instead) | Low |
| `norm_trace.py`, `universal_curve.py`, `make_report.py`, `test_a5_drop_major.py` | one-off diagnostics/reports | Low |

### Legacy job files (all `jobs/*.txt` except the 4 frepro files)
~52 files: `auto_*`, `_m_*`, `matrix_*`, `combined*`, `lr_sweep*`, `full_matrix.txt`, `matrix_reuse.txt`,
`databasis*`, `d1_*`, `dora_frontier.txt`, `gate1_seeds.txt`, `kval_kvec_grid.txt`, `phase2_combined.txt`,
`rank_sweep_lora.txt`, `salvage_evals.txt`, `uio_*`, `wd_clean*`, `mag_control_lora.txt`, `corda_kpa.txt`,
`clora_*`, `_calib.txt`, `fullscale_wd.txt`, `reeval_fast_baselines.txt`, `validate_residual.txt`,
`forensics_clora.txt`. **Risk Low** — all completed/superseded waves; provenance is in `results/` + logs.
⚠ Grep each against nothing-live before moving (none is referenced by the frepro pipeline).

### Legacy docs / misc
| File | Note | Risk |
|---|---|---|
| `README.md` (top-level) | Phase-1 "Is UIOrthoLoRA a CF mitigator?" go/no-go — goal DEAD; superseded by `WORKDIR_ALIGNMENT.md` | ⚠ Med — top-of-dir README; relabel rather than hide |
| `agent_instructions.nd` | original UIO project spec | Low |
| `models/lora_cs_l2-7b_r32/` | old small adapter checkpoint | ⚠ Med — confirm not a provenance reference |

---

## DELETE candidates (regenerable only)

| Target | Rationale | Risk / gate |
|---|---|---|
| `__pycache__/` | compiled `.pyc`, auto-regenerated on import | ⚠ **Do NOT touch while frepro3 runs** (Python may be writing `.pyc`). Safe to delete after drain; regenerates. |
| `paper/figs_v2/preview/` (if present) | downscaled preview PNGs regenerated from full figs | Low — regenerable |

---

## Items needing human confirmation (⚠) before any action

1. **CorDA++ wiring + `DEFAULT_N`** — `cordapp_init.py` is CPU-validated (14/14) but **not wired into
   `train_cs.py`** (verified: no `cordapp` reference in train_cs / make_frepro_jobs / build_lean).
   `DEFAULT_N = 8` (line 50); the plan requires **N = 5** at the next pool restart. This is a pending
   code change, **not** a cleanup action — do NOT edit while the pool runs.
2. **`metamath_prep.py` (100K)** — archive only after confirming no live/queued job references
   `metamathqa_100k.json` (frepro uses `metamathqa_395k.json`).
3. **`paper_assets.py`** — deprecated for the main table but retains useful analysis; archive, don't delete.
4. **`README.md` (top)** — stale but it is the directory's front door; prefer relabeling/redirecting to
   `WORKDIR_ALIGNMENT.md` over hiding it. (Not edited in this pass — only markdown *creation* was in scope.)
5. **`models/lora_cs_l2-7b_r32/`** — confirm it is not cited by any `results/` provenance before archiving.
6. **`jobs/` bulk archive** — 52 legacy files; move as a batch to `legacy/jobs/` only after the frepro
   pool drains (single-scheduler discipline; the pool reads only `frepro_lean.txt`).
