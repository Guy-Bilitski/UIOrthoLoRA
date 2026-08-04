# CODE_MAP — release inventory of `notebooks/catastrophic forgetting/`

Generated 2026-07-30 for release prep. One line per significant file/dir: **role | used-by | status**.
Statuses: **active** (frozen pipeline or current analysis/writing) · **archive-candidate** (superseded/one-off; see `RELEASE_PLAN.md` — nothing has been moved or deleted) · **stale-do-not-source** (contains outdated numbers/claims per `paper/writing/data/key_numbers.md` §18 / `WORKDIR_ALIGNMENT.md`) · **frozen-data** (immutable provenance).

Inputs: `README.md`, `WORKDIR_ALIGNMENT.md`, `CLEANUP_MANIFEST.md`, `key_numbers.md` §18–19 stale markers, plus import-chain inspection (read-only).

---

## 1. Core tested pipeline (the untouchable set)

Import chain (verified by grep): `train_cs.py` → `run_lib`, `norm_trace` (on `--norm_trace`), and the init modules on demand. `eval_one_gpu.py` → `run_lib`, `eval_cs`, `uio_inprocess` (`fdelta_inprocess`, `CS_DATASETS`), plus `bbh_metric_fix`, `math_eval`, and `scripts/deepseek/eval_deepseek` (medmcqa, via sys.path insert). NOTE: the PEFT fork (`UIOrthoLoRAConfig`, layer.py/config.py/train.py) lives at the repo root `/home/guyb/UIOrthoLoRA/peft/`, **outside this folder** — a release dependency.

| path | role | used-by | status |
|---|---|---|---|
| `train_cs.py` | shared trainer; every method = `--method lora\|clora\|uiortholora` + init flag | job lines in `jobs/*.txt` | active |
| `eval_one_gpu.py` | in-process evaluator (CS suite / math_faithful / medmcqa; BBH+MMLU-Pro retention; F_Δ) | job lines | active |
| `run_lib.py` | shared prompt templates (train==eval), logging, registries | train_cs, eval_one_gpu, analyze_magnitude_law | active |
| `eval_cs.py` | commonsense generative eval (`run_eval`) | eval_one_gpu | active |
| `uio_inprocess.py` | UIO-era runner; `fdelta_inprocess` + `CS_DATASETS` keep it live | eval_one_gpu | active (via those imports) |
| `math_eval.py` | faithful 0-shot GSM8K + Hendrycks MATH eval | eval_one_gpu (`--adapt_task math_faithful`) | active |
| `bbh_metric_fix.py` | patches lm-eval BBH answer-only metric normalization | eval_one_gpu, base_retention_check | active |
| `norm_trace.py` | per-step ‖ΔW‖_F + loss TrainerCallback (root copy is live; an old copy sits in `archive/analysis/`) | train_cs | active |
| `residual_save.py` | rank-2r W0-relative conversion for residual-init methods (mandatory) | train_cs save path; validate_* gates | active |
| `corda_init.py` | static CorDA-KPA init | train_cs `--corda`; reused by cordapp_init | active |
| `cordapp_init.py` | CorDA++ init (dynamic covariance/rank) | train_cs `--cordapp` | active |
| `milora_init.py` | MiLoRA bottom-r SVD init | train_cs `--milora` | active |
| `data_aware_init.py` | PiSSA top-r SVD init | train_cs `--pissa` | active |
| `sclora_init.py` | SC-LoRA D+/D− covariance init | train_cs `--sclora` | active |
| `lora_null_init.py` | LoRA-Null null-space init | train_cs `--lora_null` | active |
| `fdelta.py` | standalone F_Δ reference implementation (CLoRA Table 4 targets) | nothing imports it (eval uses `uio_inprocess.fdelta_inprocess`); kept as reference | active (reference; guardrail: do not touch) |
| `scripts/deepseek/` (12 files) | DeepSeek-V4-Flash 284B pipeline (train/eval/ce/geo/fp8_dequant/medmcqa_prep/node launchers) | `eval_deepseek.py` imported by eval_one_gpu; rest ran on DS nodes | active (eval_deepseek); rest provenance |

## 2. Orchestration / schedulers / jobs (all INERT post-evacuation 2026-07-17 — provenance + reproduction)

| path | role | used-by | status |
|---|---|---|---|
| `gpu_pool.py` | fixed-GPU scheduler, 1 job/GPU | ran `jobs/*.txt` | active (repro machinery, inert) |
| `auto_dispatch.py` | self-refilling dispatcher (`master_dispatch.txt`) | fleet | active (repro machinery, inert) |
| `make_frepro_jobs.py` | faithful-repro job generator (Tables 2/3 + LR×wd sweep) | emits `jobs/frepro_*.txt` | active (repro machinery) |
| `build_lean.py` | merges job files; done cells → eval-only | reads frepro4_* lists (now in `archive/jobs_superseded/`) | active (repro machinery) |
| `ce_batch.py` | dispatcher-native batch driver for forgetting_ce | CE backfill job lines | active (repro machinery) |
| `gen_adversarial_jobs.py` | E3/E6 job-line generator (targets a `jobs/fleet/` that no longer exists here) | one-off | archive-candidate |
| `rescale_adapters.py` | E1 interventional scale-matching adapters | one-off; E1 complete per §18.3 | archive-candidate (provenance) |
| `mem_marker.py` | done-marker for `mem_` train-only memory probes | dispatcher | archive-candidate |
| `gpu_watchdog.sh` | restarts dispatcher on idle GPUs | fleet era | archive-candidate (inert ops) |
| `sync_d002.sh` / `evacuate_qwen_adapters.sh` | node-B→A tar-over-ssh sync / adapter evacuation | fleet era | archive-candidate (inert ops; hostname leaks — see RELEASE_PLAN) |
| `results_book_loop.sh` | 30-min regenerate+push loop for `results_book/` | results_book.py | archive-candidate (output declared stale + deleted) |
| `fleet/` (24 files) | multi-node fleet bring-up/monitor/evacuate; `evac_merge.py` built the canonical merged aggregates | ran the campaign; all inert | `evac_merge.py` active-provenance; rest archive-candidate |
| `jobs/` (9 files + `ce_chunks/`) | surviving job lists; `ce_backfill_qwen.txt` (127 lines) kept as DISCLOSURE of unfillable Qwen CE cells | provenance; cited by paper limitations | frozen-data / provenance |
| `restart_staging/` | frepro4-restart patches (already applied) + Qwen 2×2 job generator | need gone post-evac | archive-candidate |
| `portable_parity_pack/` | self-contained copy of the train/eval pipeline (17 py + run_all.sh + fetch_data.py; `data/` empty) | standalone release seed | active — **verify parity first**: pack dated 07-14, root train_cs/eval_one_gpu modified 07-17 |

## 3. Analysis scripts

| path | role | used-by | status |
|---|---|---|---|
| `analyze_full_2026-07-16.py` | full-picture recompute of the magnitude relation (feeds §18) | key_numbers §18 | active |
| `analyze_adversarial_2026-07-16.py` | A1–A10 adversarial-review recomputes (knee, LR-proxy, direction) | §18.2/18.4–18.6 | active |
| `analyze_ebatch_2026-07-17.py` | E1–E7 experiment-batch analysis | §18.3 | active |
| `flag_diverged.py` | quarantine generator → `results/quarantine_diverged.txt` (71 runs) | §18 dataset definition | active |
| `forgetting_ce.py` | MiLoRA/Kalajdzievski CE-to-base metric | ce_batch | active |
| `geo_drift_phase1.py` / `geo_drift_phase2.py` | Llama base-SVD references / per-adapter geometry battery | geometry columns, `adapter_metrics*.jsonl` | active |
| `geo_drift_phase1_qwen.py` / `geo_drift_phase2_qwen.py` | Qwen variants (fingerprints never pool across families) | same | active |
| `base_retention_check.py` | base-model retention alignment vs CLoRA reference | one-off gate | active-support |
| `validate_frepro_residual.py` / `validate_residual_zero_step.py` / `validate_cordapp_cpu.py` | correctness gates (residual round-trip, CorDA++ CPU) | handoff/23; `paper/evidence/` logs | active-support |
| `retfix_retention_gate.py` / `retfix_bbh_only_report.py` | retention-axis fix diagnostics / BBH-only report | handoff/22 decision | active-support (historical fix) |
| `analyze_magnitude_law.py` | pre-freeze insight miner | superseded by analyze_full_2026-07-16 | archive-candidate |
| `analyze_matrix.py` | `mtx_`/`mtxm_`-era 2×2 analyzer; reads `campaign_summary.jsonl` (STALE store) | none current | archive-candidate; do not source its output |
| `results_book.py` | `results_book/` markdown generator (output stale + deleted 07-17) | results_book_loop.sh | archive-candidate |
| `paper_assets.py` | old table/figure generator from stale registry (deprecated 06-29) | none current | archive-candidate / stale-do-not-source |
| `paper_figs_v2.py` | pre-freeze figure generator → `paper/figs_v2/` — **flag**: older docs call it "canonical" but it reads the §18-stale registry; final figures come from `paper/writing/` scripts | none current | archive-candidate (keep for provenance) |
| `metamath_prep_395k.py` / `math_test_prep.py` | dataset builders (MetaMathQA-395K; GSM8K/MATH test, alpaca schema) | data under `repro/LLM-Adapters/` | active |
| `metamath_prep.py` | old 100K variant | superseded by 395k | archive-candidate |
| `paper/writing/analysis_final/*.py` (ladder / op_points / seed_stability / ds284b_recurrence, all `_2026-07-17`) | §19 post-freeze CPU analyses (each preflight-reproduces §18.1) | §19, docs 06–09 | active |

## 4. Paper build (`paper/` tree)

| path | role | status |
|---|---|---|
| `paper/writing/paper.tex` | current main draft (2026-07-19 framing) | active |
| `paper/writing/paper_conventional.tex` | conventional-layout rewrite (mirrors Overleaf main.tex) | active |
| `paper/writing/paper_draft.tex` | old "Wake-Up Call" draft (verdict title; violates observational framing) | stale-do-not-source |
| `paper/writing/paper_prefreeze_backup_2026-07-18.tex` | backup snapshot | archive-candidate |
| `paper/writing/references.bib` | bibliography | active |
| `paper/writing/data/key_numbers.md` | THE canonical numbers doc — quote only §18 (FINAL FREEZE) + §19 (ADDENDUM); earlier sections carry inline STALE markers | active (canonical) |
| `paper/writing/data/campaign_summary*.jsonl` | 07-02-era registry snapshots | stale-do-not-source (provenance) |
| `paper/writing/data/clora_table4_extracted.md`, `registry_cleaning_report.md` | extraction/provenance notes | active-support |
| `paper/writing/analysis_final/` (docs 01–09 + `PAPER_BLUEPRINT.md` + output logs) | the story layer; BLUEPRINT is the writing spec | active (canonical) |
| `paper/writing/fig_*.py`, `make_fig9_lr_artifact.py`, `make_figs_split_lora_null.py`, `make_table_lr_artifact.py`, `analysis_a1_a4.py`, `figstyle.py` | current figure/table/stats generators | active |
| `paper/writing/figures/` (39 files) + `tables/` (17 .tex) | generated paper exhibits — note `tables/table_grand.tex` has NO generator in-repo (lost; see RELEASE_PLAN) | active (generated) |
| `paper/writing/figures_frozen_backup/` | pre-regeneration figure backups | archive-candidate (backup) |
| `paper/writing/acl_analysis/` (9 subdirs, ~217 files) | 2026-07-19 ACL repositioning campaign reports | active (frozen analysis outputs) |
| `paper/writing/MISSING_EXPERIMENTS.md` | compute-gated relaunch list (07-19 audit) | active |
| `paper/writing/` misc md (INTERESTING_INSIGHTS, EXPERIMENTAL_FAIRNESS_AUDIT, FINAL_TABLE_PLAN, THESIS_VALIDATION_PLAN, REBUTTAL_PREP, adversarial_review_2026-07-16/17.md, fleet_findings.md, 06_reconciliation.md, artifact_status_report.html) | working docs; adversarial reviews fed §18; 06_reconciliation is key_numbers' contradiction log | active-support; pre-freeze ones (INTERESTING_INSIGHTS, FINAL_TABLE_PLAN, THESIS_VALIDATION_PLAN) — currency unverified, verify against §18 before citing |
| `paper/` root assets (fig1/fig2/fig4 pngs, table_main_*.tex/.txt, summary.txt, baseline_fidelity_audit.md) | oldest generated assets (paper_assets.py era, superseded twice) | stale-do-not-source |
| `paper/figs_v2/` | paper_figs_v2.py output (pre-freeze) | archive-candidate |
| `paper/evidence/` (2 gate logs) | validation-gate evidence for handoff/23 | frozen-data (provenance) |
| `paper/.overleaf-git/` | Overleaf git mirror (contains identity in git config — see RELEASE_PLAN) | exclude from release |

## 5. Docs

| path | role | status |
|---|---|---|
| `README.md` | directory front door (2026-07-17, post-freeze) — note it writes `/home/guy/...` (fleet host) not `/home/guyb/...` | active (internal; superseded at release by `RELEASE_README.md`) |
| `RELEASE_README.md` | release-grade README (this release prep) | active (release) |
| `WORKDIR_ALIGNMENT.md` | onboarding map; fleet sections §c/§e marked INERT | active (internal) |
| `CLEANUP_MANIFEST.md` | executed archive/deletion record (passes 07-09, 07-17) | active (internal) |
| `RELEASE_PLAN.md` | release proposal (this release prep; nothing executed) | active (internal) |
| `ACL_CAMPAIGN_INSIGHTS_2026-07-19.md` | latest insights synthesis for the PI | active (internal) |
| `agent_instructions.nd` | Phase-1 "UIOrthoLoRA go/no-go" instructions — dead original objective, pre-pivot numbers | stale-do-not-source (historical) |
| `handoff/` (40 files + `data_snapshots/`) | decision log; **`41_EVACUATION_2026-07-17.md` = current state doc**; `handoff/README.md` = index; 20/25–28/33 campaign provenance; 00–16 + 19 stale-do-not-source for numbers | mixed — see per-doc index in `handoff/README.md` |
| `papers/` (5 PDFs) | source-paper ground truth (CLoRA, CorDA++, MiLoRA, PiSSA, SC-LoRA) | frozen-data |
| `.claude/settings.local.json` | local Claude Code permissions | exclude from release |

## 6. External repro baselines

| path | role | status |
|---|---|---|
| `repro/{CLoRA, CorDA, LLM-Adapters, LoRA-Null, MiLoRA, SC-LoRA}` | vendored reference method repos + training data (commonsense_170k, metamathqa_395k, GSM8K/MATH under LLM-Adapters) | frozen-data / do-not-modify. **FLAG: on this machine all 6 dirs are EMPTY** — vendored content lives on the fleet host; release must re-vendor, submodule, or document fetch instructions (`portable_parity_pack/fetch_data.py` exists for data) |

## 7. Data / results (frozen, immutable)

| path | role | status |
|---|---|---|
| `results/` (~1,816 entries; 1,661 result dirs at freeze, 1,500 full evals) | per-run dirs, each with `summary.json` (source of record) | frozen-data |
| run-name prefixes | `frc_`/`frm_` faithful-repro Llama CS/math (307/168); `lrsw_`/`lrswm_` Llama LR×wd sweeps (240/126); `qwsw_`/`qwswm_` Qwen sweeps (182/187); `mtx_`/`mtxm_` old 2×2 matrix (126/24, pre-eval-fix — do not mix); `b4_` E4 eval-matched wave (39); `e1_` rescale interventions (24); `brl_`/`brq_` bridging; `dsv4_` DeepSeek; `base_` ceilings (22); `mem_` memory probes; UIO-era: `uioW2-4_`, `uioT_`, `a5_`, `d1_`, `databasis_`, `forensics_`, `grid_`, `lrsweep_` | frozen-data |
| `results/campaign_summary.jsonl` (645 rows) | pre-freeze registry | **stale-do-not-source (§18)** — provenance only |
| `results/forgetting_merged.jsonl`, `results/geo_drift/adapter_metrics_merged.jsonl` | canonical merged aggregates (built by `fleet/evac_merge.py`) | frozen-data (canonical) |
| `results/forgetting*.jsonl` (chunks), `results/geo_drift/` (base_svd*, permatrix*), `train_registry.jsonl`, `eval_registry.jsonl`, `quarantine_diverged.txt` (71 runs), `dsv4_adapt_n1000_logscores.jsonl` | component stores | frozen-data |
| `results/ds_adapters_evac/` | 21 DeepSeek-284B adapter tar parts + SHA256SUMS — the ONLY surviving checkpoints in the project | frozen-data (critical) |
| `results/_archive_wikitext_corda/` | quarantined contaminated old-CorDA runs | frozen-data (excluded from analysis) |

## 8. Already archived (`archive/`)

| path | contents | status |
|---|---|---|
| `archive/scripts/` (24) | dead trainers/evals/orchestrators/job-gens | archived |
| `archive/analysis/` (10) | forensics/leakage one-offs (incl. old `norm_trace.py` copy) | archived |
| `archive/jobs/` (53) + `archive/jobs_superseded/` (17) | completed job lists | archived |
| `archive/writing_2026-07-17/` (27 + handoff 34–40) | 07-02 writing suite, STATUS.md, assessments | archived; stale-do-not-source for numbers |

## Referenced but not present on this machine

`logs/`, `models/`, `results_book/` (deleted 07-17), `jobs/fleet/`, `/scratch/cf_models/` checkpoints (all 7B checkpoints destroyed per MISSING_EXPERIMENTS.md), contents of `repro/*`. Scripts hardcode the fleet host's `/home/guy/...` vs this machine's `/home/guyb/...`.

## Uncertain / needs human decision (flagged)

1. `paper_figs_v2.py` — "canonical" in older docs but reads the stale registry; recommend archive with provenance note.
2. `paper/writing/{INTERESTING_INSIGHTS, FINAL_TABLE_PLAN, THESIS_VALIDATION_PLAN, REBUTTAL_PREP}.md` — currency unverified against §18.
3. `handoff/data_snapshots/` — contents not inspected.
4. `results/` prefixes `scl2_` (24), `finalize_` (8), `fr_` (6), `grid_` (9) — undocumented in the four input docs.
5. `portable_parity_pack/` — needs a diff against root pipeline files (pack 07-14 vs root 07-17) before use as release artifact.
