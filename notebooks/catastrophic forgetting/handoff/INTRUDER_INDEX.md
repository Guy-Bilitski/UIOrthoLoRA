# Intruder-dimension campaign — index

Everything produced in the 2026-08 H200 campaign, and where it lives. Start here.

## Read in this order

| doc | what it is |
|---|---|
| **`handoff/INTRUDER_FOR_PAPER.md`** | **Start here if you are writing the paper.** What the experiment adds, precise definitions, the results table, the claims you MAY make, the claims you must NOT make (including one we withdrew), limitations, and positioning against Shuttleworth/Xie. |
| **`handoff/EXPERIMENT_FINAL.md`** | **THE spec.** Question, 7 configurations, training, evaluation, intruder criterion, the 7 arms A–F, what each contrast tests, analysis plan, wording discipline. Supersedes all earlier plans. |
| `handoff/INTRUDER_RESULTS.md` | Long-form results and methods: measured numbers, the calibration checks, and the caveats to state in the paper. |
| `handoff/TIERA_RUN_LOG.md` | Chronological log: what was run, what broke, what was ruled out. The record of *why* the design is what it is. |
| this file | Map of scripts, data and archives. |

## Scripts — analysis (the experiment)

| script | role |
|---|---|
| `intruder_pass.py` | **Intruder measurement.** Top-10 directions of `W0+dW`, max\|cos\| vs the full pretrained basis, τ=0.5. `--selftest` (6/6) must pass before trusting output. |
| `geo_fullref.py` | Builds the full base left-singular basis store that the criterion-exact comparison needs. Run once per model. |
| `intruder_ablate.py` | Builds arms **B / C / D**. |
| `arm_e_build.py` | Builds arm **E** (`--match magnitude`) and **Ep** (`--match perturbation`). |
| `arm_f_build.py` | Builds arm **F** (count-matched random non-intruder deletion, `--draw N`). |
| `build_final_queue.py` | Emits the remaining evaluation queue, deduped against `results/`. Safe to re-run; emits less as work completes. |
| `scale_adapter.py` | Uniform-scaled copies of an adapter → local magnitude curve. |
| `intruder_report.py` | R1/R2/R3 readouts plus the paired ablation read-out. |
| `magnitude_residuals.py` | Fits the magnitude law on the frozen pool, scores each config's residual against intruder energy. |
| `paper_table.py` | **The deliverable.** Prints the final tables from `results/`, pending cells as `--`. `--csv` for raw numbers. |

## Scripts — operations

| script | role |
|---|---|
| `adapter_health.py` | Chain gate: aborts before the expensive eval if an adapter is non-finite; quarantines it. |
| `nan_watchdog.sh` | Kills a run the moment `grad_norm` goes NaN. **Not used by the current queue** — the retry loop supersedes it. |
| `auto_intruder.sh` | Scores every finished adapter (CPU, polls). |
| `auto_ablate.sh` | Builds arms B–F for every finished config and queues their evals into `jobs/pending_ablation.txt` (CPU). |
| `auto_switch_go2.sh` | One-shot: switches the GPU to `jobs/tierA_go2.txt` when config 3 evacuates. |
| `evacuate_cell.sh` | Per-cell checkpoint evacuation with checksum verification (pre-existing). |
| `gen_tierA_jobs.py` | Job-file generator from the original spec. Historical — current queues are explicit files. |

## Job files (active)

- `jobs/tierA_go.txt` — running now: config 3, then the Qwen configs (retry-hardened).
- `jobs/tierA_go2.txt` — armed next: arms E/F for config 1, MiLoRA's arms, then Qwen.
- `jobs/tierA_ablation_k10.txt` — the A–F evals at the locked protocol.
- `jobs/pending_ablation.txt` — auto-appended by `auto_ablate.sh`.
- `jobs/tierA_exp2_anchors.txt` — Experiment 2 (Qwen rescale ladder), separate experiment.

## Data

| path | contents |
|---|---|
| `results/<run>/summary.json` | per-adapter task accuracy, retention, F_delta |
| `results/intruder/<run>.json` | per-adapter intruder geometry (per-matrix + aggregate) |
| `results/intruder/intruder_registry.jsonl` | one line per scored adapter |
| `results/geo_drift/base_svd[_qwen]/` | base SVD store (top/bottom 256) |
| `results/geo_drift/base_svd_fullU[_qwen]/` | **full** base left-singular basis (criterion-exact reference) |
| `/home/kfir/cf_models/<run>/` | adapters, including all intervention arms |
| `/home/kfir/tierA_evac/<run>/` | evacuated copies (checksum-verified) |
| `/home/kfir/cf_models_failed/` | quarantined NaN / partial adapters (kept, not deleted) |

## Archive — `archive/tierA_2026-08/`

Nothing is deleted; everything below is kept for the eventual repo release.

| folder | contents |
|---|---|
| `ops_scripts/` | one-shot launchers and queue-switch watchdogs that already fired (`auto_launch_tierA1.sh`, `auto_relaunch_serial.sh`, `auto_after_cell9.sh`, `auto_scale_then_slice.sh`, `auto_switch_master.sh`) |
| `job_files/` | every superseded queue (the 18-cell slice, the staged/final/master iterations, the smoke cell) |
| `diagnostics/` | the Qwen-NaN investigation: `diag_qwen_nan.py`, `run_safe_sdpa.py`. Kept because they document what was ruled out. |
| `superseded_docs/` | `SESSION_STATE_H200_2026-08-26.md`, `STATUS_FOR_AGENT.md` — both stamped with what replaced them |

Two pre-existing docs are stamped in place rather than moved, because they are still
partly current: `handoff/H200_BOOTSTRAP.md` (environment sections still accurate) and
`handoff/TIER_A_SPEC_2026-08-23.md` (Experiment 2 sections still stand).

## Known environment facts

- One process on the GPU at a time; every job waits for a free card first.
- Qwen training NaNs at batch 70 **nondeterministically**. Ruled out by controlled test:
  GPU co-tenancy, `OMP_NUM_THREADS`/`MKL_NUM_THREADS`, seed, method, learning rate,
  attention backend, left padding, and the data itself. Mitigation: retry up to 6×, each
  attempt verified by `adapter_health.py`.
- HF token: campaign processes need `HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token`
  (a stale, access-rejected token sits at `$HF_HOME/token` and would otherwise win).
