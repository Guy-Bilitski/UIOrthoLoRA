# ICLR campaign startup and continuation — 2026-09-14

Status: CPU preparation advanced; **no pretrained smoke, calibration or training
run launched**. The full P0–P8 objective is active and incomplete. This document
does not amend the authoritative experiment design or freeze a confirmation grid.

## Continuation update — 12:20 UTC

The preceding goal turn made progress: isolated CPU foundations and actual
validation artifacts were pushed at `792b7a1a`. This continuation again makes
implementation progress; it is not a wait on a live training job. No pretrained
task-training run is running, and no process restart was inferred from a timeout.

The campaign now has 69 CPU tests and these additional components:

- `checkpoints.py`: one hash-verified frozen reference plus compact per-step
  trainable states, AdamW/scheduler state, exact Python/NumPy/Torch RNG, data-order
  state and inherited trajectory history. Frozen tensors are checked on save;
  existing checkpoints are never replaced. A failed write leaves the last durable
  checkpoint pointer unchanged.
- `batching.py` and `engine.py`: fingerprinted examples, exact shuffled-order
  resumption, a fixed-step optimizer with example-weighted gradient accumulation,
  explicit paired dropout seed, accuracy-based validation selection, retained
  trajectory/best checkpoints, finite-gradient checks and bounded per-run GPU
  execution gates. Fixed and selected endpoints remain separate. Engine exits
  await independent run validation; they never automatically mark completion.
- `regularizers.py`: cached fixed projectors with objective/gradient agreement
  against the reference for all seven adapter-training P1 arms. Actual mixing
  values remain populated in the unregularized arm. No GPU speed claim is made.
- LoRA forward/merge/reload, full-FT displacement accounting for other backbone
  weights, generic common-frame diagnostics, and RoBERTa reconstruction without
  recomputing the original SVD.
- `validation.py`: independent checkpoint reload and scientific-output comparison.
  A tiny RoBERTa fixture performs optimizer steps with real spectral diagnostics
  and original-head MLM probing, then reproduces every saved checkpoint's outputs
  exactly. Those synthetic CPU fixtures are not pretrained task-training runs.

The current suite has 69 tests; static undefined/unused-name checks pass. New
checks cover exact interrupted-versus-uninterrupted optimizer/RNG/data state,
checkpoint corruption, injected disk-full failure, unchanged frozen references,
distinct selected/fixed endpoints, and independent task/P3/P8 reproduction.
One checkpoint validation is explicitly insufficient to mark a run complete:
the ledger requires whole-run scope, P3/P7/P8 and the complete artifact contract.

Immutable continuation preflights:
`data/campaign_v1/preflight/20260914T122320Z_56809356/report.json` and
`data/campaign_v1/preflight/20260914T122516Z_1d6b70e0/report.json`.
The latter passes all 69 tests, all four supplied checks and all six historical
source fingerprints. It also tests that integer counts must reproduce exactly,
even when floating-point metrics have a declared tolerance. Earlier bundles
remain unchanged; content hashes identify each tested source version.

Resource assignments remain unchanged and incomplete. At 12:20 UTC, physical
GPUs 2 and 3 both showed 0% utilization / 15 MiB used (30°C / 24.33 W and
28°C / 26.55 W respectively). The shared data filesystem reported **659 GiB
free**; `/tmp` still reported approximately 8 GiB. These are available-space
observations, not a confirmed campaign quota. Budget, persistent checkpoint
allocation and model/dataset download policy remain missing. No caches or jobs
belonging to another agent were inspected to explain changing free space.

Next independent CPU work: pinned offline-capable model/data/probe preparation,
persistent worker and monitoring integration, campaign-wide compute/storage
reservations, whole-run validation, and calibration/freeze orchestration. The
required pretrained smoke, calibration, 54 P1 and 12 early P5 runs remain at zero.
All later required/conditional phases retain their original scope. Training ETA
and GPU-hours remain unmeasured; do not infer them from CPU fixture timings.

The sections below preserve the initial inventory and reports as historical
startup observations; this continuation supersedes their implementation status.

## Isolation and provenance

- Dedicated checkout:
  `/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914`.
- Existing branch: `ortho_new`; initial and fetched remote source revision:
  `5505a244c71da72210f0e6987d292d9626f098d5`.
- The requested `git switch ortho_new` and `git pull --rebase origin ortho_new`
  completed in this clone. Its sparse checkout includes the ICLR handoff,
  delivered `src/peft`, packaging metadata and relevant legacy Python sources.
  The original shared working tree was not edited.
- All nine user-specified handoff/provenance files were read completely in order.
  While cloning, they were read from the exact Git object at the independently
  verified remote tip. The manuscript and relevant implementation were also read.
- Legacy training sources and CSVs remain unchanged. Prospective code is under
  `notebooks/iclr/campaign/`; CPU-only artifacts have a distinct dataset version,
  `data/campaign_v1/`.
- A Codex durable objective was created for the entire campaign. It has not been
  marked complete. The resource gate has been reported; it has not been bypassed.

## Resource inventory

Measured on 2026-09-14; the first check was at approximately 11:05 UTC.

| Item | Observed / assigned |
|---|---|
| Host | `eimtest-ESC4000-G4` |
| Scheduler | No `squeue`, `qstat` or `bjobs` command detected; this does not establish site policy. `tmux` and `screen` are available. |
| Explicitly assigned GPUs | Physical IDs **2 and 3**, authorized by the user |
| GPU 2 | RTX 3090, 24,576 MiB; UUID `GPU-964910cb-d7b3-9be7-b92f-2f2c4b928b42` |
| GPU 3 | RTX 3090, 24,576 MiB; UUID `GPU-3a02fb2b-53a1-7d43-5b6e-d3c8a33df6b7` |
| Host GPU count | Four RTX 3090 devices visible to inventory; only two are assigned |
| Latest assigned-device observation | Both 0% utilization and 15 MiB used; GPU 2: 32°C / 24.60 W; GPU 3: 30°C / 26.69 W |
| Driver | `560.35.03`; campaign Python Torch build `2.5.1+cu124` |
| CPU / RAM | Two Xeon Gold 6226R sockets, 64 logical CPUs; 251 GiB RAM, approximately 233 GiB available at startup |
| Data filesystem | ext4 `/dev/sda`, approximately 11 TiB total; 684 GiB available initially, 675 GiB after isolated clone/environment preparation; 94% used |
| `/tmp` filesystem | Approximately 8 GiB available; 100% used after rounding. Not a checkpoint destination. |
| Disk quota | `quota` command not found; no verified per-user quota or campaign storage allowance |
| Maximum compute budget | **Not specified**: GPU-hours or wall-clock limit required before training |
| Persistent model-output directory | **Not assigned**; the durable code checkout is not an inferred checkpoint allocation |
| Model/dataset downloads | **Permission pending**; no model or dataset downloaded |
| Existing caches | HF cache environment variables unset. Library default `/home/guyb/.cache/huggingface/hub` does not exist in the inspected environment. No broader search of another agent's paths was performed. |
| Campaign package cache | `.campaign_uv_cache/` inside this checkout; environment `.venv/`. Package downloads were individually authorized by execution approvals. |
| Other jobs | Every pre-existing process and every unassigned device is outside this campaign's scope. No unrelated process command lines, working directories or outputs were inspected; no other job was modified or stopped. |

The remaining resource question was sent while CPU work continued: maximum
GPU-hour/wall-clock budget, persistent output directory plus storage allowance,
and Hugging Face model/dataset download permission. There has been no response
to those fields. The user-provided gate in the task and GPU_HANDOFF §6 requires
them before model training. Package-install permission does not authorize model
or dataset downloads.

## CPU validation and implementation

All four supplied CPU checks passed. All six delivered training-source
fingerprints passed. The prospective suite passes **43 tests**, including:

- square and rectangular forward/delta/merge/unmerge/disable/reload comparisons;
- zero, leading, tail and cross block cases, complete energy reconstruction,
  scalar symmetries, identity initialization and a singular-value crossing;
- exact loss normalization, disconnected/coordinate projector nullspaces,
  independent Haar projectors, magnitude matching and MRPC accuracy plus F1;
- tiny randomly initialized RoBERTa module placement, identical insertion-state
  head references, all-parameter full-FT eligibility, backward gradient health
  and exact saved-state logits;
- original MLM-head logits and decoder independence from modified embeddings;
- original-frame diagnostics, block bounds, orientation nulls and seed-unit
  aggregation;
- direct forward/effective-delta correspondence with the delivered unrotated
  custom adapter, with leading and scaler switches on/off.

Immutable report:
`data/campaign_v1/preflight/20260914T114551Z_c406107e/report.json`.
The bundle includes its manifest, source-content hashes, installed package
versions, every command/stdout/stderr/exit code and a JUnit test report. These are
CPU algebra/integration observations, **not trained-model measurements**.

Post-rebase validation at 11:57 UTC also passed all checks; its separate immutable
bundle is `data/campaign_v1/preflight/20260914T115701Z_7cc2815a/report.json`.
It records source revision `9ad303ed`, which includes the complete dependency
lock and whitespace cleanup. Both validation bundles remain preserved. The source
commits contain only ICLR campaign code, documentation and CPU validation data.

Preparation failures were diagnosed and resolved without touching shared code:
GitHub/PyPI DNS were inaccessible inside the sandbox; scoped commands succeeded
after execution approvals. Shared Python 3.13 RoBERTa import failed at the
TorchVision `nms` operator. A separate Python 3.10.14 environment was installed
and locked. The delivered custom PEFT also requires bitsandbytes even for a dense
CPU import; it was installed only in the campaign environment. No quantized
execution is enabled. An extra SSH clone started during connectivity diagnosis
was cancelled; only a process started by this campaign was interrupted.

Confirmed legacy hazards: obsolete diagnostic basis attributes; zero-filled
mixing logs when coefficients are zero; hard-coded RTE metadata; automatic device
mapping; one retained checkpoint followed by deletion of the entire checkpoint
directory. The campaign code bypasses those legacy helpers. It has not altered
or relabeled old results.

## Run ledger and timing

`data/campaign_v1/run_ledger.csv` currently contains its schema and **zero model
run records**. Run artifacts and the transition ledger must be created under the
assigned output root once it exists. No failure of a CPU/environment check is
counted as a failed model run.

| Phase | Completed / planned | Running | Failed/retried model runs |
|---|---:|---|---:|
| P0 pretrained RTE smoke | 0 / 1 | none | 0 / 0 |
| Timed calibration | 0 / not yet frozen | none | 0 / 0 |
| P1 first confirmation tranche | 0 / 54 | none | 0 / 0 |
| Early P5 LoRA-8 / full FT | 0 / 12 | none | 0 / 0 |
| P2/P4/expanded P5/P6 | 0 / budget-dependent required matrix | none | 0 / 0 |
| P3/P7/P8 actual checkpoint measurements | 0 | none | 0 / 0 |

At the 11:46 UTC update, approximately 41 minutes had elapsed since startup.
The full CPU test suite took 4.35 seconds on one observed execution. This is not
step time or a run-time estimate. Model calibration time and campaign GPU-hours
are both zero. Optimistic/expected/conservative completion timestamps and a
numeric pre-calibration ETA range are **not identifiable yet**. No historical
training-call time has been substituted for measured performance on this host.

After resource assignment, smoke and calibration must measure SVD/setup,
training step median/range, diagnostics and probe cost, checkpoint/reload time,
peak memory and utilization. Forecast two independent single-GPU workers only
after each condition fits and contention is measured. The 66-run tranche is not
the completion boundary; tuning, P2/P4, expanded P5, P6 and conditional additions
must enter the frozen forecast. Largest current uncertainties: compute/storage
allocation, unfinished runner integration, full-FT persistence cost, dense
diagnostic cost, matching-grid expansion, and P6 model access/fit.

## Exact continuation

1. Work only in the dedicated clone above. Re-read this status plus authoritative
   handoff/plan if resuming without context. Confirm `git status` in this clone;
   never reset another checkout or inspect unrelated work.
2. Obtain the three outstanding resource assignments. Keep all model training
   disabled until then. Pin assigned GPU UUIDs and expose one per worker; the
   worker should see one logical `cuda:0`, never `device_map="auto"`.
3. Use this environment/check command from the research root:

   ```bash
   CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
     .venv/bin/python -m notebooks.iclr.campaign.preflight
   ```

4. Finish the production runner. The step engine, compact restart persistence,
   LoRA-8 support and independent checkpoint validator now exist, with CPU tests.
   Remaining: pinned model/data preparation and fixed splits/probe masks;
   persistent worker/monitor; global compute/storage reservation; whole-run
   validation; controlled P7 measurements; calibration grids, selection and a
   frozen manifest/forecast. Preserve every failed attempt with a new retry ID.
5. Validate an actual RoBERTa RTE smoke artifact on one assigned GPU. Do not mark
   P0 complete from the 43 CPU checks. Optimize fixed-projector loss evaluation
   with objective/gradient checks before long runs, then time seed 31415.
6. Freeze endpoints, splits, hyperparameters, ±5% matching, calibration outcomes,
   module norm distributions, run matrix and total budget before confirmation.
   Use accuracy primary and save F1 for MRPC. Do not silently inherit changed
   large-model-search learning rates from the old experiments.py.
7. Execute and monitor in the authoritative order. No campaign training session
   is currently running; there is no tmux session or model process to resume.
8. Before every push: fetch `origin/ortho_new`, rebase only scoped commits, resolve
   only conflicts in campaign files, rerun checks, and push normally to
   `ortho_new`. Do not create a remote branch or force-push.

No manuscript finding has been updated. The central scientific question remains
open, and null/reversed/failed-matching outcomes remain valid prospective results.
