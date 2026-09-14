# ICLR campaign startup and continuation — 2026-09-14

Status: **P0 smoke retry completed and independently validated**; two-task
seed-31415 throughput calibration is next. The first interrupted attempt remains preserved.
The resource gate is resolved. The full P0–P8 objective is active and incomplete. This document
does not amend the authoritative experiment design or freeze a confirmation grid.

## P0 completed — 13:31:57 UTC

Availability update at 14:02 UTC: the author coordinated with the account using
the assigned GPUs. Host telemetry verified GPUs 2/3 free, each at 15 MiB and 0%
utilization; temperatures 32/28 C. No foreign process was modified or stopped.
Earlier, at the author's explicit request, only GPU PID/account metadata was
queried to identify account `goody`; no commands, directories or outputs were
inspected. All campaign GPU leases were terminal during that occupancy.

New CPU work includes magnitude-only coefficient selection, exact initial log
grids, bounded both-outer-dose expansion proposals, three fixed RANDPROJ
orientations per dose, failure retention and hash-bound fixed-endpoint evidence
reading. Initial-grid registration/admission requires two validated timing
pilots. Confirmation and expansion-launch admission remain closed. No actual
magnitude grid or endpoint has been registered without those measurements.
The controller now admits the registered initial grid with `--purpose matching`,
`--calibration-protocol`, `--calibration-entry` and matching explicit `--steps`.
All nine P1 controls plus LoRA/full-FT are exercised by synthetic CPU workers;
these are implementation checks, not empirical runs or training-seed replicates.

Retry ID `20260914T132305Z_8729a4c83ee8`, GPU 2, ran from 13:23:05 UTC until
worker exit at 13:29:49 UTC in dedicated tmux session `smoke_gpu2_retry1` on the
`iclr_6aa54397_20260914` server. Source commit `08ffa9c2` was normally pushed to
`ortho_new` after rebase and preflight. All **eight optimizer steps** completed
with finite losses/gradients. Six saved checkpoints (steps 0,1,2,4,6,8) independently
reproduced task metrics, full P3 geometry and fixed-original-MLM-head P8 observations.
Locked fixed/selected endpoints were separately evaluated from reloaded checkpoints.
The whole-run audit reloaded every state again, verified original pretrained head
tensors and original bases, bound the numerical-reproduction reports to checkpoint/
reference/observation hashes, and checked trajectory, task split, P3/P7/P8 and
manifest/source-archive contracts. The ledger now records this P0 run completed.

Validation path relative to the output root:
`runs/iclr_6aa54397/P1_MIX/rte/seed_31415/20260914T132305Z_8729a4c83ee8/validations/9ec903673c0743ca819187d7b927f524/report.json`.
Validation SHA256: `9629cd520ce382e48ac81b0cde7fe4004ad6547f8269da6369c60198a11cdec6`.
Preflight: `data/campaign_v1/preflight/20260914T132139Z_8851cd4f/report.json`.

Observed smoke costs, not full calibration or a campaign ETA:

- Worker elapsed 394.63 seconds; supervised elapsed 403.91 seconds (6.73 minutes).
- Warmup-excluded optimizer steps: 0.624–0.671 seconds; median approximately 0.657 seconds.
- Warmup-excluded throughput approximately 6,168 nonpadding tokens/second.
- Peak allocated training CUDA memory 2,253,511,680 bytes (2.10 GiB);
  peak reserved 2,797,600,768 bytes (2.61 GiB).
- CPU SVD setup 4.37 seconds. Full trajectory diagnostics and independent reload
  measurements dominate the short smoke; they do not occur after every step in
  a long run. Initial learned-since-insertion energy is now exactly zero under
  the explicitly pinned CPU effective-delta arithmetic.

Counts: P0 completed 1, interrupted 1; throughput calibration 0/2; magnitude
calibration 0; confirmation 0/66; extensions 0. Submission readiness and full
campaign completion remain distinct; no scope reduction was authorized.

The new immutable timing protocol is
`campaign_outputs_v1/protocols/timing_20260914_v1.json`: RTE and MRPC, P1_MIX,
seed 31415, 128 optimizer steps each, effective batch 32, float32/eager attention,
four CPU diagnostic workers per GPU worker, full checkpoint/P3/P7/P8 retention.
Its explicit purpose is throughput only: no task or geometry-based hyperparameter
selection, magnitude matching or confirmation claim. The workers require the
successful P0 gate and exact protocol hashes before admission. Planned placement:
RTE GPU 2 and MRPC GPU 3. New controller code independently validates whole runs
before marking them complete. Larger magnitude-calibration and confirmation
orchestration are still pending.

## GPU P0 update — first attempt ended 13:18:07 UTC

Code and the successful 104-test/four-audit/source-fingerprint preflight were
rebased and pushed normally to `ortho_new` at `707b9d69` before launch.
Preflight: `data/campaign_v1/preflight/20260914T130653Z_13f60f28/report.json`.
The first persistent controller used the dedicated tmux server
`iclr_6aa54397_20260914`, session `smoke_gpu2`. Run ID:
`20260914T130831Z_d8fc9d01b1fb`, physical GPU 2, seed 31415, P1_MIX instrument,
P0 smoke stage. Durable path below the assigned output root:
`runs/iclr_6aa54397/P1_MIX/rte/seed_31415/20260914T130831Z_d8fc9d01b1fb/`.

Real pretrained GPU forward/effective-delta, merge/unmerge, disabled-adapter and
original-MLM-head checks passed. Maximum classifier merge error was 3.73e-8;
disabled-versus-original error 7.45e-8; original MLM-logit CPU/GPU comparison
maximum error 6.10e-5 (declared float32 P0 atol/rtol 1e-4). These checks do not
establish trained outcomes.

Initial full P3 diagnostics took **10.872–11.494 seconds per module**, mean
11.110 seconds, on GPU float64. A CPU evaluation of the first saved query layer,
with the same five cutoffs and three orientation draws, took **1.673 seconds**.
This one-layer placement comparison is not a full training-throughput estimate.
GPU initialization diagnostics also exposed tiny arithmetic-only learned-delta
energies (7.51e-19 to 7.79e-19): CPU-computed insertion buffers were subtracted
from CUDA-computed float32 deltas. These are not learned updates and their
normalized fractions must not be interpreted scientifically. The optimized
worker pins effective-delta arithmetic to CPU consistently with insertion state,
uses float64 metric algebra, and supports four ordered CPU diagnostic workers.
Training remains explicitly on the assigned CUDA device. No old observations
were replaced to hide either the overhead or roundoff issue.

An explicit stop request let the worker finish initial diagnostics and save its
step-0 optimizer/scheduler/RNG/data/model checkpoint. It exited normally with
scientific status **interrupted**, no optimizer steps performed. The supervisor
observed child exit before settling its lease. Elapsed/charged reservation:
575.39/575.40 seconds (approximately 0.160 GPU-hours); retained run bytes:
1,569,605,285. Both immutable run/engine manifests, source archive, reference,
checkpoint, observations, costs-to-date, logs and execution receipt remain.
No process from another project was inspected or touched.

The durable scientific ledger is `campaign_outputs_v1/run_ledger.jsonl`; resource
events are in `campaign_outputs_v1/accounting_v1/reservations.jsonl`.
Counts: P0 completed 0, interrupted 1; calibration 0; confirmation 0/66;
extensions 0. The next smoke must use a **new retry ID** naming this attempt.
P0 is not complete. Whole-run validation, timed calibration and full ETA remain
pending. Initial 50 GiB storage is sufficient for the smoke; an increase to
200 GiB was requested because six full-FT trajectories alone require roughly
67 GiB for uncompressed float32 model and Adam states. No increase is assumed
until the author answers.

## Continuation update — 13:05 UTC (supersedes resource-pending entries below)

The author explicitly authorized GPUs 2 and 3 for as much time as needed until
all experiments are finalized, followed by "Go ahead". The preceding proposal
specified an initial 50 GiB in this dedicated clone's `campaign_outputs_v1/`
and public model/dataset downloads. The assistant explicitly reported interpreting
"Go ahead" as approval of those storage/download terms while replacing the
proposed 4 GPU-hour cap with completion-duration authorization. The exact record
is `data/campaign_v1/RESOURCE_AUTHORIZATION_20260914.json`. Do not impose or invent
a numerical global cap; retain finite per-worker reservations. Do not exceed
50 GiB without requesting expanded storage. Other agents and GPUs remain out of scope.

The complete current CPU suite passes **104 tests**. New code covers immutable
pinned source copies, fingerprinted RTE/MRPC splits, fixed per-ID probe masks,
original-head loading without missing/new tensors, campaign-wide locked resource
reservations, concurrent reservation conflicts, retained failure costs, owned-child
supervision, safe stop boundaries and integrated tiny-model workers for six arms.
These are still synthetic/local correctness observations, not trained task results.
The P0-only `smoke.py` entry point requires an exact-source successful preflight,
committed campaign code, a resource allocation and an immutable complete job.
It records source archives and scientific run-ledger transitions; successful
children stop at `awaiting_validation` pending whole-run validation.

Input preparation completed at **12:55:43 UTC**, taking **22.21 seconds** (download
and CPU preparation only). Durable bundle:
`campaign_outputs_v1/inputs/preparation_20260914T1254Z/` in the dedicated clone.
Public model: `FacebookAI/roberta-base` commit
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`; GLUE: `nyu-mll/glue` commit
`bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`; probe corpus: `Salesforce/wikitext`
commit `b08601e04326c79dfdd32d625aee71d232d685c3`.

- RTE: 1,992 training, 498 inner-selection, 277 locked official-validation examples.
- MRPC: 2,934 training, 734 inner-selection, 408 locked official-validation examples.
- Fixed sequence length 128, stratified selection fraction 0.2, split seed 271828.
- Probe: 256 WikiText-2-raw-v1 test examples, fixed 15% masks (all selected tokens
  replaced by MASK), mask seed 161803. The corpus is held out from adaptation,
  not guaranteed unseen in RoBERTa pretraining. No autoregressive-perplexity claim.
- Every source file, prepared tensor bundle, split/sample ID and mask is retained
  and hashed in its immutable manifest. Input choices precede all training outcomes;
  they do not freeze confirmation hyperparameters or the expanded run matrix.
- Actual pinned RoBERTa MLM loads on CPU with all expected tensors and no unmatched
  tensors: 12 layers, width 768, 124,697,433 parameters.

At 12:53 UTC, assigned GPUs each used 15 MiB / 0% utilization; GPU 2 was 28°C,
24.71 W and GPU 3 was 26°C, 27.39 W. Shared data storage had 659 GiB free before
downloads. The campaign output directory occupies approximately 976 MiB after
preparation, including isolated cache and sealed copies. No unrelated jobs were
inspected. Next: final preflight, commit/rebase/push, then a persistent GPU-2 RTE
smoke and independent P0/whole-run validation. Training ETA is still unmeasured.

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
