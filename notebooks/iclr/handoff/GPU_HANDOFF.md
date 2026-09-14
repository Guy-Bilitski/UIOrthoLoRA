# GPU session: start here

Updated 2026-09-14. This document is the portable handoff for a fresh agent session. Read it and `EXPERIMENTS_REQUIRED.md` before touching a GPU. The expanded P0–P8 campaign is not complete. A P0-only persistent smoke worker is now implemented and CPU-tested; calibration/confirmation admission and whole-run validation remain pending. Historical handoff scripts reanalyze archived results and test algebra only.

Campaign continuation update, 2026-09-14: a separate server clone now contains a
prospective optimizer-step engine, compact persistent restart states, cached P1
penalties, LoRA/full-FT geometry and independent checkpoint reproduction under
`../campaign/`, now with 104 CPU algebra, input-preparation, supervision and tiny-model integration checks.
All four supplied checks and six original source fingerprints also pass in the
recorded preflights. The production training driver is still incomplete;
the first GPU P0 attempt was preserved as interrupted. Its CPU-diagnostic retry
`20260914T132305Z_8729a4c83ee8` completed eight optimizer steps and six checkpoint
reload/P3/P8 checks; whole-run validation passed at 13:31:57 UTC. P0 is complete.
Two seed-31415 throughput pilots are next; magnitude-calibration/confirmation
have not run. The user authorizes GPUs 2/3 through campaign completion; the
initial 50 GiB output allocation and public downloads are recorded in
`data/campaign_v1/RESOURCE_AUTHORIZATION_20260914.json`. Pinned RoBERTa/RTE/MRPC
and fixed held-out probe inputs are prepared. Read
[`CAMPAIGN_STATUS_20260914.md`](CAMPAIGN_STATUS_20260914.md) for exact checkout,
environment, measured resource inventory, validation artifact, empty run ledger,
remaining implementation and continuation commands. This does not change the
authoritative P0–P8 plan or promote CPU checks to trained-model evidence.

GitHub delivery correction: all files are now ordinary tracked files under `notebooks/iclr/handoff/` in UIOrthoLoRA's existing `ortho_new` branch. Start the server session in this directory after pulling the research repository. This is not a nested Git repository or submodule; do not clone Overleaf to obtain the docs. See `SNAPSHOT_PROVENANCE.md` for the exact source snapshot and portable numerical-audit fixtures.

## 1. Authority and isolation

- The author requests an ICLR analysis paper about leading/tail singular components and regularization, not a claim that UIOrthoLoRA is the best adapter.
- Authorized Overleaf project: **6aa54397e58b10444b0fa2aa only**. The Git mirror's origin must resolve to `git.overleaf.com/6aa54397e58b10444b0fa2aa` before any fetch/push.
- Another agent works on another project. Use a dedicated research checkout, environment, output root and experiment namespace `iclr_6aa54397`. Never change the other agent's branch, shared PEFT implementation, output directory, running processes or GPU allocation.
- Wait for explicit server access, assigned GPU IDs and a compute/storage budget. An idle device is not authorization. Do not kill jobs, change global CUDA/MCP settings, or overwrite caches.
- No credential is stored here. Do not copy the local Overleaf credential helper, tokens, private config or shell history onto the GPU server. Training does not need an Overleaf credential.
- This is not authorization to submit to ICLR, publish results, or contact reviewers. The files in `review_panel` are a supplied simulated review, not an actual ICLR decision.

## 2. Read these files in order

1. `GPU_HANDOFF.md` (this file).
2. `EXPERIMENTS_REQUIRED.md`: **the sole authoritative experiment design**, including exact condition meanings, calibration, matching, statistics and priorities.
3. `REVIEW_PANEL_RESPONSE.md`: review issues, fixes, pending experiments and reasoned non-adoptions.
4. `neurips_2026.tex`: the one active manuscript (legacy filename, official ICLR 2027 style), especially Sections 2–5 and the initialization/reporting appendices.
5. `data/GLUE_PROVENANCE.md`, `data/legacy_mixing/PROVENANCE.md`, and `data/legacy_layers/PROVENANCE.md`.
6. `MANUSCRIPT_AUDIT.md`: what was actually checked and which evidence is still missing.

The complete review packet resides at `../review_panel/` in the original workspace. If only the Overleaf repo is transferred, `REVIEW_PANEL_RESPONSE.md` and the experiment plan carry its operational context; the original packet is not needed to interpret the run IDs. Do not treat reviewer-suggested conclusions as facts or import reviewer synthetic numbers as experiments.

## 3. Current state: completed versus pending

Completed, with no new model training:

- All prose, proofs, tables and figure data are in `neurips_2026.tex`; no blue/gray revision markup.
- 27 archived mixing runs (nine paired task/seed units × A/B/C) reconcile with 1,296 module records (48 attention matrices/run). Raw CSVs remain unchanged.
- Main results now include A/B/C and across-seed sample SD; appendix tables expose every seed and selected step, four-block fractions and logged runtime/memory.
- Figure 2 includes the dimension-only reference; the largest-angle drift metric is supplementary because it is near ceiling under A.
- Expanded algebra includes random-projector expected penalty, initialization offset and identity-alignment definitions; CPU tests are not trained-model evidence.
- Table 2 remains in the main paper at the author's request. All 60 means/SDs are preserved. **MRPC is accuracy for every Table 2 row**, confirmed by the author on 2026-09-14; one column and Avg6 remain. Legacy mixing runs separately use MRPC F1, verified in their logs. New runs must save both metrics.
- Historical retention table is preserved in `archive/retention_2026_09_14.tex`, inactive. It is not evidence in the active paper until its evaluation/provenance gate is met.

Pending, not run and not established:

- Magnitude-only, centered-scaler, initialization-decay, random-projector and both head-only controls; more seeds; lambda dose response; fixed-step and checkpoint trajectories.
- Trained initialization-corrected/identity-residual block energies, scaler vectors, principal-angle spectra, boundary gaps and multi-cutoff drift. Scalar summary ratios do not contain the signed matrices needed to recover these.
- Same-frame LoRA/full-FT and spectral baseline comparisons; same-checkpoint functional probes; modern generative replication; controlled throughput/setup/merge cost measurements.
- Full historical broad-GLUE manifests and parameter inventories. The stated rotated large recipe does not reconcile with old 0.6M; do not fabricate replacement historical counts. Configuration-derived formulas belong in separate columns/records.

No dynamics, spectral specificity, functional retention, equivalence or acceptance claim is earned by pending experiments.

## 4. Locate and pin the research code

The handoff directory contains the archived data and analysis scripts, **not the complete training implementation**. In this GitHub delivery the surrounding UIOrthoLoRA repository contains that implementation. Use a dedicated server checkout of it; do not modify another agent's shared checkout. Do not assume a fresh pip-installed PEFT has the custom adapter. Historical paths relative to the research repository root:

- `src/peft/tuners/uiortholora/{layer.py,config.py,model.py}`: actual adapter, merge and parameterization.
- `notebooks/glue/training_new/training.py`: task/data/metric helpers.
- `notebooks/glue/training_new/newer_train/{train_ablation.py,mixing_reg.py,experiments.py,run_rte_ablation.py}`: legacy runner/diagnostics.
- `notebooks/glue/training_new/newer_train/results/`: original source of copied mixing summaries/layer logs.

Observed research Git HEAD on 2026-09-14: `bdbe00e67b05d589b78f129745fd56be2dd3a4ea`. This identifies the inspected checkout, **not a verified historical generating commit**. Relevant file hashes are in `GPU_SOURCE_FINGERPRINTS.sha256`. The shared checkout may have concurrent changes, so verify files and inspect differences rather than resetting it. Pin the chosen isolated source revision plus any local patch and dependency lock in every run manifest.

The delivery is based on published research revision `8cd4a061dad86073f4ef52997ca8a9412465b194`; the other project's 31 unpublished local commits were not included. All six inspected training-source fingerprints match this published revision. From the research root, verify with `sha256sum --check notebooks/iclr/handoff/GPU_SOURCE_FINGERPRINTS.sha256`. Do not try to check out the unpublished observed HEAD on the server; record the actual delivered research revision and any subsequent training changes instead.

Known hazards to fix/test in the isolated implementation before a pilot:

1. Legacy diagnostic API/task metadata mismatch; zero-coefficient conditions must still compute diagnostics, not placeholder zeros.
2. Legacy retention of only one checkpoint (`save_total_limit=1`) loses trajectories. Save the required checkpoint states persistently.
3. Legacy `device_map="auto"` must not place work on unassigned devices. Use the explicit allocated device mapping.
4. The local metric helper chooses MRPC F1; new common-protocol runs need both accuracy/F1 and an explicit checkpoint-selection metric.
5. Frozen leading coefficients are an **additive identity core**, not the original leading singular values or replacement of W_pre. Dropping the leading core changes the update family. Verify the resolved `drop_major`/scaler settings instead of trusting method names.
6. Model/adapter insertion, forward, `get_delta_weight`, merge/unmerge, and saved diagnostics must agree. Inspect quantization/dtype limitations; the first pilot should avoid adding unvalidated quantization.
7. Legacy global metric objects and scratch output names are unsafe for concurrent tasks in one process. Use isolated run state and collision-free paths.

## 5. Algebra and conventions that must survive the port

For a square matrix: `W_pre=U Sigma V^T`, cutoff k, r=d-k. `Delta_total=W_eff-W_pre`; `Delta_learned=W_eff-W_eff(0)`. Practical `Delta=E (U_L I V_L^T + U_T H V_T^T) D`; tail-only reference sets the leading core to zero. H is diagonal for k_vec=0 and rotated for the declared partial/full core.

`C=U^T Delta V`; cross=LT+TL, off-tail=LL+LT+TL. Square each matrix's norm ratio before averaging energy fractions. The old tables average module fractions then seeds; pooled energies are a different output. Preserve rectangular unmatched complements, ties and zero-update undefined values.

Legacy penalty: **sum** over 48 matrices of lambda_E ||U_L^T E U_T||_F² + lambda_D ||V_L^T D V_T||_F². Legacy labels A=no penalty; B=left-only; C=two-sided. New condition IDs are unambiguous in P1; never relabel legacy B as two-sided.

Reciprocal E/D scaling leaves Delta unchanged while changing raw factors. Centered-scaler and identity diagnostics, all factor norms and actual effective blocks are needed. The random-projector expected-penalty identity holds for a fixed scaler independent of the random projector, not for a jointly trained random-frame model. Uniform scalers remove cross mixing but need not preserve top-k ranking after a singular-value crossing.

## 6. First GPU session: ordered checklist

1. Read the docs above; inspect git status and the server's applicable instructions. Record machine, assigned GPU IDs/VRAM, quota, data/download permissions, scheduler and persistent output path. If any resource assignment is missing, ask the author before launch.
2. Create the dedicated source/environment/output namespace; record source fingerprints and `peft.__file__`/helper import paths. Do not edit the shared research checkout.
3. Run the small CPU checks from the paper repo:

   ```bash
   python3 scripts/build_mixing_tables.py --check
   python3 scripts/check_spectral_algebra.py
   python3 scripts/check_review_additions.py
   python3 scripts/audit_manuscript.py
   ```

   Run these commands from `notebooks/iclr/handoff/`. NumPy is needed for algebra; these commands do not launch model training. PyMuPDF is needed only for the optional PDF audit. This delivery supplies hash-verified `audit_baseline/` fixtures from the original Overleaf revision, so the historical-value audit works without that repository's Git history. Missing or altered fixtures fail the audit; they are not silently waived.

4. Implement the P0 fixes and P1 conditions in the **isolated** training copy. Create deterministic unit tests for exact penalties, offsets, zero cases, rectangular accounting and checkpoint reload. Do not assume the old runner supports the new conditions.
5. One short RTE smoke run: verify forward/delta/merge equivalence, gradients, metric fields, save/reload and device assignment. Retain it with status `smoke`, not `confirmation`. Measure setup, training, diagnostics and memory separately.
6. Timed calibration on seed 31415; include MIX/LEFT lambda sweep and nuisance-control norm overlap. Estimate the real cost of the matrix/diagnostic plan. Freeze the budget, manifest and confirmation grid before looking at confirmation outcomes.
7. Execute P1 plus early LoRA-8/full-FT comparisons: first tranche 66 confirmation runs if allocated; target five seeds for decisive comparisons after timing. This count excludes calibration, optional confirmed dose response and later replications. A reduced budget produces an explicitly smaller exploratory pilot, not a secretly incomplete confirmatory suite.
8. Compute P8's frozen-MLM-head behavior probe on the same saved checkpoints. Analyze norm matching before interpreting geometry. Preserve null/reversed findings and failed matches.
9. Update the run ledger and this handoff with commands, current source revision, completed run IDs, failed/retry IDs and persistent artifact paths. Only then proceed to the P2/P4/P5/P6 extensions agreed in the plan.

## 7. Artifact and resumption contract

Use `runs/<experiment_id>/<condition>/<task>/seed_<seed>/` under the assigned namespace. Every run saves:

- A manifest: stage (smoke/calibration/confirmation), ID, task/split fingerprints, model/tokenizer revisions, source Git hash and diff hash, dependency lock, actual imported paths, device/precision, all seeds (including projector/mask), complete hyperparameters, module list, exact effective update form/offset and penalty normalizations, checkpoint/selection/matching rules, timestamps and exit status.
- Pretrained reference and initial effective-state fingerprints, reproducible bases/projectors and tail selection; adapter and head states at all prescribed steps, optimizer/scheduler/RNG state for restart, selected/final checkpoints with hashes.
- Per-module and per-seed diagnostics including P3's signed/recoverable blocks, absolute/normalized energies, scalar means/spreads, identity alignment, bounds/gaps/drift, separate total/since-init deltas, task and P8 probe metrics; raw training and cost logs.
- All tested calibration configs, failures, exclusions, and matching decisions. A run is completed only after reload reproduces its metrics and its artifact is durable, not merely because the process exited.

Maintain a ledger with `run_id,stage,condition,task,seed,status,source_revision,manifest_path,checkpoint_path,diagnostics_path,started_utc,ended_utc,retry_of,notes`. Never overwrite an earlier run to retry it. Resume from verified state; do not regenerate missing history from final scalar summaries.

## 8. Paper synchronization

Keep new measurements separate from `data/legacy_mixing` and `data/legacy_layers`. Validate and review them before changing manuscript findings. Regenerate tables from immutable new data with explicit provenance, not by typing scores into TeX.

Overleaf author's direct edits always win: fetch/reconcile immediately before editing and again before pushing, inspect exact remote and changed files, and never force push. A GPU-only session can return an artifact bundle and patch without any Overleaf access. Do not transfer private author/review/context docs as anonymous submission material.
