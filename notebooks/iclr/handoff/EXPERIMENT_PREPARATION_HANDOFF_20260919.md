> **Current decoder plan — reviewed 19 September:** Read
> `DECODER_SUBSPACE_STUDY_20260919.md` and `CODING_AGENT_PROMPT_20260919.md`.
> Test leading/middle/tail support with our strict DIAG/ROT128 adapters; no
> LoRA/PiSSA comparison or new penalty sweep. The plan includes bounded balanced
> tuning, a two-GPU timing budget and a predeclared DIAG-only fallback. Older
> decoder commands, matrices and costs below are historical and superseded.

> **Latest author priority — modern decoder first:** Read
> `DECODER_FIRST_PRIORITY_20260919.md`. Prioritize the prepared decoder pilot,
> tuning/calibration and confirmation runs on the two assigned GPUs. Defer
> CENTER and further encoder controls. The coding agent has now delivered the
> temperature and band held-aside exports; verify/reuse them, do not rerun them.
> Older schedules below are superseded where they place controls first.

# ICLR experiment preparation — start here, 19 September 2026

Internal author/coding-agent handoff for Overleaf project **6aa54397e58b10444b0fa2aa**.
The author asked for this handoff after reviewing the ICLR system's LLM feedback.
He is starting a separate coding agent to **prepare the experiments to run**.
He reconfirmed a **two-GPU budget**. The historical machine had RTX 3090 24 GB
devices; inspect the current assignment rather than assume device IDs.

This file is the current entry point for preparation and supersedes older queue,
deadline and completion summaries. Existing registrations, data, invalidation
records and resource-isolation requirements remain binding. In particular, do
not follow old instructions to restart the already completed band block or use
four GPUs. A candidate experiment in this handoff is not a completed result or
authorization for an unattended training campaign.

## 1. Repositories and authoritative sources

- Research: `git@github.com:Guy-Bilitski/UIOrthoLoRA.git`, existing branch
  **`ortho_new`**. Use an isolated checkout and scoped commits on top of the
  latest remote. Push to the existing branch; do not force-push or include
  another project's uncommitted/unpublished work.
- Current research handoff: `notebooks/iclr/handoff/`. This file and the latest
  paper/analysis snapshot are mirrored there. `SNAPSHOT_PROVENANCE.md` records
  which Overleaf revision supplied that snapshot.
- Overleaf: `https://www.overleaf.com/project/6aa54397e58b10444b0fa2aa`;
  Git host/path `git.overleaf.com/6aa54397e58b10444b0fa2aa`, branch `main`.
  MCP project name is `default` on the **overleaf_iclr** connector. Verify the
  project ID; another Overleaf connector serves another project.
- Canonical manuscript: Overleaf `neurips_2026.tex` (legacy filename, ICLR 2027
  style). Current title: **How Pretrained Spectral Subspaces and Their
  Interactions Affect Fine-Tuning**. Scientific text was last changed in
  Overleaf `932283e1` before this handoff; later commits can be packaging/docs.
- Read `review_feedback/20260919/assessment.md` for the complete scientific
  assessment, caveats and preferred sequence. The new calibration diagnostic
  is in that same directory. It has **not** been inserted into the manuscript.
- The latest checked PDF and anonymous analysis ZIP are in
  `review_feedback/20260919/paper.pdf` and `supplement/analysis_artifact.zip`.
  The handoffs/assessment are private working documents, not submission material.

Read `AGENTS.md`, this file, the assessment, then
`FINAL_EXPORT_CLOSURE_REQUEST_20260917.md` and
`EVIDENCE_PRIORITY_HANDOFF_20260916.md`. The latter's logit schema, temperature
rule and CENTER design remain useful; its historical queue/writing status is
superseded here. Read `GPU_HANDOFF.md`, `EXPERIMENTS_REQUIRED.md`,
`REVIEW_PANEL_RESPONSE.md` and `MANUSCRIPT_AUDIT.md` for isolation, definitions
and historical hazards, treating their old status paragraphs as history.

## 2. Scope of the new coding session

Prepare concrete, tested code, protocols, export commands and a bounded schedule.
Do not stop at a high-level proposal. Inspect current code and server artifacts
first, reuse finished work, implement missing preparation in isolation, and run
the relevant CPU tests. Existing frozen-checkpoint inference was already
requested in the September 17 closure document; reconcile the live ledger and
reuse cached results before carrying it out. Do not duplicate completed scoring.

For **new training**, prioritize the prepared modern decoder pilot and its
confirmation study; defer CENTER and additional encoder controls. The author explicitly requested discussion of larger changes; the
September 19 assessment is a recommendation, not approval of all proposed runs.
Return exact pilot/campaign commands, selected design choices, validation and a
two-GPU cost estimate before committing to the new training campaign. Use any
subsequent explicit author authorization without requesting it again.

Do not rewrite the approved abstract/introduction or integrate new findings into
the paper during experiment preparation. Keep the writing agent informed through
committed status/exports and the author. No conference submission or external
messaging is requested.

## 3. What is actually complete

| Evidence block | Population | Current interpretation |
|---|---|---|
| Practical interaction study | RTE/MRPC × UNREG/MIX/NORM × seeds 17,42,123 = 18 fixed confirmations, plus separate calibration | Held-aside evaluation complete, 277 RTE / 408 MRPC examples. Preserve the existing export. |
| Strict-band study | RTE × leading/middle/tail × DIAG/ROT64 × three seeds, plus three HEAD_BASE = 21 confirmations | All completed, reloaded and frozen. All results are integrated in the paper. Two timing pilots remain separate. Both inner-selection and held-aside scores are now exported; keep the two splits separate and retain the prior-scoring disclosure. |
| Legacy mixing archive | 27 runs, nine paired task/seed groups, 48 modules/run | Separate historical protocol; 1,296 module observations are not independent training replications. |
| Broad GLUE/GPT-2 tables | Historical reports with imported baselines | Unmatched tuning/provenance limitations. No modern matched decoder comparison yet. |

Check for newer server work before declaring any item absent. The last published
ICLR-specific research handoff before this delivery was `0fb17b20`, following
complete band export `dabd48f4`; the overall branch also contains newer unrelated
project commits. Do not reset the branch to those old hashes.

The 39 tokenizer-fault campaign runs remain invalid. Use the repaired preparation
and its parity/learning gates. Do not reuse invalid dose estimates or erase their
quarantine/invalidation records. The old four-GPU authorization is not today's
resource allocation.

The central finding is lower raw held-out task NLL for MIX versus the tested size
penalty, with small average accuracy differences and much lower cross energy.
Update allocation also changes substantially; size matches are approximate.
This does not identify cross share as the sole cause, establish accuracy
equivalence, or prove preserved pretrained behavior. The masked-token probe
worsens under all arms and has the highest mean loss under MIX.

## 4. Find the implementation and saved checkpoints

Current research implementation: `notebooks/iclr/campaign/`. Inspect especially
`spectral.py`, `regularizers.py`, `modeling.py`, `engine.py`, `diagnostics.py`,
`preparation.py`, `checkpoints.py`, `run_validation.py`, `focused_plan.py`,
`band_plan.py`, `allocation.py`, `supervision.py` and `tests/`.

Recorded source snapshot in the paper: `experiment_source/`, with 46 verified
campaign Python files and `requirements.lock.txt`; see its README and
`data/band_source_audit.json`. It records executed code and is not a place to
silently replace history with new implementations. New code requires a new
source revision and preflight; old preflight tokens do not validate changed code.

Research operator scripts: `notebooks/iclr/handoff/ops/`, notably
`locked_evaluation.py`, `block_b_freeze.py`, `final_evidence_export.py` and the
launch/controller scripts. Inspect hard-coded paths and output-exists guards
before reuse; the old evaluator must not overwrite completed exports.

Last recorded server checkout:
`/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914`.
Last Overleaf checkout there: `.../notebooks/overleaf-6aa54397`.
Last output root: `campaign_outputs_confirmation_v1/`; ledgers and protocols
live inside it. These are discovery hints, not a claim that the paths or old
processes are still live. `SESSION_HANDOFF_20260916.md` contains historical server
details, named tmux socket and controller conventions. Inspect actual leases,
processes, storage and exact assigned GPU IDs before any GPU job. Do not touch
another project's processes or shared PEFT implementation.

Checkpoint populations are explicit, not inferred from filenames:

- Practical: `data/locked_evaluation_request_20260916.json` (18 fixed endpoints).
- Band/head: `data/locked_evaluation_20260916/block_b_freeze.jsonl` (21 entries).
- Supporting original outcomes, manifests and diagnostics:
  `data/final_evidence_20260916/`, `data/focused_norm/`,
  `data/locked_evaluation_20260916/`.

These paths are relative to the Overleaf root or the research handoff root.
The paper bundle contains measurements/source and checkpoint hashes, not weights.
The original bundle had 13 representative recipe manifests; the coding agent
has now exported all 39 run records and validation reports under
`data/run_records_20260919/`.

## 5. Completed frozen-checkpoint exports: verify and reuse

The coding agent has completed this request; see
`EXPERIMENT_PREPARATION_STATUS_20260919.md` and its linked exports. The contract
below documents the required checks; reuse the completed results rather than
repeat the inference described in `FINAL_EXPORT_CLOSURE_REQUEST_20260917.md`.

**Temperature scaling:** export both inner-selection and held-aside two-class
logits for all 18 practical endpoints. Follow the hash-bound schema in
`EVIDENCE_PRIORITY_HANDOFF_20260916.md`, including source-row identities and
original input/checkpoint/validation hashes. Reproduce original predictions,
accuracy/F1 and per-example losses. Fit one positive T per run on inner-selection
NLL only, log(T) in [-6,6]. Report raw/scaled held-out NLL, accuracy and 15-bin ECE,
all seeds, paired contrasts and boundary flags. Never fit or choose a temperature
on held-aside labels. The analyzer already exists:

```bash
python3 scripts/check_temperature_scaling.py
python3 scripts/analyze_temperature_scaling.py --bundle data/temperature_logits_20260916 --out data/temperature_analysis_20260916.json
```

**Band/head held-aside scoring:** evaluate all 21 registered fixed endpoints,
including every leading/middle/tail and head-only condition, as one complete
block. Preserve inner-selection outcomes and the earlier practical export.
Return per-example IDs, labels, logits, predictions/losses, aggregate scores,
checkpoint hashes and a manifest with split/source/loading checks. If any were
already scored, recover and disclose that fact. No retuning, seed omission or
checkpoint substitution based on outcomes.

Recover recorded training/selection trajectories and all individual manifests
from existing files. Preserve actual steps, fixed versus selected endpoints,
training task loss versus regularizer/total loss, and evaluation split labels.
Do not interpolate missing measurements or rerun training to fabricate history.

**Existing diagnostic:** `review_feedback/20260919/calibration_diagnostic.json`
contains raw ECE/Brier reconstructed from binary per-example CE, with source
hashes. MIX has lower ECE/Brier than NORM in all six task–seed pairs. This is
descriptive, post hoc, float32-limited evidence; it is not temperature scaling.
The portable reproducer reads only saved evidence:

```bash
python3 review_feedback/20260919/calibration_diagnostic.py --paper-root . --out /tmp/iclr-calibration-reproduction.json
```

Choose a fresh output path if that file exists. The archived evaluator text is
included only to verify its hash/loss convention; do not execute that text file.

## 6. CENTER: prepared and deferred behind decoder results

Reuse the practical family, leading core I, tail size 256, no rotations, same
48 attention matrices, task/head/data-order seeds and repaired-input recipe.
Do not constrain/reset scalers or drop the leading core. The existing code
supports `P1_CENTER`; check its implementation, plumbing and tests before adding
another regularizer implementation.

The proposed penalty is:

    gamma * sum_modules c * (||e-mean(e)||^2 + ||d-mean(d)||^2)
    c = k*(d_width-k)/((d_width-1)*(d_width+2)), d_width=768, k=512

The coefficient multiplies a **sum over modules**. This is the Haar-expected
projector penalty for fixed scalers, not a theorem about trained endpoints.
Test constant-scaler zero, translation invariance, gradients and normalization.

The complete prospective calibration rule is already in
`EVIDENCE_PRIORITY_HANDOFF_20260916.md`: seed 31415, existing MIX targets from
`data/focused_norm/selection_final.json`, initial gamma grid {1e-4,1e-3,1e-2},
up to four norms-only refinement doses per task and a 5% target tolerance.
Confirmations are RTE/MRPC × seeds 17,42,123. This means **6 confirmations plus
6–14 calibration endpoints**, not six total jobs. Freeze the rule and doses
before confirmation; preserve failures and failed matches. Prepare exact costs
and commands for discussion rather than silently expanding the queue.

Export raw/centered scaler statistics, total/initial/learned block energies,
module energy profiles, update norms, identity residuals, task/probe outcomes,
logits and reproducible checkpoint inventories. CENTER does not itself match
module allocation or establish pretrained-frame specificity.

## 7. First new GPU priority: the prepared compact decoder study

Preferred candidate for discussion: **Qwen2.5-1.5B-Instruct on GSM8K**. Neither
this model/task choice nor the full run matrix is frozen. Verify local models,
dependency compatibility, tokenization/chat format, meaningful learning signal,
memory, training/decoding time and scoring before choosing the final recipe.
Reference: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct .

Proposed confirmation matrix: UNREG, MIX, NORM, LoRA and PiSSA × three seeds =
15 confirmations, plus pilot/tuning/calibration. Use one shared model revision,
dataset/splits, adapted-module set, prompts, target masking, training budget and
evaluation script. Allow method-appropriate learning rates within a comparable
declared tuning budget. Save raw and calibrated probability measures where
well-defined, generated task success, and actual geometry. Binary-classification
temperature code is not automatically a generation evaluator.

The runner is RoBERTa-specific; this is not a model-name substitution. Square
query/output projections are a possible limited first scope shared by all arms.
Grouped-query key/value projections require explicit rectangular handling.
Pin the decision and cutoff/tail-fraction rules before confirmation. Keep proper
weight orientation, full complements and zero-update undefined fractions.
Validate initial effective weights, forward/delta/merge agreement, gradients,
save/reload and actual trainable/frozen inventories for every arm. Avoid adding
unvalidated quantization at the same time.

PiSSA initializes free factors from principal components; it does not maintain
strict confinement to the original leading span. Measure its actual effective
update relative to the same original pretrained weights, accounting for its
frozen residual and initial factor product. Do not label it a fixed-leading-core
control. A modern interaction experiment would extend the interaction finding;
it would not automatically replicate the separate band-location finding.

Measure SVD setup, reused decomposition storage, steady-state training throughput,
peak allocated/reserved memory, diagnostics and generation evaluation separately.
Cache decompositions where valid, preserving identical reference bases. Check
task feasibility without choosing the benchmark based on favorable MIX outcomes.
If the candidate decoder recipe is too costly, propose a narrower informative
decoder study first. Do not substitute additional RoBERTa controls for the
author's requested modern-model evidence without discussion.

Module-allocation-matched training, extra dose sensitivity, DoRA/OFT, larger
models, multiple generative benchmarks and further band replications are
conditional extensions, not the immediate queue. Reserve the last 48 hours of
the remaining week for analysis and writing; establish actual dates with the
author only if not available in the current session.

## 8. Historical GPT-2 provenance and publishing

Inspect `notebooks/E2E/` read-only for source scores/manifests. The current GPT-2
table's two UI rows/uncertainties cannot yet be linked to complete run populations.
One UIOrthoLoRA central row closely reproduces the rounded single-run scores in
`outputs/results/lr_0.05_svalues_256_svectors_30_seed_17_init_sigma_0.1_init_scaler_0.1/scores.txt`.
This is a provenance question, not proof that the reported row is wrong. Recover
run IDs, aggregation, counts, recipe, decoding/scorer revisions if possible; do
not fill gaps from unrelated defaults, overwrite historical scores or launch
replacement experiments without a separate decision.

Before delivery, return an inventory of recovered/missing artifacts, relevant CPU
test outcomes, implemented code paths, exact reproducible commands, unresolved
scientific choices, and a two-GPU schedule with calibration/validation/evaluation
costs included. Publish scoped code/docs and immutable exports to `ortho_new`,
mirror agreed evidence/handoff updates to the same Overleaf project, and report
both commit IDs. Keep original records and their hashes; use new output paths.
No run is complete until checkpoint reload reproduces its metrics. No successful
compile or synthetic test is evidence that a model experiment succeeded.

For paper/data updates, run the checks required by `AGENTS.md`; for the current
paper these include mixing tables, spectral algebra, review additions, manuscript
audit and, after TeX changes, the PDF audit. The clean manuscript is 38 pages
total with nine main pages. The available anonymous supplement reproduces the
analysis, not training or independent checkpoint validation.
