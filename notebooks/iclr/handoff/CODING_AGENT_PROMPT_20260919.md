> **Latest author priority — modern decoder first:** Read
> `DECODER_FIRST_PRIORITY_20260919.md`. Prioritize the prepared decoder pilot,
> tuning/calibration and confirmation runs on the two assigned GPUs. Defer
> CENTER and further encoder controls. The coding agent has now delivered the
> temperature and band held-aside exports; verify/reuse them, do not rerun them.
> Older schedules below are superseded where they place controls first.

You are preparing the next experiments for our ICLR paper, "How Pretrained
Spectral Subspaces and Their Interactions Affect Fine-Tuning."

Research repository: https://github.com/Guy-Bilitski/UIOrthoLoRA, branch
`ortho_new`. Overleaf project: `6aa54397e58b10444b0fa2aa` (the `overleaf_iclr`
connector's `default` project). Fetch the latest versions of both.

Start with `notebooks/iclr/handoff/EXPERIMENT_PREPARATION_HANDOFF_20260919.md`
in the research repo, or the same filename at the Overleaf root. It is the
entry point; read DECODER_FIRST_PRIORITY_20260919.md first for the latest
author priority, which puts modern decoder results ahead of new controls. Read the local
AGENTS.md and `review_feedback/20260919/assessment.md`, then follow the handoff's
source map and existing protocol/export contracts.

We have completed 18 practical UNREG/MIX/NORM confirmations and all 21 RTE
band/head confirmations. Do not retrain them. The coding agent has now delivered
temperature scaling and held-aside evaluation of the complete band/head block;
see EXPERIMENT_PREPARATION_STATUS_20260919.md and reuse those exports. The new raw ECE/Brier diagnostic is
available in `review_feedback/20260919/`; it is not temperature scaling.

I confirmed a two-GPU budget. Use an isolated checkout/environment/output root,
preserve other projects and their jobs, and verify the actual assigned device
IDs. Existing resource, invalidation, registration and checkpoint-validation
rules still apply. Old four-GPU instructions and unfinished-band queues are stale.

Your task is to prepare concrete, tested implementation and executable protocols,
not just write a plan. Reuse the repaired `notebooks/iclr/campaign/` runner and
prioritize the prepared compact decoder pilot, then its tuning/calibration
and confirmation study. Defer CENTER and additional encoder controls. The
prepared candidate is Qwen2.5-1.5B-Instruct/GSM8K,
with UNREG/MIX/NORM/LoRA/PiSSA as the proposed comparison. Model/task and full
training scope are not yet approved. Implement and run the relevant CPU checks;
do not launch the proposed training campaign before we agree its scope and cost.

Return what already exists, what you implemented, test results, exact commands,
remaining design decisions and a realistic two-GPU schedule including tuning,
calibration, checkpoint validation and evaluation. Reserve the final two days
for analysis/writing. Push scoped preparation updates to `ortho_new` and mirror
agreed handoff/evidence updates to this Overleaf project; report both commit IDs.

Preserve immutable evidence, all seeds and null/reversed outcomes. Fit calibration
only on inner-selection examples. Do not conflate protocols, rewrite the approved
abstract/introduction, or turn unrun experiments into manuscript claims.
