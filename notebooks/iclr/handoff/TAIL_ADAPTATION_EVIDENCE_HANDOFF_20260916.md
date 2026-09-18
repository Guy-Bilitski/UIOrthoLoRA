# Tail-adaptation question: current writing and evidence handoff

Internal author/coding-agent document, 2026-09-16. This records the author's
latest instruction and supersedes the earlier post-v4 writing freeze and the
instruction to defer all remaining strict-band work. It does not change sealed
registrations or authorize resources beyond the server's existing allocation.

## The paper's question

Can a model adapt while its backbone updates stay entirely in the lower spectral
tail? Does allowing directions inside that tail to combine change task loss,
accuracy or convergence? What changes when interaction with leading directions
is allowed and then penalized?

Distinguish within-tail rotations from leading–tail interaction throughout.
The first preserves confinement. The second relaxes it. The practical
UNREG/MIX/NORM adapter also has an additive leading core; it is not a strictly
confined tail experiment, even with no rotations or zero mixing operators.

## Writing completed in this refactor

- Title: **Adapting in the Spectral Tail**.
- The abstract/introduction begin with SVD-based adaptation, the tail question,
  and the gap between selecting a region and choosing freedom inside that region.
- Adapters come before diagnostics: coefficients → rotations within the tail →
  ambient scaling and controlled leading–tail interaction.
- A dedicated measurement section explains why each observable is needed.
  Task accuracy/loss assess usefulness; actual update blocks verify confinement
  and interaction; update magnitude and module profiles test alternative accounts.
- The main completed-results table puts held-aside accuracy and loss beside
  geometry. The detailed inner-selection confirmation table moves to the appendix.
- Full interaction bounds/proofs remain in the appendix. All reported data,
  historical GLUE means/SDs, checkpoint joins and algebraic qualifications remain.
- Strict-tail evidence is labeled incomplete. No tail superiority, head-relative
  gain, rotation benefit or convergence claim is invented from the available rows.

This is ready for the author's assessment of the story, not a declaration that
the newly central empirical question has been answered.

## Evidence actually available at the refactor

Latest synchronized input head was `e18b950`; no new server outcomes arrived in
the fetch. The strict-band outcome export has five rows: three seed-42 diagonal
confirmations (leading, middle, tail) and two seed-31415 tail timing pilots
(diagonal/rotation). The fixed tail confirmation score is 70.2811% on RTE's
inner-selection split. A trained head is present, so this is not its incremental
contribution. The later freeze ledger has eight endpoints, with no new outcome
values. Head-only/rotated confirmation outcomes and complete curves are absent.
The completed practical study has all 18 fixed held-aside evaluations.

## First action for the coding agent: refresh, then complete the core contrast

Reconcile the live controller and existing artifacts before launching anything.
These local counts are dated exports, not current job status. Preserve other
projects' jobs, allocated device IDs, disk budgets and this project's invalidation
records. This paper session did not access the server, change a scheduler or stop
or start a training process.

The central RTE block is the following **nine registered confirmations**, each
on seeds 17, 42 and 123:

| Condition | Purpose |
|---|---|
| P1_HEAD_BASE | Original backbone frozen, common task head trained |
| BAND_TAIL_DIAG | Tail coefficients trained, all backbone changes confined |
| BAND_TAIL_ROT64 | Same 256-direction tail, rotations in its last 64 directions |

Reuse validated matching runs; do not duplicate them because the narrative changed.
Prioritize missing jobs from these existing entries over newly proposed CENTER
training. Preserve all original leading/middle arms and report their status and
outcomes; focusing execution on the tail does not erase the 21-entry registration.
Leading/middle comparisons are necessary before claiming the tail is preferable
to other equal-width regions. A three-arm MRPC replication would add nine runs;
record its protocol and budget before execution. Do not silently expand tasks,
seeds or resources to meet a narrative.

Keep the repaired input preparation, common initial head and batch order, shared
optimizer-step budget, fixed endpoint, zero initial adapter delta and original
backbone reference. Count both adapter and head parameters. Rotations change
parameter count, so this is a comparison of freedom at fixed support, not equal
parameter efficiency. Validate effective-delta construction, reload/metric
reproduction and structural off-band zero. Check that rotations receive gradients
after the zero-core insertion state and that their trained core is allowed to
develop off-diagonal entries. Report a genuine no-rotation-use outcome if that is
what training produces; do not select a favorable checkpoint to force the effect.

## Export needed to answer the questions

Refresh `data/final_evidence_20260916/band_results.csv` and `band_coverage.json`
using the established exporter, preserving all pilots, confirmations, failures
and retry chains. Return manifests and checkpoint/validation/source hashes for
every new recipe and endpoint. Keep fixed and best checkpoints separate.

Also export the recorded trajectories in a separate labeled CSV/JSONL:

    task, condition, seed, purpose, run_id, optimizer_step,
    cumulative_train_examples, split, metric_name, metric_value,
    averaging_window_start_step, averaging_window_end_step,
    checkpoint_sha256 (when available), source_revision, validation_sha256

Include task training loss separately from regularizer/total objective,
inner-selection cross-entropy, accuracy and MRPC F1 where relevant. Preserve
actual evaluation steps and averaging windows; do not manufacture common steps
by interpolation. State missing steps explicitly. Export initial/fixed checkpoint
behavior, pooled update size, exact within-band off-diagonal energy, off-band
validation, inventories and learning-health. Existing shared steps are preferable
to rerunning expensive checkpoint diagnostics just to make denser plots.

Freeze all new evaluation checkpoints before any further official-split scoring.
Use cached outcomes where already evaluated and retain their provenance. Any new
held-aside block is a later labeled evaluation, not part of the earlier immutable
18-checkpoint request. Current public comparisons use inner-selection outcomes
for band runs and separately labeled held-aside outcomes for practical runs.

## Presentation fixed before the missing outcomes

Main comparison order: head-only, tail diagonal, tail rotation. Per task, show
every seed and mean/sample SD for fixed-endpoint accuracy and task NLL. Report
paired differences: diagonal minus head, rotation minus head, rotation minus
diagonal. Use three-seed intervals as descriptive uncertainty, not equivalence.
Keep pilots separate and all registered location arms in the complete supplement.

For dynamics, use aligned small panels for task training loss, inner-selection
loss and accuracy versus actual optimizer step. Use the same axis ranges across
the three conditions within a task, thin per-seed lines and an aggregate only at
shared observations. Show full recorded trajectories; do not choose a favorable
segment or loss threshold after seeing them. A lower endpoint loss is not proof
of faster convergence. Preserve reversals and null findings.

For geometry, show within-tail off-diagonal energy and update size for those same
checkpoints. Off-band zero verifies the construction; it is not a performance
result. The exact meaning of a zero-delta fraction is undefined, not zero.

## Supporting work and limits

The existing temperature-scaling export/analyzer remains useful and inference-only;
it may proceed without delaying the core tail comparison. Fit temperatures only
on inner-selection logits under the already documented rule. CENTER and random
projectors concern the mechanism of the practical mixing penalty; they are now
secondary to the author's strict-tail question. Module-norm-matched training stays
deferred. Recover an existing LoRA geometry example if inexpensive, keeping its
historical protocol separate.

Even a successful tail-only run does not identify which weight directions store
knowledge. A gain over head-only supports an incremental task contribution in
the tested setting. A rotation gain supports the usefulness of its additional
freedom, conditional on the configuration/budget. Neither proves that all tasks
can adapt in the tail, that leading directions are universally unnecessary, or
that pretrained behavior is preserved.

## Validation and reviewer artifact

`scripts/build_tail_story_tables.py --check` reproduces the new main table and
the entire available band snapshot from the unchanged exports, after the existing
evidence audits. Its `--write` mode refreshes those regions when a validated
bundle lands. All other source/algebra/data checks remain required. The refreshed
analysis archive must contain this manuscript version; the prior v4 ZIP is a
different, archived delivery. Anonymous public hosting remains unresolved.

The SVD-method framing was checked against the primary MiLoRA, PiSSA, SVFT and
Spectral Adapter papers. Their existing bibliography entries are reused. In
particular, free factor initialization in a selected span does not enforce that
span throughout training; this is an algebraic distinction, not a new empirical
evaluation of those methods.
