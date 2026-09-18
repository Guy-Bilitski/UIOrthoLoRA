# Band × flexibility presentation specification

Recorded 2026-09-16 05:27 UTC, against Overleaf manuscript/data revision `8729fe3`.
This is an internal analysis specification, not anonymous submission material.

## Information already available

The author disclosed one tail-diagonal RTE pilot (accuracy 0.703; reported off-band
energy 0.00%). No BAND confirmation outcomes were inspected to choose this
presentation. This is a prospective presentation specification for incoming
confirmation results, not a claim that the study or these choices preceded every
pilot. The execution registration, manifests and validation ledger govern the
actual experiment; this document does not change running jobs or tuning rules.

## Fixed population and endpoints

- Separate tasks: RTE first, MRPC replication if completed; never pool them or
  combine their scores with the legacy GLUE table or the practical MIX study.
- Three paired seeds: 17, 42, 123. Six arms per task: leading/middle/tail ×
  diagonal/partial rotation, plus three original-backbone head-only references.
- Bands are `[0:256)`, `[256:512)`, `[512:768)` in the pretrained descending SVD.
  Rotation acts only in the last 64 coordinates of the selected band.
- Primary endpoint: fixed-budget inner-selection accuracy, in percentage points,
  on both tasks. Report MRPC F1 alongside accuracy, explicitly labeled.
  Validation-selected checkpoints are secondary, in a separate appendix table;
  never choose the better endpoint for a condition. Preserve untouched official
  validation/test status and the registered checkpoint-selection metric.
- Include every registered seed, failure, retry and learning-health flag.
  A failed learning gate is not grounds for dropping a completed run's score.
  Only reload-validated, manifest-compatible runs enter numerical summaries.

## Main table

Use task panels and the same fixed row order in each: original-backbone head-only;
leading diagonal; leading rotation; middle diagonal; middle rotation; tail
diagonal; tail rotation. Do not sort by observed score or bold a winner.

Columns: band; flexibility; validated n/3; adapter trainable entries (task head
reported separately); fixed-step accuracy mean ± sample SD; MRPC F1 mean ± SD
where applicable; paired accuracy difference from the same-seed head-only
reference; pooled relative update norm. Counts come from validated manifests,
not inferred labels. Distinguish stored trainable entries and effective degrees
of freedom. No capacity-matched interpretation of adding rotations.

Keep all individual seed values in an appendix. A partial cell shows available
seed points and n/3, explicitly exploratory; do not present a complete-block
claim until all 21 runs for that task validate. A one-seed cell has no SD or CI.
Display missing values as pending/failed, never zero. No imputation or replacing
a failed seed with a favorable pilot.

## Figure

Fixed axes: x = leading, middle, tail in that order; separate symbols for diagonal
and partial rotation; one task panel each. Show all three seed points with stable
seed identifiers, slightly offset; connect a seed's two flexibility settings
within its band. Overlay mean ± sample SD. Show same-seed head-only references
and their mean in neutral gray. Main y-axis is fixed-step accuracy (%), using the
same scale across task panels; a separate MRPC F1 panel is secondary.

An adjacent contrast panel shows rotation minus diagonal accuracy, by band, with
all three paired differences and the mean. Do not join different tasks or infer
independent replication from modules/checkpoints. Do not condition which bands
are displayed on a favorable outcome. If only RTE finishes, label RTE explicitly.

## Contrasts and uncertainty

Report all, without choosing a contrast after viewing outcomes:

1. Each of the six arms minus the same-seed head-only reference.
2. Rotation minus diagonal within each of the three bands.
3. All three pairwise location differences (middle−leading, tail−leading,
   tail−middle) separately within each flexibility setting.
4. Differences of rotation gains for those same three location pairs, to describe
   the location × flexibility interaction.

Primary summaries are individual seed differences and mean ± sample SD.
Supplementary nominal 95% paired Student-t intervals use n=3, df=2, computed
from the paired differences. They are conditional on independent, approximately
normal seed differences, unadjusted for multiple contrasts, and descriptive;
no significance stars, confirmatory superiority, or equivalence conclusion.
Do not bootstrap modules or checkpoints as replicates. Any inferential testing
or multiplicity adjustment would require a separately declared analysis.

## Geometry and validation appendix

Per seed: achieved pooled norm; equal-module mean norm; within-band off-diagonal
energy; off-band absolute and fractional energy; total update energy; head-only
comparison; fixed and selected endpoint metrics; learning-health/validation status.
For the exact zero update, fractional energies are undefined, including at
initialization and in the frozen backbone reference. Use the implementation's
registered numerical tolerance for off-band checks and report values with enough
precision to assess it; printed 0.00% alone does not establish exact zero.

Show a 3 × 2 grid of module/layer geometry summaries in fixed band/flexibility
order if the export supports them. Pooled norm is descriptive, not evidence of
matched magnitude. Off-band confinement is an implementation check; it cannot
be the empirical novelty. The frozen MLM probe remains secondary and cannot
alone establish pretrained knowledge retention.

## Freeze and amendments

Record source/data/validation hashes and complete/partial status at each paper
refresh. Freeze evidence for the abstract at the author's stated cutoff; pending
band results remain absent from abstract findings. Log any later presentation
amendment with its reason and which outcomes were already seen. Do not stop,
expand, reorder or omit a block because its emerging scores are inconvenient.
