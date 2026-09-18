# Final export closure — 2026-09-17

For the author to forward to the coding agent. The paper has integrated the
complete 21/21 RTE band/head block from Overleaf `a88118e`. No new training is
requested here. Preserve all frozen checkpoints, invalidation records and
original evaluation exports. Check for existing results before running inference.

## 1. Complete the frozen band block's held-aside evaluation

The latest research handoff says this evaluation has not run. Evaluate all 21
fixed checkpoints in `data/locked_evaluation_20260916/block_b_freeze.jsonl`,
including every leading/middle/tail and head-only condition, in one labeled
evaluation block. Use the registered RTE held-aside split and fixed 5,670-step
endpoints. The checkpoint population is already frozen. Do not retune, choose
checkpoints, omit seeds or change the population based on the resulting scores.

Return a separate `band_held_aside_20260917/` export containing:

- CSV: task, condition, seed, run ID, example count, accuracy, task NLL,
  checkpoint SHA256, validation SHA256 and evaluation source revision.
- Per-example IDs, labels, predictions and losses; logits if available. No raw text.
- Evaluation manifest binding split/ID hashes, the 21-entry freeze ledger,
  checkpoint/loading checks and output hashes.

Keep the original 18-run practical held-aside export immutable. Band and
practical results remain separate protocol blocks even when they use the same
official task split. Report explicitly if any checkpoint was previously scored.

## 2. Close the prediction-loss calibration question

For all 18 frozen practical UNREG/MIX/NORM confirmations, provide inner-selection
and held-aside class logits with example IDs, labels and checkpoint hashes, or
the complete temperature-scaling export required by the existing analyzer.
Use cached logits/results where available. The frozen fitting rule is in
`EVIDENCE_PRIORITY_HANDOFF_20260916.md` and
`scripts/analyze_temperature_scaling.py`: fit one positive temperature per run
on inner-selection examples only; assess original and calibrated held-aside NLL
and fixed-bin calibration error. Never fit temperature on held-aside examples.
Preserve all runs, unchanged argmax accuracy and boundary-fit flags.

Either outcome is usable. The current paper reports the uncalibrated loss
finding and leaves this question open; it does not claim calibrated superiority.

## 3. Finish the evidence package from existing files

Export per-run manifests and original reload-validation reports for the 21 band
confirmations and 18 practical confirmations. The current bundle has 13
representative recipe manifests, not all 39 per-run manifests/reports. We have
recovered and verified the band experiment source from Git, so that code need
not be reconstructed. Include the sealed protocol/registration documents if
they can be distributed anonymously.

Export existing training and inner-selection metric trajectories under the
schema in `TAIL_ADAPTATION_EVIDENCE_HANDOFF_20260916.md`, retaining actual steps,
averaging windows, split labels and task/regularization losses separately. Do
not rerun training or invent/interpolate observations to fill gaps. If curves
were not recorded, state that explicitly; the paper makes endpoint claims only.

## Integration contract

Publish scoped exports to the existing authorized research branch and the same
Overleaf project, then return the commit IDs and exact export paths. Preserve
raw outcomes and fixed/best endpoint distinctions. No CENTER, MRPC replication,
LoRA or other new training is required to answer this request. Those experiments
are absent from this paper's measured claims.
