# Author-directed submission scope reduction — 2026-09-14

This records newer author instructions, not experimental findings. The author
rejected the proposed 300 GiB allocation and the 66/75-run submission package,
asked for the thinnest useful extension of existing Table 2 with additional
seeds and controls, and emphasized that this is a shared server. The abstract
working deadline is September 18 at 13:00, with the full paper one week later.
Use Asia/Jerusalem for those author working deadlines unless corrected.

These newer instructions take precedence over scheduling the full expanded
P0–P8 campaign. Preserve EXPERIMENTS_REQUIRED.md as the scientific design and
control reference; its full run matrix is not the current authorized launch
target. Do not mark its deferred experiments complete or quietly treat the
original campaign completion criteria as satisfied.

## Current execution limits

- Keep the existing 50 GiB campaign output allowance. The author did not approve
  200/300 GiB. Do not reserve that additional shared-server space.
- Do not launch the broad 66/75-run matrix, new backbones, full-FT campaign,
  method benchmark or factorial sweeps under the old plan by default.
- Preserve original experiment settings when extending an existing result.
  The proposed 1,024-step replacement was not registered or run and is not an
  authorization to append a different protocol to historical table statistics.
- Reuse verifiable existing results; target only necessary additional paired
  seeds and claim-relevant controls after identifying the table and provenance.
- Keep source, failed artifacts and existing checkpoints immutable. Retained
  campaign output is approximately 7 GiB; nothing was deleted to reduce scope.
- GPUs 2/3 remain the only assigned devices. No other user's jobs are in scope.

## Live Overleaf reconciliation and immediate execution

On the author's further instruction to use the detailed design rather than wait
for a caption, the exact Overleaf project was synchronized read-only through
its project-scoped connector. Live revision:
`f2e82eeed32985eb971cfa983281505091022aa9`. Its `EXPERIMENTS_REQUIRED.md` is
byte-identical to the research handoff (SHA256
`b146f1571166f349f2ea2588f309834b4cb22eb4d2925bc5d1ea477158ef6bed`).
The live manuscript has author changes to framing; they were read and preserved,
not overwritten by the older GitHub manuscript snapshot. Table numbering is no
longer a launch blocker. No further caption response is required.

The immediate small batch is the two central P1_MIX calibration targets,
RTE/MRPC seed 31415, followed by the specified magnitude-only control. The
immutable prospective protocol is
`campaign_outputs_v1/protocols/focused_norm_20260914_v1.json`. Its initial subset
contains ten calibration entries total: per task one MIX-0.001, one UNREG and
three NORM coefficients (0.01, 1, 100). Only the two MIX targets are being launched
first; no full 66/75-run matrix is restored. The omitted nuisance controls mean
this subset cannot support pretrained-projector specificity or equivalence.

This is a **fresh matched-control dataset**, not historical seed top-ups. It
retains the validated 128-token/batch-32/inner-split protocol, reference task
learning rates and insertion settings, and uses the manuscript reference epoch
budgets to predeclare fixed steps: RTE 5,670 (90 x ceil(1,992/32)); MRPC 2,760
(30 x ceil(2,934/32)). Evaluation is every 128 steps plus required trajectory
checkpoints. The historical recipe's 256 tokens, batch 64 and official-validation
selection are different; do not pool new measurements into those old means/SDs.
No 1,024-step replacement is used. The two worker time reservations are bounded
at six hours each; initial retained-output reservations are 3 GiB per run under
the unchanged 50 GiB campaign cap.

## Earlier table provenance finding

The checked manuscript labels the six-task base/large GLUE table
`tab:glue_results_base_large` (Table 2). It already reports six spectral seeds,
but the final per-seed manifests/results have not been recovered; see
`data/GLUE_PROVENANCE.md`. The A/B/C mixing table `tab:mixingmain` has one or two
seeds per task and complete archived paired summaries. The shared design's
central norm-control comparison can proceed without a caption. Do not invent missing seeds
or pool fresh common-protocol runs with either historical dataset.

P0 has one completed validated retry and one preserved interrupted attempt.
Both seed-31415 throughput pilots also completed and passed whole-run validation
at 14:50 UTC; these are not Table 2 seed extensions or confirmatory findings.
Their run IDs are `20260914T141540Z_e577e9ebab4f` (RTE/GPU 2) and
`20260914T141542Z_d3805e5e67f1` (MRPC/GPU 3). No magnitude-calibration protocol
or confirmation matrix has been registered, and no campaign training worker
remains active according to the settled leases and completed run ledger.

Next: execute the two registered calibration targets, monitor their actual
training progress and costs, then evaluate the fixed NORM grid within the
allocated storage. Keep historical seed/provenance reconciliation separate.
Keep analysis and writing time before
the author deadlines. Null results or failed matches remain valid outcomes.
