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
`20260914T141542Z_d3805e5e67f1` (MRPC/GPU 3). These timing workers are terminal.
The focused magnitude-calibration protocol above is now registered, and its two
MIX targets launched at 15:42:46 UTC. No confirmation matrix is registered.

## Active execution and automatic continuation — 15:51 UTC

| Task | GPU | MIX calibration run | Private tmux session |
|---|---|---|---|
| RTE | 2 | `20260914T154246Z_92f7745b1ee5` | `target_rte_gpu2_v1` |
| MRPC | 3 | `20260914T154246Z_5766195517d9` | `target_mrpc_gpu3_v1` |

The exact isolated checkout is
`/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914`;
private tmux socket `iclr_6aa54397_20260914`. Both targets use normally pushed
revision `d321e517` and the successful 129-test/four-audit/six-fingerprint
preflight `data/campaign_v1/preflight/20260914T153957Z_cafd264c/report.json`.
At the latest direct progress inspection they had reached steps 103/5,670 and
114/2,760 respectively, with finite losses and gradient norms. A 15:49:51 UTC
monitor sample recorded 3,053 MiB GPU memory each, 46/43 degrees C, no stop
reasons and approximately 10 GiB total retained campaign output. Brief 0% GPU
samples during advancing CPU spectral diagnostics are expected, not evidence
that no run is active. Optimizer-step progress and diagnostic progress must
both be inspected.

`../campaign/queue_focused.sh` is the narrow automatic continuation wrapper.
It waits for the task's exact MIX target to receive hash-verified whole-run
completion, then executes UNREG/0, NORM/0.01, NORM/1 and NORM/100 sequentially
on that task's assigned GPU. This fixed order is a scheduling decision made
before inspecting calibration endpoint norms or confirmation outcomes. It does
not change the already registered grid. Separate task-local locks prevent
duplicate queues; every queue invocation gets a new archive/log directory under
`campaign_outputs_v1/queues/`. Existing unresolved or failed attempts stop the
queue; there are no silent retries. Controller exit zero alone is insufficient:
every next launch requires durable validated completion in the run ledger.
The same source/preflight, six-hour per-run bound and 3 GiB growth reservation
remain mandatory. The wrapper stops after calibration; it cannot launch any
confirmation or expanded campaign. Do not modify campaign Python sources while
these jobs run: independent validation checks their exact source fingerprints.

The old two-lane commands below are retained as historical receipts only; do
not rerun them. Use `MINIMAL_RUNBOOK_20260914.md`, which records the corrected
four-lane GPU 0/1/2/3 commands and the newest preflight selection.

Start each task queue only once in its dedicated persistent session:

```bash
tmux -L iclr_6aa54397_20260914 new-session -d -s focused_queue_rte_v1 -c /media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914 'bash notebooks/iclr/campaign/queue_focused.sh rte'
tmux -L iclr_6aa54397_20260914 new-session -d -s focused_queue_mrpc_v1 -c /media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914 'bash notebooks/iclr/campaign/queue_focused.sh mrpc'
```

Both commands were executed successfully at **15:53:24 UTC**, from normally
pushed revision `04f79730`. The queues are active and waiting for their exact
targets; do not start duplicates. Durable queue directories are
`campaign_outputs_v1/queues/focused_rte_v1_3LapWE6Y` and
`campaign_outputs_v1/queues/focused_mrpc_v1_oT24qVPr`. Each contains its own
archived wrapper, protocol, event stream and log. The new CPU preflight
`data/campaign_v1/preflight/20260914T155131Z_453ae73a/report.json` passed all
129 tests, four supplied audits and original fingerprints. Additional queue
checks verified shell syntax, both real task/source/protocol configurations,
acceptance of a hash-validated completed timing run, and rejection of an active
or unknown run. None of these checks is a new experimental result.

### Provisional measured forecast

At roughly 15:54 UTC, warmup-excluded optimizer-step medians were about 1.5 s
(earlier direct samples: RTE 1.522 s, range 0.615–1.847; MRPC 1.460 s,
range 0.644–1.768). Completed 128-step pilots took approximately 34 minutes
per task plus whole-run audit; checkpoint diagnostics and independent reloads,
not optimization alone, account for much of that cost. The current full-length
targets have not yet completed. Initial full-run estimates are therefore
3–5 hours for RTE and 2–4 for MRPC, including checkpoint checks. Five sequential
entries per task on two assigned GPUs imply **25–45 allocated GPU-hours** and
**15–25 hours elapsed** for this calibration subset, excluding earlier setup
and timing pilots. GPU-hours here include allocated time spent in required CPU
diagnostics, not a claim of 100% device utilization.

Measured startup monitor utilization ranged 0–41% on GPU 2 and 0–43% on GPU 3;
medians were 13% and 0%, including setup/CPU checkpoint intervals. Maximum
observed memory was 3,947 and 3,055 MiB; maximum temperatures 65 and 48 degrees C.
The first few minutes are not a stable whole-run utilization estimate. No
monitor stop reasons were present at this inspection.

Using the 15:42:46 UTC launch as the forecast origin gives these provisional
calibration-completion timestamps on **September 15**:

| Scenario | UTC | Asia/Jerusalem |
|---|---|---|
| Optimistic (15 hours) | 06:45 | 09:45 |
| Expected (20 hours) | 11:45 | 14:45 |
| Conservative (25 hours) | 16:45 | 19:45 |

Confidence is medium-low until the first full runs finish. The main uncertainties
are the number of validation-improvement checkpoints, shared CPU throughput,
coefficient-dependent step cost, retries and storage growth. Recompute after
each completed task/phase and any resource change. These are calibration ETAs,
not a promised paper-completion or acceptance date. A subsequent paired-seed
confirmation matrix and its storage forecast must be frozen after norm-only
calibration selection, before inspecting confirmation outcomes; this queue does
not authorize or automatically launch it.

Check only these exact sessions and the campaign ledger, queue logs and run
monitors. Each active worker's existing supervisor samples health every ten
seconds and guards loss/gradient finiteness, progress, assigned placement,
temperature, disk growth and runtime. A stopped queue requires inspection and a
new explicitly recorded retry ID if a retry is appropriate; do not erase files,
force-close unresolved leases or touch another account's workloads.

Next: execute the two registered calibration targets, monitor their actual
training progress and costs, then evaluate the fixed NORM grid within the
allocated storage. Keep historical seed/provenance reconciliation separate.
Keep analysis and writing time before
the author deadlines. Null results or failed matches remain valid outcomes.
