# Minimal ICLR experiment runbook — 2026-09-14

This is the current resumption document for the deadline-focused campaign. It
supersedes stale historical status paragraphs in this handoff, but does not
replace `EXPERIMENTS_REQUIRED.md` as the scientific definitions document.

## Update ~19:45 UTC: author-approved core confirmation stage

The author approved, in this session: (1) the 18-run core confirmation tranche
(P1_UNREG / P1_MIX / matched P1_NORM × RTE/MRPC × seeds 42,17,123) launching
overnight once all ten calibration entries validate; (2) a separate 150 GiB
confirmation output root `campaign_outputs_confirmation_v1/` that can later be
cleaned wholesale without touching calibration evidence. The exact quote is in
`data/campaign_v1/RESOURCE_AUTHORIZATION_20260914_CONFIRMATION.json`.

Confirmation pipeline (all fail-closed, new code under `notebooks/iclr/campaign/`):

1. `focused_analysis.py` — reads only ledger-validated fixed-endpoint pooled
   norms for the registered ten-entry subset and applies the registered ±5%
   selection rule; writes an immutable selection record (all rows retained,
   failed matches reported as failed).
2. `confirmation_plan.py` — registers the confirmation protocol from the
   selection record (re-deriving the selection from the ledger and refusing on
   any disagreement); defines/validates the 18 entries and their admission.
3. `smoke.py --purpose confirmation` — same controller, whole-run validation
   unchanged; `phase_gates.py` admits stage `confirmation` only with the
   registered, author-authorized, hash-bound protocol and a declared
   confirmation seed.
4. `queue_confirmation.sh conf_gpu0..conf_gpu3` — four GPU-local lanes balanced
   by measured cost (3R+1M / 2R+2M / 2R+3M / 2R+3M). Launch only after the six
   calibration lanes exit; stopping one lane's tmux session frees exactly that
   GPU.

The six calibration NORM entries below remain the immediate prerequisite.

## Objective and stop rule

Produce a thin, reproducible extension around the existing Table 2 evidence:
paired RTE/MRPC fixed-step calibration at seed 31415, followed by the registered
norm-only controls. Do not launch the deferred 66/75-run P0–P8 matrix, new
backbones, full fine-tuning, or extra seeds unless the author explicitly changes
this scope. A norm-only result cannot establish pretrained-projector
specificity; report null, reversed, failed-match and incomplete outcomes.

Stop after the ten registered calibration entries are validated and the fixed
endpoint norms are analyzed. A later confirmation matrix is a separate decision:
freeze it only after calibration, record its storage/throughput forecast, and do
not treat this runbook as authorization to launch it automatically.

## Current durable state

- Isolated checkout: `/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914`
- Branch: `ortho_new`; latest pushed source commit: `34c3873f`
- Output namespace: `campaign_outputs_v1`, same checkout
- Campaign ledger: `campaign_outputs_v1/run_ledger.jsonl`
- Original allocation: GPUs 2/3; explicit author extension: GPUs **0/1/2/3**
- Four-GPU authorization: `notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260914_FOUR_GPUS.json`
- Allocation ledger remains immutable; its hash-bound extension is
  `campaign_outputs_v1/accounting_v1/gpu_authorization_extension.json`
- Storage cap remains 50 GiB; latest measured retained output was about 13.3 GiB
- Private tmux socket: `iclr_6aa54397_20260914`
- Prepared inputs: `campaign_outputs_v1/inputs/preparation_20260914T1254Z`
- Registered protocol: `campaign_outputs_v1/protocols/focused_norm_20260914_v1.json`

Completed and independently whole-run validated:

| Entry | Run ID | GPU | Status |
|---|---|---:|---|
| RTE/P1_MIX/0.001 | `20260914T154246Z_92f7745b1ee5` | 2 | completed |
| MRPC/P1_MIX/0.001 | `20260914T154246Z_5766195517d9` | 3 | completed |
| RTE/P1_UNREG/0 | `20260914T172943Z_e027fd2735c9` | 2 | completed |
| MRPC/P1_UNREG/0 | `20260914T170041Z_253634ee61b3` | 3 | completed |

The two original P0/timing pilots are also retained in the ledger. The first
P0 attempt is intentionally preserved as `interrupted`; it is not deleted or
reused. Every completed row has a controller validation report with checkpoint,
metrics, P3, P7 and P8 reproduction flags.

## Remaining six entries and GPU lanes

Run these four persistent lanes concurrently. They use the same fixed protocol,
reference epoch endpoints (RTE 5,670 steps; MRPC 2,760), six-hour per-run
reservation, 3 GiB retained-growth reservation and fail-closed behavior:

| Lane | GPU | Entries |
|---|---:|---|
| `rte_low` | 0 | RTE NORM/0.01, NORM/100 |
| `mrpc_low` | 1 | MRPC NORM/0.01, NORM/100 |
| `rte_mid` | 2 | RTE NORM/1 |
| `mrpc_mid` | 3 | MRPC NORM/1 |

Use the newest successful preflight report, not a guessed path:

```bash
cd /media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
preflight=$(find notebooks/iclr/handoff/data/campaign_v1/preflight -mindepth 2 -name report.json -print | sort | tail -n 1)
for lane in rte_low mrpc_low rte_mid mrpc_mid; do
  bash notebooks/iclr/campaign/queue_focused.sh "$lane" --check "$preflight"
done
```

Start each lane once in the private socket. Do not start the old `focused_queue_*`
commands from `SUBMISSION_SCOPE_20260914.md`; those commands predate the four-GPU
split and require a different argument list.

```bash
tmux -L iclr_6aa54397_20260914 new-session -d -s focused_rte_low_v2 -c "$PWD" "bash notebooks/iclr/campaign/queue_focused.sh rte_low --run $preflight"
tmux -L iclr_6aa54397_20260914 new-session -d -s focused_mrpc_low_v2 -c "$PWD" "bash notebooks/iclr/campaign/queue_focused.sh mrpc_low --run $preflight"
tmux -L iclr_6aa54397_20260914 new-session -d -s focused_rte_mid_v2 -c "$PWD" "bash notebooks/iclr/campaign/queue_focused.sh rte_mid --run $preflight"
tmux -L iclr_6aa54397_20260914 new-session -d -s focused_mrpc_mid_v2 -c "$PWD" "bash notebooks/iclr/campaign/queue_focused.sh mrpc_mid --run $preflight"
```

Each queue archives its wrapper, protocol, command log and event stream under
`campaign_outputs_v1/queues/`. A queue launches only after the corresponding
MIX/UNREG target is ledger-completed, checks exact protocol/source hashes, and
requires a hash-verified whole-run completion before considering its entry done.
It stops on an unresolved attempt, worker failure, interruption, source mismatch,
GPU collision or quota failure. Never infer completion from a process exit code;
never delete or overwrite a failed attempt. Use a new retry ID only after
diagnosing and documenting the cause.

## Validation and analysis handoff

Before launching from a resumed checkout, run the CPU preflight and record its
immutable report. After all ten entries are complete, use the focused calibration
reader/analysis to compare pooled total Frobenius norms at the fixed endpoint,
select only the predeclared ±5% match rule, and retain every per-module norm and
failure. Keep the fresh matched-control dataset separate from legacy Table 2
CSV summaries; do not pool its means or standard deviations into historical
results. MRPC must retain both accuracy (primary) and F1.

Required final checks from the research root are the four supplied scripts,
`sha256sum --check notebooks/iclr/handoff/GPU_SOURCE_FINGERPRINTS.sha256`, all
campaign tests, successful checkpoint reloads and the manuscript audit. Update
this file, `GPU_HANDOFF.md`, `CAMPAIGN_STATUS_20260914.md` and the ledger with
terminal statuses, artifact paths, measured storage and the norm-match decision.
Only then make any manuscript edit, after reconciling the live Overleaf author
revision. Push normal commits to `ortho_new` only after fetch/rebase and
validation; never force-push.

## Monitoring checklist

Inspect only this campaign's exact run directories, queue sessions and ledger.
Every supervisor samples progress, finite losses/gradients, assigned GPU UUID,
temperature, disk growth and checkpoint creation. During CPU diagnostics, zero
GPU utilization is expected; a stale heartbeat, missing checkpoint, NaN/Inf,
unassigned placement, OOM, temperature threshold or quota alert is not. Record
the observation and preserve the artifact before any retry decision.
