> **Latest author priority — modern decoder first:** Read
> `DECODER_FIRST_PRIORITY_20260919.md`. Prioritize the prepared decoder pilot,
> tuning/calibration and confirmation runs on the two assigned GPUs. Defer
> CENTER and further encoder controls. The coding agent has now delivered the
> temperature and band held-aside exports; verify/reuse them, do not rerun them.
> Older schedules below are superseded where they place controls first.

# CENTER mechanism control — executable protocol and two-GPU cost (19 September 2026)

Prepared for the author's decision. Nothing below has been launched. The design
follows `EVIDENCE_PRIORITY_HANDOFF_20260916.md` §2 and
`EXPERIMENT_PREPARATION_HANDOFF_20260919.md` §6 exactly; the implementation is
`notebooks/iclr/campaign/center_plan.py` (tests: `tests/test_center.py`, 9 passing).

## 1. What CENTER is

Same practical recipe as the 18 confirmations (leading core I, tail 256, no
rotations, all 48 attention matrices, ambient scalers free, RTE 5,670 steps at
adapter LR 1e-2 / head LR 5e-4 / initial scaler and coefficient 0.01, MRPC 2,760
steps at 1e-2 / 1e-3 / 0.1, repaired inputs `preparation_20260915T0707Z`,
head/data-order seeds paired to the training seed). The only change is the
explicit regularizer, already implemented as `P1_CENTER`:

    gamma * sum_{l=1}^{48} c * (||e_l - mean(e_l)||^2 + ||d_l - mean(d_l)||^2)
    c = k (d-k) / ((d-1)(d+2)) = 512*256 / (767*770) = 0.221930...

This is the Haar expectation of the two-sided mixing penalty for fixed scalers
(verified numerically in `test_center_is_the_haar_expectation_of_the_projector_penalty`).
It is a sum over modules, not a mean. Scaler means are never centered, reset or
constrained: constant scalers give exactly zero loss and zero gradient, the
gradient `2 gamma c (s - mean s)` sums to zero, and the loss is invariant to a
common shift (all tested). CENTER tests whether generic centered-scaler
shrinkage reproduces MIX's behavior; it does not match module allocation and
does not establish pretrained-frame specificity.

## 2. Calibration rule (frozen before any outcome)

- Seed 31415; targets = the existing MIX calibration endpoints in
  `data/focused_norm/selection_final.json` (RTE 0.0792355…, run
  `20260915T075556Z_53b69e612ef3`; MRPC 0.0412957…, run `20260915T075556Z_4edea9e8b40d`),
  hash-bound into the CENTER protocol.
- Initial grid gamma ∈ {1e-4, 1e-3, 1e-2} per task (6 endpoints). After all
  three validate: select minimum |pooled-norm relative error| (exact tie → smaller
  gamma); freeze if ≤ 5%.
- Otherwise at most FOUR added doses per task, one at a time
  (`center_plan.propose_next_dose`): geometric midpoint of the straddling pair
  whose better endpoint is closest (tie → smaller lower gamma); if no pair
  straddles, tenfold above the largest gamma when all norms exceed the target or
  tenfold below the smallest when all fall below; gamma restricted to [1e-6, 1];
  reaching the bound → stop with a labeled failed match. Stop at the first added
  dose within 5%, else at the cap. Every attempt, failure and retry is retained in
  the decision record; nothing but validated fixed-endpoint pooled norms is read.
- Confirmations: RTE/MRPC × seeds 42, 17, 123 at the frozen gamma regardless of
  match status (a failed match is run and labeled, never retuned or dropped).

Budget: 6 initial + 0–8 refinement + 6 confirmation = **12–20 endpoints**
(excluding smoke tests and retries), exactly as the handoff states.

## 3. Cost on two RTX 3090s (measured from the completed runs)

Per-run wall time from the 18 practical confirmations and 20 calibration runs
(worker elapsed, includes training, P7 costs and checkpoint reproduction):

| Task | Per run | Source |
|---|---:|---|
| RTE (5,670 steps) | 1.14–1.43 h (median ≈1.25 h) | UNREG/MIX/NORM confirmations and calibration |
| MRPC (2,760 steps) | 0.65–0.81 h (median ≈0.75 h) | same |

CENTER's regularizer is cheaper than MIX's (no projector contraction), so these
are conservative. Whole-run CPU validation (`smoke.py` → `run_validation`) is
included in the elapsed times.

| Stage | Endpoints | GPU-hours | Wall on 2 GPUs (RTE lane / MRPC lane) |
|---|---:|---:|---|
| Initial grid | 3 RTE + 3 MRPC | ≈6.0 | ≈3.9 h (3 RTE serial) / 2.3 h |
| Refinement (sequential per task) | 0–4 RTE, 0–4 MRPC | 0–8.3 | 0–5.2 h (RTE is the long pole) |
| Confirmations | 3 RTE + 3 MRPC | ≈6.0 | ≈3.9 h / 2.3 h |
| Held-aside + logit export (CPU) | 6 endpoints | 0 | ≈10 min |
| **Total** | **12–20** | **12–20** | **≈8 h typical, ≈13 h worst case** |

Storage: ≈0.5 GiB per run (compact restart checkpoints; the 81 GiB confirmation
root holds 45 runs) → ≤ 12 GiB; the template allowance is 60 GiB.

## 4. Exact commands (fill the two GPU IDs first)

All commands run from the isolated checkout
`/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914` with its
`.venv`. Output root: **new** `campaign_outputs_center_v1/` (own ledger and
accounting; the calibration and confirmation evidence roots stay untouched).

0. Authorization and preflight (after the author names the two GPUs):

```bash
cp notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_CENTER.template.json \
   notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_CENTER.json
# edit: assigned_gpu_ids -> the two assigned indices; authorization_record -> the author's words; delete "_comment"
git add notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_CENTER.json && git commit -m "CENTER two-GPU authorization"
export CENTER_RESOURCES=notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_CENTER.json
export CENTER_PREFLIGHT=notebooks/iclr/handoff/data/campaign_v1/preflight/<latest passing report>/report.json   # must match committed source hashes
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv   # both assigned GPUs must be free of foreign processes
ps -eo pid,cmd | grep -E "center_lane|band_lane|chain_lane" | grep -v grep # no surviving lane drivers
```

1. Register the initial grid (no training):

```bash
PYTHONPATH=src .venv/bin/python -m notebooks.iclr.campaign.center_plan register \
  --resources $CENTER_RESOURCES \
  --focused-protocol campaign_outputs_v1/protocols/focused_norm_20260915_v1.json \
  --selection-record campaign_outputs_v1/protocols/focused_selection_20260915_final.json \
  --authorization "<author's authorization text and date>" \
  --output campaign_outputs_center_v1/protocols/center_grid_20260919.json
```

2. Launch the grid, one serial lane per GPU (detached; survives a session clear):

```bash
P=campaign_outputs_center_v1/protocols/center_grid_20260919.json
setsid nohup bash notebooks/iclr/handoff/ops/center_lane.sh <GPU_A> $P rte/P1_CENTER/0.0001 rte/P1_CENTER/0.001 rte/P1_CENTER/0.01 > /tmp/center_lane_gpuA.log 2>&1 &
setsid nohup bash notebooks/iclr/handoff/ops/center_lane.sh <GPU_B> $P mrpc/P1_CENTER/0.0001 mrpc/P1_CENTER/0.001 mrpc/P1_CENTER/0.01 > /tmp/center_lane_gpuB.log 2>&1 &
tail -3 campaign_outputs_center_v1/run_ledger.jsonl   # a run is done only at status "completed"
```

3. Decide (norms only) and, if the rule proposes one, register and launch the next dose:

```bash
PYTHONPATH=src .venv/bin/python -m notebooks.iclr.campaign.center_plan select --resources $CENTER_RESOURCES \
  --calibration-protocol $P --output campaign_outputs_center_v1/protocols/center_decision_r0.json
# per task with match_status == unmatched_refinement_available:
PYTHONPATH=src .venv/bin/python -m notebooks.iclr.campaign.center_plan refine --resources $CENTER_RESOURCES \
  --calibration-protocol $P --decision-record campaign_outputs_center_v1/protocols/center_decision_r0.json \
  --task rte --output campaign_outputs_center_v1/protocols/center_refinement_rte_r1.json
setsid nohup bash notebooks/iclr/handoff/ops/center_lane.sh <GPU_A> campaign_outputs_center_v1/protocols/center_refinement_rte_r1.json rte/P1_CENTER/<gamma> > /tmp/center_lane_gpuA_r1.log 2>&1 &
# then select again with every refinement protocol appended:
PYTHONPATH=src .venv/bin/python -m notebooks.iclr.campaign.center_plan select --resources $CENTER_RESOURCES \
  --calibration-protocol $P --refinement-protocol campaign_outputs_center_v1/protocols/center_refinement_rte_r1.json \
  --output campaign_outputs_center_v1/protocols/center_decision_r1.json
```

Repeat until every task is `matched` or `failed_match` (cap 4 added doses or gamma bound).

4. Register and launch the six confirmations (RTE lane / MRPC lane):

```bash
D=campaign_outputs_center_v1/protocols/center_decision_final.json   # the last decision record
PYTHONPATH=src .venv/bin/python -m notebooks.iclr.campaign.center_plan confirm --resources $CENTER_RESOURCES \
  --calibration-protocol $P --decision-record $D --authorization "<author's words>" \
  --output campaign_outputs_center_v1/protocols/center_confirmation_20260919.json
C=campaign_outputs_center_v1/protocols/center_confirmation_20260919.json
setsid nohup bash notebooks/iclr/handoff/ops/center_lane.sh <GPU_A> $C rte/P1_CENTER/seed_42 rte/P1_CENTER/seed_17 rte/P1_CENTER/seed_123 > /tmp/center_lane_gpuA_conf.log 2>&1 &
setsid nohup bash notebooks/iclr/handoff/ops/center_lane.sh <GPU_B> $C mrpc/P1_CENTER/seed_42 mrpc/P1_CENTER/seed_17 mrpc/P1_CENTER/seed_123 > /tmp/center_lane_gpuB_conf.log 2>&1 &
```

5. Freeze and evaluate (CPU, after all six are ledger-completed): extend
   `ops/block_b_freeze.py`-style freezing to the six CENTER endpoints, then run a
   CENTER copy of `ops/temperature_logits_export.py` / `ops/band_held_aside_evaluation.py`
   over them (same loading checks; separately labeled block; logits for a
   later inner-only temperature fit). These two small adaptations are the only
   code still to write for CENTER and take under an hour; they are deliberately
   left until the population exists so the freeze is prospective.

## 5. Exports CENTER will deliver

runs.csv fields plus total/initial/learned block energies, pooled and
equal-module summaries, 48 module norms, identity residuals, raw scaler means
and centered norms (`scalers.left/right` in every observation), probe CE,
fixed/best task metrics and loss, inventories, source/checkpoint/validation
hashes, loading checks, signed per-pair norm error against the paired MIX
confirmation, and held-aside logits. Interpretation is fixed in advance: CENTER
reproducing MIX's pattern supports generic homogenization as sufficient in this
regime; divergence distinguishes the two finite-penalty objectives but norm,
module allocation and optimization can still differ.
