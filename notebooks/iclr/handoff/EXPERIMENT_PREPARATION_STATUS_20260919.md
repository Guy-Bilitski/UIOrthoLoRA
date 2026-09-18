# Experiment preparation status — 19 September 2026 (coding session report)

Companion to `EXPERIMENT_PREPARATION_HANDOFF_20260919.md`, which remains the entry
point. Everything below was done on CPU in the isolated checkout
`/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914` (branch
`ortho_new`, `.venv`: Python 3.10.14, torch 2.5.1+cu124, transformers 5.8.0). No
GPU was used: at inspection all four RTX 3090s carried other users' processes
(vLLM engines on GPUs 0–1, `torch_env` python on 2–3), the author has not yet
named the two assigned devices, and no training was launched. No manuscript text
was changed. The approved abstract and introduction are untouched.

## 1. Recovered versus missing (before this session)

| Item | State found | Action |
|---|---|---|
| Temperature-scaling logit export (18 practical endpoints) | Absent everywhere (only the analyzer and its synthetic check existed) | Produced by CPU inference, §2 |
| Band/head held-aside scoring (21 endpoints) | No per-example or exported scores. **Disclosure:** every band/head run's worker had computed aggregate accuracy/loss of its fixed and best checkpoints on the locked split at training time (GPU) into `locked_endpoints.json`, bound into its validation report; never exported or used for any decision | Produced by CPU inference; the training-time aggregates are recorded alongside and reproduced exactly (21/21 accuracies equal, losses within tolerance), §3 |
| Per-run manifests and reload-validation reports for all 39 runs | Only 13 representative recipe manifests in the bundle | All 39 exported, §4 |
| Training / inner-selection trajectories | Recorded in every run (`steps.jsonl`, 52–53 observation files per run); never exported | Exported at actual steps with explicit windows, §4 |
| Raw calibration diagnostic (`review_feedback/20260919/`) | Present; reproducer re-run from saved evidence reproduces every number to float rounding (max difference 1e-16) | Verified only |
| `P1_CENTER` regularizer | Implemented in `regularizers.py`/`spectral.py`, plumbed through `worker.py`, covered by the cached-vs-reference test and the synthetic worker test | Property tests added; registration/selection/confirmation plumbing was missing and is now implemented, §5 |
| Decoder runner | None (runner is RoBERTa-specific) | Isolated package implemented and CPU-tested, §6 |
| GPT-2 E2E provenance | 29 archived single-run score files, 7 with metadata | Read-only inventory, §7 |
| Historical four-GPU authorization files | Present, stale | Not reused; a two-GPU template with empty device IDs was added for CENTER |

Legacy paper tables, the 39 invalidated tokenizer-fault runs and their quarantine
records, the September 16 locked export and the band freeze ledger are untouched
(every exporter refuses to overwrite an existing output and re-verifies request,
freeze, checkpoint, validation and input-manifest hashes before reading a model).

## 2. Temperature scaling of the 18 frozen practical endpoints (complete)

Export: `data/temperature_logits_20260916/` (manifest + 18 records; schema of
`EVIDENCE_PRIORITY_HANDOFF_20260916.md` §1 plus source-split identity, sample-ID
namespace `<task>:<split>:<official-row>`, loading checks). Analysis:
`data/temperature_analysis_20260916.json` from the unchanged
`scripts/analyze_temperature_scaling.py` (its synthetic self-check
`check_temperature_scaling.py` passes). Loading checks: all 18 inner-selection
accuracies (and MRPC F1) reproduce exactly and losses within 5e-4; all 18
held-aside prediction vectors and per-example losses reproduce the September 16
export with zero difference. All 18 fitted temperatures are interior optima
(no boundary flags); argmax predictions are unchanged, so accuracy is unchanged.

| Task | Arm | T (seeds 17/42/123) | Held-out NLL before → after | ECE₁₅ before → after |
|---|---|---|---|---|
| RTE | UNREG | 9.35 / 9.39 / 10.04 | 2.429±0.071 → 0.565±0.009 | 0.265 → 0.060 |
| RTE | MIX | 5.44 / 5.33 / 5.59 | 1.229±0.089 → 0.551±0.010 | 0.225 → 0.050 |
| RTE | NORM | 9.34 / 8.09 / 7.74 | 2.193±0.327 → 0.553±0.020 | 0.270 → 0.068 |
| MRPC | UNREG | 4.70 / 4.79 / 4.70 | 0.871±0.068 → 0.335±0.013 | 0.118 → 0.048 |
| MRPC | MIX | 1.95 / 2.01 / 1.90 | 0.395±0.028 → 0.325±0.019 | 0.075 → 0.043 |
| MRPC | NORM | 4.05 / 4.35 / 4.07 | 0.755±0.064 → 0.334±0.010 | 0.116 → 0.029 |

Primary follow-up contrast (scaled MIX minus NORM held-out NLL, per seed, mean,
nominal paired t interval, n = 3, not multiplicity-adjusted):

| Task | Before scaling | After scaling |
|---|---|---|
| RTE | −1.236 / −0.764 / −0.892; mean −0.964 [−1.571, −0.357] | −0.021 / +0.020 / −0.007; mean **−0.002 [−0.054, +0.050]** |
| MRPC | −0.293 / −0.459 / −0.330; mean −0.361 [−0.578, −0.144] | +0.017 / −0.035 / −0.010; mean **−0.009 [−0.074, +0.055]** |

Reading, in the assessment's own terms: the raw held-out NLL advantage of MIX is
almost entirely reproduced by a single inner-fitted logit scale. MIX's fitted
temperatures sit closer to 1 (it trains less over-confidently), and after
scaling the three arms are indistinguishable on NLL within seed noise; scaled
ECE is descriptively lower for NORM than MIX on MRPC (+0.014 [+0.006, +0.022])
and not separable on RTE. This weakens the practical case for MIX as a
probability-quality method; it does not erase the geometric intervention, the
raw-loss finding or the allocation difference, and it says nothing about
mechanism. The paper currently reports the uncalibrated result and leaves this
question open; how to integrate this outcome is a writing decision for the
author.

## 3. Held-aside evaluation of the complete band/head block (complete)

Export: `data/band_held_aside_20260917/` (`band_held_aside_results.csv`,
`per_example/` with IDs, labels, two-class logits, predictions and losses for
the 277 held-aside RTE examples plus the inner-selection logits kept for a
possible later inner-only temperature fit, `summary_contrasts.json`,
`evaluation_manifest.json` with freeze-ledger and protocol hashes, loading
checks, prior-scoring disclosure and output hashes). All 21 inner-selection
loading checks passed; all 21 training-time GPU aggregates agree (accuracy exact,
loss within tolerance). Population and endpoints are exactly the frozen 21; no
seed, arm or checkpoint was dropped or substituted. Report alongside, never merged
with, inner-selection band outcomes or the separate practical block.

| Condition | Held-aside accuracy (seeds 17/42/123) | mean ± SD | Held-aside NLL mean ± SD |
|---|---|---|---|
| HEAD_BASE | 0.5884 / 0.5812 / 0.5740 | 0.581 ± 0.007 | 0.680 ± 0.001 |
| LEAD_DIAG | 0.7365 / 0.7112 / 0.7329 | 0.727 ± 0.014 | 1.783 ± 0.063 |
| LEAD_ROT64 | 0.7581 / 0.7401 / 0.7401 | 0.746 ± 0.010 | 1.932 ± 0.213 |
| MID_DIAG | 0.7112 / 0.7040 / 0.6823 | 0.699 ± 0.015 | 1.726 ± 0.092 |
| MID_ROT64 | 0.7292 / 0.7112 / 0.7112 | 0.717 ± 0.010 | 2.047 ± 0.130 |
| TAIL_DIAG | 0.6715 / 0.6931 / 0.7112 | 0.692 ± 0.020 | 2.007 ± 0.170 |
| TAIL_ROT64 | 0.6534 / 0.6931 / 0.7040 | 0.684 ± 0.027 | 2.250 ± 0.282 |

Pre-specified paired contrasts (accuracy, mean [nominal 95% t interval]):
tail diagonal minus head +0.111 [+0.043, +0.178]; tail rotation minus head
+0.102 [+0.019, +0.186]; tail rotation minus tail diagonal **−0.008 [−0.031,
+0.014]** (the inner-selection ordering of the tail-rotation gain does not carry
to the held-aside split); leading diagonal minus tail diagonal +0.035 [−0.030,
+0.100]. Every location and flexibility contrast is in `summary_contrasts.json`.
Head-only NLL (0.68) versus band NLL (1.7–2.3) on the same split reproduces the
"accuracy gains coexist with worse NLL" pattern within one protocol; the
cross-study comparison with the practical MIX 1.229 is still between different
adapter constructions and initializations.

## 4. Evidence package from existing files (complete)

`data/run_records_20260919/`: for all 39 frozen runs the manifest, whole-run
validation report, fixed-endpoint checkpoint-reproduction report, P0 equivalence
report, P7 cost record, training-time `locked_endpoints.json` (band runs only),
and the compressed raw per-step log; `inner_selection_trajectories.csv` (5,429
rows: task CE, accuracy, MRPC F1, fixed-probe masked CE and pooled geometry at
every recorded evaluation step); `training_loss_trajectories.csv` (8,550 rows:
task loss, regularizer loss, total objective, raw left/right mixing and pre-clip
gradient norm averaged over the actual optimizer steps between consecutive
evaluations, windows stated per row); `fixed_vs_selected_endpoints.csv` (39
rows: fixed and best-inner endpoints, metrics, checkpoint hashes, the exact list
of recorded evaluation steps). Nothing was interpolated; evaluation steps are
every 128 steps plus the prescribed checkpoint fractions and accuracy
improvements (52–53 per run). The 27 MB `training_steps/` and `p7_costs/`
directories are in the research repository only; the CSVs and reports are
mirrored to Overleaf.

## 5. CENTER control (prepared, not launched)

`notebooks/iclr/campaign/center_plan.py` + `phase_gates.py`/`smoke.py` admission
of the new purposes `center_norm_calibration` and `center_confirmation`;
`tests/test_center.py` (9 tests: coefficient c = 512·256/(767·770), exact zero
for constant scalers, translation invariance, analytic gradient 2γc(s − mean s)
with zero sum, module **sum**, Monte-Carlo Haar-expectation check, admission
tampering, refinement chain, confirmation binding). Launchers
`ops/launch_center_entry.sh` and `ops/center_lane.sh` refuse a dirty source
tree, a stale preflight, an unassigned GPU, a foreign process or a duplicate
entry; a passing preflight for the committed source exists
(`data/campaign_v1/preflight/20260918T233221Z_a3f9c1a2/`, source revision
e9324c85). The two-GPU authorization is a **template** with empty device IDs:
`data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_CENTER.template.json`.

Design, frozen rule, exact commands and measured cost: `CENTER_PROTOCOL_20260919.md`.
Cost: 12–20 endpoints, 12–20 GPU-hours, ≈8 h typical / ≈13 h worst case of
wall time on two GPUs (RTE ≈1.25 h per run, MRPC ≈0.75 h, from the completed
runs), plus ≈10 min CPU for freezing and held-aside/logit export.

## 6. Decoder pilot (prepared, not launched)

`notebooks/iclr/decoder_pilot/` with 15 passing CPU tests; pinned
Qwen2.5-1.5B-Instruct and GSM8K sealed under `campaign_outputs_decoder_v1/inputs/`
with a 56-module SVD cache; CPU feasibility report
`campaign_outputs_decoder_v1/feasibility_cpu_20260919.json`. Design, measured
feasibility numbers, proposed matrix (UNREG/MIX/NORM/LoRA/PiSSA × seeds 42, 17,
123 = 15 confirmations + 9 tuning + 4–6 calibration + 1 GPU pilot ≈ 30 runs,
≈28–30 GPU-hours, ≈1 day on two GPUs, to be corrected by the first GPU pilot)
and the five open decisions (instruct vs base model, module set, tail size,
MIX dose bracketing, fallback to LoRA/PiSSA on RTE/MRPC) are in
`DECODER_PILOT_DESIGN_20260919.md`. A registration/admission layer equivalent
to `center_plan.py` is still to be written once the design is fixed.

## 7. GPT-2 E2E provenance (read-only)

`data/gpt2_e2e_provenance_20260919/` (`e2e_result_inventory.csv`, `summary.json`)
from `notebooks/E2E`: 32 result directories, 29 with `scores.txt`, 7 with
`run_metadata.json` (the later `phase1_*`/`smart_*` screens; these record
GPT-2 Medium, beam 10, target modules attn.c_attn/attn.c_proj/mlp.c_fc/mlp.c_proj,
256 singular values), no LoRA score files. Exactly one archived single run,
`lr_0.05_svalues_256_svectors_30_seed_17_init_sigma_0.1_init_scaler_0.1`,
reproduces every rounded value of the manuscript's UIOrthoLoRA row (68.8 / 8.729
/ 46.71 / 71.6 / 2.47); **no archived file reproduces the UILinLoRA row** (68.7 /
8.732 / 46.52 / 71.45 / 2.46). The only configurations with more than one seed
are `lr_0.05/256/svectors_90` (4 seeds, BLEU 0.6767–0.6821, mean 68.1) and the
`phase1` LR screens; neither matches the table rows or their ± values. The
uncertainties of both UI rows therefore cannot be bound to a run population from
these files. No historical score was modified; correction or removal is the
author's decision.

## 8. Tests and commands run (CPU)

```bash
cd /media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest notebooks/iclr/campaign/tests notebooks/iclr/decoder_pilot/tests -q   # 176 passed
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m notebooks.iclr.campaign.preflight                                              # all checks passed
cd notebooks/iclr/handoff/scripts && CUDA_VISIBLE_DEVICES= ../../../../.venv/bin/python check_temperature_scaling.py                     # PASS
cd .. && CUDA_VISIBLE_DEVICES= ../../../.venv/bin/python review_feedback/20260919/calibration_diagnostic.py --paper-root . --out <fresh>  # reproduces
# exports (each refuses to overwrite its first output)
CUDA_VISIBLE_DEVICES= .venv/bin/python notebooks/iclr/handoff/ops/temperature_logits_export.py --threads 8
cd notebooks/iclr/handoff/scripts && CUDA_VISIBLE_DEVICES= ../../../../.venv/bin/python analyze_temperature_scaling.py --bundle ../data/temperature_logits_20260916 --out ../data/temperature_analysis_20260916.json
CUDA_VISIBLE_DEVICES= .venv/bin/python notebooks/iclr/handoff/ops/band_held_aside_evaluation.py --threads 8
CUDA_VISIBLE_DEVICES= .venv/bin/python notebooks/iclr/handoff/ops/evidence_package_export.py
.venv/bin/python notebooks/iclr/handoff/ops/gpt2_e2e_provenance_inventory.py
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.pilot prepare --root campaign_outputs_decoder_v1 --allow-downloads
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.pilot feasibility --root campaign_outputs_decoder_v1 --out campaign_outputs_decoder_v1/feasibility_cpu_20260919.json
```

Commits on `ortho_new` (research repository): `4af6033e` exporters, `e9324c85`
CENTER/decoder/evidence code, `b6bb706b` immutable exports, `d00ab113`
preflight + CENTER protocol, then this status document and the decoder design.

## 9. Two-GPU schedule proposal (from the day the author releases the GPUs)

Deadline assumption from the September 16 handoff: full paper ≈ 26 September;
the last 48 hours (24–26 September) are reserved for analysis and writing, so
GPU work must finish by the morning of 24 September.

| Day | GPU A | GPU B | CPU / writing agent |
|---|---|---|---|
| D0 (release day) | CENTER RTE grid (3 runs, ≈3.9 h) | CENTER MRPC grid (3 runs, ≈2.3 h) → decoder GPU pilot (≈0.3 h) | Decide decoder model/task from the pilot numbers; register the decoder protocol |
| D0 evening → D1 | CENTER RTE refinement doses (0–4, sequential) | CENTER MRPC refinement (0–4) then decoder LR tuning | CENTER decisions (norms only) |
| D1 → D2 | CENTER RTE confirmations (3) | CENTER MRPC confirmations (3) then decoder calibration | Freeze the six CENTER endpoints; CPU held-aside + logits |
| D2 → D3 | Decoder confirmations (15 runs across both GPUs, ≈7.5 h) | | Reload validation and geometry are in-run; export |
| D3 | Buffer for retries / fallback (LoRA+PiSSA on RTE/MRPC if the decoder is infeasible) | | Evidence exports, Overleaf mirror |
| D4–D5 | — | — | Analysis and writing only |

If the GPUs arrive later than D0 = 21 September, the decoder confirmations are
the first item to scale down (e.g. three arms UNREG/MIX/LoRA, or two seeds
reported as such), recorded before any outcome is seen.

## 10. Unresolved decisions for the author

1. Integrate the temperature-scaling outcome (gap ≈ 0 after scaling) and the
   band held-aside block into the paper: author/writing-agent decision.
2. CENTER: approve the 12–20-endpoint scope and the new output root; name the two
   GPU IDs (fill the template).
3. Decoder pilot: instruct vs base model, module set (q/o only), tail 512,
   MIX dose bracketing, max_length 640; or the LoRA/PiSSA-on-RTE/MRPC fallback.
4. GPT-2 table: keep, correct or remove the two UI rows whose uncertainties have
   no recoverable run population.
5. Whether the 27 MB raw per-step logs should also be mirrored to Overleaf.
