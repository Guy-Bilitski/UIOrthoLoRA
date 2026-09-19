# Decoder subspace study — implementation, registries and launch (19 September 2026)

Operator companion to `DECODER_SUBSPACE_STUDY_20260919.md` (design) and
`CODING_AGENT_PROMPT_20260919.md` (brief). It supersedes
`DECODER_EXECUTION_PROTOCOL_20260919.md` and the five-arm benchmark it drove;
`data/campaign_v1/DECODER_DECISIONS_20260919.proposed.json` is **not** a current
launch instruction and must not be sealed.

## 1. What is implemented

New output namespace `campaign_outputs_decoder_subspace_v1`. The sealed
Qwen2.5-1.5B-Instruct and GSM8K inputs and the 56-module SVD cache are **reused
read-only** from `campaign_outputs_decoder_v1/inputs/` through a symlink plus
`inputs_manifest.json` recording every source hash. Nothing was downloaded,
copied or re-sealed, and no revision changed.

| Module | Content |
|---|---|
| `decoder_pilot/subspace.py` | Band placement for the decoder. Reuses `campaign/band.py` (`BandConfig`/`BandLinear`) algebra unchanged; the RoBERTa helper `campaign.band.insert_band` is **not** used because it unfreezes `model.classifier`. Freezes the whole decoder including embeddings and the LM head, wraps only the scoped square projections, and refuses if anything else is trainable. Band diagnostics: strict support confinement, within-band off-diagonal energy, rotation distance from identity, per-band energy. `frozen_fingerprint` hashes arm, band bounds, q, projections, module inventory, trainable counts, SVD cache, model/dataset hashes, split fingerprints, recipe and source revision. |
| `decoder_pilot/subspace_plan.py` | Registration, admission and selection. Stages timing → tuning → confirmation, each sealed before the runs it admits, with the frozen reference row. Balanced across-band learning-rate rule, measured scope-reduction rule, generation cap audit, whole-run validation and the shared append-only ledger. |
| `decoder_pilot/subspace_runner.py` | Executes exactly one registered entry on one assigned GPU. No unregistered training mode exists. `audit` reloads a finished run's fixed endpoint and decodes the shared seeded inner-selection subset, which is how the six selected-rate endpoints are decoded at all, since tuning trains with generation disabled. `reference` scores the frozen starting checkpoint. |
| `decoder_pilot/subspace_analysis.py` | Evidence export from validated confirmation runs: per-arm tables for both primary outcomes, paired band contrasts with leading minus tail as the primary location contrast, the rotation contrast with whether rotations were active, learning curves, norms and confinement diagnostics, and the frozen reference row. Intervals labelled nominal and exploratory; missing seeds reported, never dropped. |
| `decoder_pilot/evaluate.py` | KV caching enabled for decoding and the training setting restored; generation precision pinned to the registered precision; cap hits counted from raw token/EOS boundaries rather than the answer parser. |

Every arm is `Delta = U_B H V_B^T` on one equal third of each 1536x1536
projection, `H` diagonal (DIAG) or with the band's last 128 directions rotated
on both sides (ROT128), coefficients zero and rotations identity at insertion,
no ambient scalers and no leading or complement identity core.

| Band, descending singular-value index | Coefficients only | Coefficients plus partial rotations |
|---|---|---|
| Leading [0, 512) | LEAD_DIAG | LEAD_ROT128 |
| Middle [512, 1024) | MID_DIAG | MID_ROT128 |
| Tail [1024, 1536) | TAIL_DIAG | TAIL_ROT128 |

Trainable parameters, verified against the live inventory before registration:
**28,672** for DIAG (56 x 512) and **1,863,680** for ROT128
(56 x (512 + 2 x 128 x 128)). The rotation family therefore has more parameters
than DIAG; that contrast tests added flexibility, not parameter efficiency.

## 2. Hazards addressed

- **Wrong-band reload.** `AdapterStore.restore` replaces only trainable tensors
  and compares a fingerprint. LEAD_DIAG and MID_DIAG share trainable names and
  shapes, so without the band bounds inside the fingerprint a wrong-band reload
  would have succeeded silently. Band bounds, q and the module inventory are now
  part of the checkpoint identity, and the rejection is tested.
- **KV caching.** `load_model_and_tokenizer` sets `config.use_cache=False` for
  training. Generation now enables caching explicitly and restores the training
  setting afterwards, instead of relying on a generation-config default.
- **Generation precision.** Training runs under bf16 autocast; decoding now runs
  under the same registered precision for every arm and for the frozen
  reference, rather than silently in float32.
- **Two resident models.** Reload validation previously built a fresh GPU model
  while the trained one was still resident. The trained copy is now moved off the
  device first, the fresh model is rebuilt and restored from the fixed-endpoint
  checkpoint, and **that reloaded model** is what geometry and generation use.
  One full model is resident at a time and the evaluated weights are provably
  the validated ones.
- **Dense geometry cost.** The full `U^T Delta V` decomposition runs at insertion
  and at the fixed endpoint only. The per-evaluation callback keeps cheap pooled
  scalars. No correctness check was removed.
- **Tuning generation.** Tuning entries carry `generation.enabled = false` and
  admission refuses a tuning job that would generate.

## 3. Tests

`tests/test_decoder_subspace.py` (42) and `tests/test_decoder_subspace_plan.py`
(27), with the whole suite at **280 passed**:

zero insertion for all six arms; only adapter parameters trainable with the LM
head and embeddings frozen; trainable counts against the declared
parameterization and the full-scale figures; forward/delta/merge/disable
agreement; strict support confinement per arm; disjoint bands covering the
spectrum; rotation gradients zero at the zero core and active once coefficients
move; DIAG has no within-band off-diagonal energy while ROT128 can; sparse
diagnostics skip the dense path; fingerprints separate all six arms; fresh-model
reload reproduces the update for all six; wrong-band and wrong-family reload
rejected; mismatched layer configuration rejected; design rejects an incomplete
family and off-grid evaluation steps; both primary outcomes pinned; timing
record takes the slower family; tuning refuses generation, a confirmation seed
and a changed memory setting; the balanced rule prefers the across-band mean
over a band-best rate and reports per-band sensitivity and ranking dependence;
ties take the smaller rate; incomplete grids wait; cost projection arithmetic;
scope keeps six arms when it fits, falls back to DIAG-only when it does not, and
reports an obstruction rather than cutting further; the cap audit raises the
budget only above one percent; confirmation is 6x3 (or 3x3 reduced); admission
rejects tampering and cross-stage protocol borrowing; the frozen reference must
be untrained.

## 4. Registries

Sealed design `campaign_outputs_decoder_subspace_v1/design_20260919.json`
(sha256 `9649dd84…`) from
`data/campaign_v1/DECODER_SUBSPACE_DECISIONS_20260919.json`.

| Stage | Full scope | Reduced (DIAG-only) fallback |
|---|---:|---:|
| Timing pilots (100 steps) | 2 | 2 (charged to the reduced budget) |
| Tuning (3 rates x bands x families, seed 31415) | 18 | 9 |
| Confirmations (arms x seeds 17/42/123) | 18 | 9 |
| Frozen reference (inference only) | 1 | 1 |
| Optimizer steps total | 30,512 | 15,356 |
| Full test generations (1,319 examples) | 19 states | 10 states |

## 4b. Measured on GPUs 2 and 3, 19 September 2026

The author authorized GPUs 2 and 3; GPUs 0 and 1 stayed another user's and were
untouched. The runner re-checks at launch that the named card carries no foreign
process. Four timing runs were executed, all at seed 31415 and LR 1e-3 for 100
steps, decoding the prescribed seeded 128-example inner-selection subset.

**The first attempt found a hard memory limit.** At the originally registered
microbatch 4 x 640, TAIL_DIAG peaked at 22.5 GiB reserved and survived, but
TAIL_ROT128 hit CUDA out-of-memory at step 24. Following the plan's instruction
to reduce microbatch and increase accumulation together, the microbatch was
halved to 2 and accumulation doubled to 8, keeping the effective batch at 16,
and the generation batch was halved to 16. This was applied identically to every
arm and sealed as design v2 before any tuning or confirmation. The failed
attempt is retained in the ledger as evidence.

| Run | Microbatch x accumulation | Step time | Peak reserved | Outcome |
|---|---|---:|---:|---|
| TAIL_DIAG | 4 x 4 | 1.033 s | 22.53 GiB | complete |
| TAIL_ROT128 | 4 x 4 | 1.611 s | 21.28 GiB | **CUDA OOM at step 24** |
| TAIL_DIAG | 2 x 8 | see final record | see final record | complete |
| TAIL_ROT128 | 2 x 8 | see final record | see final record | complete |

Memory, not step time, is the binding constraint on a 24 GiB card. The
151,936-token vocabulary at length 640 dominates: the logits and their gradient
are the largest single allocations, which is why halving the microbatch is what
made the rotation family fit.

Other measurements from the completed TAIL_DIAG run at 100 steps: setup 25.9 s,
one full inner-selection evaluation over 747 examples 21.1 s, decoding 128
examples 50.1 s at generation batch 32 and a 320-token cap, 3,443 tokens/s, and
417 s end to end.

Correctness on real weights, not only on the synthetic fixture: zero insertion
exact, P0 forward/delta/merge/disable agreement, fixed-endpoint reload
reproducing the recorded selection NLL, pooled off-band energy fraction
2.5e-12 and within-band off-diagonal fraction 8.3e-12 for a DIAG arm, and a
pooled relative Frobenius update norm of 0.012 after 100 steps. Trainable
parameters were 28,672 for DIAG and 1,863,680 for ROT128, matching the plan.

**Decode cap.** The DIAG timing endpoint hit the 320-token cap on 2 of 128
audited outputs, a rate of 1.6%, above the plan's 1% threshold. If the full
prescribed audit reproduces that, the registered rule raises the confirmation
decode budget to 640 tokens for every arm and for the frozen reference, and the
measured generation budget must be updated accordingly. Cap hits are counted
from raw token/EOS boundaries.

**Not a result.** These are implementation and cost measurements at a
100-step endpoint on the tail band only. They say nothing about band ordering,
and no tuning or confirmation run has been launched.

## 5. Timing worksheet

Replace every term with the measured pilot numbers; nothing here is an estimate
carried over from the superseded benchmark, whose 28-30 GPU-hour figure was for
a different study.

```
training_gpu_hours   = total_optimizer_steps x slowest_family_step_seconds / 3600
generation_gpu_hours = (confirmations + 1) x seconds_per_full_test_generation / 3600
subtotal             = training + generation + other_overhead
total                = subtotal x 1.25            # 25% contingency
budget               = min(96, 2 x hours_available_before_the_cutoff)
keep six arms if total <= budget, else the DIAG-only fallback
```

`subspace_plan.py scope-decision` computes exactly this from the timing record
and writes the frozen choice, including an explicit obstruction field if even
the reduced scope does not fit.

## 6. Launch commands

```bash
cd /media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
ROOT=$(pwd)/campaign_outputs_decoder_subspace_v1
RES=notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_SUBSPACE.json
PLAN="PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.subspace_plan"
RUN="PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.subspace_runner"
UUID=$(nvidia-smi -i <GPU> --query-gpu=uuid --format=csv,noheader)

# one registered entry on one assigned GPU (same form for every stage)
CUDA_VISIBLE_DEVICES=$UUID $RUN train --resources $RES --gpu <GPU> \
  --protocol $ROOT/protocols/<stage>.json --entry-id "<entry id>"

# after each run: whole-run validation, then the completed ledger event
CUDA_VISIBLE_DEVICES= $PLAN validate-run --resources $RES --run-directory <run_dir> --protocol <protocol>
CUDA_VISIBLE_DEVICES= $PLAN complete --resources $RES --run-id <run_id> \
  --validation-report <run_dir>/validation_report.json

# stage gates
CUDA_VISIBLE_DEVICES= $PLAN timing-record --resources $RES --timing-protocol $ROOT/protocols/timing.json \
  --output $ROOT/decisions/timing_record.json
CUDA_VISIBLE_DEVICES= $PLAN register-tuning --resources $RES --prepared $ROOT/inputs/prepared.json \
  --design $ROOT/design_20260919.json --timing-protocol $ROOT/protocols/timing.json \
  --timing-record $ROOT/decisions/timing_record.json --authorization "<author's words>" \
  --output $ROOT/protocols/tuning.json
CUDA_VISIBLE_DEVICES= $PLAN select-lr --resources $RES --tuning-protocol $ROOT/protocols/tuning.json \
  --output $ROOT/decisions/lr.json
CUDA_VISIBLE_DEVICES= $PLAN scope-decision --resources $RES --timing-record $ROOT/decisions/timing_record.json \
  --design $ROOT/design_20260919.json --hours-available-on-two-gpus <H> \
  --generation-seconds-per-state <S> --overhead-gpu-hours <O> --authorization "<author's words>" \
  --output $ROOT/decisions/scope.json
CUDA_VISIBLE_DEVICES= $PLAN generation-audit --resources $RES --design $ROOT/design_20260919.json \
  --generation-export <each selection_generation.json> --output $ROOT/decisions/generation_audit.json
CUDA_VISIBLE_DEVICES= $PLAN register-confirmation --resources $RES \
  --tuning-protocol $ROOT/protocols/tuning.json --lr-decision $ROOT/decisions/lr.json \
  --scope-decision $ROOT/decisions/scope.json --generation-audit $ROOT/decisions/generation_audit.json \
  --authorization "<author's words>" --output $ROOT/protocols/confirmation.json
CUDA_VISIBLE_DEVICES=$UUID $RUN reference --resources $RES --gpu <GPU> --protocol $ROOT/protocols/confirmation.json
```

## 7. Resource record

`data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_SUBSPACE.json` names **GPUs 2
and 3**, verified free of other users' compute processes before launch. GPUs 0
and 1 remain another user's and are untouched. The record sets
`completion_authorized = false`: it authorizes the two bounded timing pilots
only. The tuning grid and the confirmation queue require the measured timing
record, the frozen scope decision and a further author authorization. The runner
re-checks at launch that the named GPU carries no foreign process and refuses
otherwise.
