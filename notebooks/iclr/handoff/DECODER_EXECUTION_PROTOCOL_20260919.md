# Decoder study — executable protocol and cost (19 September 2026)

Companion to `DECODER_FIRST_PRIORITY_20260919.md` (order) and
`DECODER_PILOT_DESIGN_20260919.md` (design). This document is what an operator
runs. The registration/admission layer it depends on is now implemented and
tested; **no GPU run has happened** and none can start until two RTX 3090s are
free of other users' processes and an authorization record names them.

## 1. What changed in this session

`notebooks/iclr/decoder_pilot/plan.py` (new, 1,300 lines) is the
registration/admission layer the design named as the first coding step. It
mirrors `campaign/center_plan.py` and adds what the decoder needs:

| Piece | Behavior |
|---|---|
| `design(...)` | Keyword-only, **no default for any of the five open decisions** (model variant, projection set, tail size, MIX dose/bracket, fallback policy) plus the recipe knobs that must be pinned. A protocol cannot be sealed until the author fixes them, and `decision_provenance` must name the record that did. |
| Stage 1 `register_tuning` / `select_learning_rates` | Per-family learning-rate grids at seed 31415, one probe arm per family (UNREG for the spectral family, LoRA, PiSSA). Decided on inner-selection token NLL of the fixed endpoint **only**; exact match, generated text, geometry and the held-aside split are not read. Ties break to the smaller rate. The complete grid of a family must validate first. |
| Stage 2 `register_calibration` / `select_norm` / `register_norm_refinement` | The MIX endpoint at the registered dose defines the target pooled relative Frobenius norm; NORM x {1e-2, 1, 100} matches it within 5%, with at most two rule-derived refinements by the same midpoint/extension rule as CENTER, coefficient bounded to [1e-4, 1e4]. Norms only. A failed match is retained and labeled, never retuned. |
| Stage 3 `register_confirmation` | 5 arms x seeds (42, 17, 123) = 15 runs at the frozen rates and dose, held-aside generation once per run at the fixed endpoint. Registration re-derives both decisions from the current validated ledger and refuses if they disagree. |
| Pretrained reference | `reference_entry` / `validate_reference_admission` and `pilot.py reference` decode the sealed model **with no adapter and no training** on the same splits, prompts, decoding budget and scorer. It is a reference row, not an arm, and `_resolve_attempts` drops it from every decision frontier. |
| Outcome policy | Every design and confirmation protocol carries `primary_outcome = generated_answer_exact_match`, the supporting-measurement list, and the statement that lower token loss is not by itself evidence of better reasoning. Admission rejects a confirmation protocol that omits it. |
| Ledger and validation | Append-only locked ledger with the campaign's state machine; `validate_run` writes a whole-run report (P0 equivalence, both reload checks, the unique fixed endpoint, artifact hashes) and completion is refused if any validated artifact changed afterwards. |
| `pilot.py train` | Now takes `--protocol` and `--entry-id`. A registered entry fixes every scientific field: the corresponding command-line flags must be absent, the assembled job is compared field by field with `plan.materialize`, and `plan.validate_admission` runs before the model loads. Without a protocol only `--stage pilot` is possible, it needs `--acknowledge-unregistered`, and it is written `selection_allowed=false` so no decision function will ever read it. |

Tests: `notebooks/iclr/decoder_pilot/tests/test_decoder_plan.py`, 35 CPU tests
(decision requirements, admission and tampering, both selection rules, tie
breaks, incomplete grids, duplicate completed attempts, refinement direction and
bounds and cap, refinement chaining, 15-entry confirmation matrix, reference
admission, cross-stage protocol borrowing, ledger state machine, whole-run
validation round trip, artifact tampering). Full suite **211 passed**; the CPU
preflight passes for the committed source.

## 2. Frozen against the researcher's brief

Shared across every arm and the reference: one model revision, one split seed,
one prompt and system prompt, completion-only masking, 56 adapted q/o
projections, effective batch 16, max length 640, 842 optimizer steps, greedy
decoding with 320 new tokens, one scorer. Method-appropriate learning rates are
allowed only inside the declared, equal-sized grids above. **Primary outcome:
generated-answer exact match on the held-aside GSM8K test split.** Supporting:
held-out completion NLL, effective update geometry in the frozen pretrained
frame, trainable-parameter and adapter-storage counts, SVD setup time, peak
memory, step time and tokens/s, generation time. Parameter counts are reported,
not equated.

## 3. Run inventory

| Stage | Runs | Note |
|---|---:|---|
| Bounded GPU pilot (UNREG, 100 steps, selection decode) | 1 | unregistered, decides nothing, measures memory/throughput/learning |
| Pretrained reference decode (selection + held-aside) | 2 | inference only, no training |
| Learning-rate tuning (3 spectral + 3 LoRA + 3 PiSSA) | 9 | seed 31415, selection decode |
| Norm calibration (MIX target + NORM {1e-2, 1, 100}) | 4 | seed 31415, selection decode |
| Norm refinements | 0-2 | rule-derived, one per round |
| Confirmations (5 arms x 3 seeds) | 15 | held-aside decode at the fixed endpoint |
| **Total** | **31-33** | |

## 4. Cost and ETA

Per-run times below are the design's estimates (about 2.5 s per 16-example step
at roughly 300 tokens, about 15 min to decode 1,319 test questions at batch 32).
**They are estimates until the bounded pilot measures the real step time**, which
is exactly what the pilot is for and why its result is reported before the rest
is launched.

| Stage | Runs | Per run | GPU-hours | Wall on 2 GPUs |
|---|---:|---:|---:|---:|
| Bounded GPU pilot | 1 | 0.3 h | 0.3 | 0.3 h |
| Pretrained reference | 2 | 0.15 h | 0.3 | 0.2 h |
| Learning-rate tuning | 9 | 0.9 h | 8.1 | 4.1 h |
| Norm calibration | 4 | 0.9 h | 3.6 | 1.8 h |
| Norm refinements | 0-2 | 0.9 h | 0-1.8 | 0-0.9 h |
| Confirmations | 15 | 1.0 h | 15.0 | 7.5 h |
| **Total** | **31-33** | | **27.3-29.1** | **≈14 h** |

The three stages are strictly sequential, because each is registered only after
the previous decision record exists, and CPU validation and export sit between
them. Allowing about an hour of turnaround per gate and the whole-run
validation passes, the realistic figure is:

- **≈14 h of two-GPU compute**, and
- **≈18-20 h elapsed** from the moment two GPUs are released to the last
  confirmation finishing, if nothing needs a retry.

Expected memory is about 15 GiB per card (6.2 GiB float32 masters, 1.6 GiB SVD
frame, activations for 4 x 640 tokens); `--gradient-checkpointing` is available
if the pilot measures more.

Against the deadline: with the full paper at about 26 September and the last
48 hours (24-26 September) reserved for analysis and writing, GPU work must end
by the morning of 24 September. Starting the pilot any time up to about
22 September leaves the schedule intact. If the release slips past then, the
first scope reductions, frozen before any outcome is seen, are: drop the MIX
bracket (already empty), then the norm refinements, then the tuning grids to two
points each, then report two seeds instead of three and say so.

## 5. Commands

Everything below runs in the isolated checkout with the isolated environment.
`$RES` must be a real authorization naming two GPUs that carry **no other user's
process**; the template refuses to be used as-is.

```bash
cd /media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
ROOT=campaign_outputs_decoder_v1
HANDOFF=notebooks/iclr/handoff/data/campaign_v1
RES=$HANDOFF/RESOURCE_AUTHORIZATION_20260919_DECODER.json      # copied from the .template.json
DEC=$HANDOFF/DECODER_DECISIONS_20260919.json                   # copied from the .proposed.json
PY=".venv/bin/python"
UUID=$(nvidia-smi -i <GPU> --query-gpu=uuid --format=csv,noheader)

# 0. seal the author's decisions into a design record
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.plan design \
  --decisions $DEC --output $ROOT/design_20260919.json

# 1. bounded GPU pilot: memory, throughput, learning signal. Decides nothing.
CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.pilot train \
  --resources $RES --gpu <GPU> --stage pilot --acknowledge-unregistered \
  --arm UNREG --seed 31415 --lr 1e-3 --steps 100 --max-length 640 \
  --eval-every-steps 50 --generation-split selection
#    -> report step_seconds_median, tokens_per_second, training_peak_allocated/reserved,
#       the selection-NLL trend and the pilot's exact match BEFORE anything else is launched.

# 2. tuning (after the pilot's numbers are agreed)
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.plan register-tuning \
  --resources $RES --prepared $ROOT/inputs/prepared.json --design $ROOT/design_20260919.json \
  --authorization "<author's words>" --output $ROOT/protocols/tuning.json
for E in $(CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -c \
    "import json;print(' '.join(e['entry_id'] for e in json.load(open('$ROOT/protocols/tuning.json'))['entries']))"); do
  CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.pilot train \
    --resources $RES --gpu <GPU> --stage tuning --protocol $ROOT/protocols/tuning.json --entry-id "$E"
  # then, per finished run:
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.plan validate-run \
    --resources $RES --run-directory <run_dir> --protocol $ROOT/protocols/tuning.json
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.plan complete \
    --resources $RES --run-id <run_id> --validation-report <run_dir>/validation_report.json
done
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.plan select-lr \
  --resources $RES --tuning-protocol $ROOT/protocols/tuning.json --output $ROOT/decisions/lr.json

# 3. norm calibration (same run/validate/complete loop per entry)
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.plan register-calibration \
  --resources $RES --tuning-protocol $ROOT/protocols/tuning.json --lr-decision $ROOT/decisions/lr.json \
  --authorization "<author's words>" --output $ROOT/protocols/calibration.json
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.plan select-norm \
  --resources $RES --calibration-protocol $ROOT/protocols/calibration.json --output $ROOT/decisions/norm.json
# if match_status == unmatched_refinement_available:
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.plan refine-norm \
  --resources $RES --calibration-protocol $ROOT/protocols/calibration.json \
  --decision-record $ROOT/decisions/norm.json --output $ROOT/protocols/calibration_r1.json
# then re-run select-norm with --refinement-protocol $ROOT/protocols/calibration_r1.json

# 4. confirmations and the pretrained reference
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.plan register-confirmation \
  --resources $RES --calibration-protocol $ROOT/protocols/calibration.json \
  --decision-record $ROOT/decisions/norm.json --authorization "<author's words>" \
  --output $ROOT/protocols/confirmation.json
CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.pilot train \
  --resources $RES --gpu <GPU> --stage confirmation --protocol $ROOT/protocols/confirmation.json --entry-id "confirmation/<ARM>/seed_<S>"
for SPLIT in selection held_aside_test; do
  CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src $PY -m notebooks.iclr.decoder_pilot.pilot reference \
    --resources $RES --gpu <GPU> --protocol $ROOT/protocols/confirmation.json --split $SPLIT
done
```

## 6. Still the author's to decide

1. Confirm or edit `DECODER_DECISIONS_20260919.proposed.json`, in particular
   tail size 512 and MIX dose 1e-3, which the researcher's message did not fix.
   Nothing can be registered until this file is accepted as
   `DECODER_DECISIONS_20260919.json`.
2. Name the two GPU IDs in the authorization once two cards are actually free,
   and state the authorization text.
3. Approve the bounded pilot's measured cost before tuning, calibration and the
   confirmations are launched, as the brief requires.

At 2026-09-19 10:00 UTC all four RTX 3090s carried another user's processes
(three `train_contrastive.py` jobs at roughly 3-6 h elapsed and a vLLM engine at
about 46 h). A watcher is running and will report the moment two cards are free
of foreign processes; nothing will be started on a card that is not.
