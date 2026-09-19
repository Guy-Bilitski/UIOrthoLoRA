#!/bin/bash
# Run the registered frozen full-test reference on one free GPU, validate and complete it, then hand that
# card back to the confirmation queue. Failures propagate; the lane is handed back either way so the card
# never sits idle. Uses no external hold file: the queue is the single source of remaining work.
set -u
GPU="$1"
CHECKOUT=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
ROOT="$CHECKOUT/campaign_outputs_decoder_subspace_v1"
RES="$CHECKOUT/notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_SUBSPACE.json"
LOG="$ROOT/logs/reference_slot.log"
cd "$CHECKOUT" || exit 1
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
hand_back() {
  nohup bash notebooks/iclr/handoff/ops/subspace_lane.sh "$GPU" "$ROOT/protocols/confirmation.json" \
    "$ROOT/confirmation_queue.txt" > "$ROOT/logs/lane${GPU}_resume.out" 2>&1 &
  say "GPU $GPU handed back to the confirmation queue"
}
if [ "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -c .)" -ne 0 ]; then
  say "ABORT: GPU $GPU is not free; refusing to share its memory"; exit 1
fi
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
say "frozen full-test reference starting on GPU $GPU"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src \
  .venv/bin/python -m notebooks.iclr.decoder_pilot.subspace_runner reference \
    --resources "$RES" --gpu "$GPU" --protocol "$ROOT/protocols/reference_held_aside.json" \
    --eval-batch-size 4 > "$ROOT/logs/reference_held_aside.log" 2>&1
STATUS=$?
if [ $STATUS -ne 0 ]; then say "FAILED: reference process exited $STATUS"; hand_back; exit $STATUS; fi
RUNDIR=$(grep -o '"run_dir": "[^"]*"' "$ROOT/logs/reference_held_aside.log" | tail -1 | cut -d'"' -f4)
if [ -z "$RUNDIR" ]; then say "FAILED: no run directory in the reference log"; hand_back; exit 1; fi
if CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.subspace_plan \
     validate-run --resources "$RES" --run-directory "$RUNDIR" --protocol "$ROOT/protocols/reference_held_aside.json" >> "$LOG" 2>&1 \
   && CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.subspace_plan \
     complete --resources "$RES" --run-id "$(basename "$RUNDIR")" --validation-report "$RUNDIR/validation_report.json" >> "$LOG" 2>&1; then
  say "frozen full-test reference completed and validated: $RUNDIR"
  hand_back; exit 0
fi
say "FAILED: reference validation or completion did not pass"
hand_back; exit 1
