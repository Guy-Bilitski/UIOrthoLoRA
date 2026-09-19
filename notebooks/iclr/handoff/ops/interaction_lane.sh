#!/bin/bash
# One GPU lane for the decoder interaction-control study (block 2). Claims entries from a shared queue under flock,
# runs each registered entry, then validates and completes it. Refuses a GPU carrying a foreign
# process (the runner re-checks this too) and stops at the first unexplained failure.
set -u
GPU="$1"; PROTOCOL="$2"; QUEUE="$3"
CHECKOUT=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
ROOT="$CHECKOUT/campaign_outputs_decoder_interaction_v1"
RES="$CHECKOUT/notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_INTERACTION.json"
LOGS="$ROOT/logs"; STATUS="$LOGS/lane_${GPU}.status"; LOCK="$QUEUE.lock"
mkdir -p "$LOGS"; touch "$LOCK"
cd "$CHECKOUT"
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
while true; do
  ENTRY=$(flock "$LOCK" bash -c "head -n1 '$QUEUE'; sed -i '1d' '$QUEUE'")
  [ -z "$ENTRY" ] && { echo "$(date -u +%H:%M:%SZ) queue empty, lane $GPU done" >> "$STATUS"; break; }
  SAFE=$(echo "$ENTRY" | tr '/' '_')
  OUT="$LOGS/$SAFE.log"
  echo "$(date -u +%H:%M:%SZ) START $ENTRY" >> "$STATUS"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$UUID" PYTHONPATH=src \
    .venv/bin/python -m notebooks.iclr.decoder_pilot.interaction_runner train \
      --resources "$RES" --gpu "$GPU" --protocol "$PROTOCOL" --entry-id "$ENTRY" --eval-batch-size 4 > "$OUT" 2>&1
  if [ $? -ne 0 ]; then echo "$(date -u +%H:%M:%SZ) RUN-FAILED $ENTRY see $OUT" >> "$STATUS"; break; fi
  RUNDIR=$(grep -o '"run_dir": "[^"]*"' "$OUT" | head -1 | cut -d'"' -f4)
  [ -z "$RUNDIR" ] && { echo "$(date -u +%H:%M:%SZ) NO-RUNDIR $ENTRY" >> "$STATUS"; break; }
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.interaction_plan \
      validate-run --resources "$RES" --run-directory "$RUNDIR" --protocol "$PROTOCOL" > "$OUT.validate" 2>&1 \
    && CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.interaction_plan \
      complete --resources "$RES" --run-id "$(basename "$RUNDIR")" \
      --validation-report "$RUNDIR/validation_report.json" >> "$OUT.validate" 2>&1 \
    || { echo "$(date -u +%H:%M:%SZ) VALIDATE-FAILED $ENTRY see $OUT.validate" >> "$STATUS"; break; }
  echo "$(date -u +%H:%M:%SZ) DONE $ENTRY $(basename "$RUNDIR")" >> "$STATUS"
done
