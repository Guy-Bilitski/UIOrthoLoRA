#!/bin/bash
# Run the frozen full-test reference as soon as the confirmation queue is empty and no runner is active.
# Waits rather than competing for memory, so it can never OOM a confirmation.
CHECKOUT=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
ROOT="$CHECKOUT/campaign_outputs_decoder_subspace_v1"
RES="$CHECKOUT/notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_SUBSPACE.json"
cd "$CHECKOUT" || exit 1
while true; do
  LEFT=$(wc -l < "$ROOT/confirmation_queue.txt" 2>/dev/null || echo 99)
  ACTIVE=$(pgrep -fc "subspace_runner train" 2>/dev/null || echo 0)
  if [ "$LEFT" -eq 0 ] && [ "$ACTIVE" -eq 0 ]; then
    for GPU in 2 3; do
      if [ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ]; then
        UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
        echo "$(date -u +%FT%TZ) starting frozen full-test reference on GPU $GPU" >> "$ROOT/logs/reference_wait.log"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src \
          .venv/bin/python -m notebooks.iclr.decoder_pilot.subspace_runner reference \
            --resources "$RES" --gpu $GPU --protocol "$ROOT/protocols/reference_held_aside.json" \
            --eval-batch-size 4 >> "$ROOT/logs/reference_held_aside.log" 2>&1
        echo "$(date -u +%FT%TZ) frozen full-test reference exited with $?" >> "$ROOT/logs/reference_wait.log"
        exit 0
      fi
    done
  fi
  sleep 300
done
