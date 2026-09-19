#!/bin/bash
# Publish a decoder-subspace status snapshot to both repositories every 45 minutes
# until the confirmation population is complete. Read-only with respect to runs.
CHECKOUT=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
cd "$CHECKOUT" || exit 1
while true; do
  CUDA_VISIBLE_DEVICES= .venv/bin/python notebooks/iclr/handoff/ops/subspace_status.py --push >> campaign_outputs_decoder_subspace_v1/logs/status_daemon.log 2>&1
  DONE=$(python3 -c "import json;s=json.load(open('campaign_outputs_decoder_subspace_v1/STATUS.json'));print(s['completed_validated'])" 2>/dev/null || echo 0)
  POP=$(python3 -c "import json;s=json.load(open('campaign_outputs_decoder_subspace_v1/STATUS.json'));print(s['population'])" 2>/dev/null || echo 18)
  if [ "$DONE" -ge "$POP" ]; then
    echo "$(date -u +%FT%TZ) population complete, daemon exiting" >> campaign_outputs_decoder_subspace_v1/logs/status_daemon.log
    break
  fi
  sleep 2700
done
