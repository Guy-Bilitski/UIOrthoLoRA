#!/bin/bash
# Block 1's last GPU step, then hand the card to block 2.
#   1. wait until all 18 band confirmations are completed and validated
#   2. take the first free assigned card and run the exploratory NLL partition over all 19 states
#   3. start a block 2 calibration lane on that same card
# Failures stop the chain and are reported; nothing is skipped silently.
set -u
CHECKOUT=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
BAND=$CHECKOUT/campaign_outputs_decoder_subspace_v1
INTER=$CHECKOUT/campaign_outputs_decoder_interaction_v1
BAND_RES=$CHECKOUT/notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_SUBSPACE.json
INTER_RES=$CHECKOUT/notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_INTERACTION.json
LOG=$BAND/logs/block1_finish.log
cd "$CHECKOUT"
mkdir -p "$BAND/logs" "$INTER/logs"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }

say "waiting for 18/18 band confirmations"
while true; do
  done_count=$(CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python -c "
from notebooks.iclr.decoder_pilot import subspace_plan as sp
R='$BAND'
rows=sp.collect_runs(R+'/run_ledger.jsonl', R+'/protocols/confirmation.json', purpose=sp.CONFIRMATION_PURPOSE)
print(sum(1 for r in rows if r.get('status')=='completed'))
" 2>>"$LOG")
  case "$done_count" in ''|*[!0-9]*) say "could not read the ledger, retrying"; sleep 120; continue;; esac
  [ "$done_count" -ge 18 ] && { say "18/18 complete"; break; }
  sleep 120
done

# A card counts as free only when it is free on two checks 150 s apart. The block 2 watcher may be
# claiming the other card at this moment, and a lane needs about a minute to allocate; a single
# snapshot would let both schedulers pick the same card and push it into an out-of-memory failure.
say "waiting for a free assigned card, confirmed twice"
free_now() {
  local g="$1" used procs
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  procs=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c .)
  [ "$procs" -eq 0 ] && [ "$used" -lt 1024 ]
}
GPU=""
while [ -z "$GPU" ]; do
  for g in 3 2; do
    if free_now "$g"; then
      say "GPU $g looks free, re-checking in 150 s"
      sleep 150
      if free_now "$g"; then GPU="$g"; break; fi
      say "GPU $g was taken in between, looking again"
    fi
  done
  [ -z "$GPU" ] && sleep 60
done
say "using GPU $GPU for the loss partition"

CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader) PYTHONPATH=.:src \
  .venv/bin/python notebooks/iclr/handoff/ops/loss_partition_all.py \
    --root "$BAND" --resources "$BAND_RES" --gpu "$GPU" --expected 19 >> "$LOG" 2>&1 \
  || { say "PARTITION FAILED, chain stops; see $LOG"; exit 1; }
say "loss partition complete over 19 states"

say "starting the block 2 calibration lane on GPU $GPU"
nohup bash notebooks/iclr/handoff/ops/interaction_lane.sh "$GPU" "$INTER/protocols/calibration.json" \
  "$INTER/queues_calibration.txt" >> "$INTER/logs/chain_lane_${GPU}.log" 2>&1 &
say "chain done"
