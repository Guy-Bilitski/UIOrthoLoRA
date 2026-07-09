#!/bin/bash
# Evacuate FINISHED Qwen adapters from d002 (12G free, fills as keep-adapters queue
# accumulates) to Node A /scratch/cf_models (113G free, the source of truth).
# An adapter is safe to move only when its results/<run>/summary.json exists on B
# (eval done). Pull via tar-over-ssh (rsync absent), verify size locally, then delete
# the remote copy. Runs as a 30-min setsid loop. Logs to logs/evacuate.log.
set -uo pipefail
D="/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
DEST="/scratch/cf_models"
LOG="$D/logs/evacuate.log"
cd "$D" || exit 1

evac_once() {
  # list finished-but-still-present adapters on B (skip smoke)
  local list
  list=$(ssh -o ConnectTimeout=10 ubuntu@d002 '
    D="/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
    for r in "$D"/results/qw*/; do
      rn=$(basename "$r")
      [ -f "$D/results/$rn/summary.json" ] || continue
      ad="/home/ubuntu/cf_models/$rn"
      [ -d "$ad" ] && [ -f "$ad/adapter_config.json" -o -f "$ad/adapter_model.safetensors" ] && echo "$rn"
    done' 2>/dev/null)
  [ -z "$list" ] && { echo "$(date '+%F %H:%M') nothing to evacuate" >> "$LOG"; return 0; }
  local rn
  for rn in $list; do
    case "$rn" in *SMOKE*|*smoke*) continue;; esac
    [ -d "$DEST/$rn" ] && continue  # already evacuated
    # pull
    if ssh -o ConnectTimeout=10 ubuntu@d002 "cd /home/ubuntu/cf_models && tar czf - '$rn'" 2>/dev/null \
       | tar xzf - -C "$DEST" 2>/dev/null; then
      # verify non-trivial pull before deleting remote
      local sz
      sz=$(du -sm "$DEST/$rn" 2>/dev/null | cut -f1)
      if [ -n "$sz" ] && [ "$sz" -ge 10 ]; then
        ssh -o ConnectTimeout=10 ubuntu@d002 "rm -rf '/home/ubuntu/cf_models/$rn'" 2>/dev/null
        echo "$(date '+%F %H:%M') evacuated $rn (${sz}MB) and freed on B" >> "$LOG"
      else
        rm -rf "${DEST:?}/$rn"
        echo "$(date '+%F %H:%M') PULL-VERIFY FAILED for $rn (size=${sz:-0}MB) — remote kept" >> "$LOG"
      fi
    else
      echo "$(date '+%F %H:%M') tar pull failed for $rn — remote kept" >> "$LOG"
    fi
  done
  # report B disk after pass
  ssh -o ConnectTimeout=10 ubuntu@d002 'df -h /home | tail -1' 2>/dev/null \
    | awk -v d="$(date '+%F %H:%M')" '{print d" B /home now: "$5" used, "$4" free"}' >> "$LOG"
}

if [ "${1:-once}" = "loop" ]; then
  while true; do evac_once; sleep 1800; done
else
  evac_once
fi
