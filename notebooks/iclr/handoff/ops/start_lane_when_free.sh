#!/bin/bash
# Start one lane per assigned GPU as soon as that card is genuinely free.
# "Free" means: no compute process of ANY user on it, and under 1 GiB resident. The runner
# re-checks foreign occupancy itself, so this is a scheduling courtesy, not the safety check.
# Usage: MAX_LANES=n start_lane_when_free.sh <lane script> <protocol> <queue> <gpu> [gpu ...]
# MAX_LANES caps how many lanes this watcher starts, so a card can be held back for other work.
set -u
LANE="$1"; PROTOCOL="$2"; QUEUE="$3"; shift 3
CHECKOUT=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
case "$QUEUE" in /*) QUEUE_ABS="$QUEUE";; *) QUEUE_ABS="$CHECKOUT/$QUEUE";; esac
LOGDIR="$(dirname "$QUEUE_ABS")/logs"; mkdir -p "$LOGDIR"
declare -A STARTED
MAX_LANES="${MAX_LANES:-$#}"; started_count=0
while true; do
  # grep -c prints 0 AND exits 1 on no match, so `|| echo 0` would emit two lines. Take the print.
  remaining=$(grep -c . "$QUEUE" 2>/dev/null); remaining=${remaining:-0}
  pending=0
  for gpu in "$@"; do [ -z "${STARTED[$gpu]:-}" ] && pending=1; done
  [ "$started_count" -ge "$MAX_LANES" ] && { echo "$(date -u +%H:%M:%SZ) started $started_count lane(s), the cap; watcher exits"; break; }
  [ "$remaining" -eq 0 ] && { echo "$(date -u +%H:%M:%SZ) queue drained, watcher exits"; break; }
  [ "$pending" -eq 0 ] && { echo "$(date -u +%H:%M:%SZ) every assigned lane started, watcher exits"; break; }
  for gpu in "$@"; do
    [ -n "${STARTED[$gpu]:-}" ] && continue
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    procs=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -c . )
    if [ "$procs" -eq 0 ] && [ "$used" -lt 1024 ]; then
      echo "$(date -u +%H:%M:%SZ) GPU $gpu free (${used} MiB), starting lane"
      nohup bash "$CHECKOUT/$LANE" "$gpu" "$PROTOCOL" "$QUEUE" >> "$LOGDIR/watcher_lane_${gpu}.log" 2>&1 &
      STARTED[$gpu]=1; started_count=$((started_count + 1))
      [ "$started_count" -ge "$MAX_LANES" ] && break
      sleep 90   # let the lane claim its entry and allocate before judging the next card
    fi
  done
  sleep 60
done
