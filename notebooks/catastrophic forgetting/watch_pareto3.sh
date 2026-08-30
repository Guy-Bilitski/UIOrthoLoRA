#!/usr/bin/env bash
# Health heartbeat for the overnight_pareto3 chain: one status line every 30 minutes,
# plus an immediate ALERT line whenever something looks wrong.
#
# Coverage note: silence must never be mistaken for success, so every tick emits a line
# regardless of state, and the alert conditions cover death and stall as well as NaN --
# a chain that has quietly stopped produces "ALL STOPPED", not silence.
set -u
cd "$(dirname "$0")"
PERIOD="${1:-1800}"
ORCH=logs/finalize_llama5.log
LAST_FAIL=0

running() { pgrep -f "$1" >/dev/null 2>&1; }

while true; do
  ts=$(date +%H:%M)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  gmem=${gpu%%,*}; gutil=${gpu##*,}

  # which configuration is training right now (run_name is the last arg train() passes)
  cur=$(pgrep -af "[t]rain_cs.py" | grep -oE 'tia1_frc_[a-z0-9_]+' | head -1)
  stage=""; detail=""
  if [ -n "$cur" ]; then
    f="logs/train_${cur}.log"
    prog=$(tr '\r' '\n' < "$f" 2>/dev/null | grep -oE '[0-9]+/31956 \[[0-9:]+<[0-9:]+' | tail -1)
    loss=$(grep -oE "'loss': '[0-9.]+'" "$f" 2>/dev/null | tail -1 | grep -oE '[0-9.]+')
    bad=$(grep -c "'grad_norm': 'nan'\|'grad_norm': 'inf'" "$f" 2>/dev/null); bad=${bad:-0}
    stage="TRAIN $cur"
    detail="${prog:-starting} loss=${loss:-?} nan/inf=${bad}"
    if [ "${bad:-0}" -gt 0 ]; then
      echo "[$ts] ALERT $cur has ${bad} nan/inf grad lines -- the retry loop will catch it, but check"
    fi
  elif pgrep -f "[g]pu_pool.py" >/dev/null; then
    # read the tag and job file from the live pool rather than hardcoding them --
    # hardcoded tags went stale at the 07:22 queue swap and produced a false ALL STOPPED
    pl=$(pgrep -af "[g]pu_pool.py" | head -1)
    tag=$(echo "$pl" | grep -oE '\-\-tag [A-Za-z0-9]+' | awk '{print $2}')
    jf=$(echo "$pl"  | grep -oE '\-\-jobs [^ ]+'        | awk '{print $2}')
    ev=$(pgrep -af "[e]val_one_gpu.py" | grep -oE '\-\-run_name [a-zA-Z0-9_]+' | head -1 | awk '{print $2}')
    done_n=$(grep -c "DONE" "logs/${tag}_pool.log" 2>/dev/null); done_n=${done_n:-0}
    ok_n=$(grep -c "rc=0" "logs/${tag}_pool.log" 2>/dev/null); ok_n=${ok_n:-0}
    tot_n=$(grep -c "^until" "$jf" 2>/dev/null); tot_n=${tot_n:-0}
    stage="EVAL[$tag] ${done_n}/${tot_n}"
    detail="ok=${ok_n} current=${ev:-<starting>}"
  elif pgrep -f "[f]inalize_llama5.sh|[f]inish_pareto3.sh" >/dev/null; then
    stage="CPU"; detail="between stages"
  elif [ "${gmem:-0}" -ge 2000 ]; then
    # something is using the card even though no orchestrator was recognised
    stage="BUSY (unrecognised)"; detail="gpu in use, no known pool"
  else
    stage="ALL STOPPED"; detail="no trainer, no eval pool, no orchestrator"
    echo "[$ts] ALERT chain is not running -- $(tail -1 $ORCH 2>/dev/null)"
  fi

  # any new FAILED / reject lines in the orchestrator log since the last tick
  nf=$(grep -c "FAILED\|reject:" "$ORCH" 2>/dev/null); nf=${nf:-0}
  if [ "$nf" -gt "$LAST_FAIL" ]; then
    grep "FAILED\|reject:" "$ORCH" | tail -n $((nf - LAST_FAIL)) | while read -r l; do
      echo "[$ts] ALERT $l"
    done
    LAST_FAIL=$nf
  fi

  # a live orchestrator with an idle card means something is wedged
  if [ "${stage:0:3}" != "ALL" ] && [ "$stage" != "CPU" ] && [ "${gmem:-0}" -lt 2000 ]; then
    echo "[$ts] ALERT GPU idle (${gmem} MiB) while stage is '$stage'"
  fi

  echo "[$ts] $stage | $detail | gpu ${gmem}MiB ${gutil}%"
  sleep "$PERIOD"
done
