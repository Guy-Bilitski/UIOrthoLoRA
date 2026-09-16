#!/bin/bash
# Overnight watcher covering BOTH output roots (calibration + confirmation).
# Emits: new ledger events, supervisor alerts/stop-reasons, first-optimizer-step
# per run, stale progress, lane session changes, per-root storage-cap warnings.
CAMP=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
STATE=/tmp/claude-1036/-media-eimtest-data-guyb-UIOrthoLoRA-notebooks-iclr/cb615d55-7c74-42b2-a5d6-72c3f4ecb207/scratchpad/watch_state
mkdir -p "$STATE"
cd "$CAMP" || exit 1
declare -A LAST CAP
CAP[campaign_outputs_v1]=48318382080          # warn near 45 GiB of the 50 cap
CAP[campaign_outputs_confirmation_v1]=144955146240   # warn near 135 GiB of the 150 cap
for root in campaign_outputs_v1 campaign_outputs_confirmation_v1; do
  LAST[$root]=$(wc -l < "$root/run_ledger.jsonl" 2>/dev/null || echo 0)
done
prev_sessions=""
while true; do
  for root in campaign_outputs_v1 campaign_outputs_confirmation_v1; do
    ledger=$root/run_ledger.jsonl
    [ -f "$ledger" ] || continue
    cur=$(wc -l < "$ledger" 2>/dev/null || echo "${LAST[$root]}")
    if [ "$cur" -gt "${LAST[$root]}" ]; then
      tail -n +$((LAST[$root]+1)) "$ledger" | jq -rc --arg root "$root" '"LEDGER(\($root)): \(.status // "?") \(.run_id // "?") \(.condition // "") \(.task // "") seed=\(.seed // "")"' 2>/dev/null
      LAST[$root]=$cur
    fi
    bytes=$(du -sb "$root" 2>/dev/null | cut -f1)
    if [ -n "$bytes" ] && [ "$bytes" -gt "${CAP[$root]}" ] && [ ! -f "$STATE/storage_$root" ]; then
      echo "STORAGE WARNING: $root at $((bytes/1073741824)) GiB, approaching its cap"
      touch "$STATE/storage_$root"
    fi
  done
  while IFS= read -r m; do
    rid=$(basename "$(dirname "$m")")
    a=$(tail -1 "$m" | jq -rc 'select(((.alerts|length)>0) or ((.stop_reasons|length)>0)) | "alerts=\(.alerts) stop=\(.stop_reasons)"' 2>/dev/null)
    if [ -n "$a" ]; then
      h=$(echo "$a" | md5sum | cut -d' ' -f1)
      if [ "$(cat "$STATE/alert_$rid" 2>/dev/null)" != "$h" ]; then
        echo "ALERT $rid: $a"
        echo "$h" > "$STATE/alert_$rid"
      fi
    fi
    if [ ! -f "$STATE/step_$rid" ]; then
      st=$(tail -1 "$m" | jq -r 'if (.latest_step|type)=="object" then .latest_step.step else .latest_step // empty end' 2>/dev/null)
      if [ -n "$st" ] && [ "$st" != "null" ]; then
        echo "TRAINING STARTED: $rid (step $st)"
        touch "$STATE/step_$rid"
      fi
    fi
    age=$(tail -1 "$m" | jq -r '.progress_age_seconds // 0' 2>/dev/null)
    if [ -n "$age" ] && [ "${age%.*}" -gt 900 ] 2>/dev/null; then
      if [ ! -f "$STATE/stale_$rid" ]; then
        echo "STALE PROGRESS: $rid progress_age=${age}s"
        touch "$STATE/stale_$rid"
      fi
    else
      rm -f "$STATE/stale_$rid" 2>/dev/null
    fi
  done < <(find campaign_outputs_v1/runs campaign_outputs_confirmation_v1/runs -name monitor.jsonl -mmin -6 2>/dev/null)
  s=$(tmux -L iclr_6aa54397_20260914 ls 2>/dev/null | cut -d: -f1 | sort | tr '\n' ' ')
  if [ "$s" != "$prev_sessions" ]; then
    [ -n "$prev_sessions" ] && echo "LANE SESSIONS CHANGED: now [${s:-none}]"
    prev_sessions="$s"
  fi
  sleep 300
done
