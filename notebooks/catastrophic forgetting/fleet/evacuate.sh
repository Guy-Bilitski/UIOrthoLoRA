#!/bin/bash
# ============================== FLEET EVACUATION ==============================
# One-shot, ~60-minute hard-capped shutdown + harvest before the fleet is wiped.
#
#   bash fleet/evacuate.sh --dry-run   # triage + sync preview, kills nothing
#   bash fleet/evacuate.sh --launch    # the real thing (detach with setsid nohup)
#
# Policy (user-specified 2026-07-16):
#   * ALL training stops immediately (a train's adapter is useless without its eval,
#     which never fits in the window). CE/geo/derive batteries stop immediately
#     (restartable / partial rows already synced).
#   * EVALS with <= GRACE_MIN minutes remaining (last tqdm ETA in their log) are
#     GRACED: monitored to completion, their node re-synced after they finish.
#     CAVEAT: the ETA is the CURRENT tqdm stage only — a multi-stage eval can enter
#     another stage after the graced one; the T+KILL_AT_MIN hard deadline still
#     bounds it, so the 60-minute guarantee holds regardless.
#   * Hard deadline: at T+KILL_AT_MIN everything still alive is killed; by T+60
#     every GPU is idle and every result is on d001 AND git-pushed off-machine.
#
# What is harvested (priority order):
#   1. results/ from every node (summaries, geo.json, forgetting.json, geo_drift)
#      + per-node registries -> results/fleet_reg/<node>.*   [CRITICAL, git-pushed]
#   2. merged aggregates rebuilt from per-run files (fleet/evac_merge.py) [git-pushed]
#   3. /scratch/cf_models/dsv4_* adapters (284B LoRAs, small, expensive) -> d001
#   4. node logs -> results/evac_logs/<node>/ (<=50MB files)  [best-effort, NOT pushed]
#   5. 7B adapters -> /scratch/cf_models_evac/<node>/          [best-effort, time gated]
# NOTE: only git-PUSHED content (1+2) survives if d001 itself is wiped. Adapters and
# logs land on d001 only.
# ==============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.." || exit 1
WD="/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
V=/home/guy/UIOrthoLoRA/.venv/bin/python
NODES=$(grep -v '^#' "$HERE/nodes.txt")
GRACE_MIN=30
KILL_AT_MIN=50        # hard-kill graced stragglers here; leaves 10 min for final sync+push
T0=$(date +%s)
LOG="logs/evacuate_$(date -u +%Y%m%dT%H%MZ).log"
mkdir -p logs results/fleet_reg results/evac_logs /scratch/cf_models_evac
GRACEFILE=$(mktemp)

DRY=1
case "${1:-}" in
  --launch) DRY=0 ;;
  --dry-run|"") DRY=1 ;;
  *) echo "usage: evacuate.sh [--dry-run|--launch]"; exit 2 ;;
esac

log(){ echo "[evac +$((($(date +%s)-T0)/60))m $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
mins_left(){ echo $(( KILL_AT_MIN - ($(date +%s)-T0)/60 )); }
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"

log "=== EVACUATION $( [ $DRY -eq 1 ] && echo DRY-RUN || echo LAUNCH ) — grace<=${GRACE_MIN}m, hard-kill T+${KILL_AT_MIN}m ==="

# -------- Phase A: freeze all orchestration (nothing may start new work) --------
log "Phase A: freeze orchestration"
if [ $DRY -eq 0 ]; then
  pkill -TERM -f "guardian_[l]oop"        2>/dev/null
  pkill -TERM -f "collect_[l]oop"         2>/dev/null
  pkill -TERM -f "derive_[s]upervisor"    2>/dev/null
  pkill -TERM -f "derive_[l]oop"          2>/dev/null
  pkill -TERM -f "auto_[d]ispatch"        2>/dev/null
  pkill -TERM -f "gpu_[w]atchdog"         2>/dev/null
  pkill -TERM -f "launch[8]"              2>/dev/null
  pkill -TERM -f "results_book_[l]oop"    2>/dev/null
  for n in $NODES; do
    $SSH ubuntu@$n 'pkill -TERM -f "auto_[d]ispatch"; pkill -TERM -f "gpu_[w]atchdog";
                    pkill -TERM -f "derive_[l]oop"; pkill -TERM -f "run_[n]ode.sh";
                    pkill -TERM -f "finalize_[n]ode"; pkill -TERM -f "stage_[n]ode"; true' 2>/dev/null &
  done; wait
  log "orchestration frozen (d001 loops, node dispatchers/watchdogs/derive/run_node/finalize)"
else
  log "(dry) would kill: guardian/collect/derive/dispatchers/watchdogs/run_node/finalize/launch8"
fi

# -------- Phase B: triage every GPU process (d001 + all nodes) --------
# Remote classifier: for each compute PID -> KILL or GRACE (eval with tqdm ETA <= GRACE_MIN).
TRIAGE='
GRACE_S='"$((GRACE_MIN*60))"'
cd "'"$WD"'" 2>/dev/null || exit 0
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
  cmd=$(tr "\0" " " </proc/$pid/cmdline 2>/dev/null); [ -n "$cmd" ] || continue
  rn=$(echo "$cmd" | grep -oP "(?<=--run_name )\S+" | head -1)
  case "$cmd" in
    *eval_one_gpu.py*|*eval_deepseek.py*)
      lg=$(ls -t logs/*"${rn:-zzz}"*.log 2>/dev/null | head -1)
      [ -z "$lg" ] && [ -n "$rn" ] && lg=$(grep -sl -- "$rn" logs/*.log 2>/dev/null | head -1)
      rem=""
      if [ -n "$lg" ]; then
        eta=$(tail -c 6000 "$lg" | tr "\r" "\n" | grep -oE "<[0-9]{1,2}:[0-9]{2}(:[0-9]{2})?" | tail -1 | tr -d "<")
        if [ -n "$eta" ]; then
          IFS=: read -r a b c <<<"$eta"
          if [ -n "${c:-}" ]; then rem=$((10#$a*3600+10#$b*60+10#$c)); else rem=$((10#$a*60+10#$b)); fi
        fi
      fi
      if [ -n "$rem" ] && [ "$rem" -le "$GRACE_S" ]; then echo "GRACE $pid ${rem}s $rn"
      else echo "KILL $pid eval_eta=${rem:-unknown}s $rn"; fi ;;
    *) echo "KILL $pid non-eval ${rn:-${cmd:0:60}}" ;;
  esac
done'

log "Phase B: triage GPU processes"
for n in d001 $NODES; do
  if [ "$n" = d001 ]; then out=$(bash -c "$TRIAGE" 2>/dev/null); else out=$(timeout 30 $SSH ubuntu@$n "$TRIAGE" 2>/dev/null); fi
  [ -n "$out" ] || continue
  while IFS= read -r line; do
    log "  $n: $line"
    verdict=$(awk "{print \$1}" <<<"$line"); pid=$(awk "{print \$2}" <<<"$line"); rn=$(awk "{print \$NF}" <<<"$line")
    if [ "$verdict" = GRACE ]; then
      echo "$n $pid $rn" >> "$GRACEFILE"
    elif [ $DRY -eq 0 ]; then
      if [ "$n" = d001 ]; then
        [ -n "$rn" ] && pkill -TERM -f -- "--run_name $rn" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
      else
        $SSH ubuntu@$n "[ -n '$rn' ] && pkill -TERM -f -- '--run_name $rn' 2>/dev/null; kill -TERM $pid 2>/dev/null; true" 2>/dev/null
      fi
    fi
  done <<< "$out"
done
NG=$(wc -l < "$GRACEFILE" 2>/dev/null || echo 0)
log "triage done: $NG graced eval(s): $(awk '{printf "%s(%s) ", $1, $3}' "$GRACEFILE" 2>/dev/null)"
# -------- Phase C: sync sweep #1 (critical data first) --------
sync_node(){  # sync_node <node>  -> results + registries + dsv4 adapters + logs
  local n=$1 RS=""
  [ $DRY -eq 1 ] && RS="--dry-run"
  timeout 300 rsync -a --update $RS \
      --exclude 'dispatch_locks/' --exclude 'ce_locks/' --exclude 'fleet_reg/' --exclude 'evac_logs/' \
      "ubuntu@$n:$WD/results/" results/ 2>/dev/null
  for f in campaign_summary.jsonl train_registry.jsonl forgetting.jsonl; do
    timeout 60 rsync -a $RS "ubuntu@$n:$WD/results/$f" "results/fleet_reg/$n.$f" 2>/dev/null
  done
  timeout 240 rsync -a --update $RS "ubuntu@$n:/scratch/cf_models/dsv4_*" /scratch/cf_models/ 2>/dev/null
  mkdir -p "results/evac_logs/$n"
  timeout 120 rsync -a --update $RS --max-size=50m "ubuntu@$n:$WD/logs/" "results/evac_logs/$n/" 2>/dev/null
}
log "Phase C: sync sweep #1 (results+registries+dsv4-adapters+logs, all nodes, parallel)"
for n in $NODES; do sync_node "$n" & done; wait
log "sweep #1 done: $(ls results/*/summary.json 2>/dev/null | wc -l) summaries on d001"

# -------- Phase D: babysit graced evals; resync their node as each finishes --------
if [ -s "$GRACEFILE" ] && [ $DRY -eq 0 ]; then
  log "Phase D: monitoring $NG graced eval(s) until done or T+${KILL_AT_MIN}m"
  while [ -s "$GRACEFILE" ] && [ "$(mins_left)" -gt 0 ]; do
    sleep 60
    NEXT=$(mktemp)
    while read -r n pid rn; do
      if [ "$n" = d001 ]; then alive=$(kill -0 "$pid" 2>/dev/null && echo 1 || echo 0)
      else alive=$($SSH ubuntu@$n "kill -0 $pid 2>/dev/null && echo 1 || echo 0" 2>/dev/null); fi
      if [ "${alive:-0}" = 1 ]; then echo "$n $pid $rn" >> "$NEXT"
      else log "  graced eval FINISHED: $rn ($n) — resyncing node"; [ "$n" = d001 ] || sync_node "$n"; fi
    done < "$GRACEFILE"
    mv "$NEXT" "$GRACEFILE"
  done
  if [ -s "$GRACEFILE" ]; then
    log "Phase D: T+${KILL_AT_MIN}m deadline — killing remaining graced evals: $(cat "$GRACEFILE" | tr '\n' ' ')"
    while read -r n pid rn; do
      if [ "$n" = d001 ]; then pkill -KILL -f -- "--run_name $rn" 2>/dev/null
      else $SSH ubuntu@$n "pkill -KILL -f -- '--run_name $rn'; true" 2>/dev/null; fi
    done < "$GRACEFILE"
  fi
else
  log "Phase D: no graced evals (or dry-run) — skipping"
fi

# -------- Phase E: hard-stop everything left, final sweep, merge, push, verify --------
log "Phase E: final hard-stop + sweep + merge + push"
if [ $DRY -eq 0 ]; then
  for n in d001 $NODES; do
    HK='for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -KILL $pid 2>/dev/null; done; true'
    if [ "$n" = d001 ]; then bash -c "$HK"; else $SSH ubuntu@$n "$HK" 2>/dev/null & fi
  done; wait
  sleep 5
fi
for n in $NODES; do sync_node "$n" & done; wait
log "final sweep done"

if [ $DRY -eq 0 ]; then
  $V fleet/evac_merge.py 2>&1 | tee -a "$LOG"
  $V fleet/data_sanity.py 2>/dev/null | head -1 | tee -a "$LOG"
  git config user.email >/dev/null 2>&1 || git config user.email "guyb@sdsai.ai"
  git config user.name  >/dev/null 2>&1 || git config user.name  "campaign-bot"
  git add results/ 2>/dev/null
  git reset -q -- results/dispatch_locks results/ce_locks 2>/dev/null
  git commit -q -m "EVACUATION harvest: final fleet sync $(date -u +%Y-%m-%dT%H:%MZ)" 2>/dev/null
  ok=0
  for i in 1 2 3; do timeout 120 git push -q origin ortho_new 2>/dev/null && { ok=1; break; }; sleep 10; done
  if [ $ok -eq 1 ] && [ "$(git rev-parse @)" = "$(git rev-parse @{u} 2>/dev/null)" ]; then
    log "git push VERIFIED (HEAD == origin/ortho_new @ $(git rev-parse --short @))"
  else
    log "*** GIT PUSH FAILED — DATA IS NOT OFF-MACHINE. RETRY MANUALLY: git push origin ortho_new ***"
  fi
fi

# -------- verification: every GPU idle, second rsync pass transfers nothing --------
log "verification:"
FAIL=0
for n in $NODES; do
  c=$(timeout 25 $SSH ubuntu@$n 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l' 2>/dev/null)
  left=$(timeout 90 rsync -a --update --dry-run --exclude 'dispatch_locks/' --exclude 'ce_locks/' --exclude 'fleet_reg/' --exclude 'evac_logs/' \
          "ubuntu@$n:$WD/results/" results/ 2>/dev/null | wc -l)
  if [ "${c:-9}" = 0 ] && [ "${left:-9}" -le 4 ]; then log "  $n CLEAN (gpu_procs=0, unsynced=$left)"
  else log "  $n *** gpu_procs=${c:-?} unsynced_paths=${left:-?} ***"; FAIL=1; fi
done
c=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
[ "$c" = 0 ] || { log "  d001 *** gpu_procs=$c ***"; [ $DRY -eq 0 ] && FAIL=1; }
log "manifest: $(ls results/*/summary.json 2>/dev/null | wc -l) summaries | geo_merged=$(wc -l < results/geo_drift/adapter_metrics_merged.jsonl 2>/dev/null) | ce_merged=$(wc -l < results/forgetting_merged.jsonl 2>/dev/null) | dsv4_adapters=$(ls -d /scratch/cf_models/dsv4_* 2>/dev/null | wc -l) | HEAD=$(git rev-parse --short HEAD)"
if [ $FAIL -eq 0 ]; then log "=== EVACUATION COMPLETE — fleet is idle and harvested; safe to wipe ==="
else log "=== EVACUATION FINISHED WITH WARNINGS — check *** lines above before wiping ==="; fi

# -------- best-effort tail: 7B adapters, only with time to spare --------
if [ $DRY -eq 0 ] && [ "$(( ($(date +%s)-T0)/60 ))" -lt 55 ]; then
  log "time remains — best-effort 7B adapter pull to /scratch/cf_models_evac/<node>/ (background)"
  for n in $NODES; do
    ( mkdir -p "/scratch/cf_models_evac/$n"
      timeout 600 rsync -a --update --exclude 'dsv4_*' "ubuntu@$n:/scratch/cf_models/" "/scratch/cf_models_evac/$n/" 2>/dev/null ) &
  done
  # intentionally not waited on: pure bonus, dies harmlessly if the window closes
fi
exit $FAIL
