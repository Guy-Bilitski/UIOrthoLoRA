#!/usr/bin/env bash
# Early NaN kill-switch (new file; the tested pipeline is untouched).
#
# Watches the per-cell training logs of a running gpu_pool tag and, the moment a
# log shows `'grad_norm': 'nan'` (or a zero loss with nan grad), kills that job's
# train_cs.py so the pool advances instead of training a dead model for hours.
# Written after 2026-08-27: a Qwen cell NaN'd at step 70 and kept training for
# 6 more hours before its (worthless) eval also ran.
#
# The kill makes the chain exit nonzero, so eval/CE/evacuation are skipped and
# the cell is simply not marked done — safe, idempotent, re-runnable.
#
# Usage: bash nan_watchdog.sh <tag> [poll_seconds]
#   e.g. setsid bash nan_watchdog.sh tierA1s2 60 &
set -u
cd "$(dirname "$0")"
TAG="${1:?usage: nan_watchdog.sh <pool_tag> [poll_seconds]}"
POLL="${2:-60}"
LOG="logs/nan_watchdog_${TAG}.log"
echo "[nanwd] $(date -Is) watching logs/${TAG}_*.log every ${POLL}s" >> "$LOG"

# ORPHAN SWEEP: a pool killed mid-flight leaves train_cs children alive holding
# the GPU (cost 3.5 GPU-h on 2026-08-28). If no pool with our tag is running but
# train_cs processes are, they are orphans -> kill them.
SEEN_POOL=0
orphan_sweep() {
  if pgrep -f "gpu_pool.py .*--tag ${TAG}" >/dev/null; then SEEN_POOL=1; return 0; fi
  # never sweep before our pool has actually been observed running, otherwise a
  # watchdog started ahead of its pool would kill unrelated training.
  [ "$SEEN_POOL" = 1 ] || return 0
  # CRITICAL (fixed 2026-08-28 after this killed a healthy job): if ANY gpu_pool is
  # running, its train_cs children are legitimate and must not be touched. Only
  # sweep when no pool of any tag is alive -- then a train_cs really is an orphan.
  if pgrep -f "gpu_pool.py" >/dev/null; then
    echo "[nanwd] $(date -Is) our pool is gone but another pool is running -- not sweeping; exiting" >> "$LOG"
    exit 0
  fi
  for p in $(pgrep -f "train_cs.py"); do
    echo "[nanwd] $(date -Is) ORPHAN train_cs $p (no ${TAG} pool running) -> kill -9" >> "$LOG"
    kill -9 "$p" 2>/dev/null
  done
}

while true; do
  orphan_sweep
  for f in logs/${TAG}_*.log; do
    [ -e "$f" ] || continue
    # only act on logs whose training is still live
    grep -q "'grad_norm': 'nan'" "$f" 2>/dev/null || continue
    marker="${f}.nankilled"
    [ -e "$marker" ] && continue
    cmd=$(head -1 "$f" | sed 's/^# CMD: //')
    run=$(echo "$cmd" | grep -oE -- "--run_name [^ ]+" | awk '{print $2}' | head -1)
    [ -z "$run" ] && continue
    pid=$(pgrep -f "train_cs.py .*--run_name ${run}( |$)" | head -1)
    if [ -n "$pid" ]; then
      echo "[nanwd] $(date -Is) NaN detected in $f (run $run) -> SIGKILL pid $pid" >> "$LOG"
      # SIGKILL, not SIGTERM: a graceful shutdown holds GPU memory for many seconds,
      # and gpu_pool starts the NEXT job immediately -> transient co-tenancy, which
      # is what causes the next Qwen cell to NaN (observed 2026-08-28, jobs 3->4).
      kill -9 "$pid" 2>/dev/null
      touch "$marker"
    else
      # training already finished/killed; just mark so we don't re-scan forever
      touch "$marker"
    fi
  done
  sleep "$POLL"
done
