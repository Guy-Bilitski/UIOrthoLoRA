#!/bin/bash
# Robustly (re)start dispatcher+watchdog on every fleet node that lacks a REAL one.
# Fix vs launch_all: detach with (setsid nohup ... &) subshell + a settle sleep so the
# process survives ssh channel close; verify a real python auto_dispatch proc afterward.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
REPO=/home/guyb/UIOrthoLoRA
WD="$REPO/notebooks/catastrophic forgetting"
VENV=/home/guy/UIOrthoLoRA/.venv/bin/python
NODES=$(grep -v '^#' "$HERE/ready_nodes.txt")

start_one(){
  local n="$1"
  # real dispatcher already? (match python cmdline, not the ssh line)
  local have
  have=$(ssh -o BatchMode=yes -o ConnectTimeout=8 ubuntu@"$n" 'c=0; for p in $(pgrep -x python; pgrep -x python3) ; do tr "\0" " " </proc/$p/cmdline 2>/dev/null | grep -q "auto_dispatch.py --jobs jobs/fleet/'"$n"'" && c=$((c+1)); done; echo $c' 2>/dev/null)
  if [ "${have:-0}" -ge 1 ]; then echo "$n: already running ($have)"; return; fi
  # ensure shard present
  ssh -o BatchMode=yes ubuntu@"$n" "mkdir -p '$WD/jobs/fleet' '$WD/logs'" 2>/dev/null
  rsync -a "jobs/fleet/$n.txt" "ubuntu@$n:$WD/jobs/fleet/$n.txt" 2>/dev/null
  # bulletproof detached start of dispatcher + watchdog
  ssh -o BatchMode=yes ubuntu@"$n" "cd '$WD' && export HF_HOME=/scratch/hf_cache HF_HUB_DISABLE_XET=1 && \
    ( setsid nohup $VENV auto_dispatch.py --jobs jobs/fleet/$n.txt --gpus 0,1,2,3,4,5,6,7 --tag disp --hf_offline 1 >logs/disp.log 2>&1 </dev/null & ) && \
    ( pgrep -f 'gpu_watchdog.sh jobs/fleet/$n.txt' >/dev/null 2>&1 || setsid nohup bash gpu_watchdog.sh jobs/fleet/$n.txt disp loop >logs/watchdog.log 2>&1 </dev/null & ) ; \
    sleep 4" 2>/dev/null
  # verify
  local ok
  ok=$(ssh -o BatchMode=yes ubuntu@"$n" 'c=0; for p in $(pgrep -x python; pgrep -x python3); do tr "\0" " " </proc/$p/cmdline 2>/dev/null | grep -q auto_dispatch && c=$((c+1)); done; echo $c' 2>/dev/null)
  echo "$n: started, real_dispatchers=${ok:-0}"
}

export -f start_one
export WD VENV
echo "$NODES" | xargs -P 15 -I{} bash -c 'start_one "$@"' _ {}
echo "[relaunch] done"
