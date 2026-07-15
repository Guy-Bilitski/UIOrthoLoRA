#!/bin/bash
# One monitoring tick: fleet liveness + orchestration loops + data-health + d002 drain.
# Read-only (observation only — healing of dead dispatchers is left to guardian_loop;
# duplicate watchdogs are harmless and left alone). Prints a compact digest.
# Usage: bash fleet/monitor_tick.sh
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"       # working dir (…/catastrophic forgetting)
cd "$HERE" || exit 1
VENV=/home/guyb/UIOrthoLoRA/.venv/bin/python
TS=$(date -u +%Y-%m-%dT%H:%MZ)
echo "==================== monitor tick $TS ===================="

echo "---- d001 orchestration loops ----"
for pat in guardian_loop collect_loop derive_supervisor; do
  n=$(pgrep -fc "$pat" || true)
  echo "  $pat: $n alive $([ "${n:-0}" -ge 1 ] && echo OK || echo '*** DOWN ***')"
done

echo "---- fleet liveness (status.sh) ----"
timeout 200 bash fleet/status.sh 2>/dev/null | tail -30

echo "---- data health (all + last 40 min) ----"
$VENV fleet/data_sanity.py 2>/dev/null | head -1
$VENV fleet/data_sanity.py --since_min 40 2>&1 | sed 's/^/  /'

echo "---- d002 (DeepSeek target) drain ----"
timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=10 ubuntu@d002 '
  idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk "\$1<10" | wc -l)
  evals=$(pgrep -fc "eval_one_gpu.py|train_cs.py" || true)
  echo "  idle_gpus=$idle/8  running_sweep_jobs=$evals"' 2>/dev/null || echo "  d002 unreachable"
echo "==================== end tick ===================="
