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

echo "---- DeepSeek cells (21 nodes; replaces the stale d002-drain section 2026-07-16) ----"
DSN="d002 d003 d005 d006 d007 d008 d009 d010 d012 d013 d014 d015 d016 d017 d018 d019 d020 d023 d024 d029 d032"
for n in $DSN; do
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 ubuntu@$n '
    p=$(pgrep -af "train_deepseek|eval_deepseek|ce_deepseek|geo_deepseek" 2>/dev/null | grep -v pgrep | grep -oP "(?<=--run_name )\S+" | head -1)
    s=$(ls "'"$HERE"'"/results/dsv4_*/summary.json 2>/dev/null | wc -l)
    echo "  active=${p:-IDLE} local_dsv4_summaries=$s"' 2>/dev/null | sed "s/^/  $n/" &
done; wait
echo "---- E-batch summaries landed (e1/e2=fft/e3-e7 run prefixes) ----"
for pre in e1_ fft_ qwsw_lora_r16_lr7e5 qwsw_lora_r16_lr15e5 lrsw_dorawd lrsw_milorawd b4_sclora_r32_lr2e4 lrsw_lorarep05 brl_ brq_; do
  c=$(ls -d results/${pre}* 2>/dev/null | while read d; do [ -f "$d/summary.json" ] && echo x; done | wc -l)
  printf "  %-24s %s\n" "$pre" "$c"
done
echo "==================== end tick ===================="
