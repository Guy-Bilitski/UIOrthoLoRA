#!/usr/bin/env bash
# Overnight watchdog for the Qwen campaign (2026-09-14). Every 2 min:
#   * exactly one gpu_pool_dyn.py alive (restart if none, unless the queue is finished)
#   * qwen_campaign3.sh alive (relaunch if it died before queuing STOP; it adopts a running
#     training and skips finished work)
#   * report: GPU memory, processes, failed jobs, disk. Exits when STOP is queued and the
#     pool has drained, after regenerating the FINAL table.
# Every action is logged to logs/qwen_watchdog.log and logs/qwen_campaign.log.
cd "$(dirname "$0")"
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
QUEUE=jobs/qwen_dyn_queue.txt
WLOG=logs/qwen_watchdog.log
say() { echo "[watchdog] $(date -Is) $*" | tee -a "$WLOG" >> logs/qwen_campaign.log; }
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
start_pool() {
  setsid "$PY" gpu_pool_dyn.py --queue "$QUEUE" --tag tQd --slots 3 --slots_training 1 \
      --min_free_mb 40000 --stagger 150 --lock logs/gpu_slot.lock >> logs/tQd_pool.log 2>&1 < /dev/null &
  say "POOL (re)started"
}
say "watchdog start"
tick=0
while true; do
  sleep 120; tick=$((tick+1))
  npool=$(pgrep -fc "python gpu_pool_dyn.py")
  stop=$(grep -c '^STOP$' "$QUEUE" 2>/dev/null || echo 0)
  if [ "$npool" -eq 0 ]; then
    if [ "$stop" -ge 1 ] && grep -q "STOP seen and queue drained" logs/tQd_pool.log; then
      say "queue drained and pool exited normally -> regenerating table, watchdog done"
      "$PY" paper_table.py > results/FINAL_TABLE_qwen.md 2>&1
      "$PY" paper_table.py --csv results/FINAL_TABLE_qwen.csv >> results/FINAL_TABLE_qwen.md 2>&1
      say "wrote results/FINAL_TABLE_qwen.{md,csv}"; exit 0
    fi
    say "no pool running -> restart"; start_pool
  fi
  if ! pgrep -f "qwen_campaign3.sh" >/dev/null && [ "$stop" -eq 0 ]; then
    say "campaign3 not running and STOP not queued -> relaunch"
    setsid bash qwen_campaign3.sh < /dev/null > /dev/null 2>&1 &
  fi
  # stuck pool: pending work, nothing running on the GPU, nothing started for 20 min
  if [ "$(pgrep -fc 'python eval_one_gpu.py|python forgetting_ce.py|run_guarded.py')" -eq 0 ] && [ "$stop" -eq 0 ]; then
    last=$(stat -c %Y logs/tQd_pool.log); now=$(date +%s)
    if [ $((now-last)) -gt 1200 ]; then say "GPU idle 20 min with pool alive -> restart pool"; pkill -f "python gpu_pool_dyn.py"; sleep 5; start_pool; fi
  fi
  if [ $((tick % 15)) -eq 0 ]; then   # every 30 min: status line
    say "status: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) | procs: train=$(pgrep -fc run_guarded.py) eval=$(pgrep -fc 'python eval_one_gpu.py') | done=$(grep -c 'rc=0' logs/tQd_pool.log) failed=$(grep -c 'FAILED (no more' logs/tQd_pool.log) | disk free $(df -h /home/kfir | awk 'NR==2{print $4}')"
  fi
done
