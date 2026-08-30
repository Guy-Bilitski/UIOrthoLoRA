#!/usr/bin/env bash
# Finish Llama -> Qwen smoke test -> Qwen full experiment (only if the smoke test passes).
#
# Chain (Guy 2026-08-29):
#   1. wait for the current Llama eval pool to drain
#   2. run the remaining Llama evaluations (regenerated, deduped)
#   3. SMOKE TEST: Qwen with attn_implementation="eager" -- the one untested switch.
#      Must reach STEP_TARGET with zero NaN. Earlier "clean" runs were short timeouts
#      at steps 158-753, so the bar is deliberately well past that.
#   4. if and only if the smoke test passes, launch the full 4-config Qwen experiment
#      using the eager variant; otherwise STOP and leave a clear marker.
#
# Every launch waits for a genuinely free GPU first. Nothing runs two processes at once.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
LOG=logs/auto_finish_then_qwen.log
STEP_TARGET=1500

say() { echo "[chain] $(date -Is) $*" >> "$LOG"; }
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 20; done; }
pool_done() { while pgrep -f "gpu_pool.py .*--tag $1" >/dev/null; do sleep 60; done; }

say "waiting for the current Llama pool (tierAL) to drain"
pool_done tierAL
say "tierAL done"

# ---- step 2: remaining Llama evaluations -----------------------------------
"$PY" build_final_queue.py > jobs/tierA_llama_final.txt
N=$(grep -c "^until" jobs/tierA_llama_final.txt || true)
say "remaining Llama evaluations: $N"
if [ "$N" -gt 0 ]; then
  gpu_free
  setsid "$PY" gpu_pool.py --gpus 1 --tag tierALF --jobs jobs/tierA_llama_final.txt \
      > logs/tierALF_pool.log 2>&1 < /dev/null &
  sleep 30
  pool_done tierALF
  say "Llama complete"
fi

# ---- step 3: Qwen smoke test (eager attention) -----------------------------
gpu_free
say "SMOKE TEST: Qwen + attn_implementation=eager, target step $STEP_TARGET"
rm -rf /home/kfir/cf_models_failed/qwen_smoke
DIAG_ATTN=eager timeout 2400 "$PY" train_cs_eager.py \
  --method lora --lora_r 32 --lora_alpha 64 --weight_decay 0.3 \
  --learning_rate 1e-4 --cutoff_len 256 --seed 43 \
  --base_model Qwen/Qwen2.5-7B --out_root /home/kfir/cf_models_failed \
  --run_name qwen_smoke > logs/qwen_smoke.log 2>&1

NAN=$(grep -c "'grad_norm': 'nan'" logs/qwen_smoke.log || true)
STEP=$(grep -oE '[0-9]+/31956' logs/qwen_smoke.log | tail -1 | cut -d/ -f1)
STEP=${STEP:-0}
say "smoke result: reached step $STEP, nan lines $NAN"

if [ "$NAN" -eq 0 ] && [ "$STEP" -ge "$STEP_TARGET" ]; then
  say "SMOKE PASSED -> launching the full Qwen experiment with the eager variant"
  sed 's#run_safe_sdpa.py train_cs.py#train_cs_eager.py#g; s#SAFE_SDPA=math #DIAG_ATTN=eager #g' \
      jobs/tierA_qwen.txt > jobs/tierA_qwen_eager.txt
  gpu_free
  setsid "$PY" gpu_pool.py --gpus 1 --tag tierAQE --jobs jobs/tierA_qwen_eager.txt \
      > logs/tierAQE_pool.log 2>&1 < /dev/null &
  say "launched tierAQE (pid $!)"
  setsid bash nan_watchdog.sh tierAQE 60 < /dev/null > /dev/null 2>&1 &
  say "nan watchdog armed for tierAQE"
else
  touch logs/QWEN_SMOKE_FAILED.flag
  say "SMOKE FAILED (step $STEP, nan $NAN) -> NOT launching the full experiment"
fi
