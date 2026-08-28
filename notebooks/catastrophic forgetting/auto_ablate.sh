#!/usr/bin/env bash
# Build the B/C/D intervention arms for every finished slice cell (CPU only).
#
# FINAL PROTOCOL (locked with Guy 2026-08-28): every adapter configuration
# (model x method x lr x seed) gets:
#   1. intruder measurement at k=10 / tau=0.5 vs the FULL pretrained left basis
#      (auto_intruder.sh)
#   2. the full intervention set A-E, all evaluated identically:
#        A = source, unmodified (run as <run>__rl50 at the same protocol)
#        B = ALL intruders in the top-10 window removed          (shrinks ||dW||)
#        C = whole update uniformly shrunk to B's ||dW||         (size control)
#        D = B rescaled back to the source ||dW||                (size restored)
#        E = NON-intruder content removed at B's ||dW||          (specificity control)
#      B/C/E all remove the SAME energy, differing only in what is removed.
#
# This script does step 2's CPU half: it builds the arms and appends their eval
# job lines to jobs/pending_ablation.txt. The GPU evals are run as a separate
# pool once the training queue drains (deliberately NOT injected into a running
# pool -- mid-flight queue switches have caused orphaned trainers before).
#
# Usage: setsid bash auto_ablate.sh [poll_seconds] &
set -u
cd "$(dirname "$0")"
POLL="${1:-600}"
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_DISABLE_XET=1
export GEO_THREADS=6
LOG=logs/auto_ablate.log
PEND=jobs/pending_ablation.txt
touch "$PEND"
echo "[autoabl] $(date -Is) polling every ${POLL}s (k=10, ALL intruders)" >> "$LOG"

while true; do
  for d in /home/kfir/cf_models/*/; do
    run=$(basename "$d")
    case "$run" in *__*) continue ;; esac          # skip derived adapters
    [ -f "$d/.evacuated" ] || continue             # cell must be complete
    [ -f "results/intruder/${run}.json" ] || continue
    [ -d "/home/kfir/cf_models/${run}__k10allablB" ] && continue
    case "$run" in
      *_qwsw*|*_qwswm*) base="Qwen/Qwen2.5-7B" ;;
      *)                base="meta-llama/Llama-2-7b-hf" ;;
    esac
    echo "[autoabl] $(date -Is) building k10 arms for $run" >> "$LOG"
    if "$PY" intruder_ablate.py --adapter "$d" --base_model "$base" \
         --topk 10 --n_remove all --tag k10all --with-renorm >> "logs/ablate_${run}.log" 2>&1; then
      # arm E: non-intruder removal at matched magnitude (mirror of B)
      "$PY" arm_e_build.py --adapter "$d" --base_model "$base" --tag k10all \
          >> "logs/arm_e_${run}.log" 2>&1 \
        && echo "[autoabl] $(date -Is) built arm E for $run" >> "$LOG" \
        || echo "[autoabl] $(date -Is) arm E FAILED for $run" >> "$LOG"
      for arm in B C D E; do
        rn="${run}__k10allabl${arm}"
        grep -q -- "--run_name ${rn} " "$PEND" 2>/dev/null && continue
        echo "$PY eval_one_gpu.py --adapter /home/kfir/cf_models/$rn --run_name $rn --base_model $base --adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512 && $PY forgetting_ce.py --runs $rn --adapters_root /home/kfir/cf_models --base_model $base --max_length 1024 --max_blocks 0 --batch_size 2 && bash evacuate_cell.sh /home/kfir/cf_models/$rn /home/kfir/tierA_evac" >> "$PEND"
      done
      # source re-eval under the identical protocol (needed for the D-vs-source pair)
      rn="${run}__rl50"
      if [ ! -f "results/${rn}/summary.json" ] && ! grep -q -- "--run_name ${rn} " "$PEND" 2>/dev/null; then
        echo "$PY eval_one_gpu.py --adapter /home/kfir/cf_models/$run --run_name $rn --base_model $base --adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512" >> "$PEND"
      fi
      echo "[autoabl] $(date -Is) queued 4 evals for $run -> $PEND" >> "$LOG"
    else
      echo "[autoabl] $(date -Is) FAILED building arms for $run" >> "$LOG"
    fi
  done
  sleep "$POLL"
done
