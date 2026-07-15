#!/bin/bash
# Pre-cache eval datasets ONLINE into HF_HOME so subsequent --hf_offline runs find them.
# Xet disabled (its transfer path hangs on this cluster). Run: bash fleet/precache_datasets.sh
# NOTE: only d001 (head) has internet egress; on compute nodes, cache on d001 then rsync
#   /scratch/hf_cache/datasets/<name>/ out (see handoff/39).
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/scratch/hf_cache CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
V=/home/guyb/UIOrthoLoRA/.venv/bin/python
# 1) retention suite (bbh/mmlu_pro/mmlu/arc/truthfulqa) + commonsense, via one small eval.
rm -rf results/smoke_d001
$V eval_one_gpu.py --adapter /scratch/cf_models/smoke_d001 --run_name smoke_d001 \
  --base_model meta-llama/Llama-2-7b-hf --adapt_task cs --ret_suite broad \
  --eval_limit 4 --ret_limit 2
# 2) gsm8k: the lm-eval 'gsm8k' task pulls openai/gsm8k, which the cs smoke above does NOT
#    touch. Cache it explicitly so --adapt_task gsm8k cells work offline (2026-07-15 fix).
$V -c "import datasets; datasets.load_dataset('openai/gsm8k','main'); print('gsm8k cached')"
echo "PRECACHE_DONE rc=$?"
