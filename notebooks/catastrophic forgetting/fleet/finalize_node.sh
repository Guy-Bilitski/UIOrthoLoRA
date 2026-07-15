#!/bin/bash
# Run ON a node: compute ALL adapter-derived data over local /scratch/cf_models adapters that
# summary.json alone doesn't hold — geometry battery + CE-to-base. (Magnitude/F_delta is already
# in each summary.json from eval.) Idempotent: geo/ce scripts skip adapters already scored.
# Uses the node's own GPUs — intended to run when training on this node is winding down.
#   bash fleet/finalize_node.sh            (run locally on the node)
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/scratch/hf_cache HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
V=/home/guy/UIOrthoLoRA/.venv/bin/python
mkdir -p results/geo_drift/base_svd logs
log(){ echo "[finalize $(hostname) $(date -u +%H:%M:%SZ)] $*"; }

# 1) base-SVD (phase1) per base model — needed by phase2; skip if already present.
ls results/geo_drift/base_svd/*.pt >/dev/null 2>&1 || { log "geo phase1 Llama"; $V geo_drift_phase1.py >>logs/finalize_geo.log 2>&1; }
# Qwen base-SVD (phase1_qwen writes its own tagged tensors); harmless if no Qwen adapters here.
$V geo_drift_phase1_qwen.py >>logs/finalize_geo.log 2>&1 || true

# 2) geometry battery (phase2) — Llama + Qwen variants; both glob /scratch/cf_models, skip-done.
log "geo phase2 Llama"; $V geo_drift_phase2.py >>logs/finalize_geo.log 2>&1 || log "geo2 llama rc=$?"
log "geo phase2 Qwen";  $V geo_drift_phase2_qwen.py >>logs/finalize_geo.log 2>&1 || log "geo2 qwen rc=$?"

# 3) CE-to-base — Llama adapters then Qwen adapters (ce_batch is per-base-model).
NODE=$(hostname)
log "CE Llama"; $V ce_batch.py --glob 'frc_*,frm_*,lrsw_*,lrswm_*,mtx*,scl2_*,b4_*,clora_*,lora_*,dora_*,corda_*' \
  --adapters_root /scratch/cf_models --base_model meta-llama/Llama-2-7b-hf \
  --out results/forgetting_${NODE}_llama.jsonl --max_length 1024 --max_blocks 40 --batch_size 2 \
  --done_marker ce_${NODE}_llama --run_name ce_${NODE}_llama >>logs/finalize_ce.log 2>&1 || log "ce llama rc=$?"
log "CE Qwen"; $V ce_batch.py --glob 'qwsw_*,qwswm_*' \
  --adapters_root /scratch/cf_models --base_model Qwen/Qwen2.5-7B \
  --out results/forgetting_${NODE}_qwen.jsonl --max_length 1024 --max_blocks 40 --batch_size 2 \
  --done_marker ce_${NODE}_qwen --run_name ce_${NODE}_qwen >>logs/finalize_ce.log 2>&1 || log "ce qwen rc=$?"
log "DONE"
touch results/finalize_${NODE}.done
