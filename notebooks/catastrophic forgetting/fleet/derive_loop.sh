#!/bin/bash
# Continuous per-node derived-calc battery: GEOMETRY (CPU) + CE-shift (GPU-polite), for every local
# adapter, so that as soon as an adapter is trained+evaled its magnitude (already in summary.json),
# geometry and CE are computed and land as per-run files that the --ignore-existing collect loop syncs.
#
# Idempotent: geo_drift/ce_batch both skip adapters already scored. GPU-polite: CE runs on the
# emptiest GPU at nice -n 15 (7B base ~14GB co-resides under B200's headroom; ~12-15s/adapter).
# Emits per-run results/<run>/geo.json (from the append-only adapter_metrics.jsonl) and ce_batch
# already writes results/<run>/forgetting.json — both sync per-run (no reliance on aggregate jsonls).
#
# Launch detached ON a node:  setsid nohup bash fleet/derive_loop.sh 300 > logs/derive_loop.log 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."
INT="${1:-300}"
export HF_HOME=/scratch/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
V=/home/guy/UIOrthoLoRA/.venv/bin/python
NODE=$(hostname)
mkdir -p results/geo_drift/base_svd logs
log(){ echo "[derive $NODE $(date -u +%H:%M:%SZ)] $*"; }

# single-instance guard (flock held for process lifetime)
exec 9>logs/derive_loop.lock
flock -n 9 || { echo "[derive] another derive_loop already running — exiting"; exit 0; }

emptiest_gpu(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }

# one-time phase-1 base SVDs needed by phase-2 (Llama usually present; Qwen built from cached model)
ls results/geo_drift/base_svd/*.pt >/dev/null 2>&1 || { log "geo phase1 Llama"; $V geo_drift_phase1.py >>logs/derive_geo.log 2>&1 || log "geo1 llama rc=$?"; }
[ -e results/geo_drift/base_svd_qwen ] || ls results/geo_drift/base_svd/*qwen* >/dev/null 2>&1 || { log "geo phase1 Qwen"; $V geo_drift_phase1_qwen.py >>logs/derive_geo.log 2>&1 || log "geo1 qwen rc=$?"; }

while true; do
  # 1) GEOMETRY — CPU, idempotent (skip-done via adapter_metrics.jsonl)
  $V geo_drift_phase2.py      >>logs/derive_geo.log 2>&1 || log "geo2 llama rc=$?"
  $V geo_drift_phase2_qwen.py >>logs/derive_geo.log 2>&1 || log "geo2 qwen rc=$?"
  # explode the append-only aggregate into per-run geo.json so --ignore-existing collect syncs each one
  $V - >>logs/derive_geo.log 2>&1 <<'PY' || true
import json, os, glob
for agg in glob.glob("results/geo_drift/adapter_metrics*.jsonl"):
    for line in open(agg):
        try: d = json.loads(line)
        except Exception: continue
        run = d.get("run")
        if not run: continue
        rd = os.path.join("results", run)
        p = os.path.join(rd, "geo.json")
        if os.path.isdir(rd) and not os.path.exists(p):
            json.dump(d, open(p, "w"), indent=2)
PY

  # 2) CE-shift — GPU-polite: emptiest GPU, nice. ce_batch is idempotent (skip-done + per-adapter
  #    locks) and writes per-run results/<run>/forgetting.json. No --run_name/--done_marker => no
  #    fake marker cell. Llama adapters then Qwen adapters (each ce harness is per-base-model).
  G=$(emptiest_gpu); log "CE Llama on GPU$G"
  CUDA_VISIBLE_DEVICES=$G nice -n 15 $V ce_batch.py \
    --glob 'frc_*,frm_*,lrsw_*,lrswm_*,mtx*,mtxm_*,scl2_*,b4_*,clora_*,lora_*,dora_*,corda_*,uilin_*,uio*' \
    --adapters_root /scratch/cf_models --base_model meta-llama/Llama-2-7b-hf \
    --out results/forgetting_${NODE}_llama.jsonl --max_length 1024 --max_blocks 40 --batch_size 2 \
    >>logs/derive_ce.log 2>&1 || log "ce llama rc=$?"
  G=$(emptiest_gpu); log "CE Qwen on GPU$G"
  CUDA_VISIBLE_DEVICES=$G nice -n 15 $V ce_batch.py --glob 'qwsw_*,qwswm_*' \
    --adapters_root /scratch/cf_models --base_model Qwen/Qwen2.5-7B \
    --out results/forgetting_${NODE}_qwen.jsonl --max_length 1024 --max_blocks 40 --batch_size 2 \
    >>logs/derive_ce.log 2>&1 || log "ce qwen rc=$?"

  log "pass done"
  sleep "$INT"
done
