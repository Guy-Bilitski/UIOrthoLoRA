#!/bin/bash
# Continuous per-node derived-calc battery: GEOMETRY (CPU) + CE-shift (GPU-polite) + provenance, for
# every local adapter, so each adapter's full publication record (magnitude already in summary.json,
# plus geometry, CE, per-layer geo, and config) lands as per-run files the --ignore-existing collect syncs.
#
# CE protocol = FULL WikiText-103 test (--max_blocks 0), MiLoRA/Kalajdzievski-comparable (chosen for A*
# 2026-07-15). Any PRESENT adapter whose existing forgetting.json used a short slice (<100 blocks) is
# reset so it re-scores at full test (evacuated adapters keep whatever record they have).
# Idempotent; GPU-polite (emptiest GPU, nice -n15). Launch detached ON a node:
#   setsid nohup bash fleet/derive_loop.sh 300 > logs/derive_loop.log 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."
INT="${1:-300}"
FULL_MIN_BLOCKS=100          # a forgetting.json with fewer blocks than this = short-slice, re-score
export HF_HOME=/scratch/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
V=/home/guy/UIOrthoLoRA/.venv/bin/python
NODE=$(hostname)
mkdir -p results/geo_drift/base_svd logs
log(){ echo "[derive $NODE $(date -u +%H:%M:%SZ)] $*"; }

exec 9>logs/derive_loop.lock
flock -n 9 || { echo "[derive] another derive_loop already running — exiting"; exit 0; }

emptiest_gpu(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }

ls results/geo_drift/base_svd/*.pt >/dev/null 2>&1 || { log "geo phase1 Llama"; $V geo_drift_phase1.py >>logs/derive_geo.log 2>&1 || log "geo1 llama rc=$?"; }
[ -e results/geo_drift/base_svd_qwen ] || ls results/geo_drift/base_svd/*qwen* >/dev/null 2>&1 || { log "geo phase1 Qwen"; $V geo_drift_phase1_qwen.py >>logs/derive_geo.log 2>&1 || log "geo1 qwen rc=$?"; }

while true; do
  # 1) GEOMETRY — CPU, idempotent
  $V geo_drift_phase2.py      >>logs/derive_geo.log 2>&1 || log "geo2 llama rc=$?"
  $V geo_drift_phase2_qwen.py >>logs/derive_geo.log 2>&1 || log "geo2 qwen rc=$?"

  # 2) per-run PROVENANCE + geometry files (so --ignore-existing collect syncs each adapter's record):
  #    - config.json  : snapshot the adapter's run_config.json while the adapter still exists on disk
  #    - geo.json      : per-adapter geometry aggregate exploded from adapter_metrics.jsonl
  #    Also RESET short-slice CE for present adapters so they re-score at full test.
  $V - "$FULL_MIN_BLOCKS" >>logs/derive_geo.log 2>&1 <<'PY' || true
import json, os, glob, sys
FULL_MIN=int(sys.argv[1])
# config snapshot from live adapters
for cfg in glob.glob("/scratch/cf_models/*/run_config.json"):
    run=os.path.basename(os.path.dirname(cfg)); rd=os.path.join("results",run)
    if os.path.isdir(rd) and not os.path.exists(os.path.join(rd,"config.json")):
        try: json.dump(json.load(open(cfg)), open(os.path.join(rd,"config.json"),"w"), indent=2)
        except Exception: pass
# per-run geo.json
for agg in glob.glob("results/geo_drift/adapter_metrics*.jsonl"):
    for line in open(agg):
        try: d=json.loads(line)
        except Exception: continue
        run=d.get("run");
        if not run: continue
        rd=os.path.join("results",run); p=os.path.join(rd,"geo.json")
        if os.path.isdir(rd) and not os.path.exists(p): json.dump(d,open(p,"w"),indent=2)
# reset short-slice CE for adapters STILL PRESENT (so full-test re-score happens); leave evacuated ones
present=set(os.path.basename(os.path.dirname(p)) for p in glob.glob("/scratch/cf_models/*/adapter_model.safetensors"))
reset=set()
for fj in glob.glob("results/*/forgetting.json"):
    run=os.path.basename(os.path.dirname(fj))
    if run not in present: continue
    try: nb=json.load(open(fj)).get("n_blocks",0)
    except Exception: nb=0
    if nb and nb<FULL_MIN:
        os.remove(fj); reset.add(run)
if reset:
    for agg in glob.glob("results/forgetting_*.jsonl"):
        try: lines=[l for l in open(agg) if (json.loads(l).get("run_name") not in reset)]
        except Exception: continue
        open(agg,"w").writelines(lines)
    print(f"[reset] {len(reset)} present adapters cleared for full-test re-score")
PY

  # 3) CE-shift — GPU-polite, FULL WikiText-103 test (--max_blocks 0). Idempotent skip-done + per-adapter
  #    locks; writes per-run results/<run>/forgetting.json. No --run_name/--done_marker => no fake cell.
  G=$(emptiest_gpu); log "CE Llama (full test) on GPU$G"
  CUDA_VISIBLE_DEVICES=$G nice -n 15 $V ce_batch.py \
    --glob 'frc_*,frm_*,lrsw_*,lrswm_*,mtx*,mtxm_*,scl2_*,b4_*,clora_*,lora_*,dora_*,corda_*,uilin_*,uio*' \
    --adapters_root /scratch/cf_models --base_model meta-llama/Llama-2-7b-hf \
    --out results/forgetting_${NODE}_llama.jsonl --max_length 1024 --max_blocks 0 --batch_size 2 \
    >>logs/derive_ce.log 2>&1 || log "ce llama rc=$?"
  G=$(emptiest_gpu); log "CE Qwen (full test) on GPU$G"
  CUDA_VISIBLE_DEVICES=$G nice -n 15 $V ce_batch.py --glob 'qwsw_*,qwswm_*' \
    --adapters_root /scratch/cf_models --base_model Qwen/Qwen2.5-7B \
    --out results/forgetting_${NODE}_qwen.jsonl --max_length 1024 --max_blocks 0 --batch_size 2 \
    >>logs/derive_ce.log 2>&1 || log "ce qwen rc=$?"

  log "pass done"
  sleep "$INT"
done
