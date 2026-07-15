#!/bin/bash
# Bring ONE fleet node to a byte-identical clone of d001 at the SAME paths.
# Usage: bash fleet/bringup_node.sh <node>
# Requires: key auth to ubuntu@<node> already set up (fleet/bootstrap_keys.sh);
#           sudo password in env PW (used only for apt + mkdir/chown; never persisted).
set -u
NODE="$1"
PW="${PW:?set PW env with the sudo password}"
REPO=/home/guyb/UIOrthoLoRA
SCRATCH=/scratch/hf_cache
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 ubuntu@$NODE"

log(){ echo "[$NODE] $*"; }

# 1) Privileged prep: packages, dirs, symlinks (idempotent). Password via stdin to sudo -S;
#    the script itself is an ARG to sh -c (so it doesn't collide with the password on stdin).
PREP='export DEBIAN_FRONTEND=noninteractive; command -v rsync >/dev/null || apt-get install -y rsync python3.12-venv >/dev/null 2>&1; mkdir -p /home/guyb /scratch/cf_models /scratch/hf_cache; chown -R ubuntu:ubuntu /home/guyb /scratch; ln -sfn /home/guyb /home/guy; test -d /scratch/hf_cache -a -L /home/guy && echo PREP_OK'
out=$($SSH "echo '$PW' | sudo -S sh -c '$PREP'" 2>/dev/null)
echo "$out" | grep -q PREP_OK || { log "PREP FAILED"; exit 1; }
log "prep ok"

# 2) Rsync the repo (incl .venv + reconstructed data under repro/) to the SAME path.
rsync -a --delete-excluded \
  --exclude '.git/' --exclude '**/logs/' --exclude '*.log' \
  --exclude 'results/dispatch_locks/' --exclude 'results/ce_locks/' \
  --exclude '**/__pycache__/' --exclude 'jobs/fleet/' \
  "$REPO/" "ubuntu@$NODE:$REPO/" >/dev/null 2>&1 || { log "RSYNC repo FAILED"; exit 1; }
log "repo synced"

# 3) Rsync the HF model cache (Llama-2-7b + Qwen2.5-7B) to /scratch/hf_cache.
rsync -a "$SCRATCH/" "ubuntu@$NODE:$SCRATCH/" >/dev/null 2>&1 || { log "RSYNC hf_cache FAILED"; exit 1; }
log "hf_cache synced"

# 4) Verify: imports + GPU count + both models cached on disk.
V=/home/guyb/UIOrthoLoRA/.venv/bin/python
OUT=$($SSH "$V -c 'import torch,peft,transformers,trl,lm_eval; print(\"NGPU\",torch.cuda.device_count())' 2>&1; \
  ls -d /scratch/hf_cache/hub/models--meta-llama--Llama-2-7b-hf /scratch/hf_cache/hub/models--Qwen--Qwen2.5-7B >/dev/null 2>&1 && echo MODELS_OK || echo MODELS_MISSING")
echo "$OUT" | grep -q "NGPU 8" && echo "$OUT" | grep -q "MODELS_OK" \
  && { log "READY ($(echo "$OUT" | tr '\n' ' '))"; exit 0; } \
  || { log "VERIFY FAILED: $(echo "$OUT" | tr '\n' ' ')"; exit 1; }
