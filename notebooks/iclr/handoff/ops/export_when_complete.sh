#!/bin/bash
# When all 18 band confirmations are validated, write the final evidence export.
# CPU only, so it never competes with a training card. Separate from the partition chain on purpose:
# the export is the registered descriptive analysis and must not wait on an exploratory diagnostic.
set -u
CHECKOUT=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
BAND=$CHECKOUT/campaign_outputs_decoder_subspace_v1
RES=$CHECKOUT/notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_SUBSPACE.json
LOG=$BAND/logs/evidence_export.log
cd "$CHECKOUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }
while true; do
  n=$(CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python -c "
from notebooks.iclr.decoder_pilot import subspace_plan as sp
R='$BAND'
rows=sp.collect_runs(R+'/run_ledger.jsonl', R+'/protocols/confirmation.json', purpose=sp.CONFIRMATION_PURPOSE)
print(sum(1 for r in rows if r.get('status')=='completed'))
" 2>>"$LOG")
  case "$n" in ''|*[!0-9]*) sleep 120; continue;; esac
  [ "$n" -ge 18 ] && break
  sleep 120
done
say "18/18 validated, writing the final evidence export"
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python -m notebooks.iclr.decoder_pilot.subspace_analysis \
  --resources "$RES" --protocol "$BAND/protocols/confirmation.json" \
  --output-directory "$BAND/evidence_final" >> "$LOG" 2>&1 \
  || { say "EXPORT FAILED, see $LOG"; exit 1; }
DEST=$CHECKOUT/notebooks/iclr/handoff/data/decoder_subspace_evidence_20260919
mkdir -p "$DEST"; cp "$BAND"/evidence_final/* "$DEST"/
say "export written and copied to the tracked handoff directory"
