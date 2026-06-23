#!/bin/bash
# Leakage/geometry pass: after the campaign, compute weight-basis (forensics.py: out_top =
# fraction of dW energy in W0's top singular subspace) + data-basis (forensics_databasis.py:
# d_inTop, data_resp on retention-data covariance) for every real checkpoint (seed 42 reps +
# faithful SC-LoRA). Feeds paper_assets.py Fig3 / leakage table. Chained, GPU-pooled.
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting" || exit 1
PY=/home/guy/UIOrthoLoRA/.venv/bin/python
log(){ echo "[forensics $(date '+%F %T')] $*"; }
log "waiting for campaign Phase 4 (lrswmath) ALL DONE..."
while ! grep -q "ALL DONE" logs/lrswmath_pool.log 2>/dev/null; do sleep 300; done
log "campaign done -> building forensics jobs"
> jobs/forensics_pass.txt
for ck in /scratch/cf_models/mtx_*_s42 /scratch/cf_models/mtxm_*_s42 /scratch/cf_models/scl2_* /scratch/cf_models/scl2m_*; do
    [ -d "$ck" ] || continue
    rn=$(basename "$ck")
    case "$rn" in mtx_sclora*|mtxm_sclora*) continue;; esac   # deprecated buggy SC-LoRA
    echo "$PY forensics.py --adapter $ck --run_name $rn && $PY forensics_databasis.py --adapter $ck --run_name $rn --cov_source retain" >> jobs/forensics_pass.txt
done
log "$(grep -c . jobs/forensics_pass.txt) forensics jobs -> pooling on 8 GPUs"
$PY gpu_pool.py --gpus 8 --tag forensics --jobs jobs/forensics_pass.txt > logs/forensics_pool.log 2>&1
log "forensics complete. Run: python paper_assets.py  (Fig3/leakage now populate)."
