#!/usr/bin/env bash
# Run all 6 GLUE tasks on roberta-base with the best-known configs.
# Pass --no_de to disable D and E diagonal scalers.
#
# Usage:
#   ./run_training.sh           # with D and E (default)
#   ./run_training.sh --no_de   # without D and E

NO_DE_FLAG=""
RESULTS_DIR="results/glue_base"
for arg in "$@"; do
    if [ "$arg" = "--no_de" ]; then
        NO_DE_FLAG="--no_de"
        RESULTS_DIR="results/glue_base_no_de"
    fi
done

SEED=42
LOGFILE="output_$(date +'%Y%m%d_%H%M%S')${NO_DE_FLAG:+_no_de}.log"

run_task() {
    local TASK=$1 EPOCHS=$2 HEAD_LR=$3 ADAPTER_LR=$4 SCALER=$5 SIGMA=$6
    echo "[$(date +'%H:%M:%S')] Starting $TASK  epochs=$EPOCHS  head_lr=$HEAD_LR  adapter_lr=$ADAPTER_LR  scaler=$SCALER  sigma=$SIGMA  no_de=${NO_DE_FLAG:+true}"
    CUDA_VISIBLE_DEVICES=0 python main_training.py \
        --task          "$TASK"   \
        --epochs        "$EPOCHS" \
        --seed          $SEED     \
        --num_svalues_to_adapt  256 \
        --num_svectors_to_adapt 0   \
        --head_lr       "$HEAD_LR"    \
        --adapter_lr    "$ADAPTER_LR" \
        --initial_scaler "$SCALER"    \
        --initial_sigma  "$SIGMA"     \
        --batch_size    64  \
        --max_len       256 \
        --base_model_id roberta-base  \
        --model_type    uiortholora   \
        --uiortholora_alpha   1 \
        --uiortholora_dropout 0 \
        --target_modules attention.output.dense query key value \
        --results_dir   "$RESULTS_DIR" \
        $NO_DE_FLAG
    echo "[$(date +'%H:%M:%S')] Finished $TASK"
}

{
#          TASK        EPOCHS  HEAD_LR  ADAPTER_LR  SCALER  SIGMA
run_task   cola_lin      80    5e-3      3e-2        1e-2    1e-1
run_task   sst2_lin      40    1e-2      4e-2        1e-1    1e-1
run_task   mrpc_lin      30    1e-3      5e-2        1e-1    1e-1
run_task   sts-b_lin     60    5e-3      1e-2        1e-1    1e-1
run_task   qnli_lin      25    1e-3      2e-2        1e-1    1e-1
run_task   rte_lin       90    5e-4      1e-2        1e-2    1e-2
} 2>&1 | tee "$LOGFILE"

echo "All tasks done. Log: $LOGFILE"
