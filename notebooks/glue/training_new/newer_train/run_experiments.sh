#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs results outputs

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

# Edit these values to choose what this launcher runs.
MODEL_SIZE="large"
USE_DE=true
SEEDS=(42 123 2021 17 31415 1054)
TASKS=()
RESULTS_DIR="$SCRIPT_DIR/results/glue"

if [ "$USE_DE" = true ]; then
    DE_ARG="--use_de"
    DE_NAME="with_de"
else
    DE_ARG="--no_de"
    DE_NAME="no_de"
fi

LOG_FILE="logs/run_experiments.log"
PID_FILE="logs/run_experiments.pid"

RUN_ARGS=(
    --model_size "$MODEL_SIZE"
    "$DE_ARG"
    --seeds "${SEEDS[@]}"
    --results_dir "$RESULTS_DIR"
)

if [ "${#TASKS[@]}" -gt 0 ]; then
    RUN_ARGS+=(--tasks "${TASKS[@]}")
fi

RUN_ARGS+=("$@")

if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: a previous run is still alive (PID $OLD_PID)."
        echo "       Stop it first with: kill $OLD_PID"
        exit 1
    fi
fi

printf "" > "$LOG_FILE"
nohup "$PYTHON" -u experiments.py "${RUN_ARGS[@]}" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "================================================"
echo "  Experiments started"
echo "  PID:     $PID"
echo "  Log:     $SCRIPT_DIR/$LOG_FILE"
echo "  PID file:$SCRIPT_DIR/$PID_FILE"
echo ""
echo "  Monitor: tail -f $SCRIPT_DIR/$LOG_FILE"
echo "  Stop:    kill \$(cat $SCRIPT_DIR/$PID_FILE)"
echo "================================================"
