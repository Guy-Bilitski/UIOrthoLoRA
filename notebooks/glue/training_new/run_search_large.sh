#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="logs/search_large.log"
PID_FILE="logs/search_large.pid"
PYTHON="$(which python3)"

mkdir -p logs

# Warn if a previous run is still alive
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: a previous run is still alive (PID $OLD_PID)."
        echo "       Stop it first with:  kill $OLD_PID"
        exit 1
    fi
fi

nohup "$PYTHON" -u training_hpo_large_search.py > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > "$PID_FILE"

echo "================================================"
echo "  Training search started"
echo "  PID:     $PID"
echo "  Log:     $LOG_FILE"
echo ""
echo "  Monitor: tail -f $LOG_FILE"
echo "  Status:  grep '^\[20' $LOG_FILE"
echo "  Stop:    kill \$(cat $PID_FILE)"
echo "================================================"
