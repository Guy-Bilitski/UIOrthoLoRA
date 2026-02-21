#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260218_202956"
echo "Killing process group: 3645470"
kill -9 -3645470 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260218_202956"
rm -rf "logs/run_20260218_202956"
