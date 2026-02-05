#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_100446"
echo "Killing process group: 3681280"
kill -9 -3681280 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_100446"
rm -rf "logs/run_20260205_100446"
