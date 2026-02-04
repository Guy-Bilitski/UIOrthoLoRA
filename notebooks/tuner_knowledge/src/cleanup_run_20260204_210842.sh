#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260204_210842"
echo "Killing process group: 3620999"
kill -9 -3620999 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260204_210842"
rm -rf "logs/run_20260204_210842"
