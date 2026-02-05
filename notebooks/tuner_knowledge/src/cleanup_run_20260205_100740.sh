#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_100740"
echo "Killing process group: 3682165"
kill -9 -3682165 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_100740"
rm -rf "logs/run_20260205_100740"
