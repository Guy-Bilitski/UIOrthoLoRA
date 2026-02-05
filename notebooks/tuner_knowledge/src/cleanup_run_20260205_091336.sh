#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_091336"
echo "Killing process group: 3648801"
kill -9 -3648801 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_091336"
rm -rf "logs/run_20260205_091336"
