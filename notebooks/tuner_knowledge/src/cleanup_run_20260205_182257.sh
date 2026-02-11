#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_182257"
echo "Killing process group: 3741063"
kill -9 -3741063 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_182257"
rm -rf "logs/run_20260205_182257"
