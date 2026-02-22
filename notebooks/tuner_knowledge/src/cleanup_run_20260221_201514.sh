#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260221_201514"
echo "Killing process group: 1337965"
kill -9 -1337965 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260221_201514"
rm -rf "logs/run_20260221_201514"
