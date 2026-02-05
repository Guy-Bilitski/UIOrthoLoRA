#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_084539"
echo "Killing process group: 3637503"
kill -9 -3637503 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_084539"
rm -rf "logs/run_20260205_084539"
