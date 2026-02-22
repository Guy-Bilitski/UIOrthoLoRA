#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260221_201358"
echo "Killing process group: 1329133"
kill -9 -1329133 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260221_201358"
rm -rf "logs/run_20260221_201358"
