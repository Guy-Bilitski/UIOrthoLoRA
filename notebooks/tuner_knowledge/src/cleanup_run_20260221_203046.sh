#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260221_203046"
echo "Killing process group: 1482502"
kill -9 -1482502 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260221_203046"
rm -rf "logs/run_20260221_203046"
