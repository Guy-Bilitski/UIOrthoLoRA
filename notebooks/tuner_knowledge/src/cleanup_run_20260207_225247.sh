#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260207_225247"
echo "Killing process group: 3823643"
kill -9 -3823643 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260207_225247"
rm -rf "logs/run_20260207_225247"
