#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260203_185108"
echo "Killing process group: 3557223"
kill -9 -3557223 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260203_185108"
rm -rf "logs/run_20260203_185108"
