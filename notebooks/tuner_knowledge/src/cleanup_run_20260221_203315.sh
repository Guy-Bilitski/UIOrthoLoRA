#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260221_203315"
echo "Killing process group: 1514986"
kill -9 -1514986 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260221_203315"
rm -rf "logs/run_20260221_203315"
