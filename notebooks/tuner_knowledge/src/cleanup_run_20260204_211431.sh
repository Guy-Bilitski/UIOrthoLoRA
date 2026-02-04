#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260204_211431"
echo "Killing process group: 3623770"
kill -9 -3623770 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260204_211431"
rm -rf "logs/run_20260204_211431"
