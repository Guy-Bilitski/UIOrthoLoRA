#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260223_182237"
echo "Killing process group: 2421617"
kill -9 -2421617 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260223_182237"
rm -rf "logs/run_20260223_182237"
