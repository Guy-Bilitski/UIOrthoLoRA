#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260223_182819"
echo "Killing process group: 2423291"
kill -9 -2423291 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260223_182819"
rm -rf "logs/run_20260223_182819"
