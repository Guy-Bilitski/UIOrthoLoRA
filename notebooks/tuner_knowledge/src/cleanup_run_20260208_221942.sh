#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260208_221942"
echo "Killing process group: 3849940"
kill -9 -3849940 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260208_221942"
rm -rf "logs/run_20260208_221942"
