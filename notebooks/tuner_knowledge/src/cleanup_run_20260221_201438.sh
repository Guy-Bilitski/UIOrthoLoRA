#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260221_201438"
echo "Killing process group: 1330940"
kill -9 -1330940 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260221_201438"
rm -rf "logs/run_20260221_201438"
