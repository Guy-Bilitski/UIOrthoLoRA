#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260121_092915"
echo "Killing process group: 2217277"
kill -9 -2217277 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260121_092915"
rm -rf "logs/run_20260121_092915"
