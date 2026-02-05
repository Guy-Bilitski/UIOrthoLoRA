#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_110032"
echo "Killing process group: 3687240"
kill -9 -3687240 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_110032"
rm -rf "logs/run_20260205_110032"
