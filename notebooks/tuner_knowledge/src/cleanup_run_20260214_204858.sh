#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260214_204858"
echo "Killing process group: 242916"
kill -9 -242916 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260214_204858"
rm -rf "logs/run_20260214_204858"
