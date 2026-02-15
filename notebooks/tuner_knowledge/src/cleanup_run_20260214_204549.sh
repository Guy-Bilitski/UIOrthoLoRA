#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260214_204549"
echo "Killing process group: 241921"
kill -9 -241921 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260214_204549"
rm -rf "logs/run_20260214_204549"
