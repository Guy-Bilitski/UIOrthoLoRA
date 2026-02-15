#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260214_204834"
echo "Killing process group: 242511"
kill -9 -242511 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260214_204834"
rm -rf "logs/run_20260214_204834"
