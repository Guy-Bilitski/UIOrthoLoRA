#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260218_203402"
echo "Killing process group: 3647771"
kill -9 -3647771 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260218_203402"
rm -rf "logs/run_20260218_203402"
