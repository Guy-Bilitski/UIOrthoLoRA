#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260218_203059"
echo "Killing process group: 3646228"
kill -9 -3646228 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260218_203059"
rm -rf "logs/run_20260218_203059"
