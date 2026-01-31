#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260131_150126"
echo "Killing process group: 3383795"
kill -9 -3383795 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260131_150126"
rm -rf "logs/run_20260131_150126"
