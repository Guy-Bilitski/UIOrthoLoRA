#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_185357"
echo "Killing process group: 3752587"
kill -9 -3752587 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_185357"
rm -rf "logs/run_20260205_185357"
