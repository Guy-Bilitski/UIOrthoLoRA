#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260214_203904"
echo "Killing process group: 240635"
kill -9 -240635 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260214_203904"
rm -rf "logs/run_20260214_203904"
