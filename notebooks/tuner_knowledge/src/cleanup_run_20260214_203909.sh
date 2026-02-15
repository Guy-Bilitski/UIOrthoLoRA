#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260214_203909"
echo "Killing process group: 240815"
kill -9 -240815 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260214_203909"
rm -rf "logs/run_20260214_203909"
