#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260214_205031"
echo "Killing process group: 243448"
kill -9 -243448 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260214_205031"
rm -rf "logs/run_20260214_205031"
