#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260120_173837"
echo "Killing process group: 1874548"
kill -9 -1874548 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260120_173837"
rm -rf "logs/run_20260120_173837"
