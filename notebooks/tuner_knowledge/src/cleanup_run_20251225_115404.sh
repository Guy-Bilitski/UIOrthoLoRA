#!/bin/bash
set -euo pipefail
echo "Cleaning run 20251225_115404"
echo "Killing process group: 9427"
kill -9 -9427 2>/dev/null || true
echo "Removing logs for this run: logs/run_20251225_115404"
rm -rf "logs/run_20251225_115404"
