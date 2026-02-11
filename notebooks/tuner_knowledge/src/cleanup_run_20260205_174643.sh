#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_174643"
echo "Killing process group: 3738049"
kill -9 -3738049 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_174643"
rm -rf "logs/run_20260205_174643"
