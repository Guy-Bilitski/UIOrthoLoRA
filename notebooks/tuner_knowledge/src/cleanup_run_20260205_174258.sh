#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_174258"
echo "Killing process group: 3736960"
kill -9 -3736960 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_174258"
rm -rf "logs/run_20260205_174258"
