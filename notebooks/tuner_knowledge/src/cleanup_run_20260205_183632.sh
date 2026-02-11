#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_183632"
echo "Killing process group: 3743820"
kill -9 -3743820 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_183632"
rm -rf "logs/run_20260205_183632"
