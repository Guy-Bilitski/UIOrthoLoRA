#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260223_183024"
echo "Killing process group: 2443244"
kill -9 -2443244 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260223_183024"
rm -rf "logs/run_20260223_183024"
