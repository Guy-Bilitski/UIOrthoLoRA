#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_185910"
echo "Killing process group: 3758669"
kill -9 -3758669 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_185910"
rm -rf "logs/run_20260205_185910"
