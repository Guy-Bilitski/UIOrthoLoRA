#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260114_104430"
echo "Killing process group: 42802"
kill -9 -42802 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260114_104430"
rm -rf "logs/run_20260114_104430"
