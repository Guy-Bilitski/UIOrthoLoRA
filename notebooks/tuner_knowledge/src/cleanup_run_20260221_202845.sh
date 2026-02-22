#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260221_202845"
echo "Killing process group: 1468129"
kill -9 -1468129 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260221_202845"
rm -rf "logs/run_20260221_202845"
