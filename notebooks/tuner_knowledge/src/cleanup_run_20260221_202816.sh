#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260221_202816"
echo "Killing process group: 1467790"
kill -9 -1467790 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260221_202816"
rm -rf "logs/run_20260221_202816"
