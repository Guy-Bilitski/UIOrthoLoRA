#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260203_191251"
echo "Killing process group: 3559624"
kill -9 -3559624 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260203_191251"
rm -rf "logs/run_20260203_191251"
