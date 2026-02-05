#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_085118"
echo "Killing process group: 3641609"
kill -9 -3641609 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_085118"
rm -rf "logs/run_20260205_085118"
