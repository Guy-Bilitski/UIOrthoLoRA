#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260205_094923"
echo "Killing process group: 3679888"
kill -9 -3679888 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260205_094923"
rm -rf "logs/run_20260205_094923"
