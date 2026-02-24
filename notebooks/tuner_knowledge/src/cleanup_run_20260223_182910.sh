#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260223_182910"
echo "Killing process group: 2426728"
kill -9 -2426728 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260223_182910"
rm -rf "logs/run_20260223_182910"
