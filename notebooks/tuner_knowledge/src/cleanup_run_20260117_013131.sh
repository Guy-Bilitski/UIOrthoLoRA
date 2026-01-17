#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260117_013131"
echo "Killing process group: 256618"
kill -9 -256618 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260117_013131"
rm -rf "logs/run_20260117_013131"
