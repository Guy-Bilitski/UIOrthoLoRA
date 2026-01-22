#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260122_122718"
echo "Killing process group: 4059535"
kill -9 -4059535 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260122_122718"
rm -rf "logs/run_20260122_122718"
