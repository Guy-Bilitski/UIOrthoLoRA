#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260225_132007"
echo "Killing process group: 3866668"
kill -9 -3866668 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260225_132007"
rm -rf "logs/run_20260225_132007"
