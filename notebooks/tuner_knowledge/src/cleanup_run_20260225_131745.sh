#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260225_131745"
echo "Killing process group: 3865777"
kill -9 -3865777 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260225_131745"
rm -rf "logs/run_20260225_131745"
