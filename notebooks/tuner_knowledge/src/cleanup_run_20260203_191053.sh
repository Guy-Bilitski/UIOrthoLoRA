#!/bin/bash
set -euo pipefail
echo "Cleaning run 20260203_191053"
echo "Killing process group: 3559087"
kill -9 -3559087 2>/dev/null || true
echo "Removing logs for this run: logs/run_20260203_191053"
rm -rf "logs/run_20260203_191053"
