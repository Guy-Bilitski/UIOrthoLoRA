#!/bin/bash
echo "Cleaning run 20251130_104611"
echo "Killing process group: 125458"
kill -9 -125458 2>/dev/null || true
echo "Deleting logs/"
rm -rf logs
