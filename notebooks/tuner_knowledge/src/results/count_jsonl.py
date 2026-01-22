#!/usr/bin/env python3
"""Count JSON objects in a JSONL file."""

import sys

def count_jsonl(filepath):
    with open(filepath, 'r') as f:
        count = sum(1 for line in f if line.strip())
    print(f"{count} JSON objects in {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_jsonl.py <file.jsonl>")
        sys.exit(1)
    count_jsonl(sys.argv[1])
