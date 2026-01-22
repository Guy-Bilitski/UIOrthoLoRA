#!/usr/bin/env python3
"""
Simple script to deduplicate a JSONL file based on the 'id' field.
"""

import json
import sys

def dedup_jsonl(input_file, output_file=None):
    """
    Remove duplicate entries from a JSONL file based on the 'id' field.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output file (defaults to input_file with '_deduped' suffix)
    """
    if output_file is None:
        base = input_file.rsplit('.', 1)[0]
        output_file = f"{base}_deduped.jsonl"
    
    seen_ids = set()
    unique_records = []
    duplicates = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                record_id = record.get('id')
                
                if record_id is None:
                    print(f"Warning: Line {line_num} has no 'id' field, keeping it")
                    unique_records.append(record)
                elif record_id not in seen_ids:
                    seen_ids.add(record_id)
                    unique_records.append(record)
                else:
                    duplicates += 1
                    print(f"Duplicate found: id='{record_id}' at line {line_num}")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in unique_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"\nSummary:")
    print(f"  Total records read: {len(unique_records) + duplicates}")
    print(f"  Unique records: {len(unique_records)}")
    print(f"  Duplicates removed: {duplicates}")
    print(f"  Output written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dedup_jsonl.py <input.jsonl> [output.jsonl]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    dedup_jsonl(input_path, output_path)
