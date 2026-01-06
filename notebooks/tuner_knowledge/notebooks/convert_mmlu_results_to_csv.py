#!/usr/bin/env python3
"""
Convert MMLU evaluation JSON results to CSV format.
Usage: python convert_mmlu_to_csv.py <input_json> <output_csv>
"""

import json
import csv
import sys
from pathlib import Path


def json_to_csv(json_path, csv_path):
    """Convert MMLU JSON results to CSV format."""
    
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract results
    results = data.get('results', {})
    
    # Prepare CSV data
    csv_rows = []
    
    for task_name, metrics in results.items():
        row = {
            'task': task_name,
            'accuracy': metrics.get('acc,none', 0),
            'accuracy_stderr': metrics.get('acc_stderr,none', 0),
            'alias': metrics.get('alias', '')
        }
        csv_rows.append(row)
    
    # Sort by task name for consistency
    csv_rows.sort(key=lambda x: x['task'])
    
    # Write CSV file
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['task', 'accuracy', 'accuracy_stderr', 'alias']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"✓ Converted {len(csv_rows)} tasks from {json_path} to {csv_path}")
    
    # Print summary statistics
    overall = results.get('mmlu', {})
    if overall:
        print(f"\nOverall MMLU Accuracy: {overall.get('acc,none', 0):.4f} ± {overall.get('acc_stderr,none', 0):.4f}")
    
    # Print category summaries
    categories = ['mmlu_humanities', 'mmlu_social_sciences', 'mmlu_stem', 'mmlu_other']
    print("\nCategory Results:")
    for cat in categories:
        if cat in results:
            cat_data = results[cat]
            print(f"  {cat}: {cat_data.get('acc,none', 0):.4f} ± {cat_data.get('acc_stderr,none', 0):.4f}")


def process_directory(input_dir, output_dir=None):
    """Process all JSON files in a directory."""
    input_path = Path(input_dir)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    json_files = list(input_path.glob('**/*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s) to convert\n")
    
    for json_file in json_files:
        # Create corresponding CSV filename
        csv_filename = json_file.stem + '.csv'
        csv_path = output_path / csv_filename
        
        print(f"Processing: {json_file.name}")
        try:
            json_to_csv(json_file, csv_path)
            print()
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Convert single file:    python convert_mmlu_to_csv.py <input.json> [output.csv]")
        print("  Convert directory:      python convert_mmlu_to_csv.py <input_dir> [output_dir]")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    input_path = Path(input_arg)
    
    if input_path.is_file():
        # Single file conversion
        if output_arg:
            output_file = output_arg
        else:
            output_file = input_path.stem + '.csv'
        
        json_to_csv(input_arg, output_file)
    
    elif input_path.is_dir():
        # Directory conversion
        process_directory(input_arg, output_arg)
    
    else:
        print(f"Error: {input_arg} is not a valid file or directory")
        sys.exit(1)
