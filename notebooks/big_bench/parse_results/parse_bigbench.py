#!/usr/bin/env python3
"""
BigBench Results Parser
Parses evaluation results from lm_eval output and generates CSV reports.
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def find_result_files(results_dir: Path) -> List[Path]:
    """Find all results JSON files in the directory structure."""
    return list(results_dir.rglob("results_*.json"))


def extract_model_name(result_file: Path, results_dir: Path) -> str:
    """Extract model name from the directory structure."""
    # Get the parent directory of the results file
    model_dir = result_file.parent.parent
    # Get relative path from results_dir
    rel_path = model_dir.relative_to(results_dir)
    return str(rel_path)


def parse_result_file(result_file: Path) -> Dict:
    """Parse a single results JSON file."""
    with open(result_file, 'r') as f:
        return json.load(f)


def extract_task_results(data: Dict) -> List[Tuple[str, float, float]]:
    """Extract task name, accuracy, and stderr from results."""
    results = []
    for task, metrics in data.get('results', {}).items():
        acc = metrics.get('acc,none', None)
        stderr = metrics.get('acc_stderr,none', None)
        if acc is not None:
            results.append((task, acc, stderr if stderr is not None else 0.0))
    return results


def save_detailed_results(all_results: List[Tuple[str, str, float, float]], output_file: Path):
    """Save detailed results to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Task', 'Accuracy', 'Stderr'])
        for model, task, acc, stderr in sorted(all_results):
            writer.writerow([model, task, f"{acc:.6f}", f"{stderr:.6f}"])
    print(f"✓ Detailed results saved to {output_file}")


def calculate_summary(all_results: List[Tuple[str, str, float, float]]) -> Dict[str, Tuple[float, int]]:
    """Calculate average accuracy and task count per model."""
    model_scores = defaultdict(list)
    for model, task, acc, stderr in all_results:
        model_scores[model].append(acc)
    
    summary = {}
    for model, scores in model_scores.items():
        avg_acc = sum(scores) / len(scores)
        summary[model] = (avg_acc, len(scores))
    
    return summary


def save_summary(summary: Dict[str, Tuple[float, int]], output_file: Path):
    """Save summary results to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Average_Accuracy', 'Num_Tasks'])
        for model, (avg_acc, num_tasks) in sorted(summary.items(), key=lambda x: x[1][0], reverse=True):
            writer.writerow([model, f"{avg_acc:.6f}", num_tasks])
    print(f"✓ Summary saved to {output_file}")


def main(results_dir: str, output_dir: str):
    """Main execution function."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all result files
    result_files = find_result_files(results_path)
    
    if not result_files:
        print(f"Warning: No results files found in {results_dir}")
        sys.exit(0)
    
    print(f"Found {len(result_files)} result file(s)")
    
    # Parse all results
    all_results = []
    for result_file in result_files:
        model_name = extract_model_name(result_file, results_path)
        print(f"Processing: {model_name}")
        
        try:
            data = parse_result_file(result_file)
            task_results = extract_task_results(data)
            
            for task, acc, stderr in task_results:
                all_results.append((model_name, task, acc, stderr))
        
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue
    
    if not all_results:
        print("No results extracted")
        sys.exit(0)
    
    # Save outputs
    detailed_csv = output_path / "bigbench_results.csv"
    summary_csv = output_path / "bigbench_summary.csv"
    
    save_detailed_results(all_results, detailed_csv)
    
    summary = calculate_summary(all_results)
    save_summary(summary, summary_csv)
    
    print(f"\n✓ Processing complete!")
    print(f"  - {len(all_results)} task results from {len(set(r[0] for r in all_results))} models")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_bigbench.py <results_dir> <output_dir>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])