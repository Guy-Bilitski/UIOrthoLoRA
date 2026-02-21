#!/usr/bin/env python3
"""
Parse BigBench evaluation results from multiple model directories into a comparison CSV.
"""

import json
import re
import csv
from pathlib import Path
from typing import Optional


def extract_model_config(model_path: str) -> dict:
    """
    Extract configuration from model path like:
    models/google_gemma-3-12b-it_lora_trAll_lora_r3_lr1e-4
    
    Returns dict with: base_model, peft_method, rank, learning_rate, trainable_modules
    """
    config = {
        "base_model": None,
        "peft_method": None,
        "rank": None,
        "learning_rate": None,
        "trainable_modules": None,
    }
    
    # Extract from path
    path_str = str(model_path)
    
    # Try to extract base model (e.g., google_gemma-3-12b-it or google/gemma-3-12b-it)
    base_model_match = re.search(r'(google[/_]gemma-[\w-]+)', path_str)
    if base_model_match:
        config["base_model"] = base_model_match.group(1).replace("_", "/", 1)
    
    # Extract PEFT method (lora, vera, randlora, uiortholora, etc.)
    peft_match = re.search(r'_(uiortholora|ortholora|randlora|lora|vera|dora)_', path_str, re.IGNORECASE)
    if peft_match:
        config["peft_method"] = peft_match.group(1).lower()
    
    # Extract rank/size (r or s followed by number)
    # Handles: _r3_, _r8_, _r1024_, _s1024_, etc.
    rank_match = re.search(r'_[rs](\d+)_', path_str)
    if rank_match:
        config["rank"] = int(rank_match.group(1))
    
    # Extract learning rate (lr followed by scientific notation or decimal)
    lr_match = re.search(r'lr([\d.e-]+)', path_str, re.IGNORECASE)
    if lr_match:
        config["learning_rate"] = lr_match.group(1)
    
    # Extract trainable modules (trAll, trQKV, etc.)
    tr_match = re.search(r'_(tr[A-Za-z]+)_', path_str)
    if tr_match:
        config["trainable_modules"] = tr_match.group(1)
    
    return config


def parse_results_file(filepath: Path) -> Optional[dict]:
    """
    Parse a single BigBench results JSON file.
    
    Returns dict with model config and all task accuracies.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Extract model path from config
    model_args = data.get("config", {}).get("model_args", "")
    pretrained_match = re.search(r'pretrained=([^,]+)', model_args)
    model_path = pretrained_match.group(1) if pretrained_match else str(filepath.parent)
    
    # Get model configuration
    result = extract_model_config(model_path)
    result["model_path"] = model_path
    result["results_file"] = str(filepath)
    
    # Extract evaluation metadata
    result["eval_date"] = data.get("date", None)
    result["eval_duration_seconds"] = data.get("config", {}).get("total_evaluation_time_seconds", 
                                       data.get("total_evaluation_time_seconds", None))
    
    # Extract accuracies for each task
    results = data.get("results", {})
    for task_name, task_results in results.items():
        # Clean up task name for column header
        clean_name = task_name.replace("bigbench_", "").replace("_multiple_choice", "")
        
        # Get accuracy and stderr
        acc = task_results.get("acc,none", task_results.get("acc", None))
        stderr = task_results.get("acc_stderr,none", task_results.get("acc_stderr", None))
        
        result[f"acc_{clean_name}"] = acc
        result[f"stderr_{clean_name}"] = stderr
    
    # Calculate mean accuracy across all tasks
    acc_values = [v for k, v in result.items() if k.startswith("acc_") and v is not None]
    if acc_values:
        result["mean_accuracy"] = sum(acc_values) / len(acc_values)
    
    return result


def find_results_files(base_dir: Path, pattern: str = "**/results*.json") -> list[Path]:
    """Find all results JSON files in the directory tree."""
    return list(base_dir.glob(pattern))


def parse_all_results(base_dir: str | Path, output_csv: str | Path = "bigbench_comparison.csv") -> list[dict]:
    """
    Parse all BigBench results in a directory and save to CSV.
    
    Args:
        base_dir: Root directory containing model result subdirectories
        output_csv: Output CSV file path
        
    Returns:
        List of parsed result dictionaries
    """
    base_dir = Path(base_dir)
    output_csv = Path(output_csv)
    
    # Find all results files
    results_files = find_results_files(base_dir)
    print(f"Found {len(results_files)} results files in {base_dir}")
    
    if not results_files:
        print("No results files found!")
        return []
    
    # Parse all files
    all_results = []
    for filepath in sorted(results_files):
        print(f"Parsing: {filepath}")
        result = parse_results_file(filepath)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("No results parsed successfully!")
        return []
    
    # Determine all columns (union of all keys)
    all_columns = set()
    for result in all_results:
        all_columns.update(result.keys())
    
    # Order columns: config first, then mean, then individual tasks
    config_cols = ["model_path", "base_model", "peft_method", "rank", "learning_rate", 
                   "trainable_modules", "mean_accuracy"]
    acc_cols = sorted([c for c in all_columns if c.startswith("acc_")])
    stderr_cols = sorted([c for c in all_columns if c.startswith("stderr_")])
    meta_cols = ["results_file", "eval_date", "eval_duration_seconds"]
    
    ordered_columns = config_cols + acc_cols + stderr_cols + meta_cols
    # Add any remaining columns not yet included
    ordered_columns += [c for c in sorted(all_columns) if c not in ordered_columns]
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_columns, extrasaction='ignore')
        writer.writeheader()
        
        # Sort by peft_method, rank, learning_rate for easier comparison
        sorted_results = sorted(all_results, 
                                key=lambda x: (x.get("peft_method") or "", 
                                              x.get("rank") or 0, 
                                              x.get("learning_rate") or ""))
        writer.writerows(sorted_results)
    
    print(f"\nWrote {len(all_results)} results to {output_csv}")
    print(f"Columns: {len(ordered_columns)}")
    
    # Print summary
    print("\n=== Summary ===")
    for result in sorted_results:
        method = result.get("peft_method") or "?"
        rank = result.get("rank") if result.get("rank") is not None else "?"
        lr = result.get("learning_rate") or "?"
        mean_acc = result.get("mean_accuracy", 0) or 0
        print(f"{method:12} r={str(rank):<5} lr={str(lr):<10} -> mean_acc={mean_acc:.4f}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parse_bigbench_results.py <base_directory> [output.csv]")
        print("\nExample:")
        print("  python parse_bigbench_results.py ./eval_results/ comparison.csv")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "bigbench_comparison.csv"
    
    parse_all_results(base_dir, output_csv)