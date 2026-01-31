#!/usr/bin/env python3
"""
Script to process JSONL evaluation files and extract metrics to CSV.
Handles lora, uiortholora, vera, randlora adapter types.

Output CSV columns:
- base_model: e.g., "Llama-3.1-8B-Instruct"
- adapter_type: e.g., "lora", "vera", "uiortholora", "randlora"
- lr: learning rate (e.g., "1e-3")
- tr_from_name: training samples from model name
- tr_actual: actual count of trained samples (from data)
- accuracy: mean score on trained samples
- negative_shift_count: number of samples with negative shift > 0.2
"""

import json
import re
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def _sanitize(name: str) -> str:
    """Make a string safe for use as a column name."""
    return re.sub(r'[^0-9a-zA-Z]+', '_', name).strip('_')


def parse_adapter_name(adapter_name: str) -> dict:
    """
    Parse adapter model name to extract parameters.
    
    Examples:
    - "meta-llama_Llama-3.1-8B-Instruct_lora_tr100_lora_r3_lr1e-3"
    - "meta-llama_Llama-3.2-3B_uiortholora_tr500_uiortholora_s256_v64"
    - "meta-llama_Llama-3.2-3B_vera_r_tr100"
    - "meta-llama_Llama-3.2-3B_randlora_r_tr100"
    
    Returns dict with: base_model, adapter_type, lr, tr_from_name
    """
    result = {
        'base_model': None,
        'adapter_type': None,
        'lr': None,
        'tr_from_name': None,
        'rank': None,
        'raw_name': adapter_name
    }
    
    # Extract training samples from name (tr followed by number)
    tr_match = re.search(r'_tr(\d+)', adapter_name)
    if tr_match:
        result['tr_from_name'] = int(tr_match.group(1))
    
    # Extract learning rate (lr followed by scientific notation or decimal)
    lr_match = re.search(r'_lr([0-9e\-\.]+)', adapter_name, re.IGNORECASE)
    if lr_match:
        result['lr'] = lr_match.group(1)
    
    # Extract rank (r followed by number, but not in "tr")
    rank_match = re.search(r'_r(\d+)(?:_|$)', adapter_name)
    if rank_match:
        result['rank'] = int(rank_match.group(1))
    
    # Determine adapter type
    adapter_types = ['uiortholora', 'randlora', 'vera', 'lora']  # order matters: check longer names first
    for adapter in adapter_types:
        if adapter in adapter_name.lower():
            result['adapter_type'] = adapter
            break
    
    # Extract base model name
    # Pattern: organization_ModelName_adaptertype...
    # e.g., "meta-llama_Llama-3.1-8B-Instruct_lora..."
    parts = adapter_name.split('_')
    if len(parts) >= 2:
        # Try to find the model name before the adapter type appears
        model_parts = []
        for i, part in enumerate(parts):
            part_lower = part.lower()
            # Stop when we hit an adapter type keyword
            if any(a in part_lower for a in adapter_types):
                break
            model_parts.append(part)
        
        if len(model_parts) >= 2:
            # Skip organization prefix (e.g., "meta-llama")
            result['base_model'] = '_'.join(model_parts[1:]) if len(model_parts) > 1 else model_parts[0]
        elif len(model_parts) == 1:
            result['base_model'] = model_parts[0]
    
    return result


def load_jsonl(json_path: str) -> tuple:
    """
    Load JSONL file and return processed data.
    
    Returns:
        - df: DataFrame with all data
        - ft_model_names: list of FT model names found
        - model_to_cols: dict mapping model name to (score_col, train_col)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    rows = []
    ft_model_names = set()
    base_ids = set()

    # First pass: collect all FT model names and base ids
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            base_eval = obj.get("base_eval", {})
            b_id = base_eval.get("model_id")
            if b_id:
                base_ids.add(b_id)
            ft = obj.get("ft_evals", {}) or {}
            for k in ft.keys():
                ft_model_names.add(k)

    ft_model_names = sorted(ft_model_names)
    base_ids = sorted(base_ids)

    # Map FT model to (score_col, train_col)
    model_to_cols = {}
    for m in ft_model_names:
        s = _sanitize(m)
        sc_col = f"{s}_score"
        tr_col = f"{s}_train"
        model_to_cols[m] = (sc_col, tr_col)

    # Second pass: build rows
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            qid = obj.get("id")
            is_val = bool(obj.get("is_validation", False))

            base_eval = obj.get("base_eval", {}) or {}
            base_model_id = base_eval.get("model_id")
            base_score = base_eval.get("score", 0.0)

            # Rule: if base score != 0 → set validation to True
            if (base_score is not None) and (float(base_score) != 0.0):
                is_val = True

            rec = {
                "qid": qid,
                "is_validation": is_val,
                "base_model_id": base_model_id,
                "base_score": float(base_score) if base_score is not None else np.nan,
            }

            # Init FT cols
            for m in ft_model_names:
                sc_col, tr_col = model_to_cols[m]
                rec[sc_col] = np.nan
                rec[tr_col] = False

            # Fill FT values present in this record
            ft = obj.get("ft_evals", {}) or {}
            for m, payload in ft.items():
                if m not in model_to_cols:
                    continue
                sc_col, tr_col = model_to_cols[m]
                if isinstance(payload, dict):
                    if "score" in payload:
                        rec[sc_col] = float(payload["score"])
                    if "train" in payload:
                        rec[tr_col] = bool(payload["train"])

            rows.append(rec)

    df = pd.DataFrame(rows)
    return df, ft_model_names, model_to_cols


def assign_knowledge_groups(sc):
    """Assign knowledge group based on score."""
    if sc == 1.0:
        return "HK"
    elif sc == 0.0:
        return "UK"
    else:
        return "PK"


def compute_metrics_for_adapter(df: pd.DataFrame, ft_model_name: str, model_to_cols: dict, threshold: float = 0.2) -> dict:
    """
    Compute accuracy and negative shift count for a single adapter.
    
    Args:
        df: DataFrame with all data
        ft_model_name: raw FT model name
        model_to_cols: dict mapping model name to (score_col, train_col)
        threshold: threshold for counting shifts (default 0.2)
    
    Returns:
        dict with metrics
    """
    score_col, train_col = model_to_cols[ft_model_name]
    
    # Create evaluation DataFrame
    eval_df = pd.DataFrame({
        "sc_before": df["base_score"],
        "sc_after": df[score_col],
        "trained": df[train_col],
        "is_validation": df["is_validation"],
        "qid": df["qid"],
    })
    
    # Assign knowledge groups
    eval_df["group_before"] = eval_df["sc_before"].apply(assign_knowledge_groups)
    eval_df["group_after"] = eval_df["sc_after"].apply(assign_knowledge_groups)
    
    # Compute accuracy on trained samples
    trained = eval_df[eval_df["trained"]]
    if trained.empty:
        accuracy = 0.0
    else:
        accuracy = trained["sc_after"].mean()
    
    # Count actual trained samples
    tr_actual = int(eval_df["trained"].sum())
    
    # Compute shift
    eval_df["sc_shift"] = eval_df["sc_after"] - eval_df["sc_before"]
    
    # Filter for shift computation:
    # Exclude trained samples
    # For UK samples, only include if is_validation
    mask = (
        (eval_df["group_before"] != "UK") |
        ((eval_df["group_before"] == "UK") & (eval_df["is_validation"]))
    )
    filtered = eval_df[~eval_df["trained"] & mask]
    
    # Negative shift: samples where shift < -threshold
    neg_df = filtered[filtered["sc_shift"] < -threshold]
    negative_shift_count = len(neg_df)
    negative_shift_score = neg_df["sc_shift"].abs().sum()
    
    # Positive shift for completeness
    pos_df = filtered[filtered["sc_shift"] > threshold]
    positive_shift_count = len(pos_df)
    positive_shift_score = pos_df["sc_shift"].abs().sum()
    
    return {
        'tr_actual': tr_actual,
        'accuracy': accuracy,
        'negative_shift_count': negative_shift_count,
        'negative_shift_score': negative_shift_score,
        'positive_shift_count': positive_shift_count,
        'positive_shift_score': positive_shift_score,
    }


def process_jsonl_file(json_path: str, threshold: float = 0.2) -> list:
    """
    Process a single JSONL file and return list of result dicts.
    
    Args:
        json_path: path to JSONL file
        threshold: threshold for counting shifts
    """
    print(f"Processing: {json_path}")
    
    try:
        df, ft_model_names, model_to_cols = load_jsonl(json_path)
    except Exception as e:
        print(f"  Error loading file: {e}")
        return []
    
    if not ft_model_names:
        print(f"  No FT models found in file")
        return []
    
    print(f"  Found {len(ft_model_names)} adapter models")
    
    results = []
    for ft_model_name in ft_model_names:
        # Parse adapter name
        parsed = parse_adapter_name(ft_model_name)
        
        # Compute metrics
        try:
            metrics = compute_metrics_for_adapter(df, ft_model_name, model_to_cols, threshold)
        except Exception as e:
            print(f"  Error computing metrics for {ft_model_name}: {e}")
            continue
        
        # Combine parsed info with metrics
        result = {
            'source_file': os.path.basename(json_path),
            'base_model': parsed['base_model'],
            'adapter_type': parsed['adapter_type'],
            'lr': parsed['lr'],
            'rank': parsed['rank'],
            'tr_from_name': parsed['tr_from_name'],
            'tr_actual': metrics['tr_actual'],
            'accuracy': metrics['accuracy'],
            'negative_shift_count': metrics['negative_shift_count'],
            'negative_shift_score': metrics['negative_shift_score'],
            'positive_shift_count': metrics['positive_shift_count'],
            'positive_shift_score': metrics['positive_shift_score'],
            'raw_adapter_name': ft_model_name,
        }
        results.append(result)
        
        print(f"    {parsed['adapter_type']}: tr={metrics['tr_actual']}, acc={metrics['accuracy']:.3f}, neg_shift={metrics['negative_shift_count']}")
    
    return results


def main():
    """
    Main function to process all JSONL files and save results to CSV.
    """
    # Define thresholds to process
    thresholds = [0.6, 0.8]
    
    # Look for JSONL files in current directory and common locations
    search_paths = [
        "../results/gemma-12b/workdir/*.jsonl",
        # "../results/llama-8b/workdir/*.jsonl",
        "../results/llama-3b/triviaqa/workdir/*.jsonl"
    ]
    
    jsonl_files = []
    for pattern in search_paths:
        jsonl_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates
    jsonl_files = list(set(jsonl_files))
    
    if not jsonl_files:
        print("No JSONL files found. Please place your files in the current directory or specify paths.")
        print("Searched patterns:", search_paths)
        return
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f}")
    print()
    
    # Process for each threshold
    for threshold in thresholds:
        print(f"\n{'='*70}")
        print(f"Processing with threshold: {threshold}")
        print(f"{'='*70}")
        
        # Process all files with current threshold
        all_results = []
        for json_path in jsonl_files:
            results = process_jsonl_file(json_path, threshold)
            all_results.extend(results)
        
        if not all_results:
            print(f"No results extracted for threshold {threshold}.")
            continue
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns for readability
        column_order = [
            'source_file',
            'base_model',
            'adapter_type',
            'lr',
            'rank',
            'tr_from_name',
            'tr_actual',
            'accuracy',
            'negative_shift_count',
            'negative_shift_score',
            'positive_shift_count',
            'positive_shift_score',
            'raw_adapter_name',
        ]
        
        # Only include columns that exist
        column_order = [c for c in column_order if c in results_df.columns]
        results_df = results_df[column_order]
        
        # Sort by adapter_type, then by tr_actual
        results_df = results_df.sort_values(['adapter_type', 'tr_actual', 'lr'])
        
        # Create threshold-specific output directory
        threshold_dir = Path("adapters_results") / f"threshold_{threshold}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by base model and save separate CSV for each
        unique_models = results_df['base_model'].dropna().unique()
        
        print(f"\nSaving results for {len(unique_models)} models:")
        for model_name in unique_models:
            model_df = results_df[results_df['base_model'] == model_name].copy()
            
            # Save to CSV
            output_path = threshold_dir / f"adapter_results_{model_name}.csv"
            model_df.to_csv(output_path, index=False)
            
            print(f"  - {model_name}: {len(model_df)} adapters -> {output_path}")
            print(f"    Summary: avg accuracy={model_df['accuracy'].mean():.3f}, "
                  f"avg neg_shift={model_df['negative_shift_count'].mean():.1f}")
        
        print(f"\nOverall summary for threshold {threshold}:")
        print(results_df.groupby('adapter_type')[['tr_actual', 'accuracy', 'negative_shift_count']].mean().round(3))
    
    print(f"\n{'='*70}")
    print(f"Processing complete! Results saved in adapters_results/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()