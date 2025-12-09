import json
import pandas as pd
import numpy as np
import re
import glob
import os

def get_group(sc):
    """Assign knowledge group based on score."""
    if sc == 1.0:
        return "HK" # High Knowledge
    elif sc == 0.0:
        return "UK" # Unknown Knowledge
    else:
        return "PK" # Partial Knowledge

def parse_model_name(model_name):
    """
    Extracts Adapter, Base Model, and Learning Rate from the model string.
    Example: meta-llama_Llama-3.1-8B-Instruct_lora_tr100_lora_r3_lr1e-3
    """
    model_lower = model_name.lower()
    
    # 1. Extract Adapter
    # We prioritize longer names to avoid partial matching (e.g., 'uiortholora' contains 'lora')
    adapter = "unknown"
    known_adapters = ["uiortholora", "randlora", "vera", "lora"]
    
    # Sort by length descending to match 'uiortholora' before 'lora'
    for cand in sorted(known_adapters, key=len, reverse=True):
        if cand in model_lower:
            adapter = cand
            break
            
    # 2. Extract Learning Rate (lr)
    # Looks for patterns like 'lr1e-3', 'lr0.001', '_lr5e-4'
    lr_match = re.search(r'lr([0-9eE\.-]+)', model_name)
    lr = lr_match.group(1) if lr_match else "N/A"
    
    # 3. Extract Base Model Hint
    # Takes the part before the adapter name
    if adapter != "unknown":
        parts = model_name.split(adapter)
        base_part = parts[0].strip('_-')
    else:
        base_part = model_name
        
    return base_part, adapter, lr

def compute_metrics(df, ft_score_col, ft_train_col, base_score_col):
    """
    Computes Accuracy (on trained) and Negative Shifts (on untrained).
    """
    # --- 1. Accuracy on Trained Samples ---
    trained_mask = df[ft_train_col] == True
    trained_df = df[trained_mask]
    
    if trained_df.empty:
        accuracy = 0.0
        n_trained = 0
    else:
        accuracy = trained_df[ft_score_col].mean()
        n_trained = trained_mask.sum()

    # --- 2. Negative Shifts on Untrained Samples ---
    # Calculate shift
    df["sc_shift"] = df[ft_score_col] - df[base_score_col]
    
    # Determine Knowledge Groups (Before)
    df["group_before"] = df[base_score_col].apply(get_group)
    
    # Define Filter Mask (Matches your logic: Known OR (Unknown & Validation))
    # If it was Unknown (0.0) and NOT validation, we usually ignore it as 'noise' or 'training candidate'
    mask = (
        (df["group_before"] != "UK") |
        ((df["group_before"] == "UK") & (df["is_validation"]))
    )
    
    # Filter: Untrained AND satisfying the mask
    untrained_mask = (df[ft_train_col] == False) & mask
    filtered_df = df[untrained_mask]
    
    # Count Negative Shifts (score drop < -0.2)
    neg_shifts_count = (filtered_df["sc_shift"] < -0.2).sum()
    
    return n_trained, accuracy, neg_shifts_count

def main():
    # Find all jsonl files
    jsonl_files = glob.glob("../results/llama-8b/workdir/*.jsonl")
    print(f"Found files: {jsonl_files}")
    
    all_results = []
    
    for fpath in jsonl_files:
        print(f"Processing {fpath}...")
        
        # Load data into a list of dicts first
        rows = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Base Evaluation
                base_eval = obj.get("base_eval", {})
                base_score = float(base_eval.get("score", 0.0))
                
                # Validation Logic: If base score is non-zero, treat as validation
                is_val = bool(obj.get("is_validation", False))
                if base_score != 0.0:
                    is_val = True
                
                # We create a base record
                record = {
                    "base_score": base_score,
                    "is_validation": is_val
                }
                
                # Fine-Tuning Evaluations
                ft_evals = obj.get("ft_evals", {}) or {}
                for m_name, m_data in ft_evals.items():
                    # Flatten: store score and train status with model name prefix
                    record[f"{m_name}_score"] = float(m_data.get("score", 0.0))
                    record[f"{m_name}_train"] = bool(m_data.get("train", False))
                
                rows.append(record)
        
        if not rows:
            print(f"No data found in {fpath}")
            continue
            
        df = pd.DataFrame(rows)
        
        # Identify all unique FT models in this file
        # We look for columns ending in '_score' that are not 'base_score'
        ft_cols = [c for c in df.columns if c.endswith("_score") and c != "base_score"]
        ft_model_names = [c.replace("_score", "") for c in ft_cols]
        
        for m_name in ft_model_names:
            score_col = f"{m_name}_score"
            train_col = f"{m_name}_train"
            
            # Safety check if columns exist
            if score_col not in df.columns or train_col not in df.columns:
                continue
            
            # Calculate Metrics
            n_trained, acc, neg_shifts = compute_metrics(
                df.copy(), score_col, train_col, "base_score"
            )
            
            # Extract Metadata
            base_type, adapter, lr = parse_model_name(m_name)
            
            all_results.append({
                "Model Name": m_name,
                "Base Model Type": base_type,
                "Adapter": adapter,
                "Learning Rate": lr,
                "Training Samples (Count)": n_trained,
                "Accuracy": round(acc, 4),
                "Negative Shifts": neg_shifts,
                "Source File": fpath
            })

    # Save Results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_filename = "training_results_summary.csv"
        results_df.to_csv(output_filename, index=False)
        print("-" * 30)
        print(f"Successfully saved results to '{output_filename}'")
        print("-" * 30)
        print(results_df.head().to_string())
    else:
        print("No results extracted. Check if JSONL files contain 'ft_evals'.")

if __name__ == "__main__":
    main()