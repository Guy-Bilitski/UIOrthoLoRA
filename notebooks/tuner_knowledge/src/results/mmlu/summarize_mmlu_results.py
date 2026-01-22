#!/usr/bin/env python
"""
Aggregate MMLU results for all models under this directory.

Usage (from repo root):
  python results/mmlu/summarize_mmlu_results.py

Or cd into results/mmlu and run:
  python summarize_mmlu_results.py
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


def get_base_dir() -> Path:
    # If running as a script, use the script location
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    # If running in a notebook, assume current working directory is results/mmlu
    return Path(".").resolve()


def find_result_files(base_dir: Path) -> List[Path]:
    # Recursively find all json files that look like lm-eval results
    return sorted(base_dir.rglob("results_*.json"))


def extract_model_name(data: Dict[str, Any], json_path: Path) -> str:
    # Prefer explicit model_name, then sanitized, then config, then directory name
    model_name = data.get("model_name")
    if not model_name:
        model_name = data.get("model_name_sanitized")
    if not model_name:
        model_name = data.get("config", {}).get("model_args")
    if not model_name:
        # Fallback: parent directory two levels up usually encodes experiment
        # e.g. .../google_gemma-3-12b-it_lora_tr1000_lr1e-3/models__.../results_*.json
        try:
            model_name = json_path.parent.name
        except Exception:
            model_name = str(json_path)
    return str(model_name)


def extract_metrics_from_json(data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns:
      main_metrics: keys like "mmlu_acc", "mmlu_acc_stderr"
      group_metrics: keys like "mmlu_humanities_acc", "mmlu_stem_acc", ...
    """
    main_metrics: Dict[str, float] = {}
    group_metrics: Dict[str, float] = {}

    results = data.get("results", {}) or {}
    groups = data.get("groups", {}) or {}

    # Overall MMLU accuracy
    # Try results["mmlu"] first, then groups["mmlu"]
    mmlu_obj = results.get("mmlu") or groups.get("mmlu")
    if not mmlu_obj:
        # Not an mmlu run or malformed
        return main_metrics, group_metrics

    acc = mmlu_obj.get("acc,none")
    stderr = mmlu_obj.get("acc_stderr,none")

    if acc is None:
        return main_metrics, group_metrics

    main_metrics["mmlu_acc"] = float(acc)
    if stderr is not None:
        main_metrics["mmlu_acc_stderr"] = float(stderr)

    # Group metrics (humanities, stem, social_sciences, other, etc)
    for group_name, group_data in groups.items():
        acc_val = group_data.get("acc,none")
        if acc_val is None:
            continue
        col_name = f"{group_name}_acc"
        # Avoid duplicating mmlu_acc which we already put in main_metrics
        if group_name == "mmlu":
            col_name = "mmlu_acc"
        group_metrics[col_name] = float(acc_val)

    return main_metrics, group_metrics


def aggregate_results(base_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    json_files = find_result_files(base_dir)

    # For each model_name keep the latest result by "date" field if present, else by mtime
    best_by_model: Dict[str, Dict[str, Any]] = {}
    group_columns_set = set()

    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {json_path} (failed to load json: {e})")
            continue

        model_name = extract_model_name(data, json_path)
        main_metrics, group_metrics = extract_metrics_from_json(data)
        if "mmlu_acc" not in main_metrics:
            # Not a valid mmlu json for our purposes
            continue

        # Decide recency
        json_date = data.get("date")
        if isinstance(json_date, (int, float)):
            recency_key = float(json_date)
        else:
            # Fallback to file modification time
            recency_key = json_path.stat().st_mtime

        existing = best_by_model.get(model_name)
        if existing is not None and recency_key <= existing["_recency"]:
            # We already have a newer result for this model
            continue

        row: Dict[str, Any] = {}
        row["model_name"] = model_name
        # Relative path to the json so you can track where it came from
        row["json_path"] = str(json_path.relative_to(base_dir))

        # Store metrics
        row.update(main_metrics)
        row.update(group_metrics)
        row["_recency"] = recency_key

        best_by_model[model_name] = row

        # Track all group columns that appear
        for col in group_metrics.keys():
            group_columns_set.add(col)

    # Build final rows and columns
    rows: List[Dict[str, Any]] = list(best_by_model.values())

    # Sorted list of group columns but make sure we do not duplicate "mmlu_acc"
    group_columns = sorted(
        c for c in group_columns_set
        if c != "mmlu_acc"  # keep a single mmlu_acc
    )

    # Column order
    fieldnames: List[str] = ["model_name", "mmlu_acc", "mmlu_acc_stderr"]
    fieldnames.extend(group_columns)
    fieldnames.append("json_path")  # for traceability

    # Sort rows by mmlu_acc descending
    rows.sort(key=lambda r: r.get("mmlu_acc", float("-inf")), reverse=True)

    return rows, fieldnames


def write_csv(base_dir: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> Path:
    out_path = base_dir / "mmlu_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r_out = {k: v for k, v in r.items() if not k.startswith("_")}
            writer.writerow(r_out)
    return out_path


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print()
    print(f"Found {len(rows)} models with MMLU results.")
    print("Summary (sorted by mmlu_acc):")
    print()

    # Try to print the most interesting group metrics if present
    preferred_groups = [
        "mmlu_humanities_acc",
        "mmlu_social_sciences_acc",
        "mmlu_other_acc",
        "mmlu_stem_acc",
    ]

    for row in rows:
        model = row.get("model_name", "unknown")
        mmlu_acc = row.get("mmlu_acc")
        mmlu_str = f"{mmlu_acc:.4f}" if isinstance(mmlu_acc, (int, float)) else "n/a"

        parts = [f"model={model}", f"mmlu={mmlu_str}"]

        for g in preferred_groups:
            if g in row:
                val = row[g]
                parts.append(f"{g.replace('_acc', '')}={val:.4f}")

        print(" | ".join(parts))


def main() -> None:
    base_dir = get_base_dir()
    rows, fieldnames = aggregate_results(base_dir)

    if not rows:
        print(f"No valid MMLU result files found under {base_dir}")
        return

    out_csv = write_csv(base_dir, rows, fieldnames)
    print_summary(rows)
    print()
    print(f"CSV written to: {out_csv}")


if __name__ == "__main__":
    main()

