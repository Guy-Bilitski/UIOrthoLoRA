#!/usr/bin/env python
"""
Aggregate MMLU results for all models under this directory.

Usage:
  python summarize_mmlu_results.py <base_directory> [output.csv]
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def get_base_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path(".").resolve()


def find_result_files(base_dir: Path) -> List[Path]:
    return sorted(base_dir.rglob("results_*.json"))


def extract_model_name(data: Dict[str, Any], json_path: Path) -> str:
    model_name = data.get("model_name")
    if not model_name:
        model_name = data.get("model_name_sanitized")
    if not model_name:
        model_name = data.get("config", {}).get("model_args")
    if not model_name:
        try:
            model_name = json_path.parent.name
        except Exception:
            model_name = str(json_path)
    return str(model_name)


def extract_metrics_from_json(data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    main_metrics: Dict[str, float] = {}
    group_metrics: Dict[str, float] = {}

    results = data.get("results", {}) or {}
    groups = data.get("groups", {}) or {}

    mmlu_obj = results.get("mmlu") or groups.get("mmlu")
    if not mmlu_obj:
        return main_metrics, group_metrics

    acc = mmlu_obj.get("acc,none")
    stderr = mmlu_obj.get("acc_stderr,none")

    if acc is None:
        return main_metrics, group_metrics

    main_metrics["mmlu_acc"] = float(acc)
    if stderr is not None:
        main_metrics["mmlu_acc_stderr"] = float(stderr)

    for group_name, group_data in groups.items():
        acc_val = group_data.get("acc,none")
        if acc_val is None:
            continue
        col_name = f"{group_name}_acc"
        if group_name == "mmlu":
            col_name = "mmlu_acc"
        group_metrics[col_name] = float(acc_val)

    return main_metrics, group_metrics


def aggregate_results(base_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    json_files = find_result_files(base_dir)
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
            continue

        json_date = data.get("date")
        if isinstance(json_date, (int, float)):
            recency_key = float(json_date)
        else:
            recency_key = json_path.stat().st_mtime

        existing = best_by_model.get(model_name)
        if existing is not None and recency_key <= existing["_recency"]:
            continue

        row: Dict[str, Any] = {}
        row["model_name"] = model_name
        row["json_path"] = str(json_path.relative_to(base_dir))
        row.update(main_metrics)
        row.update(group_metrics)
        row["_recency"] = recency_key

        best_by_model[model_name] = row

        for col in group_metrics.keys():
            group_columns_set.add(col)

    rows: List[Dict[str, Any]] = list(best_by_model.values())

    group_columns = sorted(c for c in group_columns_set if c != "mmlu_acc")

    fieldnames: List[str] = ["model_name", "mmlu_acc", "mmlu_acc_stderr"]
    fieldnames.extend(group_columns)
    fieldnames.append("json_path")

    rows.sort(key=lambda r: r.get("mmlu_acc", float("-inf")), reverse=True)

    return rows, fieldnames


def write_csv(out_path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r_out = {k: v for k, v in r.items() if not k.startswith("_")}
            writer.writerow(r_out)


def print_summary(rows: List[Dict[str, Any]]) -> None:
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
    import sys

    if len(sys.argv) >= 2:
        base_dir = Path(sys.argv[1]).resolve()
    else:
        base_dir = get_base_dir()

    if len(sys.argv) >= 3:
        output_csv = Path(sys.argv[2])
    else:
        output_csv = base_dir / "mmlu_summary.csv"

    print(f"Searching for MMLU results in: {base_dir}")

    rows, fieldnames = aggregate_results(base_dir)

    if not rows:
        print(f"No valid MMLU result files found under {base_dir}")
        return

    write_csv(output_csv, rows, fieldnames)

    print(f"Found {len(rows)} models with MMLU results.")
    print(f"Wrote to: {output_csv}")
    print()
    print_summary(rows)


if __name__ == "__main__":
    main()