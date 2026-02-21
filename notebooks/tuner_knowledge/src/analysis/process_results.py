#!/usr/bin/env python3
"""
Process JSONL evaluation files and extract metrics to CSV.

Logic:
  For each (model, dataset) in MODELS x DATASETS:
    1. Look for ../results/{model}/{dataset}/workdir/*.jsonl
    2. Process all JSONL files found
    3. For each threshold, save to ./adapters_results/{dataset}/threshold_{t}/{model}.csv
"""

import json
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these lists as you add models/datasets
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = ["gemma-12b", "llama-3b"]
DATASETS = ["hotpotqa", "triviaqa"]
THRESHOLDS = [0.8]

RESULTS_ROOT = Path("../results")
OUTPUT_ROOT = Path("./adapters_results")

ADAPTER_TYPES = ["dora", "uiortholora", "randlora", "vera", "lora"]

CSV_COLUMNS = [
    "source_file", "adapter_type", "lr", "rank",
    "tr_from_name", "tr_actual", "accuracy",
    "negative_shift_count", "negative_shift_score",
    "positive_shift_count", "positive_shift_score",
    "raw_adapter_name",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTER NAME PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_adapter_name(name: str) -> dict:
    result = {"adapter_type": None, "lr": None, "rank": None, "tr_from_name": None}

    m = re.search(r"_tr(\d+)", name)
    if m:
        result["tr_from_name"] = int(m.group(1))

    m = re.search(r"_lr([0-9e\-\.]+)", name, re.IGNORECASE)
    if m:
        result["lr"] = m.group(1)

    m = re.search(r"_r(\d+)(?:_|$)", name)
    if m:
        result["rank"] = int(m.group(1))

    for a in ADAPTER_TYPES:
        if a in name.lower():
            result["adapter_type"] = a
            break

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> tuple:
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))

    ft_names = sorted({k for obj in raw for k in (obj.get("ft_evals") or {})})

    san = lambda s: re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    col_map = {m: (f"{san(m)}_score", f"{san(m)}_train") for m in ft_names}

    rows = []
    for obj in raw:
        be = obj.get("base_eval") or {}
        bs = be.get("score", 0.0)
        bs = float(bs) if bs is not None else np.nan
        is_val = bool(obj.get("is_validation", False))
        if not np.isnan(bs) and bs != 0.0:
            is_val = True

        rec = {"qid": obj.get("id"), "is_validation": is_val, "base_score": bs}
        for m in ft_names:
            sc, tr = col_map[m]
            rec[sc] = np.nan
            rec[tr] = False
        for m, p in (obj.get("ft_evals") or {}).items():
            if m not in col_map:
                continue
            sc, tr = col_map[m]
            if isinstance(p, dict):
                if "score" in p:
                    rec[sc] = float(p["score"])
                if "train" in p:
                    rec[tr] = bool(p["train"])
        rows.append(rec)

    return pd.DataFrame(rows), ft_names, col_map


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _kg(sc):
    if sc == 1.0: return "HK"
    if sc == 0.0: return "UK"
    return "PK"


def compute_metrics(df, ft_name, col_map, threshold):
    sc_col, tr_col = col_map[ft_name]
    ev = pd.DataFrame({
        "sc_before": df["base_score"], "sc_after": df[sc_col],
        "trained": df[tr_col], "is_validation": df["is_validation"],
    })
    ev["group_before"] = ev["sc_before"].apply(_kg)
    ev["sc_shift"] = ev["sc_after"] - ev["sc_before"]

    trained = ev[ev["trained"]]
    accuracy = trained["sc_after"].mean() if len(trained) > 0 else 0.0
    tr_actual = int(ev["trained"].sum())

    mask = (ev["group_before"] != "UK") | ((ev["group_before"] == "UK") & ev["is_validation"])
    filt = ev[~ev["trained"] & mask]
    neg = filt[filt["sc_shift"] < -threshold]
    pos = filt[filt["sc_shift"] > threshold]

    return {
        "tr_actual": tr_actual, "accuracy": accuracy,
        "negative_shift_count": len(neg), "negative_shift_score": neg["sc_shift"].abs().sum(),
        "positive_shift_count": len(pos), "positive_shift_score": pos["sc_shift"].abs().sum(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS ONE JSONL FILE
# ═══════════════════════════════════════════════════════════════════════════════

def process_file(path: str, threshold: float) -> list:
    print(f"    Processing: {Path(path).name}")
    try:
        df, ft_names, col_map = load_jsonl(path)
    except Exception as e:
        print(f"      ✗ Error: {e}")
        return []

    if not ft_names:
        print(f"      (no adapters)")
        return []

    print(f"      {len(df)} rows, {len(ft_names)} adapters")
    results = []
    for ft in ft_names:
        parsed = parse_adapter_name(ft)
        try:
            m = compute_metrics(df, ft, col_map, threshold)
        except Exception as e:
            print(f"      ✗ {ft}: {e}")
            continue
        results.append({
            "source_file": os.path.basename(path),
            "adapter_type": parsed["adapter_type"], "lr": parsed["lr"],
            "rank": parsed["rank"], "tr_from_name": parsed["tr_from_name"],
            "tr_actual": m["tr_actual"], "accuracy": m["accuracy"],
            "negative_shift_count": m["negative_shift_count"],
            "negative_shift_score": m["negative_shift_score"],
            "positive_shift_count": m["positive_shift_count"],
            "positive_shift_score": m["positive_shift_score"],
            "raw_adapter_name": ft,
        })
        print(f"        {str(parsed['adapter_type']):15s} tr={m['tr_actual']:5d} "
              f"acc={m['accuracy']:.3f} neg={m['negative_shift_count']}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"{'='*70}")
    print("RESULTS PROCESSOR")
    print(f"{'='*70}")
    print(f"Models:   {MODELS}")
    print(f"Datasets: {DATASETS}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    for model in MODELS:
        for dataset in DATASETS:
            workdir = RESULTS_ROOT / model / dataset / "workdir"
            jsonl_files = sorted(workdir.glob("*.jsonl")) if workdir.exists() else []

            print(f"\n{'─'*70}")
            print(f"  {model} / {dataset}")
            print(f"  workdir: {workdir}")
            print(f"  JSONL files: {len(jsonl_files)}")

            if not jsonl_files:
                # No data yet — save empty CSVs for all thresholds
                for t in THRESHOLDS:
                    out_dir = OUTPUT_ROOT / dataset / f"threshold_{t}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{model}.csv"
                    pd.DataFrame(columns=CSV_COLUMNS).to_csv(out_path, index=False)
                    print(f"    Saved (empty): {out_path}")
                continue

            # Process once per threshold
            for t in THRESHOLDS:
                print(f"\n  threshold={t}:")
                all_results = []
                for f in jsonl_files:
                    all_results.extend(process_file(str(f), t))

                if all_results:
                    df = pd.DataFrame(all_results)
                    df = df[[c for c in CSV_COLUMNS if c in df.columns]]
                    df = df.sort_values(["adapter_type", "tr_actual", "lr"])
                else:
                    df = pd.DataFrame(columns=CSV_COLUMNS)

                out_dir = OUTPUT_ROOT / dataset / f"threshold_{t}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{model}.csv"
                df.to_csv(out_path, index=False)

                if len(df) > 0:
                    print(f"\n    → {out_path} ({len(df)} adapters)")
                    print(f"      avg_acc={df['accuracy'].mean():.3f}  "
                          f"avg_neg={df['negative_shift_count'].mean():.1f}")
                else:
                    print(f"\n    → {out_path} (empty)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for t in THRESHOLDS:
        print(f"\n  threshold={t}:")
        for model in MODELS:
            for dataset in DATASETS:
                p = OUTPUT_ROOT / dataset / f"threshold_{t}" / f"{model}.csv"
                if p.exists():
                    df = pd.read_csv(p)
                    if len(df) > 0:
                        print(f"    {model:15s} / {dataset:10s}: {len(df):4d} adapters  "
                              f"acc={df['accuracy'].mean():.3f}  neg={df['negative_shift_count'].mean():.1f}")
                    else:
                        print(f"    {model:15s} / {dataset:10s}: (no results yet)")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()