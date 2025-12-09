import json
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any


def _sanitize(name: str) -> str:
    """Make a string safe for use as a column name."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")


def load_json_to_df(json_path: str):
    """
    Load one of the MMLU jsonl result files into a wide DataFrame.

    Returns:
      df: dataframe with base_score, one column per base model, one score/train column per FT model
      ft_model_names: list of raw ft model names as they appear in ft_evals keys
      base_model_ids: list of raw base model ids
      base_score_cols: list of base_<id>_score column names in df
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    rows: List[Dict[str, Any]] = []
    ft_model_names: set = set()
    base_ids: set = set()

    # First pass: collect all FT model names and base ids
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            base_eval = obj.get("base_eval", {}) or {}
            b_id = base_eval.get("model_id")
            if b_id:
                base_ids.add(b_id)
            ft = obj.get("ft_evals", {}) or {}
            for k in ft.keys():
                ft_model_names.add(k)

    ft_model_names = sorted(ft_model_names)
    base_ids = sorted(base_ids)

    # Map FT model to (score_col, train_col)
    model_to_cols: Dict[str, Tuple[str, str]] = {}
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

            # rule: if base score != 0 -> set validation to True
            if base_score is not None and float(base_score) != 0.0:
                is_val = True

            rec: Dict[str, Any] = {
                "qid": qid,
                "is_validation": is_val,
                "base_model_id": base_model_id,
                "base_score": float(base_score) if base_score is not None else np.nan,
            }

            # one column per base id: base_<id>_score
            for b in base_ids:
                rec[f"base_{_sanitize(b)}_score"] = np.nan
            if base_model_id:
                rec[f"base_{_sanitize(base_model_id)}_score"] = rec["base_score"]

            # init FT cols
            for m in ft_model_names:
                sc_col, tr_col = model_to_cols[m]
                rec[sc_col] = np.nan
                rec[tr_col] = False

            # fill FT values present in this record
            ft = obj.get("ft_evals", {}) or {}
            for m, payload in ft.items():
                sc_col, tr_col = model_to_cols[m]
                if isinstance(payload, dict):
                    if "score" in payload:
                        rec[sc_col] = float(payload["score"])
                    if "train" in payload:
                        rec[tr_col] = bool(payload["train"])

            rows.append(rec)

    df = pd.DataFrame(rows)
    base_score_cols = [f"base_{_sanitize(b)}_score" for b in base_ids]
    return df, ft_model_names, base_ids, base_score_cols


def compute_accuracy(eval_df: pd.DataFrame) -> float:
    """
    Accuracy on trained samples: mean sc_after over rows with trained == True.
    """
    trained = eval_df[eval_df["trained"]]
    if trained.empty:
        return 0.0
    return float(trained["sc_after"].mean())


def assign_knowledge_groups(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    HK: score exactly 1
    UK: score exactly 0
    PK: everything else
    """
    def get_group(sc: float) -> str:
        if sc == 1.0:
            return "HK"
        elif sc == 0.0:
            return "UK"
        else:
            return "PK"

    eval_df = eval_df.copy()
    eval_df["group_before"] = eval_df["sc_before"].apply(get_group)
    eval_df["group_after"] = eval_df["sc_after"].apply(get_group)
    return eval_df


def compute_shifts(eval_df: pd.DataFrame):
    """
    Compute positive and negative shift scores and counts on untrained samples,
    following your original UK/PK/HK masking rule.

    Returns:
      psc: sum of positive shifts (absolute)
      count_p: number of samples with positive shifts
      nsc: sum of negative shifts (absolute)
      count_n: number of samples with negative shifts
    """
    df = eval_df.copy()
    df["sc_shift"] = df["sc_after"] - df["sc_before"]

    mask = (
        (df["group_before"] != "UK")
        | ((df["group_before"] == "UK") & (df["is_validation"]))
    )
    # original logic: only on not trained questions that pass mask
    filtered = df[~df["trained"] & mask]

    pos_df = filtered[filtered["sc_shift"] > 0.2]
    psc = float(pos_df["sc_shift"].abs().sum())
    count_p = int(len(pos_df))

    neg_df = filtered[filtered["sc_shift"] < -0.2]
    nsc = float(neg_df["sc_shift"].abs().sum())
    count_n = int(len(neg_df))

    return psc, count_p, nsc, count_n


def parse_model_metadata(ft_model_name: str, base_model_name: str):
    """
    Heuristically parse:
      model_type  (llama, gemma, etc)
      adapter     (lora, uiortholora, vera, randlora, or 'unknown')
      n_tr_from_name  (int or None)
      lr         (string like '1e-3' or None)
    """
    name_lower = ft_model_name.lower()

    # model type from ft or base
    model_type = "unknown"
    if "llama" in name_lower or (base_model_name and "llama" in base_model_name.lower()):
        model_type = "llama"
    elif "gemma" in name_lower or (base_model_name and "gemma" in base_model_name.lower()):
        model_type = "gemma"
    else:
        # as fallback take first token
        model_type = ft_model_name.split("_", 1)[0]

    adapter = "unknown"
    for cand in ["lora", "uiortholora", "vera", "randlora"]:
        if cand in name_lower:
            adapter = cand
            break

    m_tr = re.search(r"_tr(\d+)", ft_model_name)
    n_tr_from_name = int(m_tr.group(1)) if m_tr else None

    m_lr = re.search(r"_lr([0-9.eE+-]+)", ft_model_name)
    lr = m_lr.group(1) if m_lr else None

    return model_type, adapter, n_tr_from_name, lr


def evaluate_single_ft_model(
    df: pd.DataFrame,
    ft_model_name: str,
    base_model_name: str,
    source_file: str,
):
    """
    Build eval_df for one FT model and compute:
      - n_trained (counted from 'trained' flag)
      - accuracy on trained samples
      - negative shift score and count on untrained samples
    """
    s_ft = _sanitize(ft_model_name)

    if base_model_name:
        s_base = _sanitize(base_model_name)
        base_col = f"base_{s_base}_score"
    else:
        base_col = "base_score"

    if base_col not in df.columns:
        raise KeyError(f"Baseline column '{base_col}' not found in dataframe")

    score_col = f"{s_ft}_score"
    train_col = f"{s_ft}_train"

    if score_col not in df.columns or train_col not in df.columns:
        raise KeyError(
            f"For FT model '{ft_model_name}' expected columns '{score_col}' and '{train_col}'"
        )

    eval_df = pd.DataFrame(
        {
            "sc_before": df[base_col],
            "sc_after": df[score_col],
            "trained": df[train_col],
            "is_validation": df["is_validation"],
            "qid": df["qid"],
        }
    )

    eval_df = assign_knowledge_groups(eval_df)

    acc = compute_accuracy(eval_df)
    n_trained = int(eval_df["trained"].sum())
    psc, p_count, nsc, n_count = compute_shifts(eval_df)

    model_type, adapter, n_tr_from_name, lr = parse_model_metadata(
        ft_model_name, base_model_name
    )

    return {
        "source_file": source_file,
        "base_model": base_model_name,
        "ft_model": ft_model_name,
        "model_type": model_type,
        "adapter": adapter,
        "lr": lr,
        "n_tr_from_name": n_tr_from_name,
        "n_trained": n_trained,
        "accuracy": acc,
        "neg_shift_score": nsc,
        "neg_shift_count": n_count,
        "pos_shift_score": psc,
        "pos_shift_count": p_count,
    }


def collect_results(workdir: str = ".", output_csv: str = "mmlu_ft_summary.csv"):
    """
    Walk over all jsonl files in workdir (recursively), collect FT metrics,
    and save them into a single CSV.
    """
    workdir_path = Path(workdir)
    all_rows = []

    jsonl_paths = sorted(workdir_path.rglob("../results/llama-8b/workdir/*.jsonl"))
    if not jsonl_paths:
        print(f"No jsonl files found under {workdir_path}")
        return

    for json_path in jsonl_paths:
        df, ft_model_names, base_model_ids, base_score_cols = load_json_to_df(str(json_path))

        if base_model_ids:
            # for these files you usually have a single base model id
            base_model_name = list(base_model_ids)[0]
        else:
            base_model_name = ""

        for ft_model_name in ft_model_names:
            row = evaluate_single_ft_model(
                df=df,
                ft_model_name=ft_model_name,
                base_model_name=base_model_name,
                source_file=str(json_path.relative_to(workdir_path)),
            )
            all_rows.append(row)

    if not all_rows:
        print("No FT models found in any jsonl file, nothing to write")
        return

    res_df = pd.DataFrame(all_rows)

    # a simple sort for readability
    sort_cols = [c for c in ["adapter", "model_type", "lr", "n_trained", "ft_model"] if c in res_df.columns]
    if sort_cols:
        res_df = res_df.sort_values(sort_cols)

    out_path = workdir_path / output_csv
    res_df.to_csv(out_path, index=False)
    print(f"Saved {len(res_df)} rows to {out_path}")


def main():
    # run in current directory and save mmlu_ft_summary.csv next to this script
    collect_results(".", "ft_summary_chat.csv")


if __name__ == "__main__":
    main()
