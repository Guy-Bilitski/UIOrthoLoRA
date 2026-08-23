"""10 — the Qwen k/v degeneracy: energy share, and the coordinates recomputed
without k/v.

Reason: on Qwen2.5-7B the attention key and value projections have 512 output
directions, so the top-256 and bottom-256 blocks exhaust them and e_top+e_bot=1
exactly there. Limitations 5 answers this with the k/v energy share; this script
is the reproducible source for that number, and it also performs the check the
inference stands in for: recompute each Qwen adapter's F-weighted geometry
coordinates EXCLUDING k/v, and report how much the adapter-level coordinates and
the per-family residual-geometry correlations move.

Inputs: results/geo_drift/permatrix_qwen/*.jsonl (per-matrix rows: fro, spec,
stable_rank, e_top, e_bot, ein_top, ein_bot per adapted matrix), and the pooled
frame from corr_common (ret, logfd, fam).

Outputs: qwen_kv_check.md
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 10_qwen_kv_check.py
"""
import glob
import json
import os

import numpy as np

from rq1_common import OUT
import corr_common as cc

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
PM = os.path.join(_ROOT, "results", "geo_drift", "permatrix_qwen")

KV = ("k_proj", "v_proj")
COORDS = ("e_top", "e_bot", "ein_top", "ein_bot", "stable_rank")


def per_run():
    """Per Qwen run: k/v F^2 share, and F^2-weighted coords with/without k/v."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(PM, "*.jsonl"))):
        run = os.path.basename(path)[:-6]
        recs = [json.loads(l) for l in open(path)]
        w_all = np.array([r["fro"] ** 2 for r in recs])
        kv = np.array([r["target"] in KV for r in recs])
        share = w_all[kv].sum() / w_all.sum() if w_all.sum() > 0 else np.nan
        out = {"kv_share": share}
        for c in COORDS:
            v = np.array([r[c] for r in recs])
            out[c + "_all"] = float((v * w_all).sum() / w_all.sum())
            w_ex = w_all[~kv]
            out[c + "_ex"] = float((v[~kv] * w_ex).sum() / w_ex.sum())
        rows[run] = out
    return rows


def run():
    rows = per_run()
    shares = np.array([r["kv_share"] for r in rows.values()])
    lines = [
        "# Qwen k/v degeneracy check",
        "",
        f"Per-matrix store: {len(rows)} Qwen runs.",
        "",
        "## 1. How much of the update sits in k/v",
        "",
        f"F^2-weighted share of the update's energy carried by k_proj and v_proj:",
        f"mean **{shares.mean():.3f}**, median **{np.median(shares):.3f}**, "
        f"90th percentile **{np.percentile(shares, 90):.3f}**, "
        f"max **{shares.max():.3f}**.",
        "",
        "## 2. Adapter-level coordinates, with and without k/v",
        "",
        "| coordinate | mean |delta| | median |delta| | max |delta| |",
        "|---|---|---|---|",
    ]
    for c in COORDS:
        d = np.array([abs(r[c + "_all"] - r[c + "_ex"]) for r in rows.values()])
        lines.append(f"| {c} | {d.mean():.4f} | {np.median(d):.4f} | {d.max():.4f} |")

    # 3. residual geometry association per Qwen family, both ways
    df, _ = cc.build(dedupe=True)
    df = df[df.fam.isin(("qwsw", "qwswm"))].copy()
    for c in COORDS:
        df[c + "_ex"] = [rows.get(r, {}).get(c + "_ex", np.nan) for r in df.run]
    lines += ["", "## 3. Residual association with retention, given log10 F_delta",
              "", "Partial correlation of each coordinate with retention after",
              "regressing both on log10 F_delta, per Qwen family.", "",
              "| family | coordinate | full | k/v excluded |", "|---|---|---|---|"]

    def partial(sub, col):
        s = sub.dropna(subset=["ret", "logfd", col])
        if len(s) < 10:
            return np.nan, len(s)
        x = s[col].values.astype(float)
        y = s.ret.values.astype(float)
        z = s.logfd.values.astype(float)
        rx = x - np.polyval(np.polyfit(z, x, 1), z)
        ry = y - np.polyval(np.polyfit(z, y, 1), z)
        return float(np.corrcoef(rx, ry)[0, 1]), len(s)

    pairs = {"e_top": "e_top_w", "stable_rank": "stable_rank_w"}
    for fam in ("qwsw", "qwswm"):
        sub = df[df.fam == fam]
        for c, full_col in pairs.items():
            r_full, n1 = partial(sub, full_col)
            r_ex, n2 = partial(sub, c + "_ex")
            lines.append(f"| {fam} | {c} | {r_full:+.3f} (n={n1}) "
                         f"| {r_ex:+.3f} (n={n2}) |")

    path = os.path.join(OUT, "qwen_kv_check.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", path)


if __name__ == "__main__":
    run()
