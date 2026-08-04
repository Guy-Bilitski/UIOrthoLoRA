"""Independent loader for the adversarial verification pass (2026-07-18).

Written from scratch against the data contract (key_numbers.md §0/§18,
09_verification Q4): results/<run>/summary.json JOIN
results/geo_drift/adapter_metrics_merged.jsonl (key `run`) JOIN
results/forgetting_merged.jsonl (key `run_name`); quarantine list
results/quarantine_diverged.txt. Deliberately does NOT import any code from
observatory/correlations/adjudication/insights.
"""
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd

ROOT = "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "writing", "acl_analysis", "verification")

FAMS = ["lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"]
FROZEN = {
    "lrsw": (180, -0.886), "lrswm": (120, -0.865), "qwsw": (151, -0.840),
    "qwswm": (164, -0.830), "frc": (276, -0.928), "frm": (144, -0.929),
}
STRAGGLERS = {  # ladder_2026-07-17.py, §19.4
    "lrsw_clora_k1024_lr3e4_s45",
    "qwsw_lora_null_r16_lr5e4_s43",
    "qwsw_milora_r32_lr5e5_s44",
    "qwswm_clora_k1024_lr2e5_s44",
    "qwswm_dora_r16_lr2e4_s43",
    "qwswm_lora_r16_lr1e4_s44",
    "qwswm_sclora_r32_lr3e4_s44",
}
DUPLICATE = "frc_lorawdr16_wd0p3_lr3e4_c256_s42_reeval"

BENCH = ["bbh", "mmlu_pro", "mmlu", "arc_c", "truthfulqa"]


def method_of(rn):
    body = rn.split("_", 1)[1]
    if body.startswith("lora_null"):
        return "lora_null"
    if body.startswith("lorawd"):
        return "lorawd"
    if body.startswith("milorawd"):
        return "milorawd"
    if body.startswith("dorawd"):
        return "dorawd"
    return body.split("_")[0]


def lr_of(rn):
    m = re.search(r"_lr(\d+)e(\d+)", rn)
    if not m:
        return np.nan
    return float(m.group(1)) * 10.0 ** (-int(m.group(2)))


def wd_of(rn):
    m = re.search(r"_wd0p(\d+)", rn)
    if m:
        return float("0." + m.group(1))
    if re.search(r"_wd0_", rn):
        return 0.0
    return np.nan


def k_of(rn):
    m = re.search(r"_k(\d+)_", rn)
    return int(m.group(1)) if m else np.nan


def rank_of(rn):
    m = re.search(r"_r(\d+)(?:_|$)", rn)
    return int(m.group(1)) if m else np.nan


def load_raw(drop_corda=True, drop_stragglers=True):
    rows = []
    for f in glob.glob(os.path.join(RES, "*", "summary.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rn = d.get("run_name") or os.path.basename(os.path.dirname(f))
        if "SMOKE" in rn or "smoke" in rn:
            continue
        if drop_corda and "corda" in rn:
            continue
        h = d.get("headline") or {}
        fd, ret = h.get("fdelta"), h.get("retention_mean")
        if not isinstance(fd, (int, float)) or not isinstance(ret, (int, float)):
            continue
        if not (math.isfinite(fd) and math.isfinite(ret)) or fd <= 0:
            continue
        m = re.match(r"^([a-z0-9]+)_", rn)
        fam = m.group(1) if m else "other"
        if fam not in FAMS:
            continue
        if drop_stragglers and rn in STRAGGLERS:
            continue
        sm = re.search(r"_s(4[2-9])$", rn)
        row = dict(
            run=rn, fam=fam, seed=sm.group(1) if sm else None,
            cell=re.sub(r"_s4[2-9]$", "", rn), method=method_of(rn),
            lr=lr_of(rn), wd=wd_of(rn), k=k_of(rn), rank=rank_of(rn),
            fdelta=float(fd), logfd=math.log10(fd), ret=float(ret),
        )
        for c in BENCH + ["retention_broad", "cs_avg", "dw_sv_max"]:
            v = h.get(c)
            row[c] = float(v) if isinstance(v, (int, float)) and math.isfinite(v) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["loglr"] = np.log10(df["lr"])
    return df


def load_quarantine():
    q = set()
    for line in open(os.path.join(RES, "quarantine_diverged.txt")):
        name = line.split("\t")[0].strip()
        if name:
            q.add(name)
    return q


def load_geo():
    rows = []
    for line in open(os.path.join(RES, "geo_drift", "adapter_metrics_merged.jsonl")):
        d = json.loads(line)
        rows.append(d)
    g = pd.DataFrame(rows).drop_duplicates(subset="run", keep="first")
    keep = ["run", "fro_total", "spec_max", "stable_rank_w", "eff_rank_w",
            "e_top_w", "e_bot_w", "amp_top_w"]
    g = g[[c for c in keep if c in g.columns]].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        g["logfro"] = np.log10(g["fro_total"].where(g["fro_total"] > 0))
        g["lspec"] = np.log10(g["spec_max"].where(g["spec_max"] > 0))
    return g


def load_ce():
    rows = []
    for line in open(os.path.join(RES, "forgetting_merged.jsonl")):
        d = json.loads(line)
        rows.append(dict(run=d.get("run_name"), ce=d.get("forgetting_ce"),
                         kl=d.get("forgetting_kl"),
                         base_entropy=d.get("base_entropy")))
    return pd.DataFrame(rows).dropna(subset=["run"]).drop_duplicates("run", keep="first")


def frozen_pool():
    """The §18.1 frozen n=1035 pool (quarantine included, duplicate included)."""
    df = load_raw()
    assert len(df) == 1035, f"frozen pool n={len(df)}"
    return df


def check_18_1(df, verbose=True):
    ok = True
    r_all = np.corrcoef(df.logfd, df.ret)[0, 1]
    if verbose:
        print(f"pooled r = {r_all:.3f} (frozen -0.847), n={len(df)}")
    ok &= abs(r_all - (-0.847)) < 0.0005
    for fam, (wn, wr) in FROZEN.items():
        s = df[df.fam == fam]
        r = np.corrcoef(s.logfd, s.ret)[0, 1]
        if verbose:
            print(f"  {fam}: n={len(s)} (want {wn}), r={r:.3f} (want {wr})")
        ok &= (len(s) == wn) and abs(r - wr) < 0.0005
    return ok


# ---------- stats helpers (own implementations) ----------

def ols_r2(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1.0 - resid @ resid / ((y - y.mean()) @ (y - y.mean())), beta, resid


def fam_dummies(df):
    fams = sorted(df.fam.unique())
    return [(df.fam == f).astype(float).values for f in fams[1:]]


def delta_r2(df, cols, y="ret"):
    """R2(famFE + cols) - R2(famFE)."""
    yv = df[y].values.astype(float)
    fd = fam_dummies(df)
    r0, *_ = ols_r2(fd, yv)
    r1, *_ = ols_r2(fd + [df[c].values.astype(float) for c in cols], yv)
    return r1 - r0, r0, r1


def cluster_t(df, col, y="ret"):
    """Cluster-robust (CR1) t for `col` in ret ~ famFE + col, cluster=cell."""
    yv = df[y].values.astype(float)
    X = np.column_stack([np.ones(len(df))] + fam_dummies(df)
                        + [df[col].values.astype(float)])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    u = yv - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    cells = df.cell.values
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = 0
    for c in pd.unique(cells):
        idx = cells == c
        Xg = X[idx]
        ug = u[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
        G += 1
    n, p = X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - p))
    V = adj * XtX_inv @ meat @ XtX_inv
    se = math.sqrt(V[-1, -1])
    return beta[-1] / se, G


def partial_r(y, x, controls):
    Z = np.column_stack([np.ones(len(y))] + list(controls))
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return np.corrcoef(ry, rx)[0, 1]


def mw_auc(score, label):
    """Mann-Whitney AUC = P(score_pos > score_neg), ties = 0.5. Own impl."""
    pos = np.asarray(score)[label]
    neg = np.asarray(score)[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def hinge_fit(x, y, qgrid=None):
    """Own continuous 1-knee fit; knee grid on quantiles (default 33 pts 5-95%)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if qgrid is None:
        qgrid = np.linspace(0.05, 0.95, 33)
    best = None
    for q in qgrid:
        kk = np.quantile(x, q)
        X = np.column_stack([np.ones_like(x), x, np.clip(x - kk, 0, None)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((y - X @ beta) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, kk, beta)
    return best[1], best[2]


def hinge_pred(x, kk, beta):
    x = np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x, np.clip(x - kk, 0, None)])
    return X @ beta
