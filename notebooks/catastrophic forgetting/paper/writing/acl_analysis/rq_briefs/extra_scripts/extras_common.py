"""extras_common.py — shared loader + helpers for the extra-insights pass.

Pool: insights/pool.csv (frozen n=1035). Hard preflight of key_numbers §18.1
(n=1035, pooled r(logfd, ret) = -0.847, per-family n/r to 3 decimals) before
anything is returned. Cluster-aware helpers: cell aggregation, CR1
cluster-robust OLS, cell bootstrap.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INS = os.path.abspath(os.path.join(HERE, "..", "..", "insights"))

FROZEN = {"lrsw": (180, -0.886), "lrswm": (120, -0.865), "qwsw": (151, -0.840),
          "qwswm": (164, -0.830), "frc": (276, -0.928), "frm": (144, -0.929)}
FROZEN_POOLED_R = -0.847
# canonical frozen knees, key_numbers.md section 18.2 (log10 F_delta)
KNEE_182 = dict(lrsw=-0.02, lrswm=-0.48, qwsw=-0.69, qwswm=-0.91, frc=-0.45, frm=-0.50)
FAMS = list(FROZEN)
# family -> (base model, task)
FAM_META = dict(lrsw=("llama", "cs"), frc=("llama", "cs"), lrswm=("llama", "math"),
                frm=("llama", "math"), qwsw=("qwen", "cs"), qwswm=("qwen", "math"))
GEOM_METHODS = {"clora", "sclora", "lora_null", "milora", "milorawd", "pissa", "dora"}


def load_pool(verbose=True):
    df = pd.read_csv(os.path.join(INS, "pool.csv"))
    ok = True
    for fam, (n_ref, r_ref) in FROZEN.items():
        sub = df[df.fam == fam]
        r = np.corrcoef(sub.logfd, sub.ret)[0, 1]
        m = (len(sub) == n_ref) and (round(r, 3) == r_ref)
        ok &= m
        if verbose:
            print(f"preflight {fam}: n={len(sub)} r={r:.3f} {'OK' if m else 'MISMATCH'}")
    r_all = np.corrcoef(df.logfd, df.ret)[0, 1]
    m = (len(df) == 1035) and (round(r_all, 3) == FROZEN_POOLED_R)
    ok &= m
    if verbose:
        print(f"preflight pooled: n={len(df)} r={r_all:.3f} {'OK' if m else 'MISMATCH'}")
    assert ok, "PREFLIGHT FAILED — refusing to proceed"
    return df


def cells_of(df, extra_cols=(), min_seeds=1):
    """Seed-averaged cell table. Numeric cols averaged; sd_ret / sd_logfd kept."""
    num = ["logfd", "ret", "adapt", "bbh", "mmlu_pro", "mmlu", "arc_c",
           "truthfulqa", "ret_broad", "lr", "wd", "clora_k", "rank",
           "forgetting_ce", "forgetting_kl"] + list(extra_cols)
    num = [c for c in dict.fromkeys(num) if c in df.columns]
    g = df.groupby("cell")
    out = g[num].mean()
    out["n_seeds"] = g.size()
    out["fam"] = g["fam"].first()
    out["method"] = g["method"].first()
    out["sd_ret"] = g["ret"].std()
    out["sd_logfd"] = g["logfd"].std()
    out["sd_adapt"] = g["adapt"].std()
    out["any_quar"] = g["quarantined"].any()
    return out[out.n_seeds >= min_seeds].reset_index()


def ols(X, y):
    """OLS with intercept prepended. Returns beta, residuals, XtX_inv, R2."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    xtx_inv = np.linalg.pinv(X.T @ X)
    return beta, resid, xtx_inv, r2, X


def cr1_se(X_full, resid, clusters):
    """CR1 cluster-robust SEs. X_full includes the intercept column."""
    xtx_inv = np.linalg.pinv(X_full.T @ X_full)
    meat = np.zeros((X_full.shape[1], X_full.shape[1]))
    uniq = pd.unique(clusters)
    for c in uniq:
        m = clusters == c
        s = X_full[m].T @ resid[m]
        meat += np.outer(s, s)
    G, n, k = len(uniq), len(resid), X_full.shape[1]
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    V = adj * xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.diag(V))


def dummies(series, ref):
    lv = [v for v in sorted(series.unique()) if v != ref]
    return np.column_stack([(series == v).astype(float).values for v in lv]), lv


def hinge_fit(x, y, qlo=0.1, qhi=0.9, ngrid=41):
    """SSE-optimal single-knot hinge fit; returns (knee, beta, sse)."""
    best = None
    for knee in np.quantile(x, np.linspace(qlo, qhi, ngrid)):
        X = np.column_stack([np.ones_like(x), x, np.maximum(0, x - knee)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((y - X @ beta) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, knee, beta)
    return best[1], best[2], best[0]
