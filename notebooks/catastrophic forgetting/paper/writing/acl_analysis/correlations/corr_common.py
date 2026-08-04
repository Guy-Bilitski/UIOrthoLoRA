"""Shared loader + statistics for the correlation-cartography pass.

Data contract (binding, see key_numbers.md §0/§18-19, 09_verification):
- canonical per-run data: results/<run>/summary.json
- geometry: results/geo_drift/adapter_metrics_merged.jsonl (key `run`)
- CE drift: results/forgetting_merged.jsonl (key `run_name`)
- frozen pool = ladder_2026-07-17.py conventions: drop SMOKE/smoke/corda,
  finite fdelta>0 & retention_mean, six families, exclude the 7 post-freeze
  stragglers -> n=1035 (quarantine INCLUDED, freeze convention).
- preflight MUST reproduce §18.1 (pooled r=-0.847, per-family n and r).
- After preflight, the known duplicate run
  frc_lorawdr16_wd0p3_lr3e4_c256_s42_reeval (09_verification Q4: byte-identical
  to its parent) is DROPPED for all new analyses -> n=1034 deduped pool.
- Cluster for robust SEs = recipe cell (family x method x LR x config, i.e.
  run_name minus seed suffix), per 09_verification Q1 (ICC~=0.78).
"""
import json, glob, math, os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # notebooks/catastrophic forgetting
RES = os.path.join(ROOT, "results")
OUT = HERE

STRAGGLERS = {
    "lrsw_clora_k1024_lr3e4_s45",
    "qwsw_lora_null_r16_lr5e4_s43",
    "qwsw_milora_r32_lr5e5_s44",
    "qwswm_clora_k1024_lr2e5_s44",
    "qwswm_dora_r16_lr2e4_s43",
    "qwswm_lora_r16_lr1e4_s44",
    "qwswm_sclora_r32_lr3e4_s44",
}
DUPLICATE = "frc_lorawdr16_wd0p3_lr3e4_c256_s42_reeval"

FROZEN = {
    "lrsw": (180, -0.886), "lrswm": (120, -0.865), "qwsw": (151, -0.840),
    "qwswm": (164, -0.830), "frc": (276, -0.928), "frm": (144, -0.929),
}
FROZEN_POOLED_R = -0.847
FAMS = ["lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"]

FAM_LABEL = {
    "lrsw": "Llama-2 CS", "lrswm": "Llama-2 math", "qwsw": "Qwen-2.5 CS",
    "qwswm": "Qwen-2.5 math", "frc": "Llama CS grid", "frm": "Llama math-395k",
}


def method_of(rn):
    body = rn.split("_", 1)[1]
    if body.startswith("lora_null"):
        return "lora_null"
    return body.split("_")[0]


def lr_of(rn):
    m = re.search(r"_lr(\d+)e(\d+)", rn)
    if not m:
        return np.nan
    return float(m.group(1)) * 10.0 ** (-int(m.group(2)))


def load_pool():
    """Frozen n=1035 pool as a DataFrame (duplicate still included)."""
    rows = []
    for f in glob.glob(os.path.join(RES, "*", "summary.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rn = d.get("run_name") or os.path.basename(os.path.dirname(f))
        if "SMOKE" in rn or "smoke" in rn or "corda" in rn:
            continue
        h = d.get("headline") or {}
        fd, ret = h.get("fdelta"), h.get("retention_mean")
        if not isinstance(fd, (int, float)) or not isinstance(ret, (int, float)):
            continue
        if not (math.isfinite(fd) and math.isfinite(ret)) or fd <= 0:
            continue
        m = re.match(r"^([a-z0-9]+)_", rn)
        fam = m.group(1) if m else "other"
        if fam not in FROZEN:
            continue
        if rn in STRAGGLERS:
            continue
        sm = re.search(r"_s(4[2-9])$", rn)
        rows.append(dict(
            run=rn, fam=fam, seed=sm.group(1) if sm else None,
            cell=re.sub(r"_s4[2-9]$", "", rn), method=method_of(rn),
            lr=lr_of(rn),
            fdelta=float(fd), logfd=math.log10(fd), ret=float(ret),
            ret_broad=float(h["retention_broad"]) if isinstance(h.get("retention_broad"), (int, float)) else np.nan,
            adapt=float(h["cs_avg"]) if isinstance(h.get("cs_avg"), (int, float)) else np.nan,
            dw_sv_max=float(h["dw_sv_max"]) if isinstance(h.get("dw_sv_max"), (int, float)) else np.nan,
        ))
    df = pd.DataFrame(rows)
    df["loglr"] = np.log10(df["lr"])
    return df


def load_geo():
    rows = []
    with open(os.path.join(RES, "geo_drift", "adapter_metrics_merged.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            rows.append(d)
    g = pd.DataFrame(rows)
    g = g.drop_duplicates(subset="run", keep="first")
    keep = ["run", "fro_total", "spec_max", "spec_mean", "stable_rank_w",
            "eff_rank_w", "e_top_w", "e_bot_w", "amp_top_w"]
    g = g[keep].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        g["logfro"] = np.log10(g["fro_total"].where(g["fro_total"] > 0))
        g["lspec"] = np.log10(g["spec_max"].where(g["spec_max"] > 0))
    return g


def load_ce():
    rows = []
    with open(os.path.join(RES, "forgetting_merged.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            rows.append(dict(run=d.get("run_name"), ce=d.get("forgetting_ce"),
                             base_entropy=d.get("base_entropy"),
                             kl=d.get("forgetting_kl"),
                             ce_blocks=d.get("n_blocks")))
    c = pd.DataFrame(rows).dropna(subset=["run"]).drop_duplicates(subset="run", keep="first")
    return c


def preflight(df):
    """Reproduce §18.1 or raise. Returns text report."""
    lines = ["[PREFLIGHT] frozen-pool reproduction of key_numbers.md §18.1"]
    assert len(df) == 1035, f"pool n={len(df)} != 1035"
    for fam, (wn, wr) in FROZEN.items():
        sub = df[df.fam == fam]
        r = np.corrcoef(sub.logfd, sub.ret)[0, 1]
        assert len(sub) == wn, f"{fam}: n={len(sub)} != {wn}"
        assert abs(r - wr) < 0.0005, f"{fam}: r={r:.4f} != {wr}"
        lines.append(f"  {fam}: n={len(sub)} r={r:.3f}  OK")
    rp = np.corrcoef(df.logfd, df.ret)[0, 1]
    assert abs(rp - FROZEN_POOLED_R) < 0.0005, f"pooled r={rp:.4f}"
    lines.append(f"  pooled: n={len(df)} r={rp:.3f}  OK -- §18.1 reproduced")
    return "\n".join(lines)


def build(dedupe=True, verbose=True):
    """Load, preflight, join, dedupe. Returns (df, preflight_text)."""
    df = load_pool()
    txt = preflight(df)
    if verbose:
        print(txt)
    if dedupe:
        df = df[df.run != DUPLICATE].copy()
    geo, ce = load_geo(), load_ce()
    df = df.merge(geo, on="run", how="left").merge(ce, on="run", how="left")
    return df, txt


# ---------------------------------------------------------------- statistics ---

def ols_fit(X, y):
    """Least squares via lstsq. Returns beta, resid, r2, fitted."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    resid = y - fit
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - resid @ resid / sst if sst > 0 else np.nan
    return beta, resid, r2, fit


def cluster_robust_se(X, resid, clusters):
    """CR1 cluster-robust covariance (Liang-Zeger with small-sample correction).
    Returns SE vector."""
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    ids = pd.Series(clusters).astype(str).values
    uniq = pd.unique(ids)
    G = len(uniq)
    meat = np.zeros((p, p))
    for g in uniq:
        m = ids == g
        Xg = X[m]
        ug = resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / (n - p)) if G > 1 and n > p else 1.0
    V = c * XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.diag(V)), G


def design_fe(df, terms, fe_col="fam"):
    """Design matrix: intercept + family dummies (drop first) + terms.
    Returns X (n,p), column names."""
    fams = sorted(df[fe_col].unique())
    cols = [np.ones(len(df))]
    names = ["const"]
    for f in fams[1:]:
        cols.append((df[fe_col] == f).astype(float).values)
        names.append(f"fam:{f}")
    for t in terms:
        cols.append(df[t].values.astype(float))
        names.append(t)
    return np.column_stack(cols), names


def fe_r2(df, terms, y="ret", cluster="cell"):
    """R2 of family-FE + terms model, delta-R2 over FE-only, and cluster-robust
    t for each term. Rows with NaN in any term/y are dropped."""
    need = list(terms) + [y]
    sub = df.dropna(subset=need)
    yv = sub[y].values.astype(float)
    X0, _ = design_fe(sub, [])
    _, _, r2_0, _ = ols_fit(X0, yv)
    X, names = design_fe(sub, terms)
    beta, resid, r2, _ = ols_fit(X, yv)
    se, G = cluster_robust_se(X, resid, sub[cluster].values)
    tstats = {nm: beta[i] / se[i] for i, nm in enumerate(names) if nm in terms}
    betas = {nm: beta[i] for i, nm in enumerate(names) if nm in terms}
    return dict(n=len(sub), G=G, r2=r2, r2_fe=r2_0, dr2=r2 - r2_0,
                t=tstats, beta=betas)


def demean_by(df, cols, by="fam"):
    """Family-demeaned copy of cols (for family-partialed correlations)."""
    out = df[cols + [by]].copy()
    for c in cols:
        out[c] = out[c] - out.groupby(by)[c].transform("mean")
    return out


def pearson_pairwise(df, cols):
    """Pairwise-complete Pearson matrix + n matrix."""
    k = len(cols)
    R = np.full((k, k), np.nan)
    N = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            a, b = df[cols[i]].values, df[cols[j]].values
            m = np.isfinite(a) & np.isfinite(b)
            N[i, j] = m.sum()
            if m.sum() >= 3 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                R[i, j] = np.corrcoef(a[m], b[m])[0, 1]
    return pd.DataFrame(R, index=cols, columns=cols), pd.DataFrame(N, index=cols, columns=cols)


def spearman_pairwise(df, cols):
    from scipy.stats import spearmanr
    k = len(cols)
    R = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            a, b = df[cols[i]].values, df[cols[j]].values
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 3:
                R[i, j] = spearmanr(a[m], b[m]).statistic
    return pd.DataFrame(R, index=cols, columns=cols)


# canonical variable roster and pretty labels
VARS = ["ret", "ret_broad", "adapt", "logfd", "logfro", "lspec",
        "stable_rank_w", "eff_rank_w", "e_top_w", "e_bot_w", "amp_top_w",
        "ce", "kl", "loglr"]
VLAB = {
    "ret": "retention (core)", "ret_broad": "retention (broad)",
    "adapt": "adaptation", "logfd": "log10 F_delta",
    "logfro": "log10 ||dW||_F", "lspec": "log10 spec_max",
    "stable_rank_w": "stable rank", "eff_rank_w": "effective rank",
    "e_top_w": "e_top (base top-subspace)", "e_bot_w": "e_bot",
    "amp_top_w": "amp_top", "ce": "CE drift (forgetting_ce)",
    "kl": "KL drift (CE - base H)", "loglr": "log10 LR",
}

MAG_BLOCK = ["logfd"]           # primary magnitude block
MAG_BLOCK_X = ["logfd", "lspec"]  # extended (spec_max is magnitude, 06 §5 / 09 Q1c)
GEO_BLOCK = ["stable_rank_w", "eff_rank_w", "e_top_w", "e_bot_w", "amp_top_w"]
CE_BLOCK = ["kl"]
