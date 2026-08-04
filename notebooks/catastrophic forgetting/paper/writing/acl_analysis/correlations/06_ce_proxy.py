"""06 — Can CE/KL drift (one forward pass on neutral text) substitute for full
retention evals? Per-family calibration of the KL->retention mapping + honest
error accounting, vs the same thing done with F_delta (also cheap) and combined.

Protocol notes (binding): KL = forgetting_ce - base_entropy (cross-model
comparable, §18.6). Qwen CE coverage 60-62% with seed-block missingness that is
method-balanced and ignorable for regressions, but per-seed Qwen CE analyses are
barred (09 Q4). CE store mixes two protocols (40-block vs full WikiText-103);
Q2 found this benign — we still add a protocol-dummy sensitivity line.

Analyses per family:
  1. calibration ret ~ a + b*KL and ret ~ a + b*log10(KL); model chosen by
     leave-cells-out CV RMSE (10 folds of recipe cells).
  2. same for log F_delta and for KL+logF combined.
  3. screening value: Spearman(KL, ret); AUC for detecting 'damaged' runs
     (ret < family healthy ceiling (90th pct) - 5pp) from KL alone vs logF alone.
  4. within-base-model task transfer: calibrate on one family, apply to its
     sibling (lrsw<->lrswm, frc<->frm, qwsw<->qwswm), raw and intercept-corrected.

Outputs: ce_proxy_calibration.csv, ce_proxy.md, fig_ce_proxy.png/.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import figstyle
import corr_common as cc
from scipy.stats import spearmanr

rng = np.random.default_rng(20260718)
df, _ = cc.build(dedupe=True, verbose=False)
OUT = cc.OUT

pool = df.dropna(subset=["kl", "logfd"]).copy()
pool["logkl"] = np.log10(pool["kl"].clip(lower=1e-6))


def _hinge_fit(x, y):
    """Continuous 1-knee fit y = a + b1*x + b2*max(0, x-k); k grid-searched on
    train data (10-90% quantiles). Returns (k, beta)."""
    best = None
    for q in np.linspace(0.10, 0.90, 17):
        k = np.quantile(x, q)
        X = np.column_stack([np.ones(len(x)), x, np.clip(x - k, 0, None)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = np.sum((y - X @ beta) ** 2)
        if best is None or sse < best[0]:
            best = (sse, k, beta)
    return best[1], best[2]


def _hinge_pred(x, k, beta):
    X = np.column_stack([np.ones(len(x)), x, np.clip(x - k, 0, None)])
    return X @ beta


def cv_rmse(sub, terms, k=10, hinge=None):
    """Leave-cells-out CV RMSE within a subset (single family).
    hinge='col' -> 1-knee piecewise-linear in that single column instead of OLS."""
    cells = np.array([str(c) for c in sub["cell"].unique()], dtype=object)
    r = np.random.default_rng(7)
    r.shuffle(cells)
    folds = np.array_split(cells, min(k, len(cells)))
    y = sub["ret"].values.astype(float)
    pred = np.full(len(sub), np.nan)
    for f in folds:
        te = sub["cell"].isin(f).values
        tr = ~te
        if hinge:
            kk, beta = _hinge_fit(sub.loc[tr, hinge].values.astype(float), y[tr])
            pred[te] = _hinge_pred(sub.loc[te, hinge].values.astype(float), kk, beta)
        else:
            Xtr = np.column_stack([np.ones(tr.sum())] + [sub.loc[tr, t].values for t in terms])
            Xte = np.column_stack([np.ones(te.sum())] + [sub.loc[te, t].values for t in terms])
            beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
            pred[te] = Xte @ beta
    err = y - pred
    return np.sqrt(np.mean(err ** 2)), np.mean(np.abs(err))


def auc(score, label):
    """Mann-Whitney AUC: P(score_damaged > score_healthy)."""
    s1, s0 = score[label], score[~label]
    if len(s1) == 0 or len(s0) == 0:
        return np.nan
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(s1, s0, alternative="two-sided").statistic
    return u / (len(s1) * len(s0))


rows = []
fits = {}
for fam in cc.FAMS:
    s = pool[pool.fam == fam].copy()
    n = len(s)
    # 1-2: calibrations
    cand = {"KL linear": ["kl"], "KL log": ["logkl"], "logF": ["logfd"],
            "KL+logF": ["kl", "logfd"], "KLlog+logF": ["logkl", "logfd"]}
    res = {}
    for nm, t in cand.items():
        res[nm] = cv_rmse(s, t)
    # knee (hinge) calibrations — same functional family as §18.2's magnitude knee
    res["KL knee"] = cv_rmse(s, None, hinge="logkl")
    res["logF knee"] = cv_rmse(s, None, hinge="logfd")
    best_kl = min(["KL linear", "KL log", "KL knee"], key=lambda nm: res[nm][0])
    best_f = min(["logF", "logF knee"], key=lambda nm: res[nm][0])
    best_comb = min(["KL+logF", "KLlog+logF"], key=lambda nm: res[nm][0])
    # in-sample fit of best KL calibration for the figure
    if best_kl == "KL knee":
        kk, beta = _hinge_fit(s["logkl"].values.astype(float), s["ret"].values.astype(float))
        fits[fam] = (best_kl, (kk, beta))
    else:
        t = cand[best_kl]
        X = np.column_stack([np.ones(n)] + [s[c].values for c in t])
        beta, *_ = np.linalg.lstsq(X, s["ret"].values, rcond=None)
        fits[fam] = (best_kl, beta)
    # 3: screening
    ceiling = np.percentile(s["ret"], 90)
    dam = (s["ret"] < ceiling - 5).values
    sp = spearmanr(s["kl"], s["ret"]).statistic
    rows.append(dict(
        family=fam, n=n, share_damaged=dam.mean(),
        spearman_kl=sp,
        rmse_kl=res[best_kl][0], mae_kl=res[best_kl][1], kl_form=best_kl,
        rmse_logf=res[best_f][0], mae_logf=res[best_f][1], logf_form=best_f,
        rmse_comb=res[best_comb][0], comb_form=best_comb,
        auc_kl=auc(s["kl"].values, dam), auc_logf=auc(-(-s["logfd"].values), dam),
        seed_ret_sd=s.groupby("cell")["ret"].std().dropna().mean(),
    ))

T = pd.DataFrame(rows)
T.to_csv(OUT + "/ce_proxy_calibration.csv", index=False)

# ---- KL knee locations (full-family in-sample hinge fits) -----------------------
knee_rows = []
for fam in cc.FAMS:
    s = pool[pool.fam == fam]
    kk, beta = _hinge_fit(s["logkl"].values.astype(float), s["ret"].values.astype(float))
    knee_rows.append(dict(family=fam, knee_kl_nats=10 ** kk,
                          slope_below=beta[1], slope_above=beta[1] + beta[2]))
K = pd.DataFrame(knee_rows)
K.to_csv(OUT + "/ce_proxy_knees.csv", index=False)

# ---- protocol sensitivity (Q2): does the 40-block/full split move calibration? --
prot = []
for fam in cc.FAMS:
    s = pool[pool.fam == fam]
    if s["ce_blocks"].nunique() > 1:
        for nb, g in s.groupby("ce_blocks"):
            if len(g) > 10:
                r = np.corrcoef(g["kl"], g["ret"])[0, 1]
                prot.append(f"  {fam}: n_blocks={nb} n={len(g)} r(KL,ret)={r:+.3f}")

# ---- 4: within-base-model task transfer -----------------------------------------
pairs = [("lrsw", "lrswm"), ("lrswm", "lrsw"), ("frc", "frm"), ("frm", "frc"),
         ("qwsw", "qwswm"), ("qwswm", "qwsw")]
trans = []
for a, b in pairs:
    sa, sb = pool[pool.fam == a], pool[pool.fam == b]
    for nm, terms in [("KL", ["kl"]), ("logF", ["logfd"])]:
        Xa = np.column_stack([np.ones(len(sa))] + [sa[t].values for t in terms])
        Xb = np.column_stack([np.ones(len(sb))] + [sb[t].values for t in terms])
        beta, *_ = np.linalg.lstsq(Xa, sa["ret"].values, rcond=None)
        p = Xb @ beta
        raw = np.sqrt(np.mean((sb["ret"].values - p) ** 2))
        pc = p + (sb["ret"].mean() - p.mean())
        corr = np.sqrt(np.mean((sb["ret"].values - pc) ** 2))
        trans.append(dict(calib_on=a, applied_to=b, predictor=nm,
                          rmse_raw=raw, rmse_intercept_corrected=corr))
TR = pd.DataFrame(trans)
TR.to_csv(OUT + "/ce_proxy_transfer.csv", index=False)

md = ["# CE drift as a cheap retention proxy — calibration + honest error", "",
      "KL drift costs ONE forward pass on ~40 WikiText blocks (wall_s ~= 3.4s in the",
      "store) vs a full BBH+MMLU-Pro retention eval. Question: if you calibrated the",
      "KL->retention mapping once per family, how big is the prediction error?",
      "",
      "Caveats up front: (i) CE is DOWNSTREAM — a monitor, not a knob; (ii) it is",
      "quasi-tautologically close to retention (both measure drift of base behavior);",
      "(iii) it misses channel B (format damage, 05 §2) which benchmark evals see;",
      "(iv) Qwen coverage 60-62% (seed-block missingness, ignorable per 09 Q4);",
      "(v) two CE protocols mixed, benign (09 Q2; sensitivity below).",
      "",
      "## Per-family calibration error (leave-cells-out CV, pp of retention)",
      "",
      "| family | n | Spearman(KL,ret) | RMSE KL | MAE KL | KL form | RMSE logF (form) | RMSE KL+logF | seed SD(ret) | AUC(dmg) KL | AUC(dmg) logF |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in T.iterrows():
    md.append("| %s | %d | %+.3f | %.2f | %.2f | %s | %.2f (%s) | %.2f | %.2f | %.3f | %.3f |" % (
        r.family, r.n, r.spearman_kl, r.rmse_kl, r.mae_kl, r.kl_form,
        r.rmse_logf, r.logf_form, r.rmse_comb, r.seed_ret_sd, r.auc_kl, r.auc_logf))
md += ["",
       "seed SD(ret) = mean within-cell seed SD — the noise floor of a single-seed",
       "retention measurement itself. AUC = detecting runs >5pp below the family",
       "healthy ceiling (90th pct) from KL alone.",
       "",
       "## KL knee locations (hinge fit, retention ~ log KL)",
       "",
       "| family | knee (nats KL) | slope below (pp/decade) | slope above |",
       "|---|---|---|---|"] + [
    "| %s | %.3f | %+.1f | %+.1f |" % (r.family, r.knee_kl_nats, r.slope_below, r.slope_above)
    for _, r in K.iterrows()] + [
    "",
    "Four of six families (both base models, both task types) put the knee at",
    "~0.26-0.29 nats; frc 0.40; frm has no flat region (already-steep below-knee",
    "slope -8.3, knee 1.69 is a slope change, not damage onset).",
    "",
    "## Protocol sensitivity (families with both CE protocols)", ""]
md += prot if prot else ["  (no family mixes protocols)"]
md += ["",
       "## Within-base-model task transfer (calibrate on sibling task)",
       "",
       "| calib on | applied to | predictor | RMSE raw | RMSE intercept-corrected |",
       "|---|---|---|---|---|"]
for _, r in TR.iterrows():
    md.append("| %s | %s | %s | %.2f | %.2f |" % (
        r.calib_on, r.applied_to, r.predictor, r.rmse_raw, r.rmse_intercept_corrected))

out = "\n".join(md)
open(OUT + "/ce_proxy.md", "w").write(out + "\n")
print(out)

# ---- figure ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), sharey=False)
for ax, fam in zip(axes.ravel(), cc.FAMS):
    s = pool[pool.fam == fam]
    x = s["kl"].values
    ax.scatter(x, s["ret"], s=9, alpha=0.5, color="#2a78d6", linewidths=0)
    form, fit = fits[fam]
    xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 200)
    if form == "KL knee":
        kk, beta = fit
        ys = _hinge_pred(np.log10(np.clip(xs, 1e-6, None)), kk, beta)
    else:
        Xs = np.column_stack([np.ones(200), np.log10(np.clip(xs, 1e-6, None)) if form == "KL log" else xs])
        ys = Xs @ fit
    ax.plot(xs, ys, color="#e34948", lw=1.6)
    r = T[T.family == fam].iloc[0]
    ax.set_title(f"{cc.FAM_LABEL[fam]} ({fam})  CV-RMSE {r.rmse_kl:.2f}pp ({form})", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("KL drift to base (nats, log scale)")
    ax.set_ylabel("retention (core, pp)")
fig.suptitle("Cheap-proxy calibration: retention vs KL drift, per family (fit = best CV form)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "/fig_ce_proxy.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT + "/fig_ce_proxy.pdf", bbox_inches="tight")
print("\nSaved CE-proxy figure.")
