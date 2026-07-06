#!/usr/bin/env python
"""
A1-A4 analyses for the paper polish round (2026-07-02, second pass).
Pure computation on the frozen clean registry. Writes nothing outside stdout.

Convention: LoRA-Null (`lrsw_lora_null_*`) is its OWN series (A4 relabel).
7 series x 7 LRs = 49 points, Llama-2-7B CS sweep, seed 42. CorDA withheld.

A1: slope-interaction ANCOVA (per-method slopes vs common slope).
A2: leave-one-method-out predictive check (pooled law fit on 6 series,
    predict the 7th; RMSE vs in-sample method-aware model).
A3: joint F-test on the six on-curve methods' intercepts (excl SC-LoRA).
A4: verify pooled law identical under the split convention; recompute
    within-method r, 7-series intercept ANCOVA, spline residuals, robustness.
"""
import json, re
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline

DATA = "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting/paper/writing/data/campaign_summary_clean.jsonl"

def classify(run):
    # split convention: lora_null is its own series
    m = re.match(r"lrsw_(lora_null|lorawd|lora|dora|milora|sclora|clora|corda)_", run)
    return m.group(1) if m else None

recs = {}
for line in open(DATA):
    d = json.loads(line)
    recs[d["run_name"]] = d

rows = []
for rn, d in recs.items():
    if not rn.startswith("lrsw_"):
        continue
    meth = classify(rn)
    if meth is None or meth == "corda":
        continue
    F, ret, ad = d.get("fdelta"), d.get("retention_mean"), d.get("cs_avg")
    if F is None or ret is None or F <= 0:
        continue
    rows.append(dict(run=rn, m=meth, F=F, ret=ret, adapt=ad,
                     lr=float(re.search(r"_lr([0-9])e([0-9])_", rn).group(1) + "e-" +
                              re.search(r"_lr([0-9])e([0-9])_", rn).group(2))))

METHODS = ["lora", "lora_null", "lorawd", "dora", "milora", "clora", "sclora"]
ON_CURVE = ["lora", "lora_null", "lorawd", "dora", "milora", "clora"]
x = np.array([r["F"] for r in rows]); y = np.array([r["ret"] for r in rows])
lx = np.log10(x)
mth = np.array([r["m"] for r in rows])
n = len(rows)
print(f"n = {n} points; per-series counts:",
      {m: int((mth == m).sum()) for m in METHODS})

# ---------------- A4: pooled law identical under split convention ----------------
sl, ic, r, p, se = stats.linregress(lx, y)
print(f"\n[A4] pooled law (49 pts): r={r:+.4f} R2={r*r:.4f} slope={sl:+.2f} pp/dec p={p:.2e}")
on = np.isin(mth, ON_CURVE)
sl2, ic2, r2_, p2, _ = stats.linregress(lx[on], y[on])
print(f"[A4] on-curve law (6 series, n={on.sum()}): r={r2_:+.4f} R2={r2_*r2_:.4f} slope={sl2:+.2f} p={p2:.2e}")
print("[A4] within-method r:")
for m in METHODS:
    g = mth == m
    rm = stats.pearsonr(lx[g], y[g])
    print(f"   {m:10s} r={rm[0]:+.3f} n={g.sum()}")

# spline residuals (same pooled spline as fig2: k=3, s=n*8)
order = np.argsort(lx)
spl = UnivariateSpline(lx[order], y[order], k=3, s=n * 8.0)
resid = y - spl(lx)
print("[A4] spline residuals per series (mean pp, p vs 0):")
res_by = {}
for m in METHODS:
    g = mth == m
    t, pv = stats.ttest_1samp(resid[g], 0.0)
    res_by[m] = (resid[g].mean(), pv)
    print(f"   {m:10s} mu={resid[g].mean():+.2f}  p={pv:.3f}")

# robustness (ret >= 24) per series
print("[A4] robustness (# of 7 LRs with ret >= 24):")
for m in METHODS:
    g = mth == m
    print(f"   {m:10s} {int((y[g] >= 24).sum())}/7")

# ---------------- intercept ANCOVA, 7-series convention ----------------
def design_intercepts(mask=None):
    idx = np.arange(n) if mask is None else np.where(mask)[0]
    ms = sorted(set(mth[idx]))
    M = np.zeros((len(idx), len(ms)))
    for j, m in enumerate(ms):
        M[:, j] = (mth[idx] == m).astype(float)
    return idx, ms, M

def ols(X, yy):
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    res = yy - X @ beta
    return beta, np.sum(res**2), res

# reduced: common intercept+slope; full: per-method intercepts + common slope
Xr = np.column_stack([np.ones(n), lx])
_, ssr_r, _ = ols(Xr, y)
idx, ms, M = design_intercepts()
Xf = np.column_stack([M, lx])
_, ssr_f, _ = ols(Xf, y)
ss_tot = np.sum((y - y.mean())**2)
df1 = Xf.shape[1] - Xr.shape[1]; df2 = n - Xf.shape[1]
F_int = ((ssr_r - ssr_f) / df1) / (ssr_f / df2)
p_int = 1 - stats.f.cdf(F_int, df1, df2)
print(f"\n[A4] intercept ANCOVA (7 series): R2 {1-ssr_r/ss_tot:.3f} -> {1-ssr_f/ss_tot:.3f}, "
      f"dR2={(ssr_r-ssr_f)/ss_tot:.3f}, F({df1},{df2})={F_int:.2f}, p={p_int:.2e}")

# same but excluding sclora from the intercept battery? (which method drives it)
for excl in ["sclora"]:
    mask = mth != excl
    nn = mask.sum()
    Xr_ = np.column_stack([np.ones(nn), lx[mask]])
    _, ssr_r_, _ = ols(Xr_, y[mask])
    _, ms_, M_ = design_intercepts(mask)
    Xf_ = np.column_stack([M_, lx[mask]])
    _, ssr_f_, _ = ols(Xf_, y[mask])
    d1 = Xf_.shape[1] - 2; d2 = nn - Xf_.shape[1]
    Fv = ((ssr_r_ - ssr_f_) / d1) / (ssr_f_ / d2)
    pv = 1 - stats.f.cdf(Fv, d1, d2)
    st = 1 - ssr_r_/np.sum((y[mask]-y[mask].mean())**2)
    sf = 1 - ssr_f_/np.sum((y[mask]-y[mask].mean())**2)
    print(f"[A3] on-curve-only intercept F-test (excl {excl}, n={nn}): "
          f"R2 {st:.3f} -> {sf:.3f}, F({d1},{d2})={Fv:.2f}, p={pv:.3f}")

# ---------------- A1: slope-interaction ANCOVA ----------------
# full2: per-method intercepts + per-method slopes (14 params) vs
# full : per-method intercepts + common slope (8 params)
S = np.zeros((n, len(ms)))
for j, m in enumerate(ms):
    S[:, j] = (mth == m) * lx
Xf2 = np.column_stack([M, S])
_, ssr_f2, _ = ols(Xf2, y)
d1 = Xf2.shape[1] - Xf.shape[1]; d2 = n - Xf2.shape[1]
F_slope = ((ssr_f - ssr_f2) / d1) / (ssr_f2 / d2)
p_slope = 1 - stats.f.cdf(F_slope, d1, d2)
print(f"\n[A1] slope-interaction: R2 {1-ssr_f/ss_tot:.3f} -> {1-ssr_f2/ss_tot:.3f}, "
      f"F({d1},{d2})={F_slope:.2f}, p={p_slope:.3f}")
# per-method slopes for reporting
print("[A1] per-method OLS slopes (pp/decade):")
for m in METHODS:
    g = mth == m
    s_, i_, r_, p_, _ = stats.linregress(lx[g], y[g])
    print(f"   {m:10s} slope={s_:+.1f}")
# A1 restricted to on-curve too
maskoc = np.isin(mth, ON_CURVE)
_, msoc, Moc = design_intercepts(maskoc)
Soc = np.zeros((maskoc.sum(), len(msoc)))
for j, m in enumerate(msoc):
    Soc[:, j] = (mth[maskoc] == m) * lx[maskoc]
Xoc_i = np.column_stack([Moc, lx[maskoc]])
Xoc_s = np.column_stack([Moc, Soc])
_, ssr_oi, _ = ols(Xoc_i, y[maskoc])
_, ssr_os, _ = ols(Xoc_s, y[maskoc])
d1 = Xoc_s.shape[1] - Xoc_i.shape[1]; d2 = maskoc.sum() - Xoc_s.shape[1]
Fv = ((ssr_oi - ssr_os) / d1) / (ssr_os / d2)
pv = 1 - stats.f.cdf(Fv, d1, d2)
print(f"[A1] slope-interaction on-curve only (n={maskoc.sum()}): F({d1},{d2})={Fv:.2f}, p={pv:.3f}")

# ---------------- A2: leave-one-method-out predictive check ----------------
print("\n[A2] leave-one-method-out: pooled linear law fit on 6 series -> predict 7th")
rmses = {}
for m in METHODS:
    tr = mth != m; te = ~tr
    s_, i_, *_ = stats.linregress(lx[tr], y[tr])
    pred = i_ + s_ * lx[te]
    rmse = np.sqrt(np.mean((y[te] - pred)**2))
    bias = np.mean(y[te] - pred)
    rmses[m] = rmse
    print(f"   held-out {m:10s} RMSE={rmse:5.2f} pp   mean-error={bias:+5.2f} pp")
mean_all = np.mean(list(rmses.values()))
mean_onc = np.mean([rmses[m] for m in ON_CURVE])
# reference RMSEs (in-sample)
_, ssr_pool, res_pool = ols(Xr, y)
_, ssr_meth, res_meth = ols(Xf, y)
print(f"   mean LOMO RMSE: {mean_all:.2f} pp (all 7) / {mean_onc:.2f} pp (6 on-curve; sclora {rmses['sclora']:.2f})")
print(f"   reference in-sample RMSE: pooled law {np.sqrt(ssr_pool/n):.2f} pp; "
      f"method-aware (7 intercepts) {np.sqrt(ssr_meth/n):.2f} pp")

# ---------------- extras for exhibits ----------------
print("\n[EXTRA] per-method best-adapt operating points (for tables):")
for m in METHODS:
    mr = [r for r in rows if r["m"] == m and r["adapt"] is not None]
    b = max(mr, key=lambda r: r["adapt"])
    nrob = sum(1 for r in mr if r["ret"] >= 24)
    d = recs[b["run"]]
    print(f"   {m:10s} bestLR={b['lr']:g} adapt={b['adapt']:.1f} ret={b['ret']:.1f} "
          f"broad={d.get('retention_broad'):.1f} F={b['F']:.3f} svmax={d.get('dw_sv_max'):.1f} robust={nrob}/7")
