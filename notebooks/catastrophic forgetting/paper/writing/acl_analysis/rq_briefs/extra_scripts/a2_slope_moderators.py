"""a2_slope_moderators.py — does the retention-magnitude slope have credible
moderators (rank / wd / method class / base model), cluster-aware?

Design: run-level within family (or within grid) with CR1 SEs clustered at
cell, testing the logfd x moderator interaction. Slope tests are within-family
only (raw pp scales differ across base models); the base-model contrast uses
fraction-of-base retention.
"""
import numpy as np
import pandas as pd
from extras_common import load_pool, FAMS, GEOM_METHODS, ols, cr1_se, dummies

df = load_pool()


def interaction_test(sub, mod_vals, label, cluster):
    """ret ~ logfd + mod + logfd*mod, CR1 @ cell. mod_vals: numeric array."""
    x = sub.logfd.values
    m = np.asarray(mod_vals, float)
    X = np.column_stack([x, m, x * m])
    beta, resid, _, r2, Xf = ols(X, sub.ret.values)
    se = cr1_se(Xf, resid, cluster)
    t_int = beta[3] / se[3]
    print(f"  {label:<42} n={len(sub):4d} slope={beta[1]:+6.2f} "
          f"interaction={beta[3]:+6.3f} (t={t_int:+5.2f})")
    return beta[3], t_int


print("\n==== (1) rank as slope moderator (frc grid: rank 8/16/32, lora/lorawd arms) ====")
sub = df[(df.fam == "frc") & df["rank"].isin([8, 16, 32])]
print("  n by rank:", sub.groupby("rank").size().to_dict())
interaction_test(sub, np.log2(sub["rank"]), "frc: logfd x log2(rank)", sub.cell.values)

print("\n==== (2) wd as slope moderator (frc grid, lorawd arms wd 0..0.5) ====")
sub = df[(df.fam == "frc") & df.wd.notna()]
print("  n by wd:", sub.groupby("wd").size().to_dict())
interaction_test(sub, sub.wd, "frc: logfd x wd", sub.cell.values)

print("\n==== (3) method class (geometry-constrained vs plain) per family ====")
for fam in FAMS:
    sub = df[df.fam == fam].copy()
    g = sub.method.isin(GEOM_METHODS).astype(float)
    if g.nunique() < 2:
        continue
    interaction_test(sub, g, f"{fam}: logfd x is_geometry_method", sub.cell.values)

print("\n==== (4) per-method slopes within family (cells>=?, run level, CR1) ====")
for fam in FAMS:
    sub = df[df.fam == fam]
    out = []
    for meth, gg in sub.groupby("method"):
        if gg.cell.nunique() < 4:
            continue
        beta, resid, _, _, Xf = ols(gg.logfd.values[:, None], gg.ret.values)
        se = cr1_se(Xf, resid, gg.cell.values)
        out.append((meth, len(gg), gg.cell.nunique(), beta[1], se[1]))
    slopes = [o[3] for o in out]
    print(f"{fam}: " + "; ".join(f"{m} {b:+.1f}±{s:.1f}" for m, n, nc, b, s in out)
          + f"  [range {max(slopes)-min(slopes):.1f} pp/dec]")

print("\n==== (5) full method x logfd interaction F-ish test per family (CR1 Wald per coef) ====")
for fam in FAMS:
    sub = df[df.fam == fam].copy()
    ref = "lorawd" if (sub.method == "lorawd").any() else sub.method.mode()[0]
    md, lv = dummies(sub.method, ref)
    x = sub.logfd.values
    X = np.column_stack([x, md, md * x[:, None]])
    beta, resid, _, r2, Xf = ols(X, sub.ret.values)
    se = cr1_se(Xf, resid, sub.cell.values)
    k0 = 2 + md.shape[1]
    ts = beta[k0:] / se[k0:]
    sig = [(l, b, t) for l, b, t in zip(lv, beta[k0:], ts) if abs(t) > 2.5]
    print(f"{fam}: interaction terms |t|>2.5 (of {len(lv)}): "
          + ("; ".join(f"{l} {b:+.1f} (t={t:+.1f})" for l, b, t in sig) or "none"))

print("\n==== (6) base model as moderator, fraction-of-base retention ====")
BASE_RET = {}  # family ceiling = max cell-mean ret as proxy for base ceiling
cells = df.groupby(["fam", "cell"]).agg(ret=("ret", "mean"), logfd=("logfd", "mean")).reset_index()
for fam in FAMS:
    BASE_RET[fam] = cells[cells.fam == fam].ret.max()
sub = df.copy()
sub["ret_frac"] = sub.ret / sub.fam.map(BASE_RET)
sub["is_qwen"] = sub.fam.str.startswith("qw").astype(float)
sub["is_math"] = sub.fam.isin(["lrswm", "frm", "qwswm"]).astype(float)
x = sub.logfd.values
X = np.column_stack([x, sub.is_qwen, x * sub.is_qwen, sub.is_math, x * sub.is_math])
beta, resid, _, _, Xf = ols(X, sub.ret_frac.values)
se = cr1_se(Xf, resid, sub.cell.values)
print(f"  slope (llama-cs ref) {beta[1]:+.3f}/dec; qwen shift {beta[3]:+.3f} (t={beta[3]/se[3]:+.1f}); "
      f"math shift {beta[5]:+.3f} (t={beta[5]/se[5]:+.1f})  [fraction-of-ceiling units]")
