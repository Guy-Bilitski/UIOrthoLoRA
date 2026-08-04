"""a3_benchmark_method.py — per-benchmark micro-structure: at matched
magnitude, do methods damage BBH vs MMLU-Pro differently (beyond the known
family-level fragility ordering)?

Design: per family, per benchmark, fit bench ~ logfd + hinge(logfd - knee_182)
+ method dummies (ref lorawd) at run level with CR1 @ cell. The method
coefficient is the matched-magnitude offset on that benchmark. Then contrast
each method's MMLU-Pro offset vs BBH offset in fraction-of-ceiling units
(scale-free), and check cross-family sign consistency.
"""
import numpy as np
import pandas as pd
from extras_common import load_pool, KNEE_182, FAMS, ols, cr1_se, dummies

df = load_pool()
BENCH = ["bbh", "mmlu_pro", "mmlu"]

# family benchmark ceilings (max cell mean) for fraction-of-ceiling scaling
cellm = df.groupby(["fam", "cell"])[BENCH + ["logfd"]].mean().reset_index()
CEIL = {(f, b): cellm[cellm.fam == f][b].max() for f in FAMS for b in BENCH}

rows = []
for fam in FAMS:
    sub = df[df.fam == fam].copy()
    knee = KNEE_182[fam]
    md, lv = dummies(sub.method, ref="lorawd")
    x = sub.logfd.values
    Xbase = np.column_stack([x, np.maximum(0, x - knee), md])
    for b in BENCH:
        y = (sub[b] / CEIL[(fam, b)]).values * 100.0  # % of family ceiling
        ok = np.isfinite(y)
        beta, resid, _, _, Xf = ols(Xbase[ok], y[ok])
        se = cr1_se(Xf, resid, sub.cell.values[ok])
        for i, m in enumerate(lv):
            n_m = int((sub.method == m).sum())
            rows.append(dict(fam=fam, bench=b, method=m, off=beta[3 + i],
                             se=se[3 + i], t=beta[3 + i] / se[3 + i], n=n_m))
t = pd.DataFrame(rows)

print("\n==== matched-magnitude method offsets, % of family ceiling (ref lorawd) ====")
piv = t.pivot_table(index=["method", "fam"], columns="bench", values="off")
tpiv = t.pivot_table(index=["method", "fam"], columns="bench", values="t")
piv["mmlupro_minus_bbh"] = piv["mmlu_pro"] - piv["bbh"]
print(piv.round(1).to_string())

print("\n==== cross-family consistency: does method X hit MMLU-Pro harder than BBH? ====")
for m in sorted(t.method.unique()):
    d = piv.loc[m]["mmlupro_minus_bbh"] if m in piv.index.get_level_values(0) else None
    d = piv.xs(m, level=0)["mmlupro_minus_bbh"].dropna()
    if len(d) < 2:
        continue
    print(f"{m:>10}: MMLU-Pro-minus-BBH offset per family: "
          + ", ".join(f"{f}={v:+.1f}" for f, v in d.items())
          + f"  | negative (Pro hit harder) in {(d<0).sum()}/{len(d)}")

print("\n==== offsets with |t|>2.5 (CR1), any benchmark ====")
sig = t[np.abs(t.t) > 2.5].sort_values(["method", "bench", "fam"])
print(sig.assign(off=sig.off.round(1), t=sig.t.round(1))[["method", "fam", "bench", "off", "t", "n"]].to_string(index=False))
