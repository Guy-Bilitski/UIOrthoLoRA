"""a2b_sclora_slope.py — is SC-LoRA's retention-magnitude slope credibly
steeper than LoRA+wd's, per family and pooled? CR1 @ cell; both pool
conventions; cell-level replication path.
"""
import numpy as np
import pandas as pd
from extras_common import load_pool, cells_of, FAMS, ols, cr1_se

df = load_pool()


def pair_test(pool, label):
    print(f"\n==== {label} ====")
    signs = []
    for fam in FAMS:
        sub = pool[(pool.fam == fam) & pool.method.isin(["sclora", "lorawd"])].copy()
        if (sub.method == "sclora").sum() == 0:
            print(f"{fam}: no sclora")
            continue
        sc = (sub.method == "sclora").astype(float).values
        x = sub.logfd.values
        X = np.column_stack([x, sc, x * sc])
        beta, resid, _, _, Xf = ols(X, sub.ret.values)
        se = cr1_se(Xf, resid, sub.cell.values)
        n_sc_cells = sub[sub.method == "sclora"].cell.nunique()
        dose_range = sub[sub.method == "sclora"].logfd.max() - sub[sub.method == "sclora"].logfd.min()
        print(f"{fam}: lorawd slope {beta[1]:+6.1f}; sclora extra {beta[3]:+6.1f} "
              f"(t={beta[3]/se[3]:+5.2f}; sclora cells={n_sc_cells}, dose span={dose_range:.2f} dec)")
        signs.append(beta[3] < 0)
    print(f"steeper-than-lorawd sign: {sum(signs)}/{len(signs)} families")


pair_test(df, "frozen pool, run level")
pair_test(df[~df.quarantined], "quarantine-excluded, run level")

print("\n==== cell-level replication (seed-averaged, plain OLS interaction) ====")
cells = cells_of(df)
for fam in FAMS:
    sub = cells[(cells.fam == fam) & cells.method.isin(["sclora", "lorawd"])]
    if (sub.method == "sclora").sum() < 3:
        print(f"{fam}: <3 sclora cells")
        continue
    sc = (sub.method == "sclora").astype(float).values
    x = sub.logfd.values
    X = np.column_stack([x, sc, x * sc])
    beta, resid, _, _, Xf = ols(X, sub.ret.values)
    # HC1-ish SE at cell level (cells are the clusters -> plain robust)
    se = cr1_se(Xf, resid, np.arange(len(sub)))
    print(f"{fam}: sclora extra slope {beta[3]:+6.1f} (t={beta[3]/se[3]:+5.2f}, cells={len(sub)})")

print("\n==== all-methods slope rank of sclora per family (>=4 cells, >=0.3 dec span) ====")
for fam in FAMS:
    sub = df[df.fam == fam]
    out = []
    for meth, gg in sub.groupby("method"):
        span = gg.logfd.max() - gg.logfd.min()
        if gg.cell.nunique() < 4 or span < 0.3:
            continue
        beta, resid, _, _, Xf = ols(gg.logfd.values[:, None], gg.ret.values)
        out.append((meth, beta[1]))
    out.sort(key=lambda z: z[1])
    ranks = {m: i + 1 for i, (m, b) in enumerate(out)}
    print(f"{fam}: steepest->flattest: " + ", ".join(f"{m} {b:+.1f}" for m, b in out)
          + (f"  [sclora rank {ranks['sclora']}/{len(out)}]" if "sclora" in ranks else "  [sclora not estimable]"))
