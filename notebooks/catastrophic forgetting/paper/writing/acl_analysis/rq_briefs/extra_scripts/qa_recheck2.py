"""qa_recheck2.py — second code path for the remaining candidates:

  E2 loose coupling: retention is tightly dose-determined, adaptation only
     loosely. Path 1 was cell-level quadratic R2 (a1). Path 2 here: RUN-level
     quadratic R2 per family for both outcomes, plus a knee-free hinge fit
     (free knot) so the comparison is not hostage to the quadratic form.
  A-limb: falling-limb adaptation slope re-estimated WITHOUT the quadratic:
     linear fit on runs above the family argmax dose-bin midpoint, CR1@cell.
  Sanity: which adapt task per family; both pool conventions for E2.
"""
import numpy as np
import pandas as pd
from extras_common import load_pool, cells_of, KNEE_182, FAMS, ols, cr1_se, hinge_fit

df = load_pool()
print("adapt task per family:", df.groupby("fam").adapt_task.unique().to_dict())

def quad_r2(x, y):
    _, _, _, r2, _ = ols(np.column_stack([x, x * x]), y)
    return r2

def hinge_r2(x, y):
    kn, beta, sse = hinge_fit(x, y)
    return 1 - sse / (len(y) * y.var())

for label, pool in [("frozen", df), ("quarantine-excluded", df[~df.quarantined])]:
    print(f"\n==== E2 loose coupling, run level, {label} ====")
    for fam in FAMS:
        sub = pool[(pool.fam == fam) & pool.adapt.notna() & pool.ret.notna()]
        x = sub.logfd.values
        ra_q, rr_q = quad_r2(x, sub.adapt.values), quad_r2(x, sub.ret.values)
        ra_h, rr_h = hinge_r2(x, sub.adapt.values), hinge_r2(x, sub.ret.values)
        print(f"  {fam}: adapt R2 quad {ra_q:.2f} / hinge {ra_h:.2f}  |  "
              f"ret R2 quad {rr_q:.2f} / hinge {rr_h:.2f}  (n={len(sub)})")

print("\n==== A falling limb, quadratic-free (runs above argmax bin mid, CR1) ====")
cells = cells_of(df)
for fam in FAMS:
    c = cells[(cells.fam == fam) & cells.adapt.notna()]
    b = pd.qcut(c.logfd, 5, duplicates="drop")
    bm = c.groupby(b, observed=True).adapt.mean()
    top = bm.idxmax()
    mid = (top.left + top.right) / 2
    sub = df[(df.fam == fam) & (df.logfd > mid) & df.adapt.notna()]
    beta, resid, _, _, Xf = ols(sub.logfd.values[:, None], sub.adapt.values)
    se = cr1_se(Xf, resid, sub.cell.values)
    print(f"  {fam}: slope above optimum {beta[1]:+.1f} +/- {se[1]:.1f} pp/dec "
          f"(runs={len(sub)}, cells={sub.cell.nunique()}, argmax-bin mid {mid:+.2f})")
