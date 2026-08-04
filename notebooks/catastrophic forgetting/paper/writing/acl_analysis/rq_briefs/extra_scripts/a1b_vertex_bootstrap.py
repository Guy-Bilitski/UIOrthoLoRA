"""a1b_vertex_bootstrap.py — robustify the adaptation-optimum location.

Estimates the adaptation-optimal dose per family as the vertex of a quadratic
fit at cell level (less noisy than the argmax cell), with a 95% cell-bootstrap
CI, and its distance to the frozen retention knee (§18.2). Also a 2-pp
optimal band and the same computation on the quarantine-excluded pool
(convention sensitivity).
"""
import numpy as np
import pandas as pd
from extras_common import load_pool, cells_of, KNEE_182, FAMS, ols

rng = np.random.default_rng(0)
df = load_pool()


def vertex(c):
    beta, *_ = ols(np.column_stack([c.logfd.values, c.logfd.values ** 2]), c.adapt.values)
    a, b = beta[2], beta[1]
    if a >= 0:
        return np.nan
    return -b / (2 * a)


def run(pool, label):
    cells = cells_of(pool)
    print(f"\n==== {label} ====")
    rows = []
    for fam in FAMS:
        c = cells[(cells.fam == fam) & cells.adapt.notna()].copy()
        v = vertex(c)
        # clamp check: vertex within observed dose range?
        inr = c.logfd.min() <= v <= c.logfd.max()
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(c), len(c))
            vb = vertex(c.iloc[idx])
            if np.isfinite(vb):
                boots.append(np.clip(vb, c.logfd.min() - 0.5, c.logfd.max() + 0.5))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        knee = KNEE_182[fam]
        # 2-pp optimal band from cells
        peak = c.adapt.max()
        band = c[c.adapt >= peak - 2.0]
        rows.append(dict(fam=fam, n_cells=len(c), vertex=round(v, 2),
                         ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         in_range=inr, knee=knee, vertex_minus_knee=round(v - knee, 2),
                         knee_in_ci=lo <= knee <= hi,
                         band2_lo=round(band.logfd.min(), 2),
                         band2_hi=round(band.logfd.max(), 2),
                         knee_in_band2=band.logfd.min() <= knee <= band.logfd.max()))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    d = t.vertex_minus_knee.abs()
    print(f"|vertex - knee|: median {d.median():.2f} decades, max {d.max():.2f}; "
          f"knee inside 95% CI in {int(t.knee_in_ci.sum())}/6, "
          f"knee inside 2-pp band in {int(t.knee_in_band2.sum())}/6")
    return t


t1 = run(df, "frozen pool (n=1035)")
t2 = run(df[~df.quarantined], f"quarantine-excluded (n={int((~df.quarantined).sum())})")
