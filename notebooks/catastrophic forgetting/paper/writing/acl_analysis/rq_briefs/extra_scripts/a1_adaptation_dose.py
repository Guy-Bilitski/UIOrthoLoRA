"""a1_adaptation_dose.py — what governs ADAPTATION across the pool?

Q1: does adaptation have an interior optimum on the magnitude axis (an
    optimal-dose band per family)? Where does it sit vs the retention knee?
Q2: how much adaptation variance is magnitude vs method — the adaptation
    ladder, mirrored on the retention ladder (family FE -> +dose -> +method).
All inference at cell (seed-averaged) level; CR1 clustered at cell for the
run-level mirror check.
"""
import numpy as np
import pandas as pd
from extras_common import (load_pool, cells_of, KNEE_182, FAMS, ols, cr1_se,
                           dummies, hinge_fit)

df = load_pool()
cells = cells_of(df)

print("\n================ Q1: adaptation dose-response per family ================")
rows = []
for fam in FAMS:
    c = cells[(cells.fam == fam) & cells.adapt.notna()].copy()
    knee = KNEE_182[fam]
    # peak-adapt cell and its dose
    peak = c.adapt.max()
    peak_dose = c.loc[c.adapt.idxmax(), "logfd"]
    # optimal band: dose range of cells within 1 pp of family peak
    band = c[c.adapt >= peak - 1.0]
    band_lo, band_hi = band.logfd.min(), band.logfd.max()
    # rising limb: correlation of dose with adapt at/below the peak dose
    below = c[c.logfd <= peak_dose]
    above = c[c.logfd > peak_dose]
    r_up = np.corrcoef(below.logfd, below.adapt)[0, 1] if len(below) > 3 else np.nan
    r_dn = np.corrcoef(above.logfd, above.adapt)[0, 1] if len(above) > 3 else np.nan
    # rising-limb slope (pp/decade)
    if len(below) > 3:
        b_up, *_ = ols(below.logfd.values[:, None], below.adapt.values)
        slope_up = b_up[0][1] if isinstance(b_up, tuple) else b_up[1]
    beta_up, _, _, _, _ = ols(below.logfd.values[:, None], below.adapt.values)
    beta_dn, _, _, _, _ = (ols(above.logfd.values[:, None], above.adapt.values)
                           if len(above) > 3 else (np.array([np.nan, np.nan]),) * 5)
    # hinge fit on adapt (peak location, data-driven)
    kn_adapt, hb, _ = hinge_fit(c.logfd.values, c.adapt.values)
    rows.append(dict(fam=fam, n_cells=len(c), peak_adapt=round(peak, 1),
                     peak_dose=round(peak_dose, 2), band_lo=round(band_lo, 2),
                     band_hi=round(band_hi, 2), band_width=round(band_hi - band_lo, 2),
                     ret_knee=knee, peak_minus_knee=round(peak_dose - knee, 2),
                     r_rising=round(r_up, 3), slope_rising=round(beta_up[1], 1),
                     r_falling=round(r_dn, 3),
                     slope_falling=round(beta_dn[1], 1) if len(above) > 3 else np.nan,
                     hinge_knee_adapt=round(kn_adapt, 2)))
t = pd.DataFrame(rows)
print(t.to_string(index=False))

print("\n---- inverted-U check: quadratic beats linear? (cells, per family) ----")
for fam in FAMS:
    c = cells[(cells.fam == fam) & cells.adapt.notna()]
    x = c.logfd.values
    _, _, _, r2_lin, _ = ols(x[:, None], c.adapt.values)
    _, _, _, r2_quad, _ = ols(np.column_stack([x, x ** 2]), c.adapt.values)
    print(f"{fam}: R2 linear {r2_lin:.3f} -> quadratic {r2_quad:.3f} "
          f"(concave coef sign: {'-' if ols(np.column_stack([x, x**2]), c.adapt.values)[0][2] < 0 else '+'})")

print("\n================ Q2: adaptation ladder (cells, family FE) ================")
# mirrored ladder: family FE -> + logfd + logfd^2 -> + method dummies
# built per outcome so ret and adapt use the identical sample
cc = cells[cells.adapt.notna() & cells.ret.notna()].copy()
fam_d, _ = dummies(cc.fam, ref="frc")
meth_d, meth_lv = dummies(cc.method, ref="lorawd")
x = cc.logfd.values
dose = np.column_stack([x, x ** 2])

for outcome in ["adapt", "ret"]:
    y = cc[outcome].values
    _, _, _, r2_f, _ = ols(fam_d, y)
    _, _, _, r2_fd, _ = ols(np.column_stack([fam_d, dose]), y)
    _, _, _, r2_fdm, _ = ols(np.column_stack([fam_d, dose, meth_d]), y)
    print(f"{outcome}: famFE {r2_f:.3f} -> +dose(logfd,logfd^2) {r2_fd:.3f} "
          f"(+{r2_fd - r2_f:.3f}) -> +method {r2_fdm:.3f} (+{r2_fdm - r2_fd:.3f})")

print("\n-- same ladder, run level with CR1 cell-clustered joint check on method --")
rr = df[df.adapt.notna() & df.ret.notna()].copy()
fam_dr, _ = dummies(rr.fam, ref="frc")
meth_dr, meth_lvr = dummies(rr.method, ref="lorawd")
xr = rr.logfd.values
doser = np.column_stack([xr, xr ** 2])
for outcome in ["adapt", "ret"]:
    y = rr[outcome].values
    _, _, _, r2_fd, _ = ols(np.column_stack([fam_dr, doser]), y)
    beta, resid, _, r2_fdm, Xf = ols(np.column_stack([fam_dr, doser, meth_dr]), y)
    se = cr1_se(Xf, resid, rr.cell.values)
    k0 = 1 + fam_dr.shape[1] + doser.shape[1]
    tstats = beta[k0:] / se[k0:]
    print(f"{outcome}: dR2(method)={r2_fdm - r2_fd:.3f}; method coefs (t, CR1@cell):")
    for lv, b, tt in zip(meth_lvr, beta[k0:], tstats):
        print(f"    {lv:>10}: {b:+6.2f} pp (t={tt:+5.1f})")

print("\n-- per-family dR2(method beyond dose), cells --")
for fam in FAMS:
    c = cells[(cells.fam == fam) & cells.adapt.notna()]
    if c.method.nunique() < 2:
        continue
    md, _ = dummies(c.method, ref=c.method.mode()[0])
    x = c.logfd.values
    d2 = np.column_stack([x, x ** 2])
    out = {}
    for outcome in ["adapt", "ret"]:
        y = c[outcome].values
        _, _, _, r2_d, _ = ols(d2, y)
        _, _, _, r2_dm, _ = ols(np.column_stack([d2, md]), y)
        out[outcome] = (r2_d, r2_dm - r2_d)
    print(f"{fam}: adapt dose-R2 {out['adapt'][0]:.3f} +method {out['adapt'][1]:+.3f} | "
          f"ret dose-R2 {out['ret'][0]:.3f} +method {out['ret'][1]:+.3f} "
          f"(n_cells={len(c)}, n_methods={c.method.nunique()})")
