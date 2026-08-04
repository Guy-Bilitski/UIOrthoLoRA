"""07_shareq_stability.py — seed- and family-stability of the share_q partial (insight 6).

partial r(share_q, retention | log F_delta [+ family FE]) per seed and per family.
Appends results to shareq_stability.txt.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M = pd.read_csv(os.path.join(HERE, "permatrix_features.csv"))
out = []


def partial(sub, feat, fam_fe=True):
    cols = [np.ones(len(sub)), sub.logfd.values]
    if fam_fe:
        famd = pd.get_dummies(sub.fam, drop_first=True).astype(float)
        cols += [famd[c].values for c in famd.columns]
    Z = np.column_stack(cols)
    ry = sub.ret.values - Z @ np.linalg.lstsq(Z, sub.ret.values, rcond=None)[0]
    rx = sub[feat].values - Z @ np.linalg.lstsq(Z, sub[feat].values, rcond=None)[0]
    return np.corrcoef(ry, rx)[0, 1], len(sub)


for s in (42, 43, 44, 45):
    sub = M[(M.seed == s) & M.share_q.notna()]
    if len(sub) > 30:
        r, n = partial(sub, "share_q")
        out.append(f"seed {s}: partial r(share_q, ret | logF+fam) = {r:+.3f} (n={n})")
for f in ("lrsw", "frc", "qwsw", "lrswm", "frm", "qwswm"):
    sub = M[(M.fam == f) & M.share_q.notna()]
    r, n = partial(sub, "share_q", fam_fe=False)
    out.append(f"fam {f}: partial r(share_q, ret | logF) = {r:+.3f} (n={n})")

txt = "\n".join(out)
open(os.path.join(HERE, "shareq_stability.txt"), "w").write(txt + "\n")
print(txt)
