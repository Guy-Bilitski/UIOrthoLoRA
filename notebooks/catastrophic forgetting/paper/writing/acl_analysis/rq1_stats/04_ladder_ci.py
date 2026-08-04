"""04 — Cluster-bootstrap CIs for the exact ladder steps of key_numbers §19.1.

Reason: tables/table_ladder.tex printed the geometry-step CI [+0.006, +0.032]
taken from 09-Q1's *2-metric shape-block unique share* (e_top, stable rank),
while the ladder's geometry step is the 3-metric block
(e_top, log spec_max, stable rank). This recomputes B=2000 cell-level
cluster-bootstrap CIs for the ladder's own steps so the table can carry CIs
that match its block definition.

Ladder (frozen §19.1, n=1034, run-level, family FE, quarantine-included):
  family FE -> + log10 F_delta -> + (e_top, log spec_max, stable rank)
  -> + method dummies.

Outputs: ladder_ci.md, ladder_ci.csv
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 04_ladder_ci.py
"""
import os

import numpy as np
import pandas as pd

from rq1_common import OUT
import corr_common as cc

B = 2000
SEED = 0
GEO = ["e_top_w", "lspec", "stable_rank_w"]


def r2_steps(df):
    """R2 of the four nested models on one (possibly resampled) pool."""
    sub = df.dropna(subset=["logfd", "ret"] + GEO)
    y = sub.ret.values.astype(float)
    out = []
    meth_cols = sorted(m for m in sub.method.unique())[1:]

    def fit(terms, with_method=False):
        cols = [np.ones(len(sub))]
        for f in sorted(sub.fam.unique())[1:]:
            cols.append((sub.fam == f).astype(float).values)
        for t in terms:
            cols.append(sub[t].values.astype(float))
        if with_method:
            for m in meth_cols:
                cols.append((sub.method == m).astype(float).values)
        X = np.column_stack(cols)
        _, _, r2, _ = cc.ols_fit(X, y)
        return r2

    out.append(fit([]))
    out.append(fit(["logfd"]))
    out.append(fit(["logfd"] + GEO))
    out.append(fit(["logfd"] + GEO, with_method=True))
    return np.array(out)


def run():
    df, _ = cc.build(dedupe=True)
    base = r2_steps(df)
    d_mag = base[1] - base[0]
    d_geo = base[2] - base[1]
    d_met = base[3] - base[2]
    print(f"point estimates: R2 {base.round(3)}  "
          f"dmag={d_mag:+.3f} dgeo={d_geo:+.3f} dmethod={d_met:+.3f}")

    rng = np.random.default_rng(SEED)
    cells = df.cell.unique()
    idx_by_cell = {c: np.where(df.cell.values == c)[0] for c in cells}
    boots = np.zeros((B, 3))
    order = 0
    for b in range(B):
        pick = rng.choice(len(cells), size=len(cells), replace=True)
        rows = np.concatenate([idx_by_cell[cells[i]] for i in pick])
        r = r2_steps(df.iloc[rows])
        boots[b] = [r[1] - r[0], r[2] - r[1], r[3] - r[2]]
        order += int(boots[b, 0] > boots[b, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    rows = []
    for i, (nm, pt) in enumerate([("magnitude (log10 F_delta)", d_mag),
                                  ("geometry (e_top, log spec_max, stable rank)", d_geo),
                                  ("method identity", d_met)]):
        rows.append(dict(step=nm, dr2=round(pt, 4),
                         lo=round(lo[i], 4), hi=round(hi[i], 4)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "ladder_ci.csv"), index=False)
    md = ["# Ladder-step cluster-bootstrap CIs (exact §19.1 blocks)",
          "",
          f"n={int(df.dropna(subset=['logfd','ret']+GEO).shape[0])}, family FE,",
          f"B={B} cell-level bootstrap, seed {SEED}. Geometry block is the",
          "ladder's own 3-metric block (e_top, log spec_max, stable rank),",
          "unlike 09-Q1's 2-metric shape-unique CI. Script: `04_ladder_ci.py`.",
          "",
          "| step | dR2 | 95% CI |", "|---|---|---|"]
    for _, x in t.iterrows():
        md.append(f"| {x.step} | {x.dr2:+.3f} | [{x.lo:+.3f}, {x.hi:+.3f}] |")
    md.append("")
    md.append(f"ordering magnitude > geometry: {order}/{B} replicates")
    with open(os.path.join(OUT, "ladder_ci.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[6:]))


if __name__ == "__main__":
    run()
