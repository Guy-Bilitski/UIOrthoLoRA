"""02 — Equivalence (TOST) for method offsets at matched update magnitude.

The frozen layer argues "methods are near-interchangeable at matched F_delta"
from small point estimates and n.s. tests — accepting the null. This script
tests equivalence properly: per family, regress

    ret ~ 1 + log10 F_delta + method dummies   (reference: LoRA+wd)

with CR1 cluster-robust SEs at the recipe-cell level (corr_common convention,
deduped n=1034 pool, retention = retention_mean everywhere as in the ladder).
TOST at margin m: a method is EQUIVALENT to LoRA+wd at matched magnitude iff
its 90% CI (two one-sided 5% tests) lies inside (-m, +m). Margins reported:
+/-1, +/-2, +/-3 points. Degrees of freedom for t critical values = G-1
(G = clusters in the family).

Also fits the pooled model with family FE.

Outputs: tost_offsets.csv, tost_offsets.md
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 02_equivalence_tost.py
"""
import os

import numpy as np
import pandas as pd

from rq1_common import OUT, t_crit
import corr_common as cc

MARGINS = (1.0, 2.0, 3.0)
REF = "lorawd"


def offsets(sub, label, fe=False):
    """Fit ret ~ logfd + method dummies (+ family FE if fe); return rows."""
    sub = sub.dropna(subset=["logfd", "ret"]).copy()
    methods = sorted(m for m in sub.method.unique() if m != REF)
    cols = [np.ones(len(sub)), sub.logfd.values.astype(float)]
    names = ["const", "logfd"]
    if fe:
        for f in sorted(sub.fam.unique())[1:]:
            cols.append((sub.fam == f).astype(float).values)
            names.append(f"fam:{f}")
    for m in methods:
        cols.append((sub.method == m).astype(float).values)
        names.append(m)
    X = np.column_stack(cols)
    y = sub.ret.values.astype(float)
    beta, resid, r2, _ = cc.ols_fit(X, y)
    se, G = cc.cluster_robust_se(X, resid, sub.cell.values)
    df = G - 1
    rows = []
    for i, nm in enumerate(names):
        if nm not in methods:
            continue
        b, s = beta[i], se[i]
        tc90 = t_crit(0.10, df)   # 90% CI <=> TOST at 5%
        tc95 = t_crit(0.05, df)
        lo90, hi90 = b - tc90 * s, b + tc90 * s
        n_m = int((sub.method == nm).sum())
        row = dict(scope=label, method=nm, n=n_m, G=G, beta=b, se=s,
                   lo90=lo90, hi90=hi90,
                   lo95=b - tc95 * s, hi95=b + tc95 * s, r2=r2)
        for m in MARGINS:
            row[f"equiv_{m:g}pp"] = bool(lo90 > -m and hi90 < m)
        rows.append(row)
    return rows


def run():
    df, _ = cc.build(dedupe=True)
    rows = []
    for fam in cc.FAMS:
        rows += offsets(df[df.fam == fam], fam)
    rows += offsets(df, "pooled(all six, family FE)", fe=True)
    t = pd.DataFrame(rows)
    num = t.select_dtypes(float).columns
    t[num] = t[num].round(4)
    t.to_csv(os.path.join(OUT, "tost_offsets.csv"), index=False)

    md = ["# TOST equivalence: method offsets vs LoRA+wd at matched magnitude",
          "",
          "Model: ret ~ log10 F_delta + method dummies (reference LoRA+wd),",
          "CR1 cluster-robust SEs at recipe-cell level, deduped n=1034 pool,",
          "retention = retention_mean (ladder convention). Equivalent at margin",
          "m iff the 90% CI of the offset lies inside (-m, +m).",
          "Script: `02_equivalence_tost.py`.", ""]
    for scope, sub in t.groupby("scope", sort=False):
        lab = cc.FAM_LABEL.get(scope, scope)
        md.append(f"## {lab}  (G={int(sub.G.iloc[0])} cells)")
        md.append("")
        md.append("| method | n | offset (pp) | 90% CI | 95% CI | eq +/-1 | eq +/-2 | eq +/-3 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, x in sub.sort_values("method").iterrows():
            def yn(v):
                return "YES" if v else "no"
            md.append(f"| {x.method} | {x.n} | {x.beta:+.2f} | "
                      f"[{x.lo90:+.2f}, {x.hi90:+.2f}] | [{x.lo95:+.2f}, {x.hi95:+.2f}] | "
                      f"{yn(x['equiv_1pp'])} | {yn(x['equiv_2pp'])} | {yn(x['equiv_3pp'])} |")
        md.append("")

    md.append("## Summary")
    md.append("")
    for m in MARGINS:
        pooled = t[t.scope.str.startswith("pooled")]
        k = int(pooled[f"equiv_{m:g}pp"].sum())
        md.append(f"- pooled model: {k}/{len(pooled)} methods equivalent to "
                  f"LoRA+wd within +/-{m:g} pp at matched magnitude")
    perfam = t[~t.scope.str.startswith("pooled")]
    k2 = int(perfam["equiv_2pp"].sum())
    md.append(f"- per-family: {k2}/{len(perfam)} method x family offsets "
              f"equivalent within +/-2 pp; non-equivalences listed below")
    for _, x in perfam[~perfam["equiv_2pp"]].iterrows():
        md.append(f"  - not eq at +/-2: {x.method} in {x.scope} "
                  f"(offset {x.beta:+.2f}, 90% CI [{x.lo90:+.2f}, {x.hi90:+.2f}])")
    with open(os.path.join(OUT, "tost_offsets.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[-20:]))
    print(f"\nwrote {OUT}/tost_offsets.csv, .md")


if __name__ == "__main__":
    run()
