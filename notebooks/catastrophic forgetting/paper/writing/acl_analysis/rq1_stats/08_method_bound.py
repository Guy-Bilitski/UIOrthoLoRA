"""08 — the correct bound on a between-method geometry feature.

Reason: Section 5.2 argues that a geometry feature which separates one design
from another is bounded by what method identity adds. The ladder's method step
(+0.006) is the WRONG number for that: it is measured after the geometry block
has already absorbed the between-method variation. A feature that is constant
within a method lies in the span of the method dummies, so the bound is method
identity over MAGNITUDE ALONE.

This script reproduces the frozen ladder first (key_numbers.md §19.1) and then
fits the one model the ladder omits, family FE + log10 F_delta + method dummies.

Outputs: method_bound.md
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 08_method_bound.py
"""
import os

import numpy as np

from rq1_common import OUT
import corr_common as cc

GEO = ["e_top_w", "lspec", "stable_rank_w"]


def run():
    df, _ = cc.build(dedupe=True)
    sub = df.dropna(subset=["logfd", "ret"] + GEO)
    y = sub.ret.values.astype(float)
    meth = sorted(sub.method.unique())[1:]

    def fit(terms, with_method=False):
        cols = [np.ones(len(sub))]
        for f in sorted(sub.fam.unique())[1:]:
            cols.append((sub.fam == f).astype(float).values)
        for t in terms:
            cols.append(sub[t].values.astype(float))
        if with_method:
            for m in meth:
                cols.append((sub.method == m).astype(float).values)
        _, _, r2, _ = cc.ols_fit(np.column_stack(cols), y)
        return r2

    r_fam = fit([])
    r_mag = fit(["logfd"])
    r_geo = fit(["logfd"] + GEO)
    r_all = fit(["logfd"] + GEO, with_method=True)
    r_meth_only = fit(["logfd"], with_method=True)

    # frozen ladder anchors, key_numbers.md 19.1
    for got, want, name in ((r_fam, 0.390, "family"), (r_mag, 0.785, "+magnitude"),
                            (r_geo, 0.802, "+geometry"), (r_all, 0.808, "+method")):
        if abs(got - want) > 5e-4:
            raise SystemExit(f"LADDER MISMATCH {name}: got {got:.4f}, frozen {want:.3f}")

    lines = [
        "# Bound on a between-method geometry feature",
        "",
        f"Frozen pool, run level, family fixed effects, n = {len(sub)}.",
        "",
        "| model | R2 |",
        "|---|---|",
        f"| family FE | {r_fam:.4f} |",
        f"| + log10 F_delta | {r_mag:.4f} |",
        f"| + geometry (e_top, log spec_max, stable rank) | {r_geo:.4f} |",
        f"| + method dummies | {r_all:.4f} |",
        f"| family FE + log10 F_delta + method dummies (no geometry) | {r_meth_only:.4f} |",
        "",
        f"- ladder method step, after geometry: **{r_all - r_geo:+.4f}**",
        f"- method identity over magnitude alone: **{r_meth_only - r_mag:+.4f}**",
        f"- geometry and method together over magnitude: **{r_all - r_mag:+.4f}**",
        "",
        "The middle line is the bound Section 5.2 needs. A geometry feature that is",
        "constant within a method lies in the span of the method dummies, so what",
        "method identity adds over size alone is what such a feature could at most add.",
        "The ladder's own method step is measured after the geometry block has already",
        "absorbed the between-method variation and is not a bound on it.",
    ]
    path = os.path.join(OUT, "method_bound.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print("wrote", path)


if __name__ == "__main__":
    run()
