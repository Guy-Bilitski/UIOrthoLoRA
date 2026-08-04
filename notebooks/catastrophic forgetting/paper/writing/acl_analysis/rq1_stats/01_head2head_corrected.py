"""01 — Head-to-head vs LoRA+wd with exact p-values, 95% CIs, and Holm
multiple-comparison correction.

Rebuilds the adjudication/03_head2head.py comparisons (each method's
best-adaptation cell vs LoRA+wd's, per family, paired per-seed where >=2
common seeds, Welch otherwise) and adds what the frozen layer lacks:
  - exact two-sided p per comparison (paired t df=n-1; Welch-Satterthwaite);
  - 95% CI on each retention/adaptation delta;
  - Holm-adjusted p within each family and across all comparisons per axis.

Outputs: head2head_corrected.csv, head2head_corrected.md
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 01_head2head_corrected.py
"""
import os

import numpy as np
import pandas as pd

from rq1_common import OUT, holm, paired_stats, welch_stats
from adjpool import (FAMILIES, DISPLAY, WITHHELD, load_pool, preflight_18_1,
                     family_rows, cell_table, best_cell, fmt_lr)


def seed_map(fr, mkey, lr, k, ret_field):
    sub = fr[(fr.mkey == mkey) & (fr.lr == lr)]
    if k:
        sub = sub[sub.run.str.contains(f"_{k}_")]
    out = {}
    for _, r in sub.iterrows():
        if r.seed is not None and r["adapt"] is not None and r[ret_field] is not None:
            out[int(r.seed)] = (float(r["adapt"]), float(r[ret_field]))
    return out


def axis_stats(vals_m, vals_w):
    common = sorted(set(vals_m) & set(vals_w))
    if len(common) >= 2:
        st = paired_stats([vals_m[s] - vals_w[s] for s in common])
        st["mode"] = f"paired(n={len(common)})"
        return st
    st = welch_stats(list(vals_m.values()), list(vals_w.values()))
    st["mode"] = f"welch({len(vals_m)}v{len(vals_w)})"
    return st


def run():
    df = load_pool()
    n, r = preflight_18_1(df)
    print(f"PREFLIGHT OK: n={n}, r={r:.3f}")

    rows = []
    for fk, spec in FAMILIES.items():
        fr = family_rows(df, fk)
        cells = cell_table(fr, spec["ret_field"])
        ref = best_cell(cells, "lorawd")
        vw = seed_map(fr, "lorawd", ref.lr, ref.k, spec["ret_field"])
        vw_a = {s: v[0] for s, v in vw.items()}
        vw_r = {s: v[1] for s, v in vw.items()}
        for mkey, _ in spec["specs"]:
            if mkey == "lorawd" or mkey in WITHHELD:
                continue
            bc = best_cell(cells, mkey)
            if bc is None:
                continue
            vm = seed_map(fr, mkey, bc.lr, bc.k, spec["ret_field"])
            sa = axis_stats({s: v[0] for s, v in vm.items()}, vw_a)
            sr = axis_stats({s: v[1] for s, v in vm.items()}, vw_r)
            rows.append(dict(
                family=fk, method=DISPLAY[mkey],
                method_cell=f"{fmt_lr(bc.lr)}{(' ' + bc.k) if bc.k else ''}",
                ref_cell=fmt_lr(ref.lr), mode=sr["mode"],
                d_adapt=sa["mean"], t_adapt=sa["t"], df_adapt=sa["df"],
                p_adapt=sa["p"], lo_adapt=sa["lo"], hi_adapt=sa["hi"],
                d_ret=sr["mean"], t_ret=sr["t"], df_ret=sr["df"],
                p_ret=sr["p"], lo_ret=sr["lo"], hi_ret=sr["hi"],
            ))
    t = pd.DataFrame(rows)

    # Holm within family and across all comparisons, per axis.
    for ax in ("ret", "adapt"):
        t[f"p_{ax}_holm_family"] = np.nan
        for fk in t.family.unique():
            m = t.family == fk
            t.loc[m, f"p_{ax}_holm_family"] = holm(t.loc[m, f"p_{ax}"].values)
        t[f"p_{ax}_holm_all"] = holm(t[f"p_{ax}"].values)

    def verdict(row, ax):
        p = row[f"p_{ax}_holm_all"]
        if not np.isfinite(p):
            return "n.t."   # not testable (a group with n=1)
        if p >= 0.05:
            return "n.s."
        return "WORSE" if row[f"d_{ax}"] < 0 else "BETTER"

    t["ret_verdict_holm"] = t.apply(lambda r_: verdict(r_, "ret"), axis=1)
    t["adapt_verdict_holm"] = t.apply(lambda r_: verdict(r_, "adapt"), axis=1)

    num = t.select_dtypes(float).columns
    t[num] = t[num].round(4)
    t.to_csv(os.path.join(OUT, "head2head_corrected.csv"), index=False)

    md = ["# Head-to-head vs LoRA+wd, exact p + Holm correction",
          "",
          f"Same comparisons as adjudication/03_head2head.py ({len(t)} method x",
          "family cells vs LoRA+wd at best-adaptation operating points). Paired",
          "per-seed t (df=n-1) where >=2 common seeds; Welch-Satterthwaite",
          "otherwise. Holm within family and across all comparisons per axis.",
          "Deltas in points; CI = 95%. Script: `01_head2head_corrected.py`.", ""]
    for fk in FAMILIES:
        sub = t[t.family == fk]
        if not len(sub):
            continue
        md.append(f"## {FAMILIES[fk]['title']}")
        md.append("")
        md.append("| Method (cell) | dRet [CI95] | t | p | p Holm(fam) | p Holm(all) | verdict | dAdapt [CI95] | p Holm(all) | verdict | test |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, x in sub.iterrows():
            def f(v, d=2):
                return "--" if not np.isfinite(v) else f"{v:.{d}f}"
            md.append(
                f"| {x.method} ({x.method_cell}) | {x.d_ret:+.2f} [{f(x.lo_ret)}, {f(x.hi_ret)}] | "
                f"{f(x.t_ret)} | {f(x.p_ret, 4)} | {f(x.p_ret_holm_family, 4)} | {f(x.p_ret_holm_all, 4)} | "
                f"{x.ret_verdict_holm} | {x.d_adapt:+.2f} [{f(x.lo_adapt)}, {f(x.hi_adapt)}] | "
                f"{f(x.p_adapt_holm_all, 4)} | {x.adapt_verdict_holm} | {x['mode']} |")
        md.append("")

    md.append("## Summary")
    md.append("")
    n_cmp = int(np.isfinite(t.p_ret).sum())
    n_nt = int((~np.isfinite(t.p_ret)).sum())
    better = t[t.ret_verdict_holm == "BETTER"]
    worse = t[t.ret_verdict_holm == "WORSE"]
    md.append(f"- Retention, Holm across all {n_cmp} testable comparisons "
              f"({n_nt} not testable, single-seed group; delta reported only): "
              f"**{len(better)} method(s) significantly better** than LoRA+wd, "
              f"{len(worse)} significantly worse, rest n.s.")
    for _, x in t[t.ret_verdict_holm == "n.t."].iterrows():
        md.append(f"  - not testable: {x.method} in {x.family} "
                  f"(dRet={x.d_ret:+.2f}, dAdapt={x.d_adapt:+.2f}, {x['mode']})")
    for _, x in worse.iterrows():
        md.append(f"  - worse: {x.method} in {x.family} "
                  f"(d={x.d_ret:+.2f}, p_holm={x.p_ret_holm_all:.4f})")
    for _, x in better.iterrows():
        md.append(f"  - better: {x.method} in {x.family} "
                  f"(d={x.d_ret:+.2f}, p_holm={x.p_ret_holm_all:.4f})")
    a_better = t[t.adapt_verdict_holm == "BETTER"]
    a_worse = t[t.adapt_verdict_holm == "WORSE"]
    md.append(f"- Adaptation, Holm across all: {len(a_better)} better, "
              f"{len(a_worse)} worse, rest n.s.")
    for _, x in a_better.iterrows():
        md.append(f"  - better: {x.method} in {x.family} "
                  f"(d={x.d_adapt:+.2f}, p_holm={x.p_adapt_holm_all:.4f})")
    sc = t[(t.family == "qwen_math") & (t.method == "SC-LoRA")]
    if len(sc):
        x = sc.iloc[0]
        md.append(f"- SC-LoRA on Qwen-math (the frozen layer's one exception): "
                  f"dAdapt={x.d_adapt:+.2f} raw p={x.p_adapt:.4f}, "
                  f"Holm(family)={x.p_adapt_holm_family:.4f}, "
                  f"Holm(all)={x.p_adapt_holm_all:.4f}; "
                  f"dRet={x.d_ret:+.2f} raw p={x.p_ret:.4f}.")
    with open(os.path.join(OUT, "head2head_corrected.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[-14:]))
    print(f"\nwrote {OUT}/head2head_corrected.csv, .md")


if __name__ == "__main__":
    run()
