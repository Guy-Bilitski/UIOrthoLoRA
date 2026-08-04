"""03 — Power notes: minimum detectable effects per family.

Makes "n.s. on Qwen is not evidence of no effect" quantitative. For each
adjudication family:
  - within-cell retention SD per method (median across that method's cells
    with n>=2 seeds) and its family median;
  - empirical SD of paired per-seed retention deltas vs LoRA+wd at the
    best-adaptation cells (the actual noise of the head-to-head test);
  - the minimum detectable |delta| (two-sided paired t, alpha=.05, power=.8)
    at the observed n of common seeds.

Outputs: power_notes.csv, power_notes.md
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 03_power_notes.py
"""
import os

import numpy as np
import pandas as pd

from rq1_common import OUT, mde_paired
from adjpool import (FAMILIES, DISPLAY, WITHHELD, load_pool, preflight_18_1,
                     family_rows, cell_table, best_cell)


def seed_vals(fr, mkey, lr, k, ret_field):
    sub = fr[(fr.mkey == mkey) & (fr.lr == lr)]
    if k:
        sub = sub[sub.run.str.contains(f"_{k}_")]
    return {int(r.seed): float(r[ret_field]) for _, r in sub.iterrows()
            if r.seed is not None and r[ret_field] is not None}


def run():
    df = load_pool()
    n, r = preflight_18_1(df)
    print(f"PREFLIGHT OK: n={n}, r={r:.3f}")

    rows = []
    fam_summary = []
    for fk, spec in FAMILIES.items():
        fr = family_rows(df, fk)
        rf = spec["ret_field"]
        cells = cell_table(fr, rf)
        multi = cells[cells.n >= 2]
        fam_sd = float(multi.ret_sd.median()) if len(multi) else np.nan

        ref = best_cell(cells, "lorawd")
        vw = seed_vals(fr, "lorawd", ref.lr, ref.k, rf)
        mdes = []
        for mkey, _ in spec["specs"]:
            if mkey == "lorawd" or mkey in WITHHELD:
                continue
            bc = best_cell(cells, mkey)
            if bc is None:
                continue
            vm = seed_vals(fr, mkey, bc.lr, bc.k, rf)
            common = sorted(set(vm) & set(vw))
            m_cells = multi[multi.mkey == mkey]
            m_sd = float(m_cells.ret_sd.median()) if len(m_cells) else np.nan
            if len(common) >= 2:
                d = np.array([vm[s] - vw[s] for s in common])
                sd_diff = d.std(ddof=1)
                mde = mde_paired(sd_diff, len(common))
            else:
                sd_diff, mde = np.nan, np.nan
            rows.append(dict(family=fk, method=DISPLAY[mkey],
                             n_common_seeds=len(common),
                             cell_sd_ret=m_sd, sd_paired_diff=sd_diff,
                             mde_pp=mde))
            if np.isfinite(mde):
                mdes.append(mde)
        fam_summary.append(dict(
            family=fk, median_cell_sd_ret=fam_sd,
            median_mde_pp=float(np.median(mdes)) if mdes else np.nan,
            max_mde_pp=float(np.max(mdes)) if mdes else np.nan,
            n_h2h_with_mde=len(mdes)))

    t = pd.DataFrame(rows)
    s = pd.DataFrame(fam_summary)
    for x in (t, s):
        num = x.select_dtypes(float).columns
        x[num] = x[num].round(3)
    t.to_csv(os.path.join(OUT, "power_notes.csv"), index=False)

    md = ["# Power notes: minimum detectable retention effects",
          "",
          "MDE = smallest |paired mean delta| detectable at alpha=.05 (two-sided),",
          "power=.8, at the observed number of common seeds and the empirical SD",
          "of the paired per-seed deltas vs LoRA+wd (best-adaptation cells).",
          "cell_sd_ret = median within-cell retention SD (cells with n>=2).",
          "Script: `03_power_notes.py`.", "",
          "## Family summary", "",
          "| family | median cell SD (ret) | median MDE (pp) | max MDE (pp) |",
          "|---|---|---|---|"]
    for _, x in s.iterrows():
        md.append(f"| {x.family} | {x.median_cell_sd_ret:.2f} | "
                  f"{x.median_mde_pp:.2f} | {x.max_mde_pp:.2f} |")
    md += ["", "## Per comparison", "",
           "| family | method | common seeds | cell SD | SD(paired diff) | MDE (pp) |",
           "|---|---|---|---|---|---|"]
    for _, x in t.iterrows():
        def f(v):
            return "--" if not np.isfinite(v) else f"{v:.2f}"
        md.append(f"| {x.family} | {x.method} | {x.n_common_seeds} | "
                  f"{f(x.cell_sd_ret)} | {f(x.sd_paired_diff)} | {f(x.mde_pp)} |")
    with open(os.path.join(OUT, "power_notes.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(s.to_string(index=False))
    print(f"\nwrote {OUT}/power_notes.csv, .md")


if __name__ == "__main__":
    run()
