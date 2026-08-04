"""05_adaptation_side.py — adaptation-side structure across the 8 CS datasets.

Along increasing F_delta, per-dataset adaptation accuracy (boolq..openbookqa) in the
three CS arms (lrsw, frc: Llama; qwsw: Qwen):
  - where does each dataset PEAK (argmax of rolling mean over cells)?
  - collapse onset: first logfd (after peak) where the rolling mean is >5pp below peak
  - which dataset collapses first / is most magnitude-hungry?
  - differential gain: r(dataset_acc, logfd) below vs above the family knee
  - is there an adaptation-side ordering as universal as the retention-side one?
Outputs: adaptation_side.csv/.md, fig_adaptation_side.png/.pdf
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "pool.csv"))

DS = ["cs_boolq", "cs_piqa", "cs_social_i_qa", "cs_hellaswag", "cs_winogrande",
      "cs_ARC_Easy", "cs_ARC_Challenge", "cs_openbookqa"]
CS_FAMS = ["lrsw", "frc", "qwsw"]
KNEE_182 = dict(lrsw=-0.02, qwsw=-0.69, frc=-0.45)

rows = []
lines = ["# Adaptation-side structure (8 CS datasets, cell level)", ""]
for fam in CS_FAMS:
    sub = df[df.fam == fam]
    cells = sub.groupby("cell").agg({**{d: "mean" for d in DS}, "logfd": "mean"}).dropna()
    cells = cells.sort_values("logfd")
    lines.append(f"## {fam} (n_cells={len(cells)})")
    for d in DS:
        roll = cells[d].rolling(9, center=True, min_periods=5).mean()
        ipk = roll.idxmax()
        pk, pk_x = roll.loc[ipk], cells.loc[ipk, "logfd"]
        after = cells.loc[cells.logfd >= pk_x]
        rafter = roll.loc[after.index]
        collapsed = rafter[rafter < pk - 5]
        onset = cells.loc[collapsed.index[0], "logfd"] if len(collapsed) else np.nan
        knee = KNEE_182[fam]
        lo = cells[cells.logfd <= knee]
        hi = cells[cells.logfd > knee]
        r_lo = stats.pearsonr(lo.logfd, lo[d]).statistic if len(lo) > 4 else np.nan
        r_hi = stats.pearsonr(hi.logfd, hi[d]).statistic if len(hi) > 4 else np.nan
        rows.append(dict(fam=fam, dataset=d[3:], peak_acc=pk, peak_logfd=pk_x,
                         collapse_onset_logfd=onset, r_below_knee=r_lo, r_above_knee=r_hi))
    T = pd.DataFrame([r for r in rows if r["fam"] == fam]).sort_values("collapse_onset_logfd")
    lines.append("```")
    lines.append(T.drop(columns="fam").round(3).to_string(index=False))
    lines.append("```")
    lines.append("")

T = pd.DataFrame(rows)
T.to_csv(os.path.join(HERE, "adaptation_side.csv"), index=False)

# concordance of collapse-onset ordering across the 3 CS families
piv = T.pivot_table(index="fam", columns="dataset", values="collapse_onset_logfd")
piv = piv.dropna(axis=1)
if piv.shape[1] >= 3:
    ranks = piv.rank(axis=1)
    m, k = ranks.shape
    W = 12 * ((ranks.sum(axis=0) - m * (k + 1) / 2) ** 2).sum() / (m ** 2 * (k ** 3 - k))
    lines.append(f"## Collapse-onset ordering concordance: Kendall W = {W:.3f} ({m} fams x {k} ds)")
    lines.append("mean rank (1 = collapses first): " +
                 ", ".join(f"{c}={ranks[c].mean():.2f}" for c in ranks.columns.sort_values()))

# fragile-vs-robust gap: piqa/hellaswag/winogrande (continuation/likelihood-style, high base)
# vs boolq/openbookqa/ARC (knowledge/QA)
lines.append("")
lines.append("## Above-knee slopes per dataset (pp/decade, cell level)")
for fam in CS_FAMS:
    sub = df[df.fam == fam]
    cells = sub.groupby("cell").agg({**{d: "mean" for d in DS}, "logfd": "mean"}).dropna()
    hi = cells[cells.logfd > KNEE_182[fam]]
    sl = {d[3:]: stats.linregress(hi.logfd, hi[d]).slope for d in DS}
    order = sorted(sl, key=sl.get)
    lines.append(f"- {fam}: " + ", ".join(f"{k}={sl[k]:+.1f}" for k in order))

open(os.path.join(HERE, "adaptation_side.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=False)
for ax, fam in zip(axes, CS_FAMS):
    sub = df[df.fam == fam]
    cells = sub.groupby("cell").agg({**{d: "mean" for d in DS}, "logfd": "mean"}).dropna().sort_values("logfd")
    for d in DS:
        roll = cells[d].rolling(9, center=True, min_periods=5).mean()
        ax.plot(cells.logfd, roll, label=d[3:], lw=1.3)
    ax.axvline(KNEE_182[fam], color="0.6", lw=1, ls="--")
    ax.set_title(fam); ax.set_xlabel(r"$\log_{10} F_\Delta$")
axes[0].set_ylabel("dataset accuracy")
axes[0].legend(fontsize=6)
fig.suptitle("Per-CS-dataset adaptation along the dose axis (dashed = retention knee)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_adaptation_side.png"), dpi=200)
fig.savefig(os.path.join(HERE, "fig_adaptation_side.pdf"))
print("figure saved")
