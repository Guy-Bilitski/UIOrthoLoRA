"""02_benchmark_fragility.py — which capability dies first along F_delta?

Per family x benchmark (BBH, MMLU-Pro, MMLU, ARC-c, TruthfulQA):
  - cell-level r and OLS slope vs log10 F_delta
  - normalized slope = pp/decade / family base ceiling (fraction of base per decade)
  - 2-segment (continuous hinge) fit -> per-benchmark knee
  - fragility ordering + Kendall's W concordance across families
  - TruthfulQA sign check (does forgetting RAISE TruthfulQA anywhere?)
Prior art: key_numbers Section 7 = slopes on Llama-CS n=49 single-seed only.
This is the full-n (1035), all-6-family, knee-and-ordering version.
ARC-c is contaminated in CS arms (trained-on); flagged in the table.
Outputs: benchmark_fragility.csv/.md, fig_benchmark_fragility.png/.pdf
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "pool.csv"))

BENCH = ["bbh", "mmlu_pro", "mmlu", "arc_c", "truthfulqa"]
BASE = {  # base ceilings from results/base_llama2_7b_noft & base_qwen25_7b_noft
    "llama": dict(bbh=32.96, mmlu_pro=18.82, mmlu=40.88, arc_c=44.80, truthfulqa=38.85),
    "qwen": dict(bbh=47.93, mmlu_pro=40.77, mmlu=71.80, arc_c=51.28, truthfulqa=56.28),
}
MODEL = dict(lrsw="llama", lrswm="llama", frc="llama", frm="llama", qwsw="qwen", qwswm="qwen")
CS_FAMS = {"lrsw", "qwsw", "frc"}  # ARC-c contaminated (trained on)


def hinge_fit(x, y):
    """Continuous 2-segment fit; returns (knee, slope_lo, slope_hi, r2, r2_linear)."""
    x, y = np.asarray(x), np.asarray(y)
    best = None
    for knee in np.quantile(x, np.linspace(0.1, 0.9, 33)):
        X = np.column_stack([np.ones_like(x), x, np.maximum(0, x - knee)])
        beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        sse = ((y - pred) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, knee, beta)
    sse, knee, beta = best
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sse / sst
    sl, ic, rr, *_ = stats.linregress(x, y)
    return knee, beta[1], beta[1] + beta[2], r2, rr ** 2


rows = []
for fam, sub in df.groupby("fam"):
    cells = sub.groupby("cell").agg({**{b: "mean" for b in BENCH}, "logfd": "mean"}).dropna()
    model = MODEL[fam]
    for b in BENCH:
        x, y = cells.logfd.values, cells[b].values
        sl, ic, r, p, se = stats.linregress(x, y)
        knee, slo, shi, r2h, r2l = hinge_fit(x, y)
        base = BASE[model][b]
        rows.append(dict(fam=fam, bench=b, n_cells=len(cells), r=r, slope=sl,
                         norm_slope=sl / base, knee=knee, slope_below=slo,
                         slope_above=shi, norm_slope_above=shi / base,
                         r2_hinge=r2h, r2_linear=r2l, base=base,
                         contaminated=(b == "arc_c" and fam in CS_FAMS)))

T = pd.DataFrame(rows)
T.to_csv(os.path.join(HERE, "benchmark_fragility.csv"), index=False)

lines = ["# Per-benchmark fragility along F_delta (cell level, all 6 families)", ""]
lines.append("norm_slope = pp/decade / base ceiling (fraction of base capability lost per decade of F_delta)")
lines.append("")
for fam in ["lrsw", "frc", "qwsw", "lrswm", "frm", "qwswm"]:
    sub = T[T.fam == fam].sort_values("norm_slope")
    lines.append(f"## {fam} (n_cells={sub.n_cells.iloc[0]})")
    lines.append("```")
    lines.append(sub[["bench", "r", "slope", "norm_slope", "knee", "slope_below",
                      "slope_above", "r2_hinge", "r2_linear", "contaminated"]]
                 .round(3).to_string(index=False))
    lines.append("```")
    lines.append("")

# fragility ordering concordance (exclude contaminated arc_c cells? do both)
lines.append("## Fragility ordering concordance across families")
for label, tt in [("all benchmarks", T), ("excl. contaminated ARC-c", T[~T.contaminated])]:
    piv = tt.pivot_table(index="fam", columns="bench", values="norm_slope")
    piv = piv.dropna(axis=1)
    ranks = piv.rank(axis=1)  # 1 = most negative = most fragile
    m, k = ranks.shape
    W = 12 * ((ranks.sum(0) - m * (k + 1) / 2) ** 2).sum() / (m ** 2 * (k ** 3 - k))
    chi2 = m * (k - 1) * W
    pval = stats.chi2.sf(chi2, k - 1)
    lines.append(f"- {label}: Kendall W = {W:.3f} (chi2={chi2:.1f}, p={pval:.1e}, {m} families x {k} benchmarks)")
    lines.append(f"  mean rank (1=most fragile): " +
                 ", ".join(f"{b}={ranks[b].mean():.2f}" for b in ranks.columns))

# TruthfulQA sign check
lines.append("")
lines.append("## TruthfulQA sign check (does damage RAISE TruthfulQA?)")
for fam, sub in df.groupby("fam"):
    cells = sub.groupby("cell").agg(tq=("truthfulqa", "mean"), logfd=("logfd", "mean")).dropna()
    sl, ic, r, p, se = stats.linregress(cells.logfd, cells.tq)
    hi = cells[cells.logfd > cells.logfd.median()]
    base = BASE[MODEL[fam]]["truthfulqa"]
    lines.append(f"- {fam}: r={r:+.3f} (p={p:.1e}), slope={sl:+.2f} pp/dec, base={base}, "
                 f"mean tq top-half-F_delta = {hi.tq.mean():.1f}")

open(os.path.join(HERE, "benchmark_fragility.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# figure: normalized retention (fraction of base) vs logfd per benchmark, per family grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fams = ["lrsw", "frc", "qwsw", "lrswm", "frm", "qwswm"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
colors = dict(bbh="C0", mmlu_pro="C1", mmlu="C2", arc_c="C4", truthfulqa="C3")
for ax, fam in zip(axes.ravel(), fams):
    sub = df[df.fam == fam]
    cells = sub.groupby("cell").agg({**{b: "mean" for b in BENCH}, "logfd": "mean"}).dropna()
    order = cells.sort_values("logfd")
    for b in BENCH:
        base = BASE[MODEL[fam]][b]
        # rolling mean over 9 cells for readability
        yy = (order[b] / base).rolling(9, center=True, min_periods=4).mean()
        ls = "--" if (b == "arc_c" and fam in CS_FAMS) else "-"
        ax.plot(order.logfd, yy, ls, color=colors[b], label=b, lw=1.5)
    ax.axhline(1.0, color="0.8", lw=0.8)
    ax.set_title(fam)
    ax.set_xlabel(r"$\log_{10} F_\Delta$")
axes[0, 0].set_ylabel("fraction of base ceiling")
axes[1, 0].set_ylabel("fraction of base ceiling")
axes[0, 0].legend(fontsize=7)
fig.suptitle("Per-benchmark degradation (rolling mean over cells; dashed = trained-on/contaminated)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_benchmark_fragility.png"), dpi=200)
fig.savefig(os.path.join(HERE, "fig_benchmark_fragility.pdf"))
print("figure saved")
