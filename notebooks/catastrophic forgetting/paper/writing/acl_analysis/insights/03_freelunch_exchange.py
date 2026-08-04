"""03_freelunch_exchange.py — the free-lunch region and the forgetting price of adaptation.

(a) FREE-LUNCH REGION: per family, hinge-fit retention vs log10 F_delta (cells) ->
    knee. Quantify how much adaptation is reachable at F_delta BELOW the knee
    (where measured retention is ~flat) vs the global healthy max: the fraction of
    peak adaptation that is available "for free". Per method too.
(b) EXCHANGE RATE: bin cells by F_delta; between consecutive bins compute
    delta(adapt)/|delta(retention)| -> "adaptation points bought per retention point
    paid" along the curve. Practitioner-facing marginal-price table.
Healthy filter: adapt >= 25 (drop format-collapse, convention of dyn3 sec 11).
Outputs: freelunch_table.csv/.md, exchange_rate.csv, fig_freelunch.png/.pdf
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "pool.csv"))

FAMS = ["lrsw", "frc", "qwsw", "lrswm", "frm", "qwswm"]
# CANONICAL knees from key_numbers.md section 18.2 (frozen). Own hinge fit kept as sensitivity.
KNEE_182 = dict(lrsw=-0.02, lrswm=-0.48, qwsw=-0.69, qwswm=-0.91, frc=-0.45, frm=-0.50)
BASE_RET = dict(llama=25.89, qwen=44.35)
MODEL = dict(lrsw="llama", lrswm="llama", frc="llama", frm="llama", qwsw="qwen", qwswm="qwen")


def hinge_fit(x, y):
    x, y = np.asarray(x), np.asarray(y)
    best = None
    for knee in np.quantile(x, np.linspace(0.1, 0.9, 41)):
        X = np.column_stack([np.ones_like(x), x, np.maximum(0, x - knee)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((y - X @ beta) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, knee, beta)
    return best[1], best[2]


lines = ["# Free-lunch region + exchange rate", ""]
fl_rows, ex_rows = [], []

for fam in FAMS:
    sub = df[(df.fam == fam)].copy()
    cells = sub.groupby("cell").agg(logfd=("logfd", "mean"), ret=("ret", "mean"),
                                    adapt=("adapt", "mean"), method=("method", "first")).dropna()
    healthy = cells[cells.adapt >= 25]
    knee_own, beta = hinge_fit(cells.logfd.values, cells.ret.values)
    knee = KNEE_182[fam]  # canonical frozen knee
    below = healthy[healthy.logfd <= knee]
    above = healthy[healthy.logfd > knee]
    peak = healthy.adapt.max()
    peak_below = below.adapt.max() if len(below) else np.nan
    ret_at_peak = healthy.loc[healthy.adapt.idxmax(), "ret"]
    ret_at_peak_below = below.loc[below.adapt.idxmax(), "ret"] if len(below) else np.nan
    base = BASE_RET[MODEL[fam]]
    fl_rows.append(dict(fam=fam, knee_logfd=knee, knee_fd=10 ** knee, knee_own_fit=knee_own,
                        n_below=len(below), n_above=len(above),
                        peak_adapt=peak, peak_adapt_below_knee=peak_below,
                        frac_free=peak_below / peak if peak else np.nan,
                        ret_at_global_peak=ret_at_peak, ret_at_free_peak=ret_at_peak_below,
                        base_ret=base,
                        ret_cost_global=base - ret_at_peak, ret_cost_free=base - ret_at_peak_below))
    lines.append(f"## {fam}: canonical knee (sec 18.2) F_delta={10**knee:.3f} (log {knee:+.2f}; own hinge refit {knee_own:+.2f})")
    lines.append(f"- peak adapt overall {peak:.1f} (ret {ret_at_peak:.1f}); peak adapt below knee "
                 f"{peak_below:.1f} (ret {ret_at_peak_below:.1f}) -> {100*peak_below/peak:.1f}% of peak is 'free' "
                 f"(n_below={len(below)} healthy cells)")
    # per-method free fraction
    permeth = []
    for meth, mm in healthy.groupby("method"):
        mb = mm[mm.logfd <= knee]
        if len(mm) < 3:
            continue
        permeth.append((meth, mm.adapt.max(), mb.adapt.max() if len(mb) else np.nan, len(mb), len(mm)))
    if permeth:
        lines.append("  per-method (peak, peak-below-knee, n_below/n): " +
                     "; ".join(f"{m}: {p:.1f}/{q:.1f} ({nb}/{nn})" for m, p, q, nb, nn in permeth))
    lines.append("")

    # exchange-rate bins (quintiles of logfd among healthy cells)
    try:
        healthy = healthy.copy()
        healthy["bin"] = pd.qcut(healthy.logfd, 5, labels=False, duplicates="drop")
        binstat = healthy.groupby("bin").agg(fd_med=("logfd", "median"), adapt=("adapt", "mean"),
                                             ret=("ret", "mean"), n=("ret", "size"))
        prev = None
        for b, rr in binstat.iterrows():
            price = np.nan
            if prev is not None:
                da, dr = rr.adapt - prev.adapt, rr.ret - prev.ret
                price = (-dr) / da if abs(da) > 1e-9 else np.inf
            ex_rows.append(dict(fam=fam, bin=b, fd=10 ** rr.fd_med, adapt=rr.adapt,
                                ret=rr.ret, n=rr.n,
                                d_adapt=np.nan if prev is None else rr.adapt - prev.adapt,
                                d_ret=np.nan if prev is None else rr.ret - prev.ret,
                                ret_paid_per_adapt_point=price))
            prev = rr
    except ValueError:
        pass

FL = pd.DataFrame(fl_rows)
EX = pd.DataFrame(ex_rows)
FL.to_csv(os.path.join(HERE, "freelunch_table.csv"), index=False)
EX.to_csv(os.path.join(HERE, "exchange_rate.csv"), index=False)

lines.append("## Exchange rate (marginal retention paid per adaptation point gained, quintile bins of F_delta)")
for fam in FAMS:
    sub = EX[EX.fam == fam]
    lines.append(f"### {fam}")
    lines.append("```")
    lines.append(sub[["fd", "n", "adapt", "ret", "d_adapt", "d_ret", "ret_paid_per_adapt_point"]]
                 .round(2).to_string(index=False))
    lines.append("```")

lines.append("")
lines.append("## Free-lunch summary")
lines.append("```")
lines.append(FL.round(3).to_string(index=False))
lines.append("```")
open(os.path.join(HERE, "freelunch_exchange.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# figure: adapt & ret vs logfd with knee shading, per family
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, fam in zip(axes.ravel(), FAMS):
    sub = df[df.fam == fam]
    cells = sub.groupby("cell").agg(logfd=("logfd", "mean"), ret=("ret", "mean"),
                                    adapt=("adapt", "mean")).dropna()
    healthy = cells[cells.adapt >= 25]
    knee = KNEE_182[fam]
    ax.scatter(healthy.logfd, healthy.ret, s=10, color="C3", label="retention")
    ax2 = ax.twinx()
    ax2.scatter(healthy.logfd, healthy.adapt, s=10, color="C0", alpha=0.6, label="adaptation")
    ax.axvspan(healthy.logfd.min() - 0.05, knee, color="C2", alpha=0.12)
    ax.axvline(knee, color="C2", lw=1)
    ax.set_title(f"{fam} (knee F={10**knee:.2f})")
    ax.set_xlabel(r"$\log_{10} F_\Delta$")
    ax.set_ylabel("retention", color="C3")
    ax2.set_ylabel("adaptation", color="C0")
fig.suptitle("Green = free-lunch region (below retention knee)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_freelunch.png"), dpi=200)
fig.savefig(os.path.join(HERE, "fig_freelunch.pdf"))
print("figure saved")
