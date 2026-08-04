"""06_metric_hygiene_effects.py — three quantifications supporting findings.md.

(1) TruthfulQA inside retention_broad: on Llama TQ RISES with F_delta, so
    retention_broad (mean of 5 incl. TQ) understates forgetting. Quantify slope/r
    of broad vs broad-without-TQ per family. Plus the "regression to indifference"
    test: TQ score in the top-F_delta decile per family (does damage push TQ toward
    a common band from below on Llama / from above on Qwen?).
(2) share_q effect size in pp: OLS ret ~ log F_delta + family + share_q (runs w/
    permatrix features): pp per IQR of share_q.
(3) wd mediation delta-R2 at cell level: R2(ret ~ logF) vs +wd on the lorawd cells.
Outputs: metric_hygiene.md
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "pool.csv"))
M = pd.read_csv(os.path.join(HERE, "permatrix_features.csv"))

lines = ["# Metric hygiene + effect sizes", ""]

# ---- (1) TQ in retention_broad ----
lines.append("## 1. TruthfulQA inside retention_broad")
BENCH4 = ["bbh", "mmlu_pro", "mmlu", "arc_c"]
for fam, sub in df.groupby("fam"):
    cells = sub.groupby("cell").agg(logfd=("logfd", "mean"), rb=("ret_broad", "mean"),
                                    **{b: (b, "mean") for b in BENCH4},
                                    tq=("truthfulqa", "mean")).dropna()
    rb4 = cells[BENCH4].mean(axis=1)
    s5, _, r5, _, _ = stats.linregress(cells.logfd, cells.rb)
    s4, _, r4, _, _ = stats.linregress(cells.logfd, rb4)
    top = cells[cells.logfd >= cells.logfd.quantile(0.9)]
    lines.append(f"- {fam}: broad-with-TQ slope {s5:+.2f} (r={r5:+.3f}) vs broad-no-TQ {s4:+.2f} "
                 f"(r={r4:+.3f}); TQ understates slope by {100*(1 - s5/s4):.0f}%. "
                 f"Top-decile-F_delta TQ = {top.tq.mean():.1f} (n={len(top)})")
lines.append("Base TQ: Llama 38.85, Qwen 56.28. If damage pushes TQ toward an indifference band,")
lines.append("Llama rises toward it, Qwen falls toward it.")

# extreme-damage TQ convergence, run level, most damaged runs (ret < 40% of base ret)
lines.append("")
lines.append("Most-damaged runs (retention < 40% of family base): mean TQ")
BASE_RET = dict(lrsw=25.89, lrswm=25.89, frc=25.89, frm=25.89, qwsw=44.35, qwswm=44.35)
for fam, sub in df.groupby("fam"):
    dmg = sub[sub.ret < 0.4 * BASE_RET[fam]]
    if len(dmg) >= 5:
        lines.append(f"- {fam}: TQ = {dmg.truthfulqa.mean():.1f} +- {dmg.truthfulqa.std():.1f} (n={len(dmg)})")

# ---- (2) share_q effect size ----
lines.append("")
lines.append("## 2. share_q effect size (pp of retention)")
ok = M.share_q.notna()
sub = M[ok]
famd = pd.get_dummies(sub.fam, drop_first=True).astype(float)
X = np.column_stack([np.ones(len(sub)), sub.logfd, famd.values, sub.share_q])
beta, *_ = np.linalg.lstsq(X, sub.ret.values, rcond=None)
iqr = sub.share_q.quantile(0.75) - sub.share_q.quantile(0.25)
lines.append(f"- OLS ret ~ logF + fam + share_q (n={len(sub)} runs): beta(share_q) = {beta[-1]:+.2f} pp per unit share")
lines.append(f"- share_q IQR = {iqr:.3f} -> effect across IQR = {beta[-1]*iqr:+.2f} pp "
             f"(vs magnitude beta {beta[1]:+.2f} pp/decade)")
# and within lorawd only (method-free check)
lw = sub[sub.method == "lorawd"]
famd2 = pd.get_dummies(lw.fam, drop_first=True).astype(float)
X2 = np.column_stack([np.ones(len(lw)), lw.logfd, famd2.values, lw.share_q])
b2, *_ = np.linalg.lstsq(X2, lw.ret.values, rcond=None)
r_lw = np.corrcoef(lw.ret - X2[:, :-1] @ np.linalg.lstsq(X2[:, :-1], lw.ret.values, rcond=None)[0],
                   lw.share_q - X2[:, :-1] @ np.linalg.lstsq(X2[:, :-1], lw.share_q.values, rcond=None)[0])[0, 1]
lines.append(f"- lorawd-only (n={len(lw)}): beta(share_q) = {b2[-1]:+.2f}, partial r = {r_lw:+.3f}")

# ---- (3) wd mediation delta-R2 ----
lines.append("")
lines.append("## 3. wd adds nothing beyond the F_delta it produces (cell level)")
frc = df[(df.fam == "frc") & (df.method == "lorawd") & df.wd.notna()]
wdc = frc.groupby("cell").agg(wd=("wd", "mean"), logfd=("logfd", "mean"), ret=("ret", "mean")).dropna()
X1 = np.column_stack([np.ones(len(wdc)), wdc.logfd])
X2 = np.column_stack([X1, wdc.wd])
for name, X in [("ret ~ logF", X1), ("ret ~ logF + wd", X2)]:
    b, *_ = np.linalg.lstsq(X, wdc.ret.values, rcond=None)
    r2 = 1 - ((wdc.ret.values - X @ b) ** 2).sum() / ((wdc.ret - wdc.ret.mean()) ** 2).sum()
    lines.append(f"- {name}: R2 = {r2:.3f}")

open(os.path.join(HERE, "metric_hygiene.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
