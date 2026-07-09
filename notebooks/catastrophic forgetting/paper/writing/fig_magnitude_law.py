#!/usr/bin/env python
"""FIGURE 1 - THE MAGNITUDE LAW (headline).

Left  : retention (core = mean BBH-AO + MMLU-Pro, base ceiling 26.0) vs log10 F_Delta
        for the Commonsense LR sweep (lrsw_, Llama-2-7B, s42, 7 adapters x 7 LRs),
        coloured by adapter, with a SATURATING (hockey-stick) fit, the base-ceiling
        line and the annotated knee.
Right : the same retention vs LEARNING RATE (a weaker proxy, R^2~0.32) to contrast.

Reads data live from campaign_summary_clean.jsonl (== results/campaign_summary).
Prints the key stats it drew.  Supports paper section: The Magnitude Law / Fairness.
"""
import json
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import figstyle as fs
fs.apply_rc()

# --------------------------------------------------------------- load data -----
recs = {}
for line in open(fs.CAMPAIGN):
    d = json.loads(line)
    recs[d["run_name"]] = d

rows = []
for rn, d in recs.items():
    if not rn.startswith("lrsw_"):
        continue
    meth = fs.method_from_run(rn)
    if meth is None or meth in ("CorDA", "PiSSA"):   # CorDA excluded per key_numbers section 8
        continue
    F, ret = d.get("fdelta"), d.get("retention_mean")
    lr = fs.lr_from_run(rn)
    if F is None or ret is None or F <= 0 or lr is None:
        continue
    rows.append(dict(m=meth, F=F, ret=ret, lr=lr))

F   = np.array([r["F"] for r in rows])
ret = np.array([r["ret"] for r in rows])
lr  = np.array([r["lr"] for r in rows])
lx  = np.log10(F)
n   = len(rows)
CEIL = 26.0

# --------------------------------------------------------------- statistics ----
pear = stats.pearsonr(lx, ret)[0]
spear = stats.spearmanr(F, ret)[0]
lin_slope, lin_ic, lin_r, lin_p, _ = stats.linregress(lx, ret)


def hockey(lxv, a, b, lk, s):
    """Saturating hockey-stick: plateau ~a near the ceiling for small F_Delta,
    then a straight decline of slope -b in log space beyond the knee lk."""
    return a - b * s * np.log1p(np.exp((lxv - lk) / s))


popt, _ = curve_fit(hockey, lx, ret, p0=[26.0, 15.0, -0.4, 0.3], maxfev=40000)
a_fit, b_fit, lk_fit, s_fit = popt
knee_F = 10 ** lk_fit
hs_pred = hockey(lx, *popt)
hs_r2 = 1 - np.sum((ret - hs_pred) ** 2) / np.sum((ret - ret.mean()) ** 2)

lr_slope, lr_ic, lr_r, lr_p, _ = stats.linregress(np.log10(lr), ret)
lr_r2 = lr_r ** 2

# ------------------------------------------------------------------ figure ------
fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(12.4, 5.2),
    gridspec_kw=dict(width_ratios=[2.35, 1.0], wspace=0.28))

# ----- LEFT: magnitude law -----
methods = [m for m in fs.PALETTE if m in {r["m"] for r in rows}]
for m in methods:
    mask = np.array([r["m"] == m for r in rows])
    axL.scatter(F[mask], ret[mask], s=64, marker=fs.marker(m),
                facecolor=fs.color(m), edgecolor="white", linewidth=0.8,
                alpha=0.95, label=m, zorder=3)

xs = np.logspace(np.log10(F.min()) - 0.02, np.log10(F.max()) + 0.02, 300)
axL.plot(xs, hockey(np.log10(xs), *popt), color=fs.FIT_C, lw=2.4, zorder=4,
         label=f"saturating fit ($R^2$={hs_r2:.2f})")
axL.axhline(CEIL, color=fs.CEILING_C, lw=1.8, ls=(0, (6, 3)), zorder=2)
axL.text(F.max(), CEIL + 0.35, "base ceiling  26.0", color=fs.CEILING_C,
         ha="right", va="bottom", fontsize=10, fontweight="bold")

# knee annotation
axL.axvline(knee_F, color=fs.MUTED, lw=1.2, ls=":", zorder=1)
axL.annotate(f"knee  $F_\\Delta\\approx{knee_F:.2f}$",
             xy=(knee_F, hockey(np.log10(knee_F), *popt)),
             xytext=(knee_F * 1.7, 14.5), color=fs.INK, fontsize=10.5,
             arrowprops=dict(arrowstyle="->", color=fs.MUTED, lw=1.1))

axL.set_xscale("log")
axL.set_xlabel(r"effective update magnitude  $F_\Delta$  (CLoRA Eq. 3, log scale)")
axL.set_ylabel("retention  (BBH-AO + MMLU-Pro,  base = 26.0)")
axL.set_title("Forgetting is governed by update magnitude")
axL.grid(True, which="both", alpha=0.55)
axL.text(0.03, 0.05,
         f"Spearman $\\rho$ = {spear:.2f}\nPearson $r$ = {pear:.2f}"
         f"\nslope {lin_slope:.1f} pp/decade\n$n$ = {n}",
         transform=axL.transAxes, va="bottom", ha="left", fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f4", ec=fs.GRID))
handles, labels = axL.get_legend_handles_labels()

# ----- RIGHT: LR is a weaker proxy -----
for m in methods:
    mask = np.array([r["m"] == m for r in rows])
    axR.scatter(lr[mask], ret[mask], s=48, marker=fs.marker(m),
                facecolor=fs.color(m), edgecolor="white", linewidth=0.7,
                alpha=0.95, zorder=3)
lxr = np.log10(np.array(sorted(set(lr))))
axR.plot(10 ** lxr, lr_ic + lr_slope * lxr, color=fs.MUTED, lw=2.0, ls="--",
         zorder=2)
axR.axhline(CEIL, color=fs.CEILING_C, lw=1.4, ls=(0, (6, 3)), zorder=1)
axR.set_xscale("log")
axR.set_xlabel("learning rate  (log scale)")
axR.set_ylabel("retention")
axR.set_title("Learning rate: a weaker proxy")
axR.set_ylim(axL.get_ylim())
axR.grid(True, which="both", alpha=0.55)
axR.text(0.05, 0.05, f"$R^2$ = {lr_r2:.2f}\n(vs {hs_r2:.2f} for $F_\\Delta$)",
         transform=axR.transAxes, va="bottom", ha="left", fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f4", ec=fs.GRID))

fig.legend(handles, labels, loc="lower center", ncol=len(handles),
           bbox_to_anchor=(0.5, -0.03), handletextpad=0.3, columnspacing=1.2,
           frameon=False)
fig.subplots_adjust(bottom=0.16)

pdf, png = fs.save(fig, "fig_magnitude_law")
plt.close(fig)

# --------------------------------------------------------------- report --------
print("FIGURE 1  magnitude law")
print(f"  n = {n} (7 adapters x 7 LRs, CorDA excluded)")
print(f"  Pearson r(ret, log F_Delta) = {pear:.3f}")
print(f"  Spearman rho                = {spear:.3f}")
print(f"  linear slope                = {lin_slope:.2f} pp/decade  (R2={lin_r**2:.3f})")
print(f"  saturating fit: plateau a={a_fit:.2f}  post-knee slope={b_fit:.1f} pp/dec"
      f"  knee F_Delta={knee_F:.3f}  R2={hs_r2:.3f}")
print(f"  LR panel: R2(ret vs log LR) = {lr_r2:.3f}  (r={lr_r:.3f})")
print(f"  wrote {pdf}\n        {png}")
