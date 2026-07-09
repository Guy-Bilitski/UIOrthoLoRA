#!/usr/bin/env python
"""FIGURE 5 - THE LAW IS ALREADY IN THE 2025 LITERATURE (cross-paper overlay).

Panel A: CLoRA (Kim et al. 2025) Table 4 -- effective update magnitude F_Delta
         vs BBH retention across THEIR OWN methods (LoRA / LoRA-r8/r16 /
         LoRA-L2 / MiLoRA / CLoRA-k128..k2048).  One line: r=-0.98, slope
         ~-14.7 pp/decade -- indistinguishable from our -14.8 (two datasets).
Panel B: MiLoRA (Wang et al. 2024) Table 7 (||dW||) vs Table 8 (CE-to-base):
         LoRA->MiLoRA follows the magnitude trend; PiSSA (principal-space init)
         forgets more than its magnitude predicts -- the same outlier our
         geometry fingerprints flag.
Plus a cited note from LoRA-Null (Table 4b): capacity costs retention.

All numbers here are PUBLISHED anchors, each cited in-code and on the figure.
This is the only figure that hardcodes numbers (published-paper tables).
Supports paper section: Related work / external replication.
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import figstyle as fs
fs.apply_rc()

# ---- [EXTERNAL] CLoRA Table 4  (F_Delta, BBH); 'reference' row excluded -------
# source: CLoRA (Kim et al., 2025), Table 4.  (F_Delta = their forgetting metric,
# lower=better; F = BBH retention.)  Reference/full-FT row (2.42/34.91) is off the
# adapter trend and excluded from the fit.
CLORA_BASE_BBH = 34.91
CLORA_T4 = [
    ("LoRA",        0.79, 26.69, "LoRA"),
    ("LoRA-r8",     0.95, 26.90, "LoRA"),
    ("LoRA-r16",    1.03, 26.73, "LoRA"),
    ("LoRA-L2",     0.29, 32.93, "LoRA+wd"),
    ("MiLoRA",      0.92, 25.14, "MiLoRA"),
    ("CLoRA k128",  0.36, 30.82, "CLoRA"),
    ("CLoRA k256",  0.34, 31.92, "CLoRA"),
    ("CLoRA k512",  0.27, 34.32, "CLoRA"),
    ("CLoRA k1024", 0.21, 36.49, "CLoRA"),
    ("CLoRA k2048", 0.14, 38.67, "CLoRA"),
]
# ---- [EXTERNAL] MiLoRA Table 7 (||dW||) & Table 8 (CE-to-base) ----------------
# source: MiLoRA (Wang et al., 2024), Tables 7 & 8 (CE-to-base = Kalajdzievski 2024).
MILORA_T78 = [("LoRA", 68.2, 3.24), ("PiSSA", 55.8, 6.07), ("MiLoRA", 44.9, 2.54)]

OUR_SLOPE = -14.8   # our CS pooled magnitude-law slope (key_numbers.md sec.1)

# --------------------------------------------------------------- figure --------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 5.4),
                               gridspec_kw=dict(width_ratios=[1.7, 1.0],
                                                wspace=0.26))

# ===== Panel A: CLoRA Table 4 =====
F = np.array([r[1] for r in CLORA_T4])
B = np.array([r[2] for r in CLORA_T4])
r_p = stats.pearsonr(np.log10(F), B)[0]
sl, ic, _, pval, _ = stats.linregress(np.log10(F), B)
xs = np.logspace(np.log10(F.min()) - 0.06, np.log10(F.max()) + 0.06, 100)
axA.plot(xs, ic + sl * np.log10(xs), color=fs.FIT_C, lw=2.2, zorder=2)
LEG = {"LoRA+wd": "LoRA-L2 (=LoRA+wd)"}   # bridge their naming to ours
seen = set()
for name, f, b, fam in CLORA_T4:
    lab = LEG.get(fam, fam) if fam not in seen else None
    seen.add(fam)
    axA.scatter(f, b, s=110, marker=fs.marker(fam), facecolor=fs.color(fam),
                edgecolor="white", linewidth=1.0, label=lab, zorder=3)
# label the CLoRA k-series + the LoRA-L2 anchor
for name, f, b, fam in CLORA_T4:
    if name.startswith("CLoRA k") or name == "LoRA-L2":
        axA.annotate(name.replace("CLoRA ", ""), (f, b),
                     textcoords="offset points", xytext=(-4, 7), fontsize=7.8,
                     color=fs.INK2, ha="right")
axA.axhline(CLORA_BASE_BBH, color=fs.CEILING_C, lw=1.6, ls=(0, (6, 3)), zorder=1)
axA.text(F.max(), CLORA_BASE_BBH + 0.25, "their base BBH 34.91",
         color=fs.CEILING_C, ha="right", va="bottom", fontsize=9, fontweight="bold")
axA.set_xscale("log")
axA.set_xlabel(r"$F_\Delta$  (CLoRA forgetting metric, log scale)")
axA.set_ylabel("BBH retention  (published)")
axA.set_title("A   CLoRA's own Table 4 is the magnitude law", loc="left")
axA.grid(True, which="both", alpha=0.5)
axA.legend(loc="lower left", fontsize=9)
axA.text(0.97, 0.95,
         f"$r$ = {r_p:.2f}   slope {sl:.1f} pp/decade\n"
         f"(our sweep: {OUR_SLOPE:.1f} pp/decade)",
         transform=axA.transAxes, ha="right", va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f4", ec=fs.GRID))

# ===== Panel B: MiLoRA Table 7 vs 8 =====
nx = np.array([r[1] for r in MILORA_T78])
ce = np.array([r[2] for r in MILORA_T78])
# magnitude trend through the two non-principal-init methods (LoRA, MiLoRA)
on = [i for i, r in enumerate(MILORA_T78) if r[0] in ("LoRA", "MiLoRA")]
sl2, ic2 = np.polyfit(nx[on], ce[on], 1)
xr = np.array([nx.min() - 3, nx.max() + 3])
axB.plot(xr, ic2 + sl2 * xr, color=fs.MUTED, lw=1.8, ls="--", zorder=2)
for name, x, y in MILORA_T78:
    axB.scatter(x, y, s=150, marker=fs.marker(name),
                facecolor=fs.color(name), edgecolor="white", linewidth=1.0,
                zorder=3)
    axB.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6),
                 fontsize=9.5, color=fs.INK)
axB.annotate("principal-space init:\nforgets above its magnitude\n(our geometry outlier)",
             xy=(55.8, 6.07), xytext=(46, 4.6), fontsize=8.4, color=fs.INK,
             arrowprops=dict(arrowstyle="->", color=fs.MUTED, lw=1.0))
axB.set_xlabel(r"$\|\Delta W\|$  (MiLoRA Table 7)")
axB.set_ylabel("CE to base  (MiLoRA Table 8)")
axB.set_title("B   MiLoRA's Tables 7 & 8 agree", loc="left")
axB.grid(True, alpha=0.5)

# LoRA-Null cited note (qualitative; no scatter-able points published)
fig.text(0.5, -0.02,
         "LoRA-Null (Table 4b): CorDA retention falls 89% -> 73% as rank 4 -> 256"
         " — more capacity (larger updates) costs retention, same direction.",
         ha="center", va="top", fontsize=8.6, color=fs.INK2, style="italic")

pdf, png = fs.save(fig, "fig_cross_literature")
plt.close(fig)

# --------------------------------------------------------------- report --------
print("FIGURE 5  cross-literature")
print(f"  A  CLoRA Table 4 (10 adapter rows): r(log F_Delta, BBH) = {r_p:.3f},"
      f" slope = {sl:.2f} pp/decade (p={pval:.1e}); our sweep slope {OUR_SLOPE}")
print(f"  B  MiLoRA Tab7 ||dW|| vs Tab8 CE: LoRA(68.2,3.24) MiLoRA(44.9,2.54)"
      f" on trend; PiSSA(55.8,6.07) above (principal-init outlier)")
print(f"  wrote {pdf}\n        {png}")
