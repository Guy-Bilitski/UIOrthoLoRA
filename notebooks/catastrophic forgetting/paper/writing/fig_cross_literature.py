#!/usr/bin/env python
"""CROSS-LITERATURE OVERLAY - the magnitude law in two independent datasets.

BBH-to-BBH overlay (both axes are BBH retention, so the two datasets are directly
comparable; we do NOT mix our BBH+MMLU-Pro core against their BBH-only):

  * OUR cloud   : the Llama-2 commonsense LR sweep (lrsw_*, CorDA excluded),
                  per-run (F_Delta, BBH answer-only), n=49.  Own fit line.
  * CLoRA cloud : CLoRA (Kim et al., 2025) Table 4 published rows,
                  (F_Delta, BBH).  Own fit line.

Two point clouds, each with its OWN fitted line => PARALLEL slopes, NOT one shared
line.  Guards baked in per the PI-critic:
  - lead the annotation with r and direction; CLoRA robust in every subset (r<=-0.95).
  - report slope as a RANGE (CLoRA pooled -14.7 blends a -12.7 baseline family and a
    -18.9 k-series; ours -14.3 BBH<->BBH).
  - CLoRA's LoRA-baseline F_Delta (0.79) ~ ours; only the constrained k-series runs
    lower  (do NOT claim "~2x lower F_Delta").
  - annotate that CLoRA k1024/k2048 EXCEED their own base BBH (34.91).

Our numbers are read live from campaign_summary_clean.jsonl; the CLoRA rows are
published anchors (cited in-code).  F_Delta = CLoRA Eq. 3 (mean ||dW x||/||x||).
Supports paper section: The Magnitude Law / external replication (Fig. crosslit).
"""
import re
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import figstyle as fs
fs.apply_rc()

# ---- OUR data: Llama-2 CS LR sweep, per-run (F_Delta, BBH), CorDA excluded -----
OUR_BASE_BBH = 33.10   # answer-only base ceiling (key_numbers.md sec.0) [EXTERNAL]
our = []
for line in open(fs.CAMPAIGN):
    if not line.strip():
        continue
    r = json.loads(line)
    rn = r.get("run_name", "")
    if not rn.startswith("lrsw_"):          # Llama-2 commonsense LR sweep only
        continue
    if fs.method_from_run(rn) == "CorDA":   # CorDA excluded from every law claim
        continue
    f, b = r.get("fdelta"), r.get("bbh")
    if not f or f <= 0 or b is None or b <= 0:
        continue
    our.append((f, b))
OF = np.array([x[0] for x in our])
OB = np.array([x[1] for x in our])
o_lx = np.log10(OF)
o_r = stats.pearsonr(o_lx, OB)[0]
o_sl, o_ic, *_ = stats.linregress(o_lx, OB)

# ---- [EXTERNAL] CLoRA Table 4  (F_Delta, BBH) ---------------------------------
# source: CLoRA (Kim et al., 2025), Table 4.  Reference/full-FT row (2.42/34.91) is
# the base, drawn as a line, not a fit point.
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
CF = np.array([r[1] for r in CLORA_T4])
CB = np.array([r[2] for r in CLORA_T4])
c_lx = np.log10(CF)
c_r = stats.pearsonr(c_lx, CB)[0]
c_sl, c_ic, *_ = stats.linregress(c_lx, CB)
# subset slopes for the range claim
base_i = [i for i, r in enumerate(CLORA_T4) if r[3] != "CLoRA"]
k_i = [i for i, r in enumerate(CLORA_T4) if r[0].startswith("CLoRA k")]
base_sl = stats.linregress(c_lx[base_i], CB[base_i])[0]
k_sl = stats.linregress(c_lx[k_i], CB[k_i])[0]
# robustness: worst drop-one |r|
drop1 = max(stats.pearsonr(np.delete(c_lx, j), np.delete(CB, j))[0]
            for j in range(len(CLORA_T4)))

# --------------------------------------------------------------- figure --------
fig, ax = plt.subplots(figsize=(8.6, 6.2))
OUR_C = "#2a78d6"
CLO_C = "#4a3aa7"

# our cloud + fit
xs_o = np.logspace(o_lx.min() - 0.03, o_lx.max() + 0.03, 100)
ax.scatter(OF, OB, s=34, marker="o", facecolor=OUR_C, edgecolor="white",
           linewidth=0.5, alpha=0.55, zorder=3,
           label=f"ours: Llama-2 CS sweep (n={len(our)})")
ax.plot(xs_o, o_ic + o_sl * np.log10(xs_o), color=OUR_C, lw=2.4, ls="--", zorder=4)

# CLoRA cloud + fit
xs_c = np.logspace(c_lx.min() - 0.05, c_lx.max() + 0.05, 100)
ax.plot(xs_c, c_ic + c_sl * np.log10(xs_c), color=CLO_C, lw=2.4, zorder=4)
ax.scatter(CF, CB, s=120, marker="D", facecolor=CLO_C, edgecolor="white",
           linewidth=1.0, zorder=5, label="CLoRA Table 4 (published)")
# label only the load-bearing rows to avoid clutter
LABELPTS = {"CLoRA k2048": (-6, 6), "CLoRA k1024": (7, -2), "LoRA-L2": (-6, -11),
            "LoRA": (8, -3)}
for name, f, b, fam in CLORA_T4:
    if name in LABELPTS:
        dx, dy = LABELPTS[name]
        ax.annotate(name.replace("CLoRA ", ""), (f, b),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.8,
                    color=fs.INK2, ha="left" if dx > 0 else "right")

# base BBH reference lines (each dataset its own base; NOT lined up)
ax.axhline(CLORA_BASE_BBH, color=CLO_C, lw=1.2, ls=(0, (6, 3)), alpha=0.7, zorder=1)
ax.text(OF.max(), CLORA_BASE_BBH + 0.25, "CLoRA base BBH 34.91",
        color=CLO_C, ha="right", va="bottom", fontsize=8.2)
ax.axhline(OUR_BASE_BBH, color=OUR_C, lw=1.2, ls=(0, (2, 2)), alpha=0.7, zorder=1)
ax.text(OF.max(), OUR_BASE_BBH - 0.35, "our base BBH 33.10",
        color=OUR_C, ha="right", va="top", fontsize=8.2)

# k1024/k2048 exceed their own base -- annotate so it can't be weaponised
ax.annotate("k1024 & k2048 exceed their own base BBH\n"
            "(they gain out-of-domain accuracy by shrinking $F_\\Delta$)",
            xy=(0.17, 37.6), xytext=(0.62, 39.0), fontsize=8.2, color=CLO_C,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=CLO_C, lw=0.9))

ax.set_ylim(4, 41.5)
ax.set_xscale("log")
ax.set_xlabel(r"effective update magnitude  $F_\Delta$  (CLoRA Eq. 3, log scale)")
ax.set_ylabel("BBH retention (%)")
ax.set_title("The magnitude law in two independent datasets (BBH-to-BBH)",
             loc="left", pad=10)
ax.grid(True, which="both", alpha=0.45)
ax.legend(loc="lower left", fontsize=9.5)
ax.text(0.975, 0.34,
        "same direction, parallel slopes (two fits, not one line):\n"
        f"CLoRA Table 4:  $r$ = {c_r:.2f}   slope {c_sl:.1f} pp/decade\n"
        f"   (robust: drop-one $r\\leq{drop1:.2f}$; baseline {base_sl:.1f}, "
        f"$k$-series {k_sl:.1f})\n"
        f"ours (BBH$\\leftrightarrow$BBH):  $r$ = {o_r:.2f}   slope {o_sl:.1f} pp/decade",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", fc="#f7f7f4", ec=fs.GRID))

fig.text(0.5, -0.015,
         "CLoRA's LoRA baseline sits at $F_\\Delta$=0.79, essentially our own LoRA "
         "magnitude; only its null-space-constrained $k$-series runs lower.",
         ha="center", va="top", fontsize=8.4, color=fs.INK2, style="italic")

pdf, png = fs.save(fig, "fig_cross_literature")
plt.close(fig)

# --------------------------------------------------------------- report --------
print("FIGURE cross-literature (BBH<->BBH overlay)")
print(f"  ours : n={len(our)}  r={o_r:.3f}  slope={o_sl:.2f} pp/decade  "
      f"(F range {OF.min():.2f}-{OF.max():.2f}, BBH {OB.min():.1f}-{OB.max():.1f})")
print(f"  CLoRA: n={len(CLORA_T4)}  r={c_r:.3f}  slope={c_sl:.2f}  "
      f"(baseline {base_sl:.2f}, k-series {k_sl:.2f}; drop-one worst r={drop1:.3f})")
print(f"  CLoRA base BBH 34.91; k1024=36.49 k2048=38.67 exceed base")
print(f"  wrote {pdf}\n        {png}")
