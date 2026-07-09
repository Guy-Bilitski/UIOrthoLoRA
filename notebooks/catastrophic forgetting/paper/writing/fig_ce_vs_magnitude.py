#!/usr/bin/env python
"""FIGURE 4 - CE-TO-BASE vs MAGNITUDE (a third-party-comparable forgetting axis).

Main : our runs' soft cross-entropy of the fine-tuned next-token distribution to
       the BASE distribution on WikiText-103 (MiLoRA Table-8 / Kalajdzievski 2024
       metric) vs effective update magnitude F_Delta.  CE rises monotonically with
       magnitude; MiLoRA ~ LoRA once magnitude is matched (LR 3e-4, F_Delta~1.27).
Right: MiLoRA's own PUBLISHED Table-8 CE-to-base numbers (LoRA 3.24, PiSSA 6.07,
       MiLoRA 2.54) on the SAME y-axis -- their PiSSA>LoRA>MiLoRA ordering is the
       magnitude ordering; MiLoRA's published edge came from a lower-magnitude op.

Reads results/forgetting.jsonl live.  Published points are cited anchors.
Supports paper section: CE / independent-metric corroboration.
"""
import re
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import figstyle as fs
fs.apply_rc()

# ---------------------------------------------------------------- load ours ----
def label(run):
    # frm_lorawd_wd0_* is wd=0 == plain LoRA; frm_lorawd_wd0pX_* is LoRA+wd
    if "lorawd" in run:
        m = re.search(r"wd([0-9]+)(p([0-9]+))?", run)
        wd = float(m.group(1) + ("." + m.group(3) if m.group(3) else ""))
        return "LoRA+wd" if wd > 0 else "LoRA"
    return fs.method_from_run(run)

pts = []
for l in open(fs.FORGETTING):
    d = json.loads(l)
    pts.append(dict(m=label(d["run_name"]), F=d["fdelta"], ce=d["forgetting_ce"],
                    run=d["run_name"]))
F = np.array([p["F"] for p in pts])
CE = np.array([p["ce"] for p in pts])
rho = stats.spearmanr(F, CE)[0]

# matched-magnitude pair
milora = next(p for p in pts if p["m"] == "MiLoRA")
lora_m = min((p for p in pts if p["m"] == "LoRA"),
             key=lambda p: abs(p["F"] - milora["F"]))

# -------------------------------- [EXTERNAL] MiLoRA Table 8 (cited anchors) ----
# Wang et al., MiLoRA, Table 8 (CE-to-base on WikiText-103, Kalajdzievski 2024).
PUB = [("LoRA", 3.24), ("PiSSA", 6.07), ("MiLoRA", 2.54)]

# ------------------------------------------------------------------ figure -----
fig, (ax, axp) = plt.subplots(1, 2, figsize=(11.6, 5.2), sharey=True,
                              gridspec_kw=dict(width_ratios=[2.5, 1.0],
                                               wspace=0.06))

# fit line (log F)
sl, ic, r, p, _ = stats.linregress(np.log10(F), CE)
xs = np.logspace(np.log10(F.min()) - 0.05, np.log10(F.max()) + 0.05, 100)
ax.plot(xs, ic + sl * np.log10(xs), color=fs.FIT_C, lw=2.0, zorder=2)
for p_ in pts:
    ax.scatter(p_["F"], p_["ce"], s=120, marker=fs.marker(p_["m"]),
               facecolor=fs.color(p_["m"]), edgecolor="white", linewidth=1.0,
               label=p_["m"], zorder=3)
# the matched-magnitude MiLoRA point coincides with LoRA -> draw it on top so both show
ax.scatter(milora["F"], milora["ce"], s=150, marker=fs.marker("MiLoRA"),
           facecolor=fs.color("MiLoRA"), edgecolor=fs.INK, linewidth=1.3, zorder=6)
# de-dup legend
h, lb = ax.get_legend_handles_labels()
seen = dict(zip(lb, h))
ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=9.5)
ax.set_xscale("log")
ax.set_xlabel(r"effective update magnitude  $F_\Delta$  (log scale)")
ax.set_ylabel("CE to base  (WikiText-103, MiLoRA Tab. 8 metric)")
ax.set_title("Forgetting (CE) rises with update magnitude", loc="left")
ax.grid(True, which="both", alpha=0.5)
ax.text(0.97, 0.05, f"Spearman $\\rho$ = {rho:.2f}\n$n$ = {len(pts)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f4", ec=fs.GRID))
# annotate the matched-magnitude near-tie
ax.annotate("MiLoRA ≈ LoRA\nat matched magnitude\n"
            f"(CE {milora['ce']:.2f} vs {lora_m['ce']:.2f},\n"
            f" $F_\\Delta$≈{milora['F']:.2f}, LR 3e-4)",
            xy=(milora["F"], milora["ce"]), xytext=(0.30, 6.3),
            fontsize=9.2, color=fs.INK,
            arrowprops=dict(arrowstyle="->", color=fs.MUTED, lw=1.1))

# ----- right: published Table-8 -----
xp = np.arange(len(PUB))
for i, (m, ce) in enumerate(PUB):
    axp.scatter(i, ce, s=150, marker=fs.marker(m), facecolor=fs.color(m),
                edgecolor="white", linewidth=1.0, zorder=3)
    axp.plot([i, i], [0, ce], color=fs.color(m), lw=2.0, alpha=0.5, zorder=2)
    axp.text(i, ce + 0.18, f"{ce:.2f}", ha="center", va="bottom", fontsize=9.5,
             fontweight="bold", color=fs.INK)
axp.set_xticks(xp)
axp.set_xticklabels([m for m, _ in PUB], rotation=25, ha="right", fontsize=9)
axp.set_title("MiLoRA Tab. 8\n(published)", loc="left", fontsize=11)
axp.set_xlim(-0.6, len(PUB) - 0.4)
axp.grid(True, axis="y", alpha=0.5)
axp.text(0.5, 0.02, "same magnitude\nordering", transform=axp.transAxes,
         ha="center", va="bottom", fontsize=8.6, color=fs.INK2, style="italic")

pdf, png = fs.save(fig, "fig_ce_vs_magnitude")
plt.close(fig)

# --------------------------------------------------------------- report --------
print("FIGURE 4  CE-to-base vs magnitude")
print(f"  n = {len(pts)} our runs;  Spearman rho(CE, F_Delta) = {rho:.3f}")
for p_ in sorted(pts, key=lambda z: z["F"]):
    print(f"     {p_['m']:8s} F_Delta={p_['F']:7.3f}  CE={p_['ce']:.3f}  ({p_['run']})")
print(f"  matched magnitude: MiLoRA CE {milora['ce']:.2f} ~ LoRA CE {lora_m['ce']:.2f}"
      f" at F_Delta~{milora['F']:.2f}")
print(f"  published (MiLoRA Tab.8): " + ", ".join(f"{m} {c}" for m, c in PUB))
print(f"  wrote {pdf}\n        {png}")
