#!/usr/bin/env python
"""FIGURE 2 - GEOMETRY: magnitude first, geometry as a measurement tool.
Follows the 4-panel spec in handoff/27_GEOMETRY_DRIFT_2026-07-09.md.

A  magnitude law scatter (retention vs log F_Delta, 303 saved adapters, by method).
B  stress-test bars: partial r(retention, metric | log F_Delta) across
   [all] / [drop PiSSA+SC-LoRA] / [on-curve six] -- amp_top / ein_top / e_top
   collapse & FLIP, stable_rank STAYS (the honest rank second-order effect).
C  method-fingerprint heatmap: z-scored SVD-alignment metrics x method -- the
   metrics recover each method's init design from the *trained* adapter.
D  per-layer input-side concentration (SC-LoRA q/k early-layer spike vs LoRA
   baseline) + inset: SC-LoRA's constraint ERODES with learning rate (r=-0.96).

Reads results/geo_drift/master_labeled.jsonl (+ summary.json F_Delta join +
permatrix/ per-layer).  Prints the key stats it drew.
Supports paper section: Geometry / Measurement-tool.
"""
import os
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import figstyle as fs
fs.apply_rc()

# ----------------------------------------------------------------- load --------
geo = [json.loads(l) for l in open(fs.GEO_MASTER) if l.strip()]
rows = []
for r in geo:
    if r.get("retention") is None:
        continue
    f = fs.summary_fdelta(r["run"])
    if not f or f <= 0:
        continue
    r["fdelta"] = f
    r["logf"] = np.log10(f)
    rows.append(r)

ON_CURVE = {"LoRA", "LoRA+wd", "LoRA-Null", "MiLoRA", "CLoRA", "DoRA"}
METRICS = ["e_top", "amp_top", "ein_top", "stable_rank_w"]
MLABEL = {"e_top": r"$e_{\rm top}$", "amp_top": r"amp$_{\rm top}$",
          "ein_top": r"$e^{\rm in}_{\rm top}$", "stable_rank_w": "stable rank"}


def partial_r(sub, metric):
    lx = np.array([x["logf"] for x in sub])
    y = np.array([x["retention"] for x in sub])
    m = np.array([x[metric] for x in sub], float)

    def resid(v):
        s, i, *_ = stats.linregress(lx, v)
        return v - (i + s * lx)
    return stats.pearsonr(resid(m), resid(y))[0]


def subset(task="cs", drop=(), only_on=False):
    out = []
    for x in rows:
        if task and x["task"] != task:
            continue
        if x["method"] in drop:
            continue
        if only_on and x["method"] not in ON_CURVE:
            continue
        out.append(x)
    return out


# --------------------------------------------------------------- figure --------
fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.6))
(axA, axB), (axC, axD) = axes
fig.subplots_adjust(hspace=0.32, wspace=0.24, left=0.07, right=0.985,
                    top=0.94, bottom=0.09)

# ===== Panel A: magnitude law scatter =====
F = np.array([r["fdelta"] for r in rows])
RET = np.array([r["retention"] for r in rows])
LX = np.log10(F)
pear = stats.pearsonr(LX, RET)[0]
spear = stats.spearmanr(F, RET)[0]
for m in fs.PALETTE:
    mask = np.array([r["method"] == m for r in rows])
    if not mask.any():
        continue
    axA.scatter(F[mask], RET[mask], s=42, marker=fs.marker(m),
                facecolor=fs.color(m), edgecolor="white", linewidth=0.6,
                alpha=0.9, label=m, zorder=3)
sl, ic, *_ = stats.linregress(LX, RET)
xs = np.logspace(LX.min(), LX.max(), 100)
axA.plot(xs, ic + sl * np.log10(xs), color=fs.FIT_C, lw=2.0, zorder=4)
axA.set_xscale("log")
axA.set_xlabel(r"effective update magnitude  $F_\Delta$  (log scale)")
axA.set_ylabel("retention")
axA.set_title("A   Magnitude is the 1st-order lever", loc="left")
axA.set_ylim(-3, 41)
axA.grid(True, which="both", alpha=0.5)
axA.text(0.03, 0.05,
         f"pooled $r$ = {pear:.2f}\nSpearman $\\rho$ = {spear:.2f}\n$n$ = {len(rows)}",
         transform=axA.transAxes, va="bottom", ha="left", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f4", ec=fs.GRID))
axA.legend(loc="upper right", ncol=2, fontsize=8.3, handletextpad=0.25,
           columnspacing=0.8, borderaxespad=0.3)

# ===== Panel B: stress-test bars =====
subs = [("all", subset("cs")),
        ("drop\nPiSSA+SC-LoRA", subset("cs", drop=("PiSSA", "SC-LoRA"))),
        ("on-curve\nsix", subset("cs", only_on=True))]
vals = {met: [partial_r(s, met) for _, s in subs] for met in METRICS}
xpos = np.arange(len(METRICS))
w = 0.26
sub_colors = ["#256abf", "#eda100", "#1baf7a"]   # three subsets (sequential-ish, distinct)
for j, (name, _) in enumerate(subs):
    axB.bar(xpos + (j - 1) * w, [vals[m][j] for m in METRICS], w,
            color=sub_colors[j], edgecolor="white", linewidth=0.8, label=name,
            zorder=3)
axB.axhline(0, color=fs.INK, lw=1.0, zorder=2)
axB.set_xticks(xpos)
axB.set_xticklabels([MLABEL[m] for m in METRICS])
axB.set_ylabel(r"partial $r$(retention, metric | $\log F_\Delta$)")
axB.set_title("B   Alignment effects collapse; rank stays", loc="left")
axB.grid(True, axis="y", alpha=0.5)
axB.legend(loc="upper left", fontsize=9, ncol=1)
# annotate the amp/ein FLIP and the stable-rank persistence
axB.annotate("flips sign\nwhen 2 outliers\nremoved",
             xy=(xpos[1] + 0 * w, vals["amp_top"][1]), xytext=(1.15, 0.28),
             fontsize=9, color=fs.INK, ha="left",
             arrowprops=dict(arrowstyle="->", color=fs.MUTED, lw=1.0))
axB.annotate("robust\n(rank 2nd-order)", xy=(xpos[3], vals["stable_rank_w"][2]),
             xytext=(2.35, -0.5), fontsize=9, color=fs.INK, ha="left",
             arrowprops=dict(arrowstyle="->", color=fs.MUTED, lw=1.0))

# ===== Panel C: method fingerprint heatmap =====
hm_metrics = ["e_top", "e_bot", "ein_top", "ein_bot", "stable_rank_w"]
hm_lab = [r"$e_{\rm top}$", r"$e_{\rm bot}$", r"$e^{\rm in}_{\rm top}$",
          r"$e^{\rm in}_{\rm bot}$", "stable rank"]
methods_c = [m for m in ["LoRA", "LoRA+wd", "LoRA-Null", "MiLoRA", "CLoRA",
                         "DoRA", "SC-LoRA", "CorDA"]]
cs_rows = subset("cs")
raw = np.zeros((len(hm_metrics), len(methods_c)))
for jc, m in enumerate(methods_c):
    grp = [r for r in cs_rows if r["method"] == m]
    for ic_, met in enumerate(hm_metrics):
        raw[ic_, jc] = np.mean([g[met] for g in grp])
# z-score each metric row ACROSS methods
z = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
div = LinearSegmentedColormap.from_list(
    "bwr_div", ["#2a78d6", "#f0efec", "#e34948"])
vmax = np.abs(z).max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = axC.imshow(z, cmap=div, norm=norm, aspect="auto")
axC.set_xticks(range(len(methods_c)))
axC.set_xticklabels(methods_c, rotation=35, ha="right", fontsize=9)
axC.set_yticks(range(len(hm_metrics)))
axC.set_yticklabels(hm_lab, fontsize=10)
axC.set_title("C   Fingerprints of each method's init (trained adapter)", loc="left")
for ic_ in range(len(hm_metrics)):
    for jc in range(len(methods_c)):
        axC.text(jc, ic_, f"{raw[ic_, jc]:.2f}", ha="center", va="center",
                 fontsize=7.6,
                 color="white" if abs(z[ic_, jc]) > 1.4 else fs.INK)
axC.grid(False)
cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.02)
cb.set_label("z-score across methods", fontsize=9)
cb.ax.tick_params(labelsize=8)

# ===== Panel D: per-layer concentration + erosion inset =====
def layer_profile(run, targets, metric="ein_top"):
    fn = os.path.join(fs.GEO_PERMATRIX, run + ".jsonl")
    if not os.path.exists(fn):
        return None
    d = {}
    for l in open(fn):
        x = json.loads(l)
        if x["target"] in targets:
            d.setdefault(x["layer"], []).append(x[metric])
    return np.array([np.mean(d[k]) for k in sorted(d)])

qk = ("q_proj", "k_proj")
sc_prof = layer_profile("lrsw_sclora_r32_lr2e5_s42", qk)
lora_prof = layer_profile("lrsw_lora_r16_lr2e4_s42", qk)
milora_prof = layer_profile("lrsw_milora_r32_lr3e4_s42", qk)
L = np.arange(len(sc_prof))
axD.plot(L, sc_prof, color=fs.color("SC-LoRA"), lw=2.2, marker="P", ms=5,
         label="SC-LoRA (lr 2e-5)", zorder=4)
if milora_prof is not None:
    axD.plot(L, milora_prof, color=fs.color("MiLoRA"), lw=1.8, marker="^",
             ms=4, label="MiLoRA", zorder=3)
axD.plot(L, lora_prof, color=fs.color("LoRA"), lw=1.8, marker="o", ms=4,
         label="LoRA (random baseline)", zorder=3)
axD.set_xlabel("transformer layer (q/k projections)")
axD.set_ylabel(r"input-side top-subspace energy  $e^{\rm in}_{\rm top}$")
axD.set_title("D   SC-LoRA concentrates in early-layer principal dirs", loc="left")
axD.grid(True, alpha=0.5)
axD.legend(loc="center right", fontsize=8.6)
axD.set_ylim(0, 1.05)

# inset: erosion vs LR (lrsw_sclora sweep)
scl = sorted([r for r in geo if r["run"].startswith("lrsw_sclora_")],
             key=lambda r: r["lr"])
elr = np.array([r["lr"] for r in scl])
etop = np.array([r["ein_top"] for r in scl])
er = stats.pearsonr(np.log10(elr), etop)[0]
axins = axD.inset_axes([0.09, 0.33, 0.42, 0.40])
axins.scatter(elr, etop, s=34, color=fs.color("SC-LoRA"), marker="P",
              edgecolor="white", linewidth=0.6, zorder=3)
sle, ice, *_ = stats.linregress(np.log10(elr), etop)
xr = np.array([elr.min(), elr.max()])
axins.plot(xr, ice + sle * np.log10(xr), color=fs.INK, lw=1.6, zorder=2)
axins.set_xscale("log")
axins.set_title(f"erosion with LR ($r$={er:.2f})", fontsize=8.5)
axins.set_xlabel("learning rate", fontsize=8)
axins.set_ylabel(r"$e^{\rm in}_{\rm top}$", fontsize=8)
axins.tick_params(labelsize=7)
axins.grid(True, which="both", alpha=0.4)

pdf, png = fs.save(fig, "fig_geometry_4panel")
plt.close(fig)

# --------------------------------------------------------------- report --------
print("FIGURE 2  geometry 4-panel")
print(f"  A  pooled r(ret,logF)={pear:.3f} spearman={spear:.3f} n={len(rows)}")
print("  B  partial r(ret, metric | logF)  [all | drop PiSSA+SC-LoRA | on-curve]:")
for m in METRICS:
    print(f"       {m:14s} " + "  ".join(f"{v:+.2f}" for v in vals[m]))
print("  C  fingerprint means (z-scored across 8 methods): MiLoRA e_bot>e_top,"
      " SC-LoRA ein_top spike, CorDA ein_bot spike, LoRA-Null e_top high")
print(f"  D  SC-LoRA q/k ein_top early-layer peak={sc_prof[:6].max():.2f}"
      f"  vs LoRA baseline mean={lora_prof.mean():.2f};"
      f"  erosion r(ein_top,logLR)={er:.3f} ({etop[0]:.2f}->{etop[-1]:.2f})")
print(f"  wrote {pdf}\n        {png}")
