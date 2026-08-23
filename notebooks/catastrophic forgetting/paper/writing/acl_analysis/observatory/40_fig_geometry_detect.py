#!/usr/bin/env python3
"""Body Figure 2 (chosen 2026-08-07 over the stated-target view, which is now
the appendix figure from 38_fig_geometry_panels.py).

Emit figures/fig_geometry_detect.pdf|png: the "detection" view of RQ2 geometry.

The question this answers is the one the body actually asks: can we tell which
method produced an update from its geometry alone? That is not the same as
"does each design hit the subspace it states", and it is the version that
covers every method rather than the three with a measurable stated target.

Panel (a): for each method, the largest fold-difference across the four energy
columns between that method's median and the pooled median of the methods that
state no target subspace (plain LoRA, LoRA+wd, DoRA), on the same column and
the same run family. 1.0 means the update is indistinguishable from a
no-target update. The subspace where each method differs most is printed on
its bar, so the reader sees both HOW distinctive and WHERE.

  Note this deliberately does not care whether the distinctive subspace is the
  one the design states. SC-LoRA's largest difference is in the principal input
  directions, which is a mechanical side effect of its initialization rather
  than its stated target (see 37_geometry_battery_table.py). For detection that
  is still a valid fingerprint; for "the design does what it says" it is not.
  Those are different claims and this figure makes only the first.

Panel (b): the nested variance ladder (frozen anchors, key_numbers.md 19.1;
CIs from acl_analysis/rq1_stats/04_ladder_ci.py), i.e. how little the geometry
adds to predicting retention once the update's size is known.

Both commonsense sweeps are shown (Llama `lrsw`, Qwen `qwsw`) so the split is
visibly not a one-model artifact. Energy shares are invariant to multiplying
the update by a constant.

Usage: python3 40_fig_geometry_detect.py
"""
import csv
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
WRITING = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, WRITING)
import figstyle                                             # noqa: E402

POOL = os.path.join(WRITING, "acl_analysis", "insights", "pool.csv")
OUTDIR = os.path.join(WRITING, "figures")

NO_TARGET = ("lora", "lorawd", "dora")
ADJ = os.path.join(WRITING, "acl_analysis", "adjudication", "tables")
OPFILE = {"lrsw": "op_points_llama_cs.csv", "qwsw": "op_points_qwen_cs.csv"}
NAME2POOL = {"LoRA": "lora", "LoRA+wd": "lorawd", "DoRA": "dora",
             "CLoRA": "clora", "MiLoRA": "milora", "SC-LoRA": "sclora",
             "LoRA-Null": "lora_null"}
# 99th percentile of the same statistic computed within the no-target runs
# alone, resampling training cells (4000 draws, seed 0). Anything below this
# is indistinguishable from sampling noise.
NULL_P99 = {"lrsw": 1.28, "qwsw": 1.20}
COLS = ["e_top_w", "e_bot_w", "ein_top_w", "ein_bot_w"]
SUBSPACE = {"e_top_w": "principal output", "e_bot_w": "minor output",
            "ein_top_w": "principal input", "ein_bot_w": "minor input"}
DISPLAY = {"lora": "LoRA", "lorawd": "LoRA+wd", "dora": "DoRA",
           "clora": "CLoRA", "milora": "MiLoRA", "sclora": "SC-LoRA",
           "lora_null": "LoRA-Null"}
ARMS = [("lrsw", "Llama-2-7B", "#2a78d6"), ("qwsw", "Qwen2.5-7B", "#eda100")]
MIN_RUNS = 5

LADDER = [("update size", 0.395, 0.309, 0.483),
          ("update geometry", 0.017, 0.007, 0.032),
          ("which method", 0.006, 0.003, 0.018)]


def load():
    return [r for r in csv.DictReader(open(POOL))
            if r.get("quarantined", "").lower() not in ("true", "1")]


def op_points(fam):
    """method -> operating-point learning rate, the cell Table 1 reports."""
    out = {}
    with open(os.path.join(ADJ, OPFILE[fam])) as fh:
        for o in csv.DictReader(fh):
            pm = NAME2POOL.get(o["method"])
            if pm:
                out[pm] = float(o["best_lr"])
    return out


def at_rate(rows, fam, methods, lr):
    out = []
    for r in rows:
        if r["fam"] != fam or r["method"] not in methods:
            continue
        try:
            if abs(float(r["lr"]) - lr) / lr > 1e-3:
                continue
        except (ValueError, ZeroDivisionError, TypeError):
            continue
        out.append(r)
    return out


def distinctiveness(rows, fam):
    """method -> (fold, column, n) at that method's reported operating point,
    against the plain adapters trained at the SAME learning rate."""
    ops = op_points(fam)
    out = {}
    for m, lr in ops.items():
        mr = at_rate(rows, fam, (m,), lr)
        rr = at_rate(rows, fam, NO_TARGET, lr)
        if len(mr) < 2 or len(rr) < 3:
            continue
        best, bestc = 1.0, None
        for c in COLS:
            a = [float(x[c]) for x in mr if x.get(c)]
            b = [float(x[c]) for x in rr if x.get(c)]
            if not a or not b:
                continue
            fold = st.median(a) / st.median(b)
            sc = max(fold, 1 / fold)
            if sc > best:
                best, bestc = sc, c
        out[m] = (best, bestc, len(mr))
    return out


def main():
    rows = load()
    per_arm = {fam: distinctiveness(rows, fam) for fam, _l, _c in ARMS}

    methods = [m for m in per_arm[ARMS[0][0]] if m in per_arm[ARMS[1][0]]]
    constrains = [m for m in methods if m not in NO_TARGET]
    plain = [m for m in methods if m in NO_TARGET]
    constrains.sort(key=lambda m: -max(per_arm[f][m][0] for f, _l, _c in ARMS))
    plain.sort(key=lambda m: -max(per_arm[f][m][0] for f, _l, _c in ARMS))
    order = constrains + plain

    # Single column width (PI decision 2026-08-09): the variance-ladder panel
    # was dropped because its three numbers are printed in the sentence beside
    # it and its intervals are in Table 7; the figure is now one panel.
    fig, axes = plt.subplots(1, 1, figsize=(3.3, 2.9))
    axes = [axes]

    ax = axes[0]
    h = 0.36
    ypos = {}
    y = 0.0
    for i, m in enumerate(order):
        if i == len(constrains):
            y += 0.75                      # gap between the two groups
        ypos[m] = y
        y += 1.0
    for (fam, label, colr), off in zip(ARMS, (+h / 2, -h / 2)):
        d = per_arm[fam]
        ax.barh([ypos[m] + off for m in order], [d[m][0] for m in order],
                height=h, color=colr, edgecolor="white", linewidth=.6,
                zorder=3, label=label)
    ax.axvline(1.0, color=figstyle.BASELINE, lw=1.0, ls="--", zorder=2)
    band = max(NULL_P99.values())
    ax.axvspan(0, band, color=figstyle.GRID, alpha=.55, zorder=1, lw=0)
    ax.annotate("sampling noise", (band, -0.75), textcoords="offset points",
                xytext=(-3, 0), ha="right", va="center", fontsize=6.2,
                color=figstyle.INK2, style="italic")
    for m in order:
        if m in NO_TARGET:
            continue
        # label the column that won on the LONGEST bar, which is the bar the
        # text sits beside; and if the two models disagree, say so rather than
        # silently showing one of them.
        best_fam = max(ARMS, key=lambda a: per_arm[a[0]][m][0])[0]
        other_fam = [a[0] for a in ARMS if a[0] != best_fam][0]
        best = per_arm[best_fam][m][0]
        col = per_arm[best_fam][m][1]
        if not col:
            continue
        txt = SUBSPACE[col]
        if per_arm[other_fam][m][1] != col:
            txt += "*"
        ax.annotate(txt, (best, ypos[m]), textcoords="offset points",
                    xytext=(5, -2.4), fontsize=6.2, color=figstyle.INK2)
    ax.set_yticks([ypos[m] for m in order])
    ax.set_yticklabels([DISPLAY[m] for m in order], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 5.9)
    ax.set_ylim(ypos[order[-1]] + 0.6, -1.35)   # headroom for "sampling noise"
    ax.set_xlabel("largest fold-difference across four energy measures\n"
                  r"($1\times$ = indistinguishable)",
                  fontsize=7.2)
    ax.set_title("distance from a no-target update", fontsize=8,
                 loc="left")
    ax.tick_params(labelsize=7)
    # two-line group labels: single lines collide with the per-bar subspace
    # annotations on the same row
    ax.annotate("constrains\nthe geometry", (5.8, ypos[constrains[-1]]),
                fontsize=6.5, color=figstyle.INK2, ha="right", va="center",
                style="italic")
    ax.annotate("states no\ntarget subspace", (5.8, ypos[plain[1]]),
                fontsize=6.5, color=figstyle.INK2, ha="right", va="center",
                style="italic")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right",
              handlelength=1.1, handletextpad=.5,
              bbox_to_anchor=(1.0, -0.02))

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="x", color=figstyle.GRID, lw=.6, zorder=0)
        a.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, "fig_geometry_detect." + ext), dpi=300)

    for fam, label, _c in ARMS:
        print("\n%s (%s)" % (label, fam))
        for m in order:
            f, c, n = per_arm[fam][m]
            print("   %-10s n=%2d  %5.2fx  %-18s %s"
                  % (DISPLAY[m], n, f, SUBSPACE[c] if c else "-",
                     "" if f > NULL_P99[fam] else "<- inside sampling noise"))
    print("\nwrote", os.path.join(OUTDIR, "fig_geometry_detect.pdf"))


if __name__ == "__main__":
    main()
