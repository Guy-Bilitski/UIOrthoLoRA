#!/usr/bin/env python3
"""Emit figures/fig0_hero.png|pdf: body Figure 1 (fig:hero).

Reconstructed generator (2026-08-08). The committed fig0_hero.png had no
working generator in the repo: paper_figs_v2.py and
make_figs_split_lora_null.py both emit a DIFFERENT figure (verdict title,
slope annotation, split legend, pre-rename axis label). This script
reproduces the committed figure from the frozen pool; the only intended
deviation is a [%] unit on the y label (2026-08 exhibit audit).

Content: core retention against effective update magnitude F_delta (log x)
for the Llama-2-7B commonsense pool (families lrsw + frc,
quarantine-included, the correlational-pool convention of app:pool).
456 runs total; the 8 with F_delta > 10 (all retention <= 3) are outside
the frame and disclosed by the in-figure note, so 448 are plotted, which
is the caption's count. Overlays: two-segment least-squares fit in
log10(F_delta) with the breakpoint chosen by scan (lands near
F_delta ~ 0.45, the caption's "knee near 0.4"); the base ceiling 26.0
(rq1_stats/07_make_grand_table.py block table); a star at LoRA+wd's
Table-1 operating point (fdelta 0.399, ret 25.86,
adjudication/tables/op_points_llama_cs.csv).

Method display mapping is explicit (regex classify would misroute
milorawd via its lorawd substring): lorawdr16 -> LoRA+wd,
milorawd -> MiLoRA, matching the committed 8-entry legend.

Usage: python3 41_fig_hero.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WRITING = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, WRITING)
import figstyle                                             # noqa: E402

POOL = os.path.join(WRITING, "acl_analysis", "insights", "pool.csv")
OUTDIR = os.path.join(WRITING, "figures")

FAMS = ("lrsw", "frc")
FRAME_FDMAX = 10.0
BASE_CEILING = 26.0
OP_STAR = (0.399, 25.86)          # LoRA+wd, op_points_llama_cs.csv
DISPLAY = {"lora": "LoRA", "lorawd": "LoRA+wd", "lorawdr16": "LoRA+wd",
           "milora": "MiLoRA", "milorawd": "MiLoRA", "lora_null": "LoRA-Null",
           "clora": "CLoRA", "sclora": "SC-LoRA", "dora": "DoRA",
           "pissa": "PiSSA"}
LEGEND_ORDER = ["LoRA", "LoRA+wd", "MiLoRA", "LoRA-Null",
                "CLoRA", "SC-LoRA", "DoRA", "PiSSA"]


def two_segment_fit(xs, ys):
    """Least-squares continuous two-segment fit in x; scan breakpoints."""
    import math
    pts = sorted(zip(xs, ys))
    best = None
    cands = sorted({x for x, _ in pts})
    for b in cands[10:-10]:
        # basis: 1, x, max(0, x-b)
        n = len(pts)
        X = [(1.0, x, max(0.0, x - b)) for x, _ in pts]
        # normal equations, 3x3
        A = [[sum(r[i] * r[j] for r in X) for j in range(3)] for i in range(3)]
        v = [sum(r[i] * y for r, (_, y) in zip(X, pts)) for i in range(3)]
        try:
            import numpy as np
            coef = np.linalg.solve(np.array(A), np.array(v))
        except Exception:
            continue
        sse = sum((y - (coef[0] + coef[1] * x + coef[2] * max(0.0, x - b))) ** 2
                  for x, y in pts)
        if best is None or sse < best[0]:
            best = (sse, b, coef)
    return best[1], best[2]


def main():
    import math
    rows = [r for r in csv.DictReader(open(POOL))
            if r["fam"] in FAMS and r.get("ret") and r.get("fd")]
    total = len(rows)
    beyond = [r for r in rows if float(r["fd"]) > FRAME_FDMAX]
    plot = [r for r in rows if float(r["fd"]) <= FRAME_FDMAX]
    max_ret_beyond = max(float(r["ret"]) for r in beyond)

    fig, ax = plt.subplots(figsize=(5.4, 3.75))
    for disp in LEGEND_ORDER:
        sub = [r for r in plot if DISPLAY.get(r["method"]) == disp]
        if not sub:
            continue
        ax.scatter([float(r["fd"]) for r in sub],
                   [float(r["ret"]) for r in sub],
                   s=16, c=figstyle.color(disp), marker=figstyle.marker(disp),
                   alpha=.85, edgecolors="white", linewidths=.3,
                   label=disp, zorder=3)

    xs = [math.log10(float(r["fd"])) for r in plot]
    ys = [float(r["ret"]) for r in plot]
    b, coef = two_segment_fit(xs, ys)
    import numpy as np
    fx = np.linspace(min(xs), max(xs), 200)
    fy = coef[0] + coef[1] * fx + coef[2] * np.maximum(0.0, fx - b)
    ax.plot(10 ** fx, fy, color=figstyle.FIT_C, lw=2.0, zorder=4)

    ax.axhline(BASE_CEILING, color=figstyle.CEILING_C, lw=1.2, ls=":",
               zorder=2)
    ax.annotate("base ceiling %.1f" % BASE_CEILING,
                (FRAME_FDMAX * 0.95, BASE_CEILING),
                textcoords="offset points", xytext=(0, 5), ha="right",
                fontsize=8, color=figstyle.CEILING_C)
    ax.scatter([OP_STAR[0]], [OP_STAR[1]], marker="*", s=260,
               facecolors="none", edgecolors=figstyle.INK, linewidths=1.2,
               zorder=5)
    # committed figure said "retention <= 3", but the max beyond frame is
    # 3.04; ceil keeps the note true if the pool ever shifts
    ax.annotate("%d collapsed runs with $F_\\Delta > %d$\n"
                "(retention $\\leq$ %d) beyond frame"
                % (len(beyond), int(FRAME_FDMAX), math.ceil(max_ret_beyond)),
                (FRAME_FDMAX * 0.95, 11.8), ha="right", fontsize=7,
                color=figstyle.INK2)

    ax.set_xscale("log")
    ax.set_xlabel(r"effective update magnitude $F_\Delta$ (log scale)",
                  fontsize=9)
    ax.set_ylabel("core retention (mean of BBH, MMLU-Pro) [%]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(color=figstyle.GRID, lw=.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", ncol=2, frameon=True, fontsize=7.5,
              framealpha=.9, edgecolor=figstyle.GRID)

    fig.tight_layout(pad=0.4)
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, "fig0_hero." + ext), dpi=300)

    print("pool %d = plotted %d + beyond-frame %d" % (total, len(plot),
                                                      len(beyond)))
    print("two-segment breakpoint: F_delta = %.3f" % (10 ** b))
    print("wrote", os.path.join(OUTDIR, "fig0_hero.png"))


if __name__ == "__main__":
    main()
