#!/usr/bin/env python3
"""Emit figures/fig_geometry.pdf|png: the three-panel RQ2 geometry exhibit.

Appendix figure fig:placement. (The BODY geometry exhibit is
40_fig_geometry_detect.py; this one answers the stricter stated-target
question.) One figure, three panels, one per part of the RQ2 geometry
claim:

  (a) how far each design's update sits above the level of the methods that
      state no target subspace, on the subspace that design targets;
  (b) the same quantity against update size;
  (the variance ladder that used to be panel (c) now lives in the body
   figure from 40_fig_geometry_detect.py, so it is not repeated here)

WHAT (a) AND (b) ACTUALLY SHOW on the Llama commonsense sweep, after the
2026-08-07 target-column correction: MiLoRA sits at 2.10x the no-target level
and falls from 3.84x to 1.35x across its own magnitude terciles; LoRA-Null
sits at 1.17x and falls from 1.48x to 1.11x; SC-LoRA sits at 1.26x and does
NOT vary with magnitude (1.29x -> 1.28x). Only MiLoRA carries the effect
strongly. Do not restore the earlier, much larger SC-LoRA numbers: those were
measured on ein_top, which is a side effect of its init, not its target.

Panels (a) and (b) use the Llama-2-7B commonsense sweep (family `lrsw`), the
same family as Figure 1, from acl_analysis/insights/pool.csv (unquarantined).
Panel (c) restates the frozen ladder anchors of key_numbers.md 19.1, whose CIs
come from acl_analysis/rq1_stats/04_ladder_ci.py.

Each design is scored on the base-weight subspace its own paper targets (see
the DESIGNS comment below for the init-code verification):
  MiLoRA     minor input directions             ein_bot_w
  SC-LoRA    principal output directions        e_top_w
  LoRA-Null  least-activated input directions   ein_bot_w
The reference in both panels is the pooled set of runs whose method states no
target subspace (plain LoRA, LoRA+wd, DoRA), measured on the SAME column, so
the comparison is within one metric.

Energy shares are invariant to multiplying the update by a constant, so the
trend in panel (b) cannot follow the update's size by construction.

DISCLOSURE: SC-LoRA's seed-42 series in `lrsw`/`qwsw` sits at roughly twice the
update magnitude of its seeds 43-45 at every learning rate, while those three
agree to three decimals; it is retained here and flagged in the caption, not
silently dropped.

Colors and markers come from paper/writing/figstyle.py (validated: worst
adjacent CVD pair dE 7.2 protan, carried by distinct marker shapes and direct
labels). Pure stdlib + matplotlib.

Usage: python3 38_fig_geometry_panels.py
"""
import csv
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
WRITING = os.path.dirname(os.path.dirname(HERE))            # paper/writing
sys.path.insert(0, WRITING)
import figstyle                                             # noqa: E402

POOL = os.path.join(WRITING, "acl_analysis", "insights", "pool.csv")
OUTDIR = os.path.join(WRITING, "figures")

FAMILY = "lrsw"
NO_TARGET = ("lora", "lorawd", "dora")
# Target columns follow 37_geometry_battery_table.py SIGNATURE, which was
# verified against the initialization code in the 2026-08-06 review pass:
#   sclora_init.py sets B = Q_r, the top eigenvectors of the OUTPUT second
#     moment, so SC-LoRA's proxy is e_top. Its much larger ein_top share is a
#     mechanical side effect of A = Q_r^T W0 and is NOT its stated target.
#   lora_null_init.py puts A's rows in the least-activated INPUT directions,
#     so LoRA-Null's proxy is ein_bot, not e_top.
# The body RQ2 prose used to name principal-input for SC-LoRA and
# principal-output for LoRA-Null; that mapping predates the review and is
# what this generator must not follow.
DESIGNS = [
    ("milora",    "MiLoRA",    "ein_bot_w", "minor input"),
    ("sclora",    "SC-LoRA",   "e_top_w",   "principal output"),
    ("lora_null", "LoRA-Null", "ein_bot_w", "least-activated input"),
]

# Frozen ladder anchors (key_numbers.md 19.1; CIs from 04_ladder_ci.py).
LADDER = [
    ("update size",      0.395, 0.309, 0.483),
    ("update geometry",  0.017, 0.007, 0.032),
    ("which method",     0.006, 0.003, 0.018),
]
LADDER_BASE = 0.390   # run family alone


def load():
    rows = []
    with open(POOL) as fh:
        for r in csv.DictReader(fh):
            if r.get("quarantined", "").lower() in ("true", "1"):
                continue
            if r["fam"] != FAMILY:
                continue
            rows.append(r)
    return rows


def vals(rows, methods, col):
    out = []
    for r in rows:
        if r["method"] in methods and r.get(col) and r.get("logfd"):
            out.append((float(r["logfd"]), float(r[col])))
    return out


def main():
    rows = load()
    if not rows:
        sys.exit("no rows for family %s" % FAMILY)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))

    # Each design is scored against the no-target level on its OWN metric, so
    # the three designs share one comparable axis: 1.0 means "the same share a
    # run that states no target subspace puts there".
    ref = {}
    for _meth, disp, col, _s in DESIGNS:
        ref[col] = st.median([v for _, v in vals(rows, NO_TARGET, col)])

    # ---------------------------------------------------------- panel (a) --
    ax = axes[0]
    report = []
    for i, (meth, disp, col, _sub) in enumerate(DESIGNS):
        c, m = figstyle.color(disp), figstyle.marker(disp)
        tv = [v / ref[col] for _, v in vals(rows, (meth,), col)]
        nv = [v / ref[col] for _, v in vals(rows, NO_TARGET, col)]
        ax.scatter([i + 0.19] * len(nv), nv, s=8, c=figstyle.MUTED,
                   marker="o", alpha=.40, linewidths=0, zorder=2)
        ax.scatter([i - 0.19] * len(tv), tv, s=17, c=c, marker=m, alpha=.85,
                   edgecolors="white", linewidths=.35, zorder=3)
        above = sum(1 for x in nv if x >= min(tv))
        report.append((disp, col, len(tv), st.median(tv), len(nv),
                       st.median(nv), above))
    ax.axhline(1.0, color=figstyle.BASELINE, lw=1.0, ls="--", zorder=1)
    ax.annotate("no-target level", (-0.5, 1.0), textcoords="offset points",
                xytext=(1, 5), ha="left", fontsize=6.3, color=figstyle.INK2)
    ax.set_xticks(range(len(DESIGNS)))
    ax.set_xticklabels([d[1] for d in DESIGNS], fontsize=7)
    ax.set_xlim(-0.55, len(DESIGNS) - 0.45)
    ax.set_ylabel("energy in the targeted subspace\n"
                  r"($\times$ a no-target update)", fontsize=7.2)
    ax.set_title("(a) placement in its own target", fontsize=8, loc="left")
    ax.tick_params(labelsize=7)

    # ---------------------------------------------------------- panel (b) --
    ax = axes[1]
    nt_all = []
    for _meth, disp, col, _s in DESIGNS:
        nt_all += [(x, v / ref[col]) for x, v in vals(rows, NO_TARGET, col)]
    ax.scatter([x for x, _ in nt_all], [y for _, y in nt_all], s=8,
               c=figstyle.MUTED, marker="o", alpha=.30, linewidths=0, zorder=2)
    ax.axhline(1.0, color=figstyle.BASELINE, lw=1.0, ls="--", zorder=1)
    trends = []
    for meth, disp, col, _s in DESIGNS:
        c, m = figstyle.color(disp), figstyle.marker(disp)
        pts = sorted((x, v / ref[col]) for x, v in vals(rows, (meth,), col))
        ax.scatter([x for x, _ in pts], [y for _, y in pts], s=17, c=c,
                   marker=m, alpha=.85, edgecolors="white", linewidths=.35,
                   zorder=3)
        k = max(1, len(pts) // 3)
        seg = [pts[:k], pts[k:-k] or pts[k:], pts[-k:]]
        tx = [st.median([x for x, _ in s]) for s in seg if s]
        ty = [st.median([y for _, y in s]) for s in seg if s]
        ax.plot(tx, ty, color=c, lw=1.5, zorder=4)
        trends.append((disp, ty[0], ty[-1]))
    ax.annotate("no-target level", (min(x for x, _ in nt_all), 1.0),
                textcoords="offset points", xytext=(0, -9), ha="left",
                fontsize=6.3, color=figstyle.INK2)
    ax.set_xlabel(r"update magnitude $F_\Delta$ (log$_{10}$)", fontsize=7.2)
    ax.set_ylabel("energy in the targeted subspace\n"
                  r"($\times$ a no-target update)", fontsize=7.2)
    ax.set_xlim(right=max(x for x, _ in nt_all) + 0.10)
    ax.set_title("(b) the same, vs update size", fontsize=8, loc="left")
    ax.tick_params(labelsize=7)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=figstyle.GRID, lw=.6, zorder=0)
        ax.set_axisbelow(True)

    handles = [Line2D([], [], color=figstyle.color(d[1]),
                      marker=figstyle.marker(d[1]), lw=1.5, ms=4.5,
                      markeredgecolor="white", markeredgewidth=.35,
                      label="%s: %s" % (d[1], d[3])) for d in DESIGNS]
    handles.append(Line2D([], [], color=figstyle.MUTED, marker="o", lw=0,
                          ms=3.6, alpha=.6,
                          label="states no target subspace"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=6.3, handletextpad=.4, columnspacing=1.1,
               bbox_to_anchor=(0.5, -0.015))

    fig.tight_layout(pad=0.5, w_pad=1.9, rect=(0, 0.075, 1, 1))
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, "fig_geometry." + ext), dpi=300)

    print("family:", FAMILY)
    print("\npanel (a) separation")
    for disp, col, nt, mt, nn, mn, above in report:
        print("  %-10s %-11s design n=%3d median %.3f | no-target n=%3d "
              "median %.3f | no-target runs at or above design min: %d"
              % (disp, col, nt, mt, nn, mn, above))
    print("\npanel (b) tercile medians (small -> large update)")
    for disp, lo, hi in trends:
        print("  %-10s %.3f -> %.3f  (%.2fx)" % (disp, lo, hi, hi / lo))
    print("\nwrote", os.path.join(OUTDIR, "fig_geometry.pdf"))


if __name__ == "__main__":
    main()
