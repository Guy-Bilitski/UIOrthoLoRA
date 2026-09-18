"""Render the matched-control figure: cross-energy share vs achieved update norm.

Reads only data/focused_norm/runs.csv (hash-verified snapshot). Calibration-seed
points draw the NORM dose frontier plus the MIX and UNREG references; if
confirmation rows exist, their per-seed points are overlaid as open markers.
Run with any Python that has matplotlib (the campaign venv does not).
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
# Distinct marker shapes and grayscale preserve the manuscript's print style.
C_NORM, C_MIX, C_UNREG = "#666666", "#000000", "#999999"
INK, MUTED = "#1a1a1a", "#898989"

rows = list(csv.DictReader(open(ROOT / "data/focused_norm/runs.csv")))
rows = [r for r in rows if r['condition'] in {'P1_UNREG', 'P1_MIX', 'P1_NORM'}]
selection = json.loads((ROOT / "data/focused_norm/selection_final.json").read_text())["selection"]


def cross(row):
    return 100 * (float(row["pooled_LT"]) + float(row["pooled_TL"]))


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, task, label in zip(axes, ("rte", "mrpc"), ("RTE", "MRPC")):
    ax.axhline(100*4/9, color=".45", linewidth=.8, linestyle="--", zorder=1, label="Dimension-only reference")
    cal = [r for r in rows if r["stage"] == "calibration" and r["task"] == task]
    norm = sorted([r for r in cal if r["condition"] == "P1_NORM"], key=lambda r: float(r["rho_pooled"]))
    mix = [r for r in cal if r["condition"] == "P1_MIX"]
    unreg = [r for r in cal if r["condition"] == "P1_UNREG"]
    ax.plot(
        [float(r["rho_pooled"]) for r in norm],
        [cross(r) for r in norm],
        "-o", color=C_NORM, linewidth=1.8, markersize=5.5, zorder=3,
        label="Frobenius control (dose frontier)",
    )
    if unreg:
        ax.plot(
            [float(r["rho_pooled"]) for r in unreg], [cross(r) for r in unreg],
            "s", color=C_UNREG, markersize=7, zorder=4, label="Unregularized",
        )
    if mix:
        ax.plot(
            [float(r["rho_pooled"]) for r in mix], [cross(r) for r in mix],
            "D", color=C_MIX, markersize=8, zorder=5, label="Two-sided mixing",
        )
        target = float(mix[0]["rho_pooled"])
        ax.axvline(target, color=MUTED, linewidth=0.8, linestyle=":", zorder=1)
    conf = [r for r in rows if r["stage"] == "confirmation" and r["task"] == task]
    for condition, color, marker in (("P1_NORM", C_NORM, "^"), ("P1_MIX", C_MIX, "v"), ("P1_UNREG", C_UNREG, ">")):
        pts = [r for r in conf if r["condition"] == condition]
        if pts:
            ax.plot(
                [float(r["rho_pooled"]) for r in pts], [cross(r) for r in pts],
                marker, markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.4,
                markersize=7, linestyle="none", zorder=6,
                label=None,
            )
    ax.set_xscale("log")
    norms = [float(r['rho_pooled']) for r in rows]
    ax.set_xlim(.8 * min(norms), 1.15 * max(norms))
    ax.set_xticks([.02, .05, .1, .2], ["0.02", "0.05", "0.10", "0.20"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(label, fontsize=10, color=INK)
    ax.set_xlabel(r"Achieved pooled relative norm $\rho_F$ (log)", fontsize=8.5, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(True, which="major", color="#e5e5e5", linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
axes[0].set_ylabel("Cross-block energy share (%)", fontsize=8.5, color=INK)
axes[0].set_ylim(bottom=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=8, frameon=False)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"focused_cross_vs_norm.{ext}", dpi=200, bbox_inches="tight")
print(f"figure written to {OUT}/focused_cross_vs_norm.[pdf,png]; "
      f"calibration rows: {sum(r['stage']=='calibration' for r in rows)}, "
      f"confirmation rows: {sum(r['stage']=='confirmation' for r in rows)}")
