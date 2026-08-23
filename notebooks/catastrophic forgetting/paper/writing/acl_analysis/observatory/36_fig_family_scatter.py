"""Per-family retention-vs-magnitude scatter (appendix exhibit).

Six panels, one per run family (lrsw, lrswm, qwsw, qwswm, frc, frm):
every on-pool run as a dot (figstyle method colour + marker), a
binned-median trend line, and a dashed base-model ceiling line.

Data: observatory/m3_master.csv, on_pool rows only. (Verified identical,
per family, to insights/pool.csv: same n and the same Pearson r between
log10 F_Delta and retention in all six families.)

Ceilings, all verified against main.tex "Retention benchmarks" paragraph:
  Llama-2-7B core (BBH+MMLU-Pro mean) = 26.0  -> all four Llama families
  Qwen2.5-7B core (BBH+MMLU-Pro mean) = 44.4  -> both Qwen families
  (retention_mean is CORE retention for every family, math included: the
  per-family maxima are lrswm/frm 26.4 and qwswm 45.9, far below the BBH-only
  ceilings 33.1/47.9, so the BBH-only math ceilings do not apply to this axis;
  the BBH-only convention governs operating-point exhibits, not this pool.)

Outputs: figures/fig_family_scatter.{png,pdf} (PNG at 300 dpi) and prints
per-family n and Pearson r for verification against the paper's frozen
correlations.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WRITING = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, WRITING)
import figstyle  # noqa: E402

figstyle.apply_rc()

FIGDIR = os.path.join(WRITING, "figures")
MASTER = os.path.join(HERE, "m3_master.csv")

FAMS = ["lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"]
FAM_TITLE = {
    "lrsw":  "Llama-2 CS sweep (lrsw)",
    "lrswm": "Llama-2 math sweep (lrswm)",
    "qwsw":  "Qwen2.5 CS sweep (qwsw)",
    "qwswm": "Qwen2.5 math sweep (qwswm)",
    "frc":   "Llama-2 CS faithful grid (frc)",
    "frm":   "Llama-2 math faithful grid (frm)",
}
# base-model ceilings (see module docstring for provenance)
CEILING = {"lrsw": 26.0, "frc": 26.0, "lrswm": 26.0, "frm": 26.0,
           "qwsw": 44.4, "qwswm": 44.4}

# method colour/marker: figstyle palette; +wd / r16 variants borrow the
# parent hue (same convention as obs_common.py)
_P = figstyle.PALETTE
STYLE = {
    "LoRA":        _P["LoRA"],
    "LoRA+wd":     _P["LoRA+wd"],
    "LoRA+wd-r16": ("#7fd4b4", "s"),
    "LoRA-Null":   _P["LoRA-Null"],
    "MiLoRA":      _P["MiLoRA"],
    "MiLoRA+wd":   ("#b58a2a", "^"),
    "CLoRA":       _P["CLoRA"],
    "DoRA":        _P["DoRA"],
    "SC-LoRA":     _P["SC-LoRA"],
    "PiSSA":       _P["PiSSA"],
}
METHOD_ORDER = list(STYLE)


def binned_median(logfd, ret, nbins=9):
    """Quantile-binned median trend: (bin-median x, bin-median y) pairs."""
    q = np.quantile(logfd, np.linspace(0, 1, nbins + 1))
    q = np.unique(q)
    idx = np.clip(np.digitize(logfd, q[1:-1]), 0, len(q) - 2)
    xs, ys = [], []
    for b in range(len(q) - 1):
        m = idx == b
        if m.sum() >= 3:
            xs.append(np.median(logfd[m]))
            ys.append(np.median(ret[m]))
    return np.array(xs), np.array(ys)


def main():
    df = pd.read_csv(MASTER)
    df = df[df.on_pool].dropna(subset=["log10_fdelta", "retention_mean"]).copy()

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharex=False)

    print("family  n     r")
    for ax, fam in zip(axes.flat, FAMS):
        g = df[df.family == fam]
        r = np.corrcoef(g.log10_fdelta, g.retention_mean)[0, 1]
        print(f"{fam:6s}  {len(g):4d}  {r:+.3f}")

        for meth in METHOD_ORDER:
            gm = g[g.method_display == meth]
            if gm.empty:
                continue
            c, mk = STYLE[meth]
            ax.scatter(gm.log10_fdelta, gm.retention_mean, s=16, marker=mk,
                       c=c, alpha=0.55, linewidths=0, zorder=2)

        bx, by = binned_median(g.log10_fdelta.values, g.retention_mean.values)
        ax.plot(bx, by, color=figstyle.FIT_C, lw=1.8, zorder=3)

        ax.axhline(CEILING[fam], color=figstyle.CEILING_C, ls="--", lw=1.1,
                   zorder=1)

        ax.set_title(FAM_TITLE[fam], fontsize=11)
        ax.text(0.03, 0.06, f"n = {len(g)},  r = {r:.2f}",
                transform=ax.transAxes, fontsize=9, color=figstyle.INK2)

    for ax in axes[1]:
        ax.set_xlabel(r"$\log_{10}\,F_\Delta$")
    for ax in axes[:, 0]:
        ax.set_ylabel("retention")

    handles = [Line2D([], [], ls="none", marker=STYLE[m][1],
                      color=STYLE[m][0], markersize=6, label=m)
               for m in METHOD_ORDER
               if (df.method_display == m).any()]
    handles += [
        Line2D([], [], color=figstyle.FIT_C, lw=1.8, label="binned median"),
        Line2D([], [], color=figstyle.CEILING_C, ls="--", lw=1.1,
               label="base ceiling"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 6), bbox_to_anchor=(0.5, -0.045),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    os.makedirs(FIGDIR, exist_ok=True)
    png = os.path.join(FIGDIR, "fig_family_scatter.png")
    pdf = os.path.join(FIGDIR, "fig_family_scatter.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
