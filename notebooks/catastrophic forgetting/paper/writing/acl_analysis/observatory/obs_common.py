"""Shared helpers for the METRIC OBSERVATORY scripts (M1-M4).

Loads master_runs.csv (built by 00_build_master.py, preflight-checked against
key_numbers.md SS18.1) and provides consistent style (figstyle.py palette),
method/family ordering, descriptive-table builders, best-adaptation operating
points, and matched-F_Delta bins.

Statistical conventions respected (analysis_final/09_verification_2026-07-18.md):
- seeds within a recipe cell are correlated (ICC~0.78) -> every mean+-sd states
  what it is over (runs = cells x seeds, vs seeds within one cell);
- per-seed Qwen CE analysis is barred (seed-block deletion) -> no seed-level
  CE splits are produced for qwsw/qwswm;
- spec_max is treated as a MAGNITUDE metric (M2), not geometry.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
OUT = os.path.join(ROOT, "paper", "writing", "acl_analysis", "observatory")
sys.path.insert(0, os.path.join(ROOT, "paper", "writing"))
import figstyle  # noqa: E402

figstyle.apply_rc()

FAMS = ["lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"]
FAM_LABEL = {
    "lrsw": "Llama-2 CS (lrsw)", "lrswm": "Llama-2 Math (lrswm)",
    "qwsw": "Qwen-2.5 CS (qwsw)", "qwswm": "Qwen-2.5 Math (qwswm)",
    "frc": "Llama-2 CS grid (frc)", "frm": "Llama-2 Math grid (frm)",
}
ADAPT_LABEL = {"cs": "Commonsense-8 acc [%]", "math": "GSM8K acc [%]"}

# canonical method order: assessed 10, then withheld CorDA family
METHODS = ["lora", "lora_null", "lorawd", "lorawdr16", "milora", "milorawd",
           "clora", "dora", "sclora", "pissa"]
WITHHELD = ["corda", "cordapp"]
DISPLAY = {
    "lora": "LoRA", "lora_null": "LoRA-Null", "lorawd": "LoRA+wd",
    "lorawdr16": "LoRA+wd-r16", "milora": "MiLoRA", "milorawd": "MiLoRA+wd",
    "clora": "CLoRA", "dora": "DoRA", "sclora": "SC-LoRA", "pissa": "PiSSA",
    "corda": "CorDA (withheld)", "cordapp": "CorDA++ (withheld)",
}
# colors/markers from figstyle.PALETTE; variants borrow the parent hue.
_P = figstyle.PALETTE
COL = {
    "lora": _P["LoRA"][0], "lora_null": _P["LoRA-Null"][0],
    "lorawd": _P["LoRA+wd"][0], "lorawdr16": "#7fd4b4",
    "milora": _P["MiLoRA"][0], "milorawd": "#b58a2a",
    "clora": _P["CLoRA"][0], "dora": _P["DoRA"][0],
    "sclora": _P["SC-LoRA"][0], "pissa": _P["PiSSA"][0],
    "corda": _P["CorDA"][0], "cordapp": "#f0956a",
}
MARK = {
    "lora": _P["LoRA"][1], "lora_null": _P["LoRA-Null"][1],
    "lorawd": _P["LoRA+wd"][1], "lorawdr16": "s",
    "milora": _P["MiLoRA"][1], "milorawd": "^",
    "clora": _P["CLoRA"][1], "dora": _P["DoRA"][1],
    "sclora": _P["SC-LoRA"][1], "pissa": _P["PiSSA"][1],
    "corda": _P["CorDA"][1], "cordapp": "<",
}

NOTE_RUNS = ("sd is over all on-pool runs of the cell (configs x LRs x seeds); "
             "seeds within a cell are correlated (ICC~0.78), so this sd mixes "
             "cell-to-cell and seed noise.")
NOTE_SEEDS = "mean+-sd over seeds within the single best-adaptation cell."


def load_master(on_pool_only=False, keep_withheld=True):
    df = pd.read_csv(os.path.join(OUT, "master_runs.csv"))
    df["family"] = pd.Categorical(df["family"], FAMS, ordered=True)
    order = METHODS + WITHHELD
    df["method"] = pd.Categorical(df["method"], order, ordered=True)
    if on_pool_only:
        df = df[df.on_pool].copy()
    elif not keep_withheld:
        df = df[~df.withheld].copy()
    return df


def pool(df=None):
    df = load_master() if df is None else df
    return df[df.on_pool].copy()


def dist_table(df, value, extra_group=()):
    """Per family x method (+extras) distribution of `value`."""
    g = df.groupby(["family", "method", *extra_group], observed=True)[value]
    t = g.agg(n="count", mean="mean", sd="std", min="min", max="max").reset_index()
    t = t[t.n > 0]
    for c in ["mean", "sd", "min", "max"]:
        t[c] = t[c].round(3)
    return t


def op_points(df, adapt_col="adapt"):
    """Best-adaptation operating point per (family, method): the recipe cell
    (config x LR) whose seed-mean adaptation is highest. Returns the run rows
    of those cells with an op_cell marker column."""
    d = df[df[adapt_col].notna()].copy()
    cm = (d.groupby(["family", "method", "cell"], observed=True)[adapt_col]
            .mean().reset_index())
    best = (cm.sort_values(adapt_col)
              .groupby(["family", "method"], observed=True).tail(1))
    keys = set(zip(best.family, best.method, best.cell))
    d["is_op"] = [
        (f, m, c) in keys for f, m, c in zip(d.family, d.method, d.cell)]
    return d[d.is_op].copy()


def op_table(df, values, adapt_col="adapt"):
    """Summary at the best-adaptation operating point: mean+-sd over seeds
    within that one cell, for each metric in `values`."""
    op = op_points(df, adapt_col)
    agg = {"lr": ("lr", "first"), "cell": ("cell", "first"),
           "n_seeds": ("run", "count")}
    for v in values:
        agg[v + "_mean"] = (v, "mean")
        agg[v + "_sd"] = (v, "std")
    t = (op.groupby(["family", "method"], observed=True)
           .agg(**agg).reset_index())
    num = [c for c in t.select_dtypes("number").columns if c != "lr"]
    t[num] = t[num].round(3)
    t["lr"] = t["lr"].map(lambda v: f"{v:g}" if pd.notna(v) else "")
    return t


def matched_fdelta_bins(df, value, binw=0.5, min_methods=3, min_runs=2):
    """Per family: bin log10 F_Delta into width-`binw` bins; keep bins where
    >= min_methods methods each have >= min_runs runs; report per-method mean
    of `value` inside each kept bin (matched-magnitude comparison)."""
    d = df[df[value].notna() & df.log10_fdelta.notna()].copy()
    d["fd_bin"] = (np.floor(d.log10_fdelta / binw) * binw).round(2)
    g = (d.groupby(["family", "fd_bin", "method"], observed=True)[value]
           .agg(n="count", mean="mean", sd="std").reset_index())
    g = g[g.n >= min_runs]
    keep = (g.groupby(["family", "fd_bin"], observed=True)["method"]
              .nunique().reset_index(name="k"))
    keep = keep[keep.k >= min_methods]
    g = g.merge(keep[["family", "fd_bin"]], on=["family", "fd_bin"])
    for c in ["mean", "sd"]:
        g[c] = g[c].round(3)
    return g


def _md_table(t):
    """Minimal pipe-table renderer (no tabulate dependency)."""
    cols = [str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in t.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_md(t, path, title, note=""):
    with open(path, "w") as fh:
        fh.write(f"# {title}\n\n")
        if note:
            fh.write(f"> {note}\n\n")
        fh.write(_md_table(t))
        fh.write("\n")


def save_fig(fig, stem):
    png = os.path.join(OUT, stem + ".png")
    pdf = os.path.join(OUT, stem + ".pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def methods_present(df):
    order = METHODS + WITHHELD
    have = set(df.method.astype(str))
    return [m for m in order if m in have]


def legend_handles(methods):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], marker=MARK[m], color=COL[m], lw=0, ms=8,
                   markeredgecolor="k", markeredgewidth=0.4,
                   label=DISPLAY[m]) for m in methods]


def lr_curve_grid(df, value, ylabel, stem, title, logy=False,
                  agg="mean", withheld_df=None):
    """Small multiples: `value` vs LR, one panel per family, one line per
    method (seed/config-mean per LR; faint dots = individual runs)."""
    fams = [f for f in FAMS if (df.family == f).any()]
    n = len(fams)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.7 * nrows),
                             squeeze=False)
    for i, fam in enumerate(fams):
        ax = axes[i // ncols][i % ncols]
        sub = df[(df.family == fam) & df[value].notna() & df.lr.notna()]
        for m in methods_present(sub):
            ms = sub[sub.method == m]
            line = ms.groupby("lr")[value].agg(agg)
            ax.plot(line.index, line.values, "-", color=COL[m], lw=1.6,
                    marker=MARK[m], ms=6, markeredgecolor="k",
                    markeredgewidth=0.35, alpha=0.95, zorder=4)
            ax.scatter(ms.lr, ms[value], s=9, color=COL[m], alpha=0.25,
                       linewidths=0, zorder=2)
        if withheld_df is not None:
            ws = withheld_df[(withheld_df.family == fam)
                             & withheld_df[value].notna()
                             & withheld_df.lr.notna()]
            if len(ws):
                ax.scatter(ws.lr, ws[value], s=26, facecolors="none",
                           edgecolors=COL["corda"], linewidths=1.0,
                           marker="<", zorder=3,
                           label="CorDA/CorDA++ (withheld)")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(FAM_LABEL[fam], fontsize=11)
        ax.set_xlabel("learning rate")
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, y=1.0, fontsize=13)
    hs = legend_handles(methods_present(df))
    fig.legend(handles=hs, loc="lower center", ncol=min(len(hs), 6),
               bbox_to_anchor=(0.5, -0.015), fontsize=8.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    return save_fig(fig, stem)


def box_grid(df, value, ylabel, stem, title, logy=False, withheld_df=None):
    """Per-family panels: box + strip of `value` per method."""
    fams = [f for f in FAMS if (df.family == f).any()]
    ncols = 3
    nrows = int(np.ceil(len(fams) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.8 * nrows),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for i, fam in enumerate(fams):
        ax = axes[i // ncols][i % ncols]
        sub = df[(df.family == fam) & df[value].notna()]
        meths = methods_present(sub)
        if withheld_df is not None:
            wsub = withheld_df[(withheld_df.family == fam)
                               & withheld_df[value].notna()]
            wmeths = methods_present(wsub)
        else:
            wsub, wmeths = None, []
        allm = meths + wmeths
        data = []
        for m in meths:
            data.append(sub.loc[sub.method == m, value].values)
        for m in wmeths:
            data.append(wsub.loc[wsub.method == m, value].values)
        if not data:
            ax.axis("off")
            continue
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        widths=0.62, medianprops=dict(color="k", lw=1.2))
        for patch, m in zip(bp["boxes"], allm):
            patch.set_facecolor(COL[m])
            patch.set_alpha(0.30 if m not in WITHHELD else 0.15)
            patch.set_edgecolor(COL[m])
            if m in WITHHELD:
                patch.set_hatch("///")
        for k, (vals, m) in enumerate(zip(data, allm)):
            jit = rng.normal(0, 0.06, len(vals))
            ax.scatter(np.full(len(vals), k + 1) + jit, vals, s=12,
                       color=COL[m], edgecolor="k", linewidth=0.25,
                       alpha=0.75, zorder=5)
        ax.set_xticks(range(1, len(allm) + 1))
        ax.set_xticklabels([DISPLAY[m].replace(" (withheld)", "*")
                            .replace("CorDA++*", "CorDA++*")
                            for m in allm], rotation=40, ha="right",
                           fontsize=8)
        if logy:
            ax.set_yscale("log")
        ax.set_title(FAM_LABEL[fam], fontsize=11)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    for j in range(len(fams), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    star = "  (* = CorDA/CorDA++, withheld from assessment)" if withheld_df is not None else ""
    fig.suptitle(title + star, y=1.0, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_fig(fig, stem)
