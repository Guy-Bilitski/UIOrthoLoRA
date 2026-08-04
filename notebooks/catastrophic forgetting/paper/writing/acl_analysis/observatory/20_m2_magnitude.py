#!/usr/bin/env python
"""METRIC OBSERVATORY M2 — update-magnitude family.

Primary metric: fdelta (CLoRA Eq-3 effective update magnitude F_Delta — NOT a
Frobenius norm). Family: fro_total, spec_max, dw_sv_max (dw_sv_mean/spec_mean
kept in the master CSV). spec_max is treated as MAGNITUDE (verification memo:
r ~ +0.93 with log F_Delta), not geometry.

Outputs: m2_master.csv, m2_dist_all.{csv,md}, m2_op_points.{csv,md},
m2_corr_structure.{csv,md}, m2_matched_fdelta_specmax.{csv,md},
m2_fig_fdelta_vs_lr, m2_fig_box_fdelta, m2_fig_specmax_vs_fdelta.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import obs_common as oc

OUT = oc.OUT
ID_COLS = ["run", "family", "family_label", "model", "task", "recipe",
           "method", "method_display", "lr", "seed", "cell", "withheld",
           "quarantined", "on_pool"]
M2_COLS = ["fdelta", "log10_fdelta", "fro_total", "log10_fro_total",
           "spec_max", "log10_spec_max", "spec_mean", "dw_sv_max",
           "log10_dw_sv_max", "dw_sv_mean", "retention_mean"]


def main():
    df = oc.load_master()
    pool = df[df.on_pool].copy()
    wh = df[df.withheld].copy()

    df[ID_COLS + M2_COLS].to_csv(os.path.join(OUT, "m2_master.csv"),
                                 index=False)

    # ---------- (a) distributions ----------
    parts = []
    for v in ["log10_fdelta", "log10_fro_total", "log10_spec_max",
              "log10_dw_sv_max"]:
        t = oc.dist_table(pool, v)
        t.insert(0, "metric", v)
        parts.append(t)
        tw = oc.dist_table(wh, v)
        if len(tw):
            tw.insert(0, "metric", v)
            tw["method"] = tw["method"].astype(str) + " [WITHHELD]"
            parts.append(tw)
    dist = pd.concat(parts, ignore_index=True)
    dist.to_csv(os.path.join(OUT, "m2_dist_all.csv"), index=False)
    oc.write_md(dist, os.path.join(OUT, "m2_dist_all.md"),
                "M2 magnitude-family distributions per method x family "
                "(log10 units; all on-pool runs; CorDA flagged WITHHELD)",
                oc.NOTE_RUNS)

    # ---------- (b) op points ----------
    opt = oc.op_table(pool, ["adapt", "fdelta", "spec_max", "fro_total",
                             "dw_sv_max"])
    opt = opt.sort_values(["family", "fdelta_mean"])
    opt.to_csv(os.path.join(OUT, "m2_op_points.csv"), index=False)
    oc.write_md(opt, os.path.join(OUT, "m2_op_points.md"),
                "M2 magnitude at each method's best-adaptation operating "
                "point", oc.NOTE_SEEDS)

    # ---------- correlation structure of the magnitude family ----------
    mags = ["log10_fdelta", "log10_fro_total", "log10_spec_max",
            "log10_dw_sv_max"]
    rows = []
    scopes = [("pooled", pool)] + [(f, pool[pool.family == f])
                                   for f in oc.FAMS]
    for name, s in scopes:
        row = {"scope": name, "n": len(s)}
        for m2 in mags[1:]:
            ok = s[mags[0]].notna() & s[m2].notna()
            row[f"r(logFd,{m2.replace('log10_', '')})"] = round(
                np.corrcoef(s.loc[ok, mags[0]], s.loc[ok, m2])[0, 1], 3)
        for m in mags:
            ok = s[m].notna() & s.retention_mean.notna()
            row[f"r(ret,{m.replace('log10_', 'log ')})"] = round(
                np.corrcoef(s.loc[ok, m], s.loc[ok, "retention_mean"])[0, 1], 3)
        rows.append(row)
    corr = pd.DataFrame(rows)
    corr.to_csv(os.path.join(OUT, "m2_corr_structure.csv"), index=False)
    oc.write_md(corr, os.path.join(OUT, "m2_corr_structure.md"),
                "M2 correlation structure of the magnitude family "
                "(Pearson on on-pool runs)",
                "spec_max tracks log F_Delta (a magnitude metric, per the "
                "verification memo), and F_Delta is the most retention-"
                "predictive axis.")
    print("[corr structure]\n" + corr.to_string(index=False))

    # ---------- (c) spec_max shape at matched F_Delta ----------
    mb = oc.matched_fdelta_bins(pool, "log10_spec_max")
    mb.to_csv(os.path.join(OUT, "m2_matched_fdelta_specmax.csv"), index=False)
    piv = mb.pivot_table(index=["family", "fd_bin"], columns="method",
                         values="mean", observed=True).round(2).reset_index()
    oc.write_md(piv, os.path.join(OUT, "m2_matched_fdelta_specmax.md"),
                "M2 mean log10 spec_max per method inside matched "
                "log10(F_Delta) bins",
                "How 'spiky' each method's update is at the same effective "
                "magnitude.")

    # ---------- transmission: fdelta per method at shared LRs ----------
    print("\n[transmission] seed/config-mean log10 F_Delta at lr=1e-4 / 3e-4:")
    for fam in oc.FAMS:
        s = pool[(pool.family == fam)]
        for lr in (1e-4, 3e-4):
            sl = s[np.isclose(s.lr, lr)]
            if not len(sl):
                continue
            m = sl.groupby("method", observed=True)["log10_fdelta"].mean()
            m = m.sort_values()
            txt = ", ".join(f"{k}={v:.2f}" for k, v in m.items())
            print(f"  {fam:6s} lr={lr:g}: {txt}")

    # ---------- figures ----------
    print("[fig]", oc.lr_curve_grid(
        pool, "fdelta", r"$F_\Delta$ (log)", "m2_fig_fdelta_vs_lr",
        "M2  Effective update magnitude $F_\\Delta$ vs learning rate "
        "(same LR $\\to$ different magnitude per method)", logy=True,
        withheld_df=wh))
    print("[fig]", oc.box_grid(
        pool, "fdelta", r"$F_\Delta$ (log)", "m2_fig_box_fdelta",
        "M2  $F_\\Delta$ distribution per method (all on-pool runs)",
        logy=True, withheld_df=wh))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    for i, fam in enumerate(oc.FAMS):
        ax = axes[i // 3][i % 3]
        sub = pool[(pool.family == fam) & pool.log10_spec_max.notna()]
        for m in oc.methods_present(sub):
            ms = sub[sub.method == m]
            ax.scatter(ms.log10_fdelta, ms.log10_spec_max, s=22,
                       color=oc.COL[m], marker=oc.MARK[m], edgecolor="k",
                       linewidth=0.25, alpha=0.8, zorder=4)
        ok = sub.log10_fdelta.notna() & sub.log10_spec_max.notna()
        r = np.corrcoef(sub.log10_fdelta[ok], sub.log10_spec_max[ok])[0, 1]
        ax.text(0.03, 0.95, f"r={r:+.2f}", transform=ax.transAxes,
                va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
        ax.set_title(oc.FAM_LABEL[fam], fontsize=11)
        ax.set_xlabel(r"log10 $F_\Delta$")
        if i % 3 == 0:
            ax.set_ylabel(r"log10 spec_max($\Delta W$)")
    fig.suptitle("M2  spec_max rides the magnitude axis "
                 "(spec_max is magnitude, not geometry)", y=1.0, fontsize=13)
    hs = oc.legend_handles(oc.methods_present(pool))
    fig.legend(handles=hs, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.012), fontsize=8.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.965])
    print("[fig]", oc.save_fig(fig, "m2_fig_specmax_vs_fdelta"))


if __name__ == "__main__":
    main()
