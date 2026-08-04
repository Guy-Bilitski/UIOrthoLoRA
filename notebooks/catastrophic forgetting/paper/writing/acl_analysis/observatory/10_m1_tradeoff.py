#!/usr/bin/env python
"""METRIC OBSERVATORY M1 — retention / adaptation trade-off.

Metrics: retention_mean (core = mean BBH + MMLU-Pro), retention_broad
(secondary), adapt (cs_avg for CS families, GSM8K for math families).

Outputs (OUTDIR): m1_master.csv, m1_dist_all.{csv,md}, m1_op_points.{csv,md},
m1_matched_fdelta.{csv,md}, m1_seed_noise.{csv,md},
m1_fig_scatter_tradeoff.{png,pdf}, m1_fig_retention_vs_lr.{png,pdf},
m1_fig_adapt_vs_lr.{png,pdf}, m1_fig_box_retention.{png,pdf}.
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
M1_COLS = ["retention_mean", "retention_broad", "adapt", "bbh", "mmlu_pro",
           "mmlu", "arc_c", "truthfulqa", "fdelta", "log10_fdelta"]


def main():
    df = oc.load_master()
    pool = df[df.on_pool].copy()
    wh = df[df.withheld].copy()

    df[ID_COLS + M1_COLS].to_csv(os.path.join(OUT, "m1_master.csv"), index=False)

    # ---------- (a) distributions over all on-pool runs ----------
    parts = []
    for v in ["retention_mean", "retention_broad", "adapt"]:
        t = oc.dist_table(pool, v)
        t.insert(0, "metric", v)
        parts.append(t)
        tw = oc.dist_table(wh, v)
        if len(tw):
            tw.insert(0, "metric", v)
            tw["method"] = tw["method"].astype(str) + " [WITHHELD]"
            parts.append(tw)
    dist = pd.concat(parts, ignore_index=True)
    dist.to_csv(os.path.join(OUT, "m1_dist_all.csv"), index=False)
    oc.write_md(dist, os.path.join(OUT, "m1_dist_all.md"),
                "M1 distributions per method x family (all on-pool runs; "
                "CorDA rows flagged WITHHELD)", oc.NOTE_RUNS)

    # ---------- (b) best-adaptation operating points ----------
    opt = oc.op_table(pool, ["adapt", "retention_mean", "retention_broad",
                             "fdelta"])
    opt = opt.sort_values(["family", "adapt_mean"],
                          ascending=[True, False])
    opt.to_csv(os.path.join(OUT, "m1_op_points.csv"), index=False)
    oc.write_md(opt, os.path.join(OUT, "m1_op_points.md"),
                "M1 best-adaptation operating point per method x family",
                oc.NOTE_SEEDS + " Op point = recipe cell (config x LR) with "
                "highest seed-mean adaptation.")

    # ---------- (c) matched F_Delta bins ----------
    mb = oc.matched_fdelta_bins(pool, "retention_mean")
    mb.to_csv(os.path.join(OUT, "m1_matched_fdelta.csv"), index=False)
    piv = mb.pivot_table(index=["family", "fd_bin"], columns="method",
                         values="mean", observed=True).round(2).reset_index()
    oc.write_md(piv, os.path.join(OUT, "m1_matched_fdelta.md"),
                "M1 mean retention_core per method inside matched "
                "log10(F_Delta) bins (width 0.5; bins with >=3 methods, "
                ">=2 runs/method)",
                "Comparing methods at matched update magnitude. sd per cell "
                "in the csv.")

    # ---------- seed noise ----------
    rows = []
    for v in ["retention_mean", "adapt"]:
        cs = (pool.groupby(["family", "cell"], observed=True)[v]
                  .agg(["count", "std"]).reset_index())
        cs = cs[cs["count"] >= 2]
        s = cs.groupby("family", observed=True)["std"].agg(
            n_cells="count", mean_sd="mean", median_sd="median").reset_index()
        s.insert(0, "metric", v)
        rows.append(s)
    sn = pd.concat(rows, ignore_index=True).round(3)
    sn.to_csv(os.path.join(OUT, "m1_seed_noise.csv"), index=False)
    oc.write_md(sn, os.path.join(OUT, "m1_seed_noise.md"),
                "M1 within-cell seed noise (sd over seeds inside each recipe "
                "cell, averaged over cells with >=2 seeds)",
                "Cross-check vs key_numbers.md SS18.1 within-cell SD(ret).")
    print("[seed noise]\n" + sn.to_string(index=False))

    # ---------- console stats for findings ----------
    print("\n[corr] run-level Pearson r per family (on-pool):")
    for fam in oc.FAMS:
        s = pool[pool.family == fam]
        ok = s.adapt.notna() & s.retention_mean.notna()
        r_ar = np.corrcoef(s.adapt[ok], s.retention_mean[ok])[0, 1]
        okl = s.lr.notna() & s.retention_mean.notna()
        r_lr = np.corrcoef(np.log10(s.lr[okl]), s.retention_mean[okl])[0, 1]
        r_fd = np.corrcoef(s.log10_fdelta, s.retention_mean)[0, 1]
        print(f"  {fam:6s} r(adapt,ret)={r_ar:+.3f}  r(log lr,ret)={r_lr:+.3f}"
              f"  r(log fd,ret)={r_fd:+.3f}  n={len(s)}")

    # ---------- figure: adaptation vs retention scatter ----------
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    for i, fam in enumerate(oc.FAMS):
        ax = axes[i // 3][i % 3]
        sub = pool[(pool.family == fam) & pool.adapt.notna()
                   & pool.retention_mean.notna()]
        for m in oc.methods_present(sub):
            ms = sub[sub.method == m]
            ax.scatter(ms.adapt, ms.retention_mean, s=26, color=oc.COL[m],
                       marker=oc.MARK[m], edgecolor="k", linewidth=0.3,
                       alpha=0.8, zorder=4)
        wsub = wh[(wh.family == fam) & wh.adapt.notna()
                  & wh.retention_mean.notna()]
        if len(wsub):
            ax.scatter(wsub.adapt, wsub.retention_mean, s=30,
                       facecolors="none", edgecolors=oc.COL["corda"],
                       marker="<", linewidths=1.0, zorder=3)
        # star each method's best-adapt op point
        ops = oc.op_points(sub)
        opm = ops.groupby("method", observed=True)[
            ["adapt", "retention_mean"]].mean()
        for m, row in opm.iterrows():
            ax.scatter([row.adapt], [row.retention_mean], s=190, marker="*",
                       color=oc.COL[m], edgecolor="k", linewidth=0.7,
                       zorder=6)
        task = sub.task.iloc[0] if len(sub) else "cs"
        ax.set_title(oc.FAM_LABEL[fam], fontsize=11)
        ax.set_xlabel(oc.ADAPT_LABEL[task])
        if i % 3 == 0:
            ax.set_ylabel("Retention core (BBH+MMLU-Pro mean) [%]")
    fig.suptitle("M1  Adaptation vs retention, all on-pool runs "
                 "(star = best-adaptation cell; open '<' = CorDA, withheld)",
                 y=1.0, fontsize=13)
    hs = oc.legend_handles(oc.methods_present(pool))
    fig.legend(handles=hs, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.012), fontsize=8.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.965])
    print("[fig]", oc.save_fig(fig, "m1_fig_scatter_tradeoff"))

    # ---------- figures: metric vs LR; boxes ----------
    print("[fig]", oc.lr_curve_grid(
        pool, "retention_mean", "Retention core [%]",
        "m1_fig_retention_vs_lr",
        "M1  Retention core vs learning rate (line = mean over runs at that "
        "LR; dots = runs)", withheld_df=wh))
    print("[fig]", oc.lr_curve_grid(
        pool, "adapt", "Adaptation [%]", "m1_fig_adapt_vs_lr",
        "M1  Adaptation vs learning rate (line = mean over runs at that LR; "
        "dots = runs)", withheld_df=wh))
    print("[fig]", oc.box_grid(
        pool, "retention_mean", "Retention core [%]", "m1_fig_box_retention",
        "M1  Retention-core distribution per method (all on-pool runs)",
        withheld_df=wh))


if __name__ == "__main__":
    main()
