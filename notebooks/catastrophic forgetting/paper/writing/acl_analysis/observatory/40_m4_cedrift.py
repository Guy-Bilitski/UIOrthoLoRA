#!/usr/bin/env python
"""METRIC OBSERVATORY M4 — CE drift (base-corpus forgetting).

Metrics: forgetting_ce, base_entropy, forgetting_kl
(identity: kl = ce - base_entropy; checked here).

Statistical guardrails: CE coverage on Qwen sweeps is ~62% with a SEED-BLOCK
deletion pattern -> per-seed Qwen CE analysis is barred; all Qwen CE numbers
below are pooled over whatever seeds have CE, never split by seed.

Outputs: m4_master.csv, m4_dist_all.{csv,md}, m4_op_points.{csv,md},
m4_matched_fdelta_kl.{csv,md}, m4_fig_kl_vs_lr, m4_fig_box_kl,
m4_fig_kl_vs_retention.
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
M4_COLS = ["forgetting_ce", "base_entropy", "forgetting_kl",
           "log10_fdelta", "retention_mean"]


def main():
    df = oc.load_master()
    pool = df[df.on_pool].copy()
    wh = df[df.withheld].copy()

    df[ID_COLS + M4_COLS].to_csv(os.path.join(OUT, "m4_master.csv"),
                                 index=False)

    # ---------- identity + coverage checks ----------
    d = pool.dropna(subset=["forgetting_ce", "base_entropy", "forgetting_kl"])
    dev = (d.forgetting_ce - d.base_entropy - d.forgetting_kl).abs().max()
    print(f"[identity] max |ce - H0 - kl| = {dev:.2e} (n={len(d)})")
    cov = pool.groupby("family", observed=True)["forgetting_kl"].agg(
        n_pool="size", n_ce="count")
    cov["pct"] = (100 * cov.n_ce / cov.n_pool).round(1)
    print("[coverage] CE per family (on-pool):\n" + cov.to_string())
    print("[base entropy] per family mean±sd over runs:")
    print(pool.groupby("family", observed=True)["base_entropy"]
              .agg(["mean", "std"]).round(4).to_string())

    # ---------- (a) distributions ----------
    parts = []
    for v in ["forgetting_kl", "forgetting_ce", "base_entropy"]:
        t = oc.dist_table(pool, v)
        t.insert(0, "metric", v)
        parts.append(t)
        tw = oc.dist_table(wh, v)
        if len(tw):
            tw.insert(0, "metric", v)
            tw["method"] = tw["method"].astype(str) + " [WITHHELD]"
            parts.append(tw)
    dist = pd.concat(parts, ignore_index=True)
    dist.to_csv(os.path.join(OUT, "m4_dist_all.csv"), index=False)
    oc.write_md(dist, os.path.join(OUT, "m4_dist_all.md"),
                "M4 CE-drift distributions per method x family "
                "(all on-pool runs with CE; CorDA flagged WITHHELD)",
                oc.NOTE_RUNS + " Qwen families: ~60% CE coverage with a "
                "seed-block deletion pattern -- pooled only, never per-seed.")

    # ---------- (b) op points ----------
    opt = oc.op_table(pool, ["adapt", "forgetting_kl", "forgetting_ce"])
    opt = opt.sort_values(["family", "forgetting_kl_mean"])
    opt.to_csv(os.path.join(OUT, "m4_op_points.csv"), index=False)
    oc.write_md(opt, os.path.join(OUT, "m4_op_points.md"),
                "M4 CE drift at each method's best-adaptation operating "
                "point", oc.NOTE_SEEDS + " Qwen cells may have <3 seeds "
                "with CE (seed-block deletion) -- treat sd as indicative.")

    # ---------- (c) matched F_Delta bins ----------
    mb = oc.matched_fdelta_bins(pool, "forgetting_kl")
    mb.to_csv(os.path.join(OUT, "m4_matched_fdelta_kl.csv"), index=False)
    piv = mb.pivot_table(index=["family", "fd_bin"], columns="method",
                         values="mean", observed=True).round(3).reset_index()
    oc.write_md(piv, os.path.join(OUT, "m4_matched_fdelta_kl.md"),
                "M4 mean forgetting_kl per method inside matched "
                "log10(F_Delta) bins",
                "CE drift at matched update magnitude.")

    # ---------- correlations ----------
    print("\n[corr] per family (on-pool, CE subset):")
    for fam in oc.FAMS:
        s = pool[(pool.family == fam)].dropna(
            subset=["forgetting_kl", "retention_mean", "log10_fdelta"])
        lkl = np.log10(s.forgetting_kl.clip(lower=1e-6))
        r_kr = np.corrcoef(s.forgetting_kl, s.retention_mean)[0, 1]
        r_lkr = np.corrcoef(lkl, s.retention_mean)[0, 1]
        r_kf = np.corrcoef(lkl, s.log10_fdelta)[0, 1]
        print(f"  {fam:6s} n={len(s):3d}  r(kl,ret)={r_kr:+.3f}  "
              f"r(log kl,ret)={r_lkr:+.3f}  r(log kl,log fd)={r_kf:+.3f}")

    # ---------- figures ----------
    print("[fig]", oc.lr_curve_grid(
        pool, "forgetting_kl", "KL(FT || base) on base corpus [nats]",
        "m4_fig_kl_vs_lr", "M4  CE drift (KL to base) vs learning rate",
        logy=True, withheld_df=wh))
    print("[fig]", oc.box_grid(
        pool, "forgetting_kl", "KL(FT || base) [nats]", "m4_fig_box_kl",
        "M4  CE-drift (KL) distribution per method (all on-pool runs "
        "with CE)", logy=True, withheld_df=wh))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    for i, fam in enumerate(oc.FAMS):
        ax = axes[i // 3][i % 3]
        sub = pool[(pool.family == fam) & pool.forgetting_kl.notna()
                   & pool.retention_mean.notna()]
        for m in oc.methods_present(sub):
            ms = sub[sub.method == m]
            ax.scatter(ms.forgetting_kl, ms.retention_mean, s=22,
                       color=oc.COL[m], marker=oc.MARK[m], edgecolor="k",
                       linewidth=0.25, alpha=0.8, zorder=4)
        if len(sub) > 3:
            lkl = np.log10(sub.forgetting_kl.clip(lower=1e-6))
            r = np.corrcoef(lkl, sub.retention_mean)[0, 1]
            ax.text(0.03, 0.05, f"r(log KL, ret)={r:+.2f}\nn={len(sub)}",
                    transform=ax.transAxes, va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="0.6"))
        ax.set_xscale("log")
        ax.set_title(oc.FAM_LABEL[fam], fontsize=11)
        ax.set_xlabel("KL(FT || base) [nats, log]")
        if i % 3 == 0:
            ax.set_ylabel("Retention core [%]")
    fig.suptitle("M4  CE drift vs benchmark retention (Qwen panels: pooled "
                 "over available seeds; ~60% CE coverage)", y=1.0,
                 fontsize=13)
    hs = oc.legend_handles(oc.methods_present(pool))
    fig.legend(handles=hs, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.012), fontsize=8.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.965])
    print("[fig]", oc.save_fig(fig, "m4_fig_kl_vs_retention"))


if __name__ == "__main__":
    main()
