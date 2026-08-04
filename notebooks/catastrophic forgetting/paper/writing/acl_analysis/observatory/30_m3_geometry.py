#!/usr/bin/env python
"""METRIC OBSERVATORY M3 — geometry / major-minor component structure.

Metrics: stable_rank_w, eff_rank_w, e_top_w vs e_bot_w (energy split),
amp_top_w. Framing guardrail: magnitude first, geometry second — geometry is
a real but second-order axis (key_numbers.md SS19.1: dR2 +0.017 after
magnitude; SS19.2: e_top/stable_rank seed-stable, spec_max-based direction
effects are not).

Outputs: m3_master.csv, m3_dist_all.{csv,md}, m3_op_points.{csv,md},
m3_matched_fdelta_geom.{csv,md}, m3_residual_corr.{csv,md},
m3_fig_stablerank_vs_lr, m3_fig_box_stablerank, m3_fig_box_etop,
m3_fig_etop_vs_ebot.
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
GEOM = ["stable_rank_w", "eff_rank_w", "e_top_w", "e_bot_w", "amp_top_w"]
M3_COLS = GEOM + ["log10_fdelta", "retention_mean"]


def residualize(y, X):
    """OLS residual of y on X (with intercept)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def main():
    df = oc.load_master()
    pool = df[df.on_pool].copy()
    wh = df[df.withheld].copy()

    df[ID_COLS + M3_COLS].to_csv(os.path.join(OUT, "m3_master.csv"),
                                 index=False)

    # ---------- (a) distributions ----------
    parts = []
    for v in GEOM:
        t = oc.dist_table(pool, v)
        t.insert(0, "metric", v)
        parts.append(t)
        tw = oc.dist_table(wh, v)
        if len(tw):
            tw.insert(0, "metric", v)
            tw["method"] = tw["method"].astype(str) + " [WITHHELD]"
            parts.append(tw)
    dist = pd.concat(parts, ignore_index=True)
    dist.to_csv(os.path.join(OUT, "m3_dist_all.csv"), index=False)
    oc.write_md(dist, os.path.join(OUT, "m3_dist_all.md"),
                "M3 geometry distributions per method x family "
                "(all on-pool runs; CorDA flagged WITHHELD)", oc.NOTE_RUNS)

    # ---------- (b) op points ----------
    opt = oc.op_table(pool, ["adapt", "stable_rank_w", "eff_rank_w",
                             "e_top_w", "e_bot_w", "amp_top_w"])
    opt = opt.sort_values(["family", "stable_rank_w_mean"])
    opt.to_csv(os.path.join(OUT, "m3_op_points.csv"), index=False)
    oc.write_md(opt, os.path.join(OUT, "m3_op_points.md"),
                "M3 geometry at each method's best-adaptation operating "
                "point", oc.NOTE_SEEDS)

    # ---------- (c) matched F_Delta bins: the shape fingerprint ----------
    parts = []
    for v in ["stable_rank_w", "e_top_w"]:
        mb = oc.matched_fdelta_bins(pool, v)
        mb.insert(0, "metric", v)
        parts.append(mb)
    mbl = pd.concat(parts, ignore_index=True)
    mbl.to_csv(os.path.join(OUT, "m3_matched_fdelta_geom.csv"), index=False)
    piv = (mbl[mbl.metric == "stable_rank_w"]
           .pivot_table(index=["family", "fd_bin"], columns="method",
                        values="mean", observed=True).round(2).reset_index())
    oc.write_md(piv, os.path.join(OUT, "m3_matched_fdelta_geom.md"),
                "M3 mean stable_rank_w per method inside matched "
                "log10(F_Delta) bins (e_top_w rows in the csv)",
                "The update-shape fingerprint at matched magnitude.")

    # ---------- second-order strength: raw vs magnitude-residualized ----------
    rows = []
    for fam in oc.FAMS:
        s = pool[(pool.family == fam)].dropna(
            subset=["retention_mean", "log10_fdelta"] + GEOM)
        res_ret = residualize(s.retention_mean.values,
                              s.log10_fdelta.values[:, None])
        row = {"family": fam, "n": len(s)}
        for v in GEOM:
            r_raw = np.corrcoef(s[v], s.retention_mean)[0, 1]
            res_v = residualize(s[v].values, s.log10_fdelta.values[:, None])
            r_par = np.corrcoef(res_v, res_ret)[0, 1]
            row[f"r_raw({v})"] = round(r_raw, 3)
            row[f"r_partial({v}|logFd)"] = round(r_par, 3)
        rows.append(row)
    rc = pd.DataFrame(rows)
    rc.to_csv(os.path.join(OUT, "m3_residual_corr.csv"), index=False)
    oc.write_md(rc, os.path.join(OUT, "m3_residual_corr.md"),
                "M3 geometry vs retention: raw r and partial r after "
                "removing log10 F_Delta (per family, run level)",
                "Second-order check: geometry adds little once magnitude is "
                "controlled (key_numbers SS19.1: dR2 +0.017). Partial rs "
                "here are descriptive; seeds within cells are correlated.")
    print("[residual corr]\n" + rc.to_string(index=False))

    # ---------- method ordering of stable rank (pooled fingerprint) ----------
    print("\n[fingerprint] pooled median stable_rank_w per method (on-pool):")
    med = (pool.groupby("method", observed=True)["stable_rank_w"]
               .median().sort_values())
    print(med.round(2).to_string())

    # ---------- figures ----------
    print("[fig]", oc.lr_curve_grid(
        pool, "stable_rank_w", "stable rank of $\\Delta W$",
        "m3_fig_stablerank_vs_lr",
        "M3  Stable rank vs learning rate", withheld_df=wh))
    print("[fig]", oc.box_grid(
        pool, "stable_rank_w", "stable rank of $\\Delta W$",
        "m3_fig_box_stablerank",
        "M3  Stable-rank distribution per method (update-shape fingerprint)",
        withheld_df=wh))
    print("[fig]", oc.box_grid(
        pool, "e_top_w", "top-component energy share $e_{top}$",
        "m3_fig_box_etop",
        "M3  Major-component energy share per method", withheld_df=wh))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    for i, fam in enumerate(oc.FAMS):
        ax = axes[i // 3][i % 3]
        sub = pool[(pool.family == fam) & pool.e_top_w.notna()
                   & pool.e_bot_w.notna()]
        for m in oc.methods_present(sub):
            ms = sub[sub.method == m]
            ax.scatter(ms.e_top_w, ms.e_bot_w, s=22, color=oc.COL[m],
                       marker=oc.MARK[m], edgecolor="k", linewidth=0.25,
                       alpha=0.8, zorder=4)
        wsub = wh[(wh.family == fam) & wh.e_top_w.notna()]
        if len(wsub):
            ax.scatter(wsub.e_top_w, wsub.e_bot_w, s=28, facecolors="none",
                       edgecolors=oc.COL["corda"], marker="<",
                       linewidths=1.0, zorder=3)
        ax.set_title(oc.FAM_LABEL[fam], fontsize=11)
        ax.set_xlabel("$e_{top}$ (major-component energy)")
        if i % 3 == 0:
            ax.set_ylabel("$e_{bot}$ (minor-component energy)")
    fig.suptitle("M3  Major vs minor component energy split "
                 "(open '<' = CorDA, withheld)", y=1.0, fontsize=13)
    hs = oc.legend_handles(oc.methods_present(pool))
    fig.legend(handles=hs, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.012), fontsize=8.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.965])
    print("[fig]", oc.save_fig(fig, "m3_fig_etop_vs_ebot"))


if __name__ == "__main__":
    main()
