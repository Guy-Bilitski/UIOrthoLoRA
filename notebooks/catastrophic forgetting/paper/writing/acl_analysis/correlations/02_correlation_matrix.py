"""02 — Full correlation matrix over the deduped frozen pool (n=1034).

Outputs:
  corr_pooled_pearson.csv        raw pooled Pearson (pairwise-complete)
  corr_pooled_pearson_partial.csv family-partialed (family-demeaned) Pearson
  corr_pooled_spearman.csv       raw pooled Spearman
  corr_pooled_spearman_partial.csv family-demeaned Spearman
  corr_family_<fam>_pearson.csv  per-family Pearson
  corr_vs_retention.md/.csv      compact table: every metric vs retention_mean,
                                 pooled-partialed + per family + cell-level
  fig_corr_heatmap.png/.pdf      2-panel heatmap (raw pooled | family-partialed)
Notes: raw CE pools two base models + two protocols; KL is the cross-model
comparable drift measure (§18.6). Correlations here are descriptive; inference
lives in 03 (cluster-robust).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import figstyle  # noqa: F401  (applies rcParams; palette not needed for heatmap)
import corr_common as cc

df, _ = cc.build(dedupe=True, verbose=False)
V = cc.VARS
OUT = cc.OUT

# ---- pooled matrices ---------------------------------------------------------
R_raw, N_raw = cc.pearson_pairwise(df, V)
dm = cc.demean_by(df, V)
R_par, _ = cc.pearson_pairwise(dm, V)
S_raw = cc.spearman_pairwise(df, V)
S_par = cc.spearman_pairwise(dm, V)

R_raw.to_csv(OUT + "/corr_pooled_pearson.csv")
R_par.to_csv(OUT + "/corr_pooled_pearson_partial.csv")
S_raw.to_csv(OUT + "/corr_pooled_spearman.csv")
S_par.to_csv(OUT + "/corr_pooled_spearman_partial.csv")
N_raw.to_csv(OUT + "/corr_pooled_n.csv")

# ---- per family ---------------------------------------------------------------
fam_ret = {}
for fam in cc.FAMS:
    sub = df[df.fam == fam]
    Rf, _ = cc.pearson_pairwise(sub, V)
    Rf.to_csv(OUT + f"/corr_family_{fam}_pearson.csv")
    fam_ret[fam] = Rf["ret"]

# ---- cell-level (seed-averaged) family-partialed, retention row ---------------
cells = df.groupby(["fam", "cell"], as_index=False)[V].mean()
dm_c = cc.demean_by(cells, V)
R_cell, _ = cc.pearson_pairwise(dm_c, V)

# ---- compact vs-retention table ------------------------------------------------
rows = []
for v in V:
    if v == "ret":
        continue
    row = dict(metric=v, label=cc.VLAB[v],
               pooled_raw=R_raw.loc[v, "ret"], pooled_partial=R_par.loc[v, "ret"],
               spearman_partial=S_par.loc[v, "ret"], cell_partial=R_cell.loc[v, "ret"],
               n=N_raw.loc[v, "ret"])
    for fam in cc.FAMS:
        row[fam] = fam_ret[fam][v]
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(OUT + "/corr_vs_retention.csv", index=False)

md = ["# Every metric vs retention (core) — Pearson r", "",
      "Pooled-partial = family-demeaned (family FE partialed out). Cell = seed-averaged cells, family-demeaned.",
      "Raw CE pools two base models/protocols — prefer KL pooled (disclosed).", "",
      "| metric | pooled raw | pooled partial | Spearman partial | cell partial | " +
      " | ".join(cc.FAMS) + " | n |", "|" + "---|" * (12)]
for _, r in T.iterrows():
    md.append("| %s | %+.3f | %+.3f | %+.3f | %+.3f | %s | %d |" % (
        r.label, r.pooled_raw, r.pooled_partial, r.spearman_partial, r.cell_partial,
        " | ".join("%+.3f" % r[f] for f in cc.FAMS), r.n))
open(OUT + "/corr_vs_retention.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

# ---- heatmap -------------------------------------------------------------------
SHORT = {"ret": "retention (core)", "ret_broad": "retention (broad)",
         "adapt": "adaptation", "logfd": "log F_delta", "logfro": "log ||dW||_F",
         "lspec": "log spec_max", "stable_rank_w": "stable rank",
         "eff_rank_w": "effective rank", "e_top_w": "e_top", "e_bot_w": "e_bot",
         "amp_top_w": "amp_top", "ce": "CE drift", "kl": "KL drift",
         "loglr": "log LR"}
labels = [SHORT[v] for v in V]


def heat(ax, M, title):
    im = ax.imshow(M.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(len(V)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(V)), labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    for i in range(len(V)):
        for j in range(len(V)):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "−."),
                        ha="center", va="center", fontsize=5.2,
                        color="white" if abs(v) > 0.55 else "#0b0b0b")
    return im


fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.6))
fig.subplots_adjust(wspace=0.42)
heat(axes[0], R_raw, "Pooled Pearson (raw, n=1034; pairwise-complete)")
im = heat(axes[1], R_par, "Pooled Pearson, family partialed out (demeaned)")
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("Pearson r")
fig.suptitle("Correlation structure of retention, magnitude, geometry, CE drift, LR (frozen pool, deduped n=1034)",
             fontsize=11)
fig.savefig(OUT + "/fig_corr_heatmap.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT + "/fig_corr_heatmap.pdf", bbox_inches="tight")
print("\nSaved heatmap + CSVs.")
