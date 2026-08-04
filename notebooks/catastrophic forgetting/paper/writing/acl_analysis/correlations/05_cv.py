"""05 — Does combining metrics beat magnitude alone? Grouped cross-validation.

Pool: frozen(deduped) ∩ geometry ∩ CE (n=911), so all model families compete on
the same sample. Models:
  FE-only, FE+M (log F_delta), FE+M+G (shape block), FE+M+C (KL), FE+M+G+C, FE+C
Schemes:
  (a) leave-cells-out: 10-fold GroupKFold on recipe cells (no seed leakage;
      family dummies kept — every family present in every training fold).
  (b) leave-one-family-out (LOFO): no family FE (global intercept); reported raw
      and with an intercept-oracle correction (held-out family mean offset
      removed) to separate level-transfer failure from slope transfer.
Out-of-sample R2 = 1 - SSE/SST over pooled held-out predictions.

Outputs: cv_results.csv, cv_results.md, fig_cv_pred_vs_actual.png/.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import figstyle
import corr_common as cc

rng = np.random.default_rng(20260718)
df, _ = cc.build(dedupe=True, verbose=False)
OUT = cc.OUT

M, G, C = cc.MAG_BLOCK, cc.GEO_BLOCK, cc.CE_BLOCK
pool = df.dropna(subset=M + G + C).reset_index(drop=True)
y = pool["ret"].values.astype(float)

MODELS = {
    "FE only": [],
    "magnitude": M,
    "magnitude+geometry": M + G,
    "magnitude+CE": M + C,
    "magnitude+geometry+CE": M + G + C,
    "CE only": C,
}


def Xmat(sub, terms, fe):
    if fe:
        X, _ = cc.design_fe(sub, terms)
        # design_fe drops first family alphabetically based on sub — for CV we must
        # keep a FIXED dummy basis across folds:
        fams = sorted(pool["fam"].unique())
        cols = [np.ones(len(sub))]
        for f in fams[1:]:
            cols.append((sub["fam"] == f).astype(float).values)
        for t in terms:
            cols.append(sub[t].values.astype(float))
        return np.column_stack(cols)
    cols = [np.ones(len(sub))] + [sub[t].values.astype(float) for t in terms]
    return np.column_stack(cols)


# ---------- (a) leave-cells-out --------------------------------------------------
cells = np.array([str(c) for c in pool["cell"].unique()], dtype=object)
rng.shuffle(cells)
folds = np.array_split(cells, 10)
pred_store = {}
rows = []
for name, terms in MODELS.items():
    pred = np.full(len(pool), np.nan)
    for f in folds:
        te = pool["cell"].isin(f).values
        tr = ~te
        Xtr, Xte = Xmat(pool[tr], terms, fe=True), Xmat(pool[te], terms, fe=True)
        beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        pred[te] = Xte @ beta
    sse = np.sum((y - pred) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst
    rmse = np.sqrt(sse / len(y))
    rows.append(dict(scheme="leave-cells-out", model=name, r2_oos=r2, rmse=rmse, n=len(y)))
    pred_store[name] = pred.copy()

# ---------- (b) leave-one-family-out ---------------------------------------------
for name, terms in MODELS.items():
    if name == "FE only":
        continue
    pred = np.full(len(pool), np.nan)
    pred_or = np.full(len(pool), np.nan)
    for fam in cc.FAMS:
        te = (pool["fam"] == fam).values
        tr = ~te
        Xtr, Xte = Xmat(pool[tr], terms, fe=False), Xmat(pool[te], terms, fe=False)
        beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        p = Xte @ beta
        pred[te] = p
        pred_or[te] = p + (y[te].mean() - p.mean())  # intercept-oracle
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / sst
    # oracle version scored within family (slope transfer)
    sst_w = np.sum((y - pool.groupby("fam")["ret"].transform("mean").values) ** 2)
    r2_or = 1 - np.sum((y - pred_or) ** 2) / sst_w
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    rmse_or = np.sqrt(np.mean((y - pred_or) ** 2))
    rows.append(dict(scheme="LOFO raw", model=name, r2_oos=r2, rmse=rmse, n=len(y)))
    rows.append(dict(scheme="LOFO intercept-oracle (within-family)", model=name,
                     r2_oos=r2_or, rmse=rmse_or, n=len(y)))

R = pd.DataFrame(rows)
R.to_csv(OUT + "/cv_results.csv", index=False)

md = ["# Grouped cross-validation — does anything beat magnitude alone?", "",
      f"Pool n={len(pool)} (frozen∩geometry∩CE). OOS R2 = 1 - SSE/SST on pooled held-out",
      "predictions. Leave-cells-out: 10 folds of recipe cells (seeds never split across",
      "train/test). LOFO: family FE removed; 'intercept-oracle' removes the held-out",
      "family's mean offset (scores slope transfer only, vs within-family SST).", ""]
for scheme in ["leave-cells-out", "LOFO raw", "LOFO intercept-oracle (within-family)"]:
    md += [f"## {scheme}", "", "| model | OOS R2 | RMSE (pp) |", "|---|---|---|"]
    s = R[R.scheme == scheme]
    for _, r in s.iterrows():
        md.append(f"| {r.model} | {r.r2_oos:+.3f} | {r.rmse:.2f} |")
    md.append("")

# per-family LOFO detail for the full model vs magnitude
md += ["## LOFO per held-out family (magnitude vs full model, intercept-oracle R2 within family)",
       "", "| held-out family | magnitude | mag+geo+CE | CE only |", "|---|---|---|---|"]
for fam in cc.FAMS:
    line = [fam]
    for name in ["magnitude", "magnitude+geometry+CE", "CE only"]:
        terms = MODELS[name]
        te = (pool["fam"] == fam).values
        tr = ~te
        Xtr, Xte = Xmat(pool[tr], terms, fe=False), Xmat(pool[te], terms, fe=False)
        beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        p = Xte @ beta
        p = p + (y[te].mean() - p.mean())
        r2 = 1 - np.sum((y[te] - p) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)
        line.append(f"{r2:+.3f}")
    md.append("| " + " | ".join(line) + " |")

out = "\n".join(md)
open(OUT + "/cv_results.md", "w").write(out + "\n")
print(out)

# ---------- figure: predicted vs actual (leave-cells-out) -------------------------
show = ["magnitude", "magnitude+geometry", "magnitude+CE", "magnitude+geometry+CE"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), sharex=True, sharey=True)
fam_colors = {f: c for f, c in zip(cc.FAMS,
              ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7", "#eb6834"])}
for ax, name in zip(axes, show):
    p = pred_store[name]
    for fam in cc.FAMS:
        m = (pool["fam"] == fam).values
        ax.scatter(y[m], p[m], s=8, alpha=0.55, color=fam_colors[fam],
                   label=cc.FAM_LABEL[fam], linewidths=0)
    lo, hi = y.min() - 1, y.max() + 1
    ax.plot([lo, hi], [lo, hi], color="#c3c2b7", lw=1, zorder=0)
    r2 = R[(R.scheme == "leave-cells-out") & (R.model == name)].r2_oos.iloc[0]
    rmse = R[(R.scheme == "leave-cells-out") & (R.model == name)].rmse.iloc[0]
    ax.set_title(f"{name}\nOOS R²={r2:.3f}, RMSE={rmse:.2f}pp", fontsize=9)
    ax.set_xlabel("actual retention (pp)")
axes[0].set_ylabel("predicted retention (pp)")
axes[0].legend(fontsize=7, loc="upper left", frameon=False)
fig.suptitle("Leave-cells-out CV: predicted vs actual retention (n=911; family FE models)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "/fig_cv_pred_vs_actual.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT + "/fig_cv_pred_vs_actual.pdf", bbox_inches="tight")
print("\nSaved CV figure.")
