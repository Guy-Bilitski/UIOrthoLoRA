"""04_permatrix_layers.py — per-layer / per-projection structure of the update.

Mines results/geo_drift/permatrix/ (+ _qwen) — 160 matrices per adapter
(32 layers x {q,k,v,up,down}) with fro/spec/e_top/... per matrix. Nobody has
used these per-layer beyond fig_geometry_4panel panel D (SC-LoRA ein_top profile).

Per run we compute update-ENERGY (fro^2) composition features:
  share_q/k/v/up/down  — projection-type energy shares
  depth_centroid       — sum(layer * fro^2) / sum(fro^2) / 31   (0=input, 1=output)
  depth_gini           — concentration across the 160 matrices (Herfindahl->effective count)
  eff_n_mat            — 1/sum(share_i^2): effective number of matrices carrying the update

Questions:
 A. Method fingerprints — do methods have distinct layer/projection signatures?
 B. Does any composition feature predict retention BEYOND log F_delta (+family FE)?
    (run level descriptive + cell level for inference; guardrail: magnitude first.)
Outputs: permatrix_features.csv, permatrix_layers.md, fig_permatrix.png/.pdf
"""
import os, json, glob
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
GD = os.path.join(ROOT, "results", "geo_drift")

pool = pd.read_csv(os.path.join(HERE, "pool.csv"))
runs = set(pool.rn)

feat_rows = []
prof_by_method = {}   # method -> list of per-layer energy profiles (llama CS only for fig)
for d in ("permatrix", "permatrix_qwen"):
    for f in glob.glob(os.path.join(GD, d, "*.jsonl")):
        rn = os.path.basename(f)[:-6]
        if rn not in runs:
            continue
        try:
            rows = [json.loads(l) for l in open(f)]
        except Exception:
            continue
        if not rows:
            continue
        fro2 = np.array([r["fro"] ** 2 for r in rows], float)
        tot = fro2.sum()
        if tot <= 0:
            continue
        share = fro2 / tot
        layer = np.array([r["layer"] for r in rows], float)
        tgt = np.array([r["target"] for r in rows])
        nl = layer.max() or 1.0
        feat = dict(rn=rn,
                    depth_centroid=float((share * layer).sum() / nl),
                    eff_n_mat=float(1.0 / (share ** 2).sum()))
        for t in ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj"):
            feat["share_" + t.split("_")[0]] = float(share[tgt == t].sum())
        # per-layer profile (summed over targets), normalized
        prof = np.zeros(int(nl) + 1)
        for l, s in zip(layer.astype(int), share):
            prof[l] += s
        feat["prof"] = prof
        feat_rows.append(feat)

F = pd.DataFrame(feat_rows)
profs = {r.rn: r.prof for r in F.itertuples()}
F = F.drop(columns=["prof"])
M = pool.merge(F, on="rn", how="inner")
M.to_csv(os.path.join(HERE, "permatrix_features.csv"), index=False)

lines = [f"# Per-layer / per-projection update composition (n={len(M)} runs joined of {len(pool)} pool)", ""]

FEATS = ["share_q", "share_k", "share_v", "share_up", "share_down", "depth_centroid", "eff_n_mat"]

# ---- A. method fingerprints (Llama families, then Qwen) ----
lines.append("## A. Method fingerprints — mean composition per method (all 6 families pooled)")
fp = M.groupby("method")[FEATS].mean().round(3)
fp["n"] = M.groupby("method").size()
lines.append("```")
lines.append(fp.to_string())
lines.append("```")
# one-way ANOVA eta^2 per feature (how method-determined is composition?)
lines.append("")
lines.append("Method-determination of composition (eta^2 of method, one-way, runs):")
for ft in FEATS:
    groups = [g[ft].dropna().values for _, g in M.groupby("method") if len(g) > 5]
    grand = np.concatenate(groups)
    ssb = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
    sst = ((grand - grand.mean()) ** 2).sum()
    lines.append(f"- {ft}: eta^2 = {ssb / sst:.3f}")

# ---- B. beyond-magnitude prediction ----
def partial_r(y, x, Z):
    Z = np.column_stack([np.ones(len(y))] + list(Z))
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    r = np.corrcoef(ry, rx)[0, 1]
    dof = len(y) - 2 - (Z.shape[1] - 1)
    t = r * np.sqrt(dof / max(1e-12, 1 - r * r))
    return r, t, 2 * stats.t.sf(abs(t), dof)

lines.append("")
lines.append("## B. Does composition predict retention beyond log F_delta + family?")
famd = pd.get_dummies(M.fam, drop_first=True).astype(float)
Zbase = [M.logfd.values] + [famd[c].values for c in famd.columns]
for ft in FEATS:
    ok = M[ft].notna()
    r, t, p = partial_r(M.ret[ok].values, M[ft][ok].values,
                        [z[ok.values] for z in Zbase])
    # also within-method (add method dummies): is it composition or just method identity?
    md = pd.get_dummies(M.method, drop_first=True).astype(float)
    Zm = Zbase + [md[c].values for c in md.columns]
    r2, t2, p2 = partial_r(M.ret[ok].values, M[ft][ok].values,
                           [z[ok.values] for z in Zm])
    lines.append(f"- {ft}: partial r | (logF, fam) = {r:+.3f} (t={t:.1f}, p={p:.1e}); "
                 f"| (+method) = {r2:+.3f} (t={t2:.1f}, p={p2:.1e})")
lines.append("(run level; seeds within cells are correlated — treat |t|<4 as suggestive only)")

# cell-level for the strongest features
Mc = M.groupby("cell").agg({**{f: "mean" for f in FEATS}, "logfd": "mean", "ret": "mean",
                            "fam": "first", "method": "first"}).reset_index()
famdc = pd.get_dummies(Mc.fam, drop_first=True).astype(float)
Zc = [Mc.logfd.values] + [famdc[c].values for c in famdc.columns]
lines.append("")
lines.append("Cell-level (seed-averaged) partials | (logF, fam):")
for ft in FEATS:
    ok = Mc[ft].notna()
    r, t, p = partial_r(Mc.ret[ok].values, Mc[ft][ok].values, [z[ok.values] for z in Zc])
    lines.append(f"- {ft}: r = {r:+.3f} (t={t:.1f}, p={p:.1e}, n={ok.sum()})")

# ---- depth profile by method (Llama CS: lrsw+frc), energy vs layer ----
lines.append("")
lines.append("## C. Depth profiles (Llama CS arms, mean energy share per layer)")
sel = M[(M.fam.isin(["lrsw", "frc"]))]
prof_meth = {}
for meth, g in sel.groupby("method"):
    ps = [profs[rn] for rn in g.rn if rn in profs and len(profs[rn]) == 32]
    if len(ps) >= 5:
        prof_meth[meth] = np.mean(ps, axis=0)
for meth, p in sorted(prof_meth.items()):
    top = np.argsort(p)[-3:][::-1]
    lines.append(f"- {meth}: top layers {list(top)} carry {p[top].sum()*100:.0f}% of energy; "
                 f"first8 {p[:8].sum()*100:.0f}% / last8 {p[24:].sum()*100:.0f}%")

open(os.path.join(HERE, "permatrix_layers.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
ax = axes[0]
for meth, p in sorted(prof_meth.items()):
    ax.plot(np.arange(32), p * 100, label=meth, lw=1.4)
ax.set_xlabel("layer"); ax.set_ylabel("% of update energy")
ax.set_title("A  depth profile of update energy (Llama CS)", loc="left")
ax.legend(fontsize=6, ncol=2)

ax = axes[1]
fp2 = M[M.fam.isin(["lrsw", "frc"])].groupby("method")[["share_q", "share_k", "share_v", "share_up", "share_down"]].mean()
bottom = np.zeros(len(fp2))
for col, c in zip(fp2.columns, ["C0", "C1", "C2", "C4", "C3"]):
    ax.bar(fp2.index, fp2[col] * 100, bottom=bottom, label=col.replace("share_", ""), color=c)
    bottom += fp2[col].values * 100
ax.set_ylabel("% of update energy"); ax.tick_params(axis="x", rotation=45)
ax.set_title("B  projection-type composition (Llama CS)", loc="left")
ax.legend(fontsize=6)

ax = axes[2]
ok = M.depth_centroid.notna()
sc = ax.scatter(M.logfd[ok], M.depth_centroid[ok], c=M.ret[ok], s=8, cmap="viridis")
ax.set_xlabel(r"$\log_{10} F_\Delta$"); ax.set_ylabel("depth centroid (0=input,1=output)")
plt.colorbar(sc, ax=ax, label="retention")
ax.set_title("C  depth centroid vs magnitude", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_permatrix.png"), dpi=200)
fig.savefig(os.path.join(HERE, "fig_permatrix.pdf"))
print("figure saved; joined", len(M))
