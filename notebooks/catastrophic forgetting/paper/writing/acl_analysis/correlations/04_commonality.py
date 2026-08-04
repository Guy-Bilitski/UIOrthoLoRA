"""04 — Three-block commonality decomposition: magnitude / geometry-shape / CE drift.

Extends 06 §5's two-block split to CE. All R2s are dR2 over the family-FE
baseline on the CE∩geometry pool (n=911). Blocks:
  M = magnitude = [log10 F_delta]           (variant Mx adds log spec_max)
  G = geometry-shape = [stable_rank, eff_rank, e_top, e_bot, amp_top]
  C = CE drift = [KL]                       (variant: raw CE — identical under FE)

FRAMING CAVEAT (binding): CE drift is measured on base-model text and is
quasi-tautologically close to retention (both measure change to base behavior),
and it is DOWNSTREAM of training (a consequence, not a knob). CE = proximal
channel / early-warning signal; magnitude = upstream controllable variable.

Also: two-block M vs C split ("what does CE capture that magnitude doesn't"),
pooled + per family, and a cell-level cluster bootstrap CI (B=2000) on the key
unique components (09 Q1 rule: never naive F/p).

Outputs: commonality.md, commonality_3block.csv, commonality_mc_per_family.csv
"""
import itertools
import numpy as np
import pandas as pd
import corr_common as cc

rng = np.random.default_rng(20260718)
df, _ = cc.build(dedupe=True, verbose=False)
OUT = cc.OUT

M = cc.MAG_BLOCK
G = cc.GEO_BLOCK
C = cc.CE_BLOCK
pool = df.dropna(subset=M + G + C + ["lspec"]).copy()
print(f"pool n={len(pool)}")


def dr2(sub, terms, fe=True):
    y = sub["ret"].values.astype(float)
    if fe:
        X0, _ = cc.design_fe(sub, [])
        base = cc.ols_fit(X0, y)[2]
        X, _ = cc.design_fe(sub, terms)
    else:
        base = 0.0
        X = np.column_stack([np.ones(len(sub))] + [sub[t].values for t in terms])
    return cc.ols_fit(X, y)[2] - base


def commonality3(sub, Mb, Gb, Cb, fe=True):
    R = {}
    for name, terms in [("M", Mb), ("G", Gb), ("C", Cb), ("MG", Mb + Gb),
                        ("MC", Mb + Cb), ("GC", Gb + Cb), ("MGC", Mb + Gb + Cb)]:
        R[name] = dr2(sub, terms, fe=fe)
    out = {
        "unique_M": R["MGC"] - R["GC"],
        "unique_G": R["MGC"] - R["MC"],
        "unique_C": R["MGC"] - R["MG"],
        "shared_MG": R["MC"] + R["GC"] - R["MGC"] - R["C"],
        "shared_MC": R["MG"] + R["GC"] - R["MGC"] - R["G"],
        "shared_GC": R["MG"] + R["MC"] - R["MGC"] - R["M"],
        "shared_MGC": (R["M"] + R["G"] + R["C"] - R["MG"] - R["MC"] - R["GC"] + R["MGC"]),
        "total": R["MGC"], "R_M": R["M"], "R_G": R["G"], "R_C": R["C"],
    }
    return out, R


com, Rsets = commonality3(pool, M, G, C)
com_x, _ = commonality3(pool, ["logfd", "lspec"], G, C)

# ---- cell-level cluster bootstrap on the primary decomposition -----------------
cells = pool["cell"].unique()
B = 2000
keys = ["unique_M", "unique_G", "unique_C", "shared_MC"]
boot = {k: np.empty(B) for k in keys}
order_win = 0
bycell = {c: pool[pool.cell == c] for c in cells}
for b in range(B):
    pick = rng.choice(cells, size=len(cells), replace=True)
    sb = pd.concat([bycell[c] for c in pick], ignore_index=True)
    try:
        cb, _ = commonality3(sb, M, G, C)
    except Exception:
        cb = {k: np.nan for k in keys}
    for k in keys:
        boot[k][b] = cb[k]
    if cb["unique_M"] > cb["unique_G"] and cb["unique_M"] > cb["unique_C"]:
        order_win += 1
ci = {k: (np.nanpercentile(boot[k], 2.5), np.nanpercentile(boot[k], 97.5)) for k in keys}

# ---- two-block M vs C, pooled + per family -------------------------------------
def mc_split(sub, fe):
    rM = dr2(sub, M, fe); rC = dr2(sub, C, fe); rMC = dr2(sub, M + C, fe)
    return dict(unique_M=rMC - rC, unique_C=rMC - rM, shared=rM + rC - rMC,
                R_M=rM, R_C=rC, R_MC=rMC, n=len(sub))

mc_pool = mc_split(pool, fe=True)
fam_rows = []
for fam in cc.FAMS:
    s = pool[pool.fam == fam]
    d = mc_split(s, fe=False)
    d["fam"] = fam
    fam_rows.append(d)
F = pd.DataFrame(fam_rows)[["fam", "n", "R_M", "R_C", "R_MC", "unique_M", "unique_C", "shared"]]
F.to_csv(OUT + "/commonality_mc_per_family.csv", index=False)

rows = []
for nm, c in [("primary (M=logF)", com), ("extended (M=logF+log spec_max)", com_x)]:
    rows.append(dict(variant=nm, **{k: v for k, v in c.items()}))
pd.DataFrame(rows).to_csv(OUT + "/commonality_3block.csv", index=False)

md = ["# Commonality decomposition — magnitude / geometry-shape / CE drift",
      "",
      f"Pool: frozen(deduped) ∩ geometry ∩ CE, n={len(pool)}. All components are dR2 over",
      "family FE. Blocks: M=[log10 F_delta]; G=[stable_rank, eff_rank, e_top, e_bot,",
      "amp_top]; C=[KL drift].",
      "",
      "## FRAMING (binding caveat)",
      "CE/KL drift is measured on base-model text; it is quasi-tautologically close to",
      "retention (both quantify change to base behavior) and is DOWNSTREAM — a",
      "consequence of the update, not a knob. Read CE as the proximal channel /",
      "early-warning signal, magnitude as the upstream controllable variable. Its",
      "'unique share' is diagnostic value, not causal leverage.",
      "",
      "## Three-block decomposition (sums to the full-model dR2)",
      "",
      "| component | primary | extended (M incl. log spec_max) |",
      "|---|---|---|"]
for k, lab in [("unique_M", "unique: magnitude"), ("unique_G", "unique: geometry-shape"),
               ("unique_C", "unique: CE drift"), ("shared_MG", "shared: M∩G only"),
               ("shared_MC", "shared: M∩C only"), ("shared_GC", "shared: G∩C only"),
               ("shared_MGC", "shared: M∩G∩C"), ("total", "TOTAL (full model dR2)")]:
    md.append(f"| {lab} | {com[k]:+.3f} | {com_x[k]:+.3f} |")
md += ["", f"Single-block dR2 same-sample: M {com['R_M']:+.3f}, G {com['R_G']:+.3f}, C {com['R_C']:+.3f}.",
       "",
       "## Cell-level cluster bootstrap (B=2000, resample recipe cells)",
       "",
       "| quantity | point | 95% CI |", "|---|---|---|"]
for k, lab in [("unique_M", "unique magnitude"), ("unique_G", "unique geometry-shape"),
               ("unique_C", "unique CE drift"), ("shared_MC", "shared magnitude∩CE")]:
    md.append(f"| {lab} | {com[k]:+.3f} | [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}] |")
md += ["", f"Ordering unique_M > max(unique_G, unique_C): {order_win}/{B} bootstrap replicates.",
       "",
       "## What does CE capture that magnitude doesn't (and vice versa)? Two-block M vs C.",
       "",
       f"Pooled (family FE, n={mc_pool['n']}): R2(M)={mc_pool['R_M']:+.3f}, R2(C)={mc_pool['R_C']:+.3f},",
       f"R2(M+C)={mc_pool['R_MC']:+.3f} -> unique(M)={mc_pool['unique_M']:+.3f},",
       f"unique(C)={mc_pool['unique_C']:+.3f}, shared={mc_pool['shared']:+.3f}.",
       "",
       "| family | n | R2(M) | R2(C=KL) | R2(M+C) | unique M | unique C | shared |",
       "|---|---|---|---|---|---|---|---|"]
for _, r in F.iterrows():
    md.append("| %s | %d | %.3f | %.3f | %.3f | %+.3f | %+.3f | %+.3f |" % (
        r.fam, r.n, r.R_M, r.R_C, r.R_MC, r.unique_M, r.unique_C, r.shared))
md += ["",
       "Reading: pooled, magnitude keeps a large unique share beyond CE while CE adds",
       "little beyond magnitude; WITHIN families the balance shifts (05's mediation:",
       "KL is the proximal channel in the math/Qwen arms) — quote both granularities."]
out = "\n".join(md)
open(OUT + "/commonality.md", "w").write(out + "\n")
print(out)
