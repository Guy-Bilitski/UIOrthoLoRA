"""B4: league table (same-sample n=911 dRsq over family FE) + commonality
unique/shared components. Independent recompute."""
import numpy as np
import pandas as pd

import verify_common as vc

df = vc.frozen_pool()
assert vc.check_18_1(df, verbose=False)
df = df[df.run != vc.DUPLICATE]  # dedupe (correlations convention)

geo = vc.load_geo()
ce = vc.load_ce()
m = df.merge(geo, on="run", how="left").merge(ce[["run", "kl", "ce"]], on="run", how="left")

# same-sample pool: CE and geometry both present
p = m[np.isfinite(m.kl) & np.isfinite(m.lspec) & np.isfinite(m.stable_rank_w)].copy()
print(f"same-sample pool n = {len(p)} (claim 911)")

metrics = {
    "log10 F_delta": "logfd",
    "log10 spec_max": "lspec",
    "log10 fro_total": "logfro",
    "KL drift": "kl",
    "CE drift": "ce",
    "log10 LR": "loglr",
    "stable rank": "stable_rank_w",
    "amp_top": "amp_top_w",
    "eff rank": "eff_rank_w",
    "e_top": "e_top_w",
    "e_bot": "e_bot_w",
}
print("\nLEAGUE TABLE — dR2 over family FE, same sample")
print("(claims: F 0.420, spec 0.349, fro 0.348, CE/KL 0.340, LR 0.207, srank 0.116)")
for name, col in metrics.items():
    sub = p[np.isfinite(p[col])]
    if len(sub) < len(p):
        print(f"  {name:16s} [n={len(sub)}]", end="")
    dr2, r0, r1 = vc.delta_r2(sub, [col])
    t, G = vc.cluster_t(sub, col)
    print(f"  {name:16s} dR2 = {dr2:+.3f}   cluster-t = {t:+.1f} (G={G} cells)")

# per-family R2 for logfd and loglr (claims 0.69-0.86 and 0.22-0.52)
print("\nper-family R2 (single regressor):")
for col in ["logfd", "loglr", "kl"]:
    vals = {}
    for f, s in p.groupby("fam"):
        s = s[np.isfinite(s[col])]
        r2, *_ = vc.ols_r2([s[col].values.astype(float)], s.ret.values.astype(float))
        vals[f] = round(float(r2), 2)
    print(f"  {col}: {vals}")

# ---- two-block commonality: M = (logfd,) vs C = (kl,) ----
print("\nTWO-BLOCK COMMONALITY (M = log F_delta [+spec ext], C = KL)")
y = "ret"


def r2_fe(sub, cols):
    dr2, r0, r1 = vc.delta_r2(sub, cols)
    return dr2


M = ["logfd"]
C = ["kl"]
rM = r2_fe(p, M)
rC = r2_fe(p, C)
rMC = r2_fe(p, M + C)
print(f"  R2(M)={rM:.3f} (claim .420)  R2(C)={rC:.3f} (claim .340)  R2(M+C)={rMC:.3f}")
print(f"  unique(C beyond M) = {rMC - rM:+.3f} (claim +0.005)")
print(f"  unique(M beyond C) = {rMC - rC:+.3f} (claim +0.085)")
print(f"  shared = {rM + rC - rMC:+.3f} (claim +0.335)")

# ---- three-block: M, G (5 shape metrics), C ----
G5 = ["stable_rank_w", "eff_rank_w", "e_top_w", "e_bot_w", "amp_top_w"]
sub = p[np.isfinite(p[G5]).all(axis=1)]
print(f"\nTHREE-BLOCK (n={len(sub)}): M=(logfd) G=({','.join(G5)}) C=(kl)")
rM = r2_fe(sub, M)
rG = r2_fe(sub, G5)
rC = r2_fe(sub, C)
rMG = r2_fe(sub, M + G5)
rMC = r2_fe(sub, M + C)
rGC = r2_fe(sub, G5 + C)
rMGC = r2_fe(sub, M + G5 + C)
uM = rMGC - rGC
uG = rMGC - rMC
uC = rMGC - rMG
print(f"  total dR2 = {rMGC:+.3f} (claim +0.456)")
print(f"  unique M = {uM:+.3f} (claim +0.033)")
print(f"  unique G = {uG:+.3f} (claim +0.031)")
print(f"  unique C = {uC:+.3f} (claim +0.009)")
# pairwise shared via inclusion-exclusion (Newton commonality)
sMC = rM + rC - rMC - (rM + rG - rMG) * 0  # do proper commonality below
# full commonality decomposition (3 predictors blocks):
c_MG = rMGC - rC - uM - uG  # not standard; do standard formulas:
# standard: shared(M,C only) = rMC + rGC + rMG ... use classic commonality:
S = dict(M=rM, G=rG, C=rC, MG=rMG, MC=rMC, GC=rGC, MGC=rMGC)
u_M = S["MGC"] - S["GC"]
u_G = S["MGC"] - S["MC"]
u_C = S["MGC"] - S["MG"]
c_MG_only = S["MC"] + S["GC"] - S["MGC"] - S["C"]
c_MC_only = S["MG"] + S["GC"] - S["MGC"] - S["G"]
c_GC_only = S["MG"] + S["MC"] - S["MGC"] - S["M"]
c_MGC = S["MGC"] - u_M - u_G - u_C - c_MG_only - c_MC_only - c_GC_only
print(f"  shared M&C only = {c_MC_only:+.3f} (claim +0.181)")
print(f"  shared M&G&C    = {c_MGC:+.3f} (claim +0.154)")
print(f"  shared M&G only = {c_MG_only:+.3f} (claim +0.052)")
print(f"  shared G&C only = {c_GC_only:+.3f} (claim -0.004)")

# ---- M-vs-G exogenous split (the frozen §19/06 anchor) on n=1034 ----
g2 = df.merge(geo, on="run", how="inner")
g2 = g2[np.isfinite(g2[["lspec", "stable_rank_w", "e_top_w"]]).all(axis=1)]
print(f"\nM-vs-G exogenous split on frozen∩geometry (n={len(g2)}):")
Mfull = ["logfd"]
Gsh = ["stable_rank_w", "eff_rank_w", "e_top_w", "e_bot_w", "amp_top_w"]
gsub = g2[np.isfinite(g2[Gsh]).all(axis=1)]
rM = r2_fe(gsub, Mfull)
rG = r2_fe(gsub, Gsh)
rMG = r2_fe(gsub, Mfull + Gsh)
print(f"  unique(M)={rMG - rG:+.3f} (claim +0.296)  unique(G)={rMG - rM:+.3f} (claim +0.016)"
      f"  shared={rM + rG - rMG:+.3f} (claim +0.099)")

# ---- ladder reproduction (19.1) ----
lad = g2[np.isfinite(g2[["e_top_w", "lspec", "stable_rank_w"]]).all(axis=1)]
print(f"\nLADDER (n={len(lad)}, claim 1034): "
      f"famFE R2={vc.delta_r2(lad, [])[1]:.3f}")
_, r0, r1 = vc.delta_r2(lad, ["logfd"])
print(f"  +logfd: {r1:.3f} (claim 0.785)")
_, _, r2v = vc.delta_r2(lad, ["logfd", "e_top_w", "lspec", "stable_rank_w"])
print(f"  +geometry: {r2v:.3f} (claim 0.802)")
