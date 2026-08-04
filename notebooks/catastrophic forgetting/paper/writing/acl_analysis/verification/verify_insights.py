"""B7: insights claims — fragility ordering W=1.000, TQ sign flip + attenuation,
three-knobs dose-response, free-lunch, k=2048 adaptation tax."""
import numpy as np
import pandas as pd
from scipy import stats

import verify_common as vc

df = vc.frozen_pool()
assert vc.check_18_1(df, verbose=False)

BASE = {
    "llama": dict(bbh=32.96, mmlu_pro=18.82, mmlu=40.88, arc_c=44.80, truthfulqa=38.85),
    "qwen": dict(bbh=47.93, mmlu_pro=40.77, mmlu=71.80, arc_c=51.28, truthfulqa=56.28),
}
MODEL = dict(lrsw="llama", lrswm="llama", frc="llama", frm="llama",
             qwsw="qwen", qwswm="qwen")

print("=" * 70)
print("7a. FRAGILITY ORDERING (cell level, normalized slope = slope/base)")
print("=" * 70)
B4 = ["mmlu_pro", "bbh", "mmlu", "truthfulqa"]
ranks = []
for fam in vc.FAMS:
    s = df[df.fam == fam]
    cells = s.groupby("cell").agg({**{b: "mean" for b in B4 + ["arc_c"]}, "logfd": "mean"}).dropna()
    ns = {}
    for b in B4 + ["arc_c"]:
        sl, ic, r, p, se = stats.linregress(cells.logfd, cells[b])
        ns[b] = sl / BASE[MODEL[fam]][b]
    order = sorted(B4, key=lambda b: ns[b])  # most negative first = most fragile
    ranks.append([order.index(b) + 1 for b in B4])
    print(f"  {fam}: norm slopes {[f'{b}={ns[b]:+.3f}' for b in B4]} -> order {order}"
          f"  TQ r: {stats.pearsonr(cells.logfd, cells.truthfulqa)[0]:+.2f}")
R = np.array(ranks)
k, m = R.shape[1], R.shape[0]
S = ((R.sum(axis=0) - m * (k + 1) / 2) ** 2).sum()
W = 12 * S / (m ** 2 * (k ** 3 - k))
print(f"  Kendall W (4 benchmarks, ARC-c excluded) = {W:.3f} (claim 1.000)")

print()
print("=" * 70)
print("7b. TRUTHFULQA SIGN + BROAD ATTENUATION")
print("=" * 70)
for fam in vc.FAMS:
    s = df[df.fam == fam]
    cells = s.groupby("cell").agg(tq=("truthfulqa", "mean"), logfd=("logfd", "mean"),
                                  bbh=("bbh", "mean"), mp=("mmlu_pro", "mean"),
                                  mmlu=("mmlu", "mean"), arc=("arc_c", "mean"),
                                  broad=("retention_broad", "mean")).dropna()
    r, p = stats.pearsonr(cells.logfd, cells.tq)
    slope_tq = stats.linregress(cells.logfd, cells.tq).slope
    slope_broad = stats.linregress(cells.logfd, cells.broad).slope
    broad_no_tq = cells[["bbh", "mp", "mmlu", "arc"]].mean(axis=1)
    slope_no_tq = stats.linregress(cells.logfd, broad_no_tq).slope
    att = 1 - slope_broad / slope_no_tq
    print(f"  {fam}: r(TQ,logF)={r:+.2f} (p={p:.1e}) slope_TQ={slope_tq:+.1f}pp/dec | "
          f"broad slope {slope_broad:+.2f} vs no-TQ {slope_no_tq:+.2f} -> attenuation {att * 100:.0f}%")

print()
print("=" * 70)
print("7c. THREE KNOBS (frc grid)")
print("=" * 70)
frc = df[df.fam == "frc"]
# knob 1: weight decay (lorawd cells with wd)
wd = frc[(frc.method == "lorawd") & np.isfinite(frc.wd)]
wdc = wd.groupby("cell").agg(wd=("wd", "mean"), lr=("lr", "mean"),
                             logfd=("logfd", "mean"), ret=("ret", "mean")).reset_index()
r1 = vc.partial_r(wdc.logfd.values, wdc.wd.values, [np.log10(wdc.lr.values)])
dof = len(wdc) - 3
t1 = r1 * np.sqrt(dof / (1 - r1 ** 2))
r2 = vc.partial_r(wdc.ret.values, wdc.wd.values, [wdc.logfd.values])
# dR2 of wd beyond logfd
r2a, *_ = vc.ols_r2([wdc.logfd.values], wdc.ret.values)
r2b, *_ = vc.ols_r2([wdc.logfd.values, wdc.wd.values], wdc.ret.values)
print(f"  wd: n_cells={len(wdc)} stage-1 partial r(logF, wd | logLR) = {r1:+.3f} (t={t1:.1f}) "
      f"(claim -0.762, t=-6.4, n=33)")
print(f"      residual partial r(ret, wd | logF) = {r2:+.2f} (claim -0.25 ns); "
      f"dR2 wd beyond logF = {r2b - r2a:+.3f} on R2 {r2a:.3f}->{r2b:.3f} (claim 0.006, 0.902->0.908)")
# knob 2: CLoRA k at lr 3e-4
cl = frc[(frc.method == "clora") & np.isclose(frc.lr, 3e-4) & np.isfinite(frc.k)]
clc = cl.groupby("cell").agg(k=("k", "mean"), fdelta=("fdelta", "mean"),
                             ret=("ret", "mean"), adapt=("cs_avg", "mean"),
                             n=("run", "count")).sort_values("k")
print(f"  CLoRA k cells (lr3e-4):\n{clc.round(3).to_string()}")
rho_f = stats.spearmanr(np.log10(clc.k), clc.fdelta).statistic
rho_r = stats.spearmanr(np.log10(clc.k), clc.ret).statistic
print(f"      Spearman rho(log k, F)={rho_f:+.2f} (claim -1.00), rho(log k, ret)={rho_r:+.2f} (claim +1.00)")
print(f"      k tax: adapt k128={clc.adapt.iloc[0]:.1f} ... k2048={clc.adapt.iloc[-1]:.1f} "
      f"(claim 76.8 -> 69.4, peak k256-512)")
# knob 3: rank at lr 3e-4 (plain lora r8/16/32)
rk = frc[(frc.method == "lora") & np.isclose(frc.lr, 3e-4) & np.isfinite(frc.rank)]
rkc = rk.groupby("cell").agg(rank=("rank", "mean"), fdelta=("fdelta", "mean"),
                             ret=("ret", "mean"), n=("run", "count")).sort_values("rank")
print(f"  rank cells (lr3e-4):\n{rkc.round(3).to_string()}")
# on-curve residuals: hinge fit on all frc cells
cells_all = frc.groupby("cell").agg(logfd=("logfd", "mean"), ret=("ret", "mean")).dropna()
kk, beta = vc.hinge_fit(cells_all.logfd.values, cells_all.ret.values)
r_cell = np.corrcoef(cells_all.logfd, cells_all.ret)[0, 1]
print(f"  frc cell-level r = {r_cell:.3f} (claim -0.951); hinge knee log10 = {kk:+.2f}")
for name, sub in [("wd", wdc), ("k", clc.assign(logfd=np.log10(clc.fdelta))),
                  ("rank", rkc.assign(logfd=np.log10(rkc.fdelta)))]:
    resid = sub.ret.values - vc.hinge_pred(sub.logfd.values, kk, beta)
    print(f"      mean on-curve residual [{name}] = {resid.mean():+.2f} pp (claims +0.26/+0.53/+1.73)")

print()
print("=" * 70)
print("7d. FREE LUNCH (knees from §18.2) + reachability")
print("=" * 70)
KNEE = dict(lrsw=-0.02, lrswm=-0.48, qwsw=-0.69, qwswm=-0.91, frc=-0.45, frm=-0.50)
for fam in ["lrsw", "frc", "qwsw", "lrswm", "frm", "qwswm"]:
    s = df[(df.fam == fam) & np.isfinite(df.cs_avg)]
    cells = s.groupby("cell").agg(adapt=("cs_avg", "mean"), logfd=("logfd", "mean")).reset_index()
    healthy = cells[cells.adapt >= 25]
    below = healthy[healthy.logfd < KNEE[fam]]
    pk_b = below.adapt.max() if len(below) else np.nan
    pk_g = healthy.adapt.max()
    print(f"  {fam}: peak-below-knee {pk_b:.1f} vs global {pk_g:.1f} ({100 * pk_b / pk_g:.1f}%)")
# reachability at lr3e-4 in frc
print("  frc @ lr3e-4, cells below knee (-0.45):")
for meth in ["lora", "sclora", "clora", "lorawd"]:
    s = frc[(frc.method == meth) & np.isclose(frc.lr, 3e-4)]
    cells = s.groupby("cell").agg(logfd=("logfd", "mean"), adapt=("cs_avg", "mean")).dropna()
    nb = int((cells.logfd < KNEE["frc"]).sum())
    print(f"    {meth}@3e-4: {nb}/{len(cells)} cells below knee")
s = frc[frc.method == "lorawd"]
cells = s.groupby("cell").agg(logfd=("logfd", "mean"), adapt=("cs_avg", "mean")).dropna()
nb = int((cells.logfd < KNEE["frc"]).sum())
print(f"    lorawd ALL LRs: {nb}/{len(cells)} cells below knee; "
      f"max adapt below knee = {cells[cells.logfd < KNEE['frc']].adapt.max():.1f} (claims 12/31, 81.4)")
