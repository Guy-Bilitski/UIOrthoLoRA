"""A1-A3 + B8: pool conventions, adaptation-retention sign, KL-vs-F within family,
observatory spot checks. Independent recompute (see verify_common.py)."""
import numpy as np
import pandas as pd

import verify_common as vc

pd.set_option("display.width", 200)

print("=" * 70)
print("A1. POOL CONVENTIONS")
print("=" * 70)

df = vc.frozen_pool()
assert vc.check_18_1(df), "frozen §18.1 preflight FAILED"
print("[OK] frozen pool n=1035 reproduces §18.1 exactly\n")

# observatory master: + corda + stragglers, all finite rows
full = vc.load_raw(drop_corda=False, drop_stragglers=False)
print(f"observatory-master expectation (finite rows incl corda + stragglers): n={len(full)}"
      f"  [observatory claims 1097]")
obs_master = pd.read_csv(vc.os.path.join(vc.ROOT, "paper/writing/acl_analysis/observatory/master_runs.csv"))
print(f"their master_runs.csv rows: {len(obs_master)}; on_pool sum = {int(obs_master.on_pool.sum())}")

# duplicate check
dup = df[df.run == vc.DUPLICATE]
parent = df[df.run == vc.DUPLICATE.replace("_reeval", "")]
print(f"\nduplicate present in frozen pool: {len(dup) == 1}; parent present: {len(parent) == 1}")
if len(dup) and len(parent):
    same = np.allclose(dup[["fdelta", "ret"]].values, parent[["fdelta", "ret"]].values)
    print(f"duplicate values identical to parent (fdelta, ret): {same}")

quar = vc.load_quarantine()
df["quar"] = df.run.isin(quar)
print(f"quarantined-but-finite runs inside frozen pool: {int(df.quar.sum())} (claim: 32)")

# convention robustness of headline pooled/per-family r
variants = {
    "frozen (n=1035, quar-in)": df,
    "deduped (n=1034)": df[df.run != vc.DUPLICATE],
    "quarantine-excluded": df[~df.quar],
    "current pool (stragglers in)": vc.load_raw(drop_stragglers=False),
}
print("\npooled and per-family r(logF, ret) under each pool convention:")
for name, d in variants.items():
    rp = np.corrcoef(d.logfd, d.ret)[0, 1]
    fam_r = {f: round(float(np.corrcoef(s.logfd, s.ret)[0, 1]), 3)
             for f, s in d.groupby("fam")}
    print(f"  {name:32s} n={len(d):4d} pooled r={rp:.3f}  {fam_r}")

print()
print("=" * 70)
print("A2. ADAPTATION-RETENTION SIGN")
print("=" * 70)
# observatory claim (frozen pool, run level): +0.16 lrsw, -0.22 lrswm, +0.49 qwsw,
# +0.71 qwswm, +0.24 frc, +0.86 frm. correlations claim: family-partialed +0.39.
for name, d in variants.items():
    dd = d.dropna(subset=["cs_avg"])
    per = {f: round(float(np.corrcoef(s.cs_avg, s.ret)[0, 1]), 2)
           for f, s in dd.groupby("fam")}
    fdum = [(dd.fam == f).astype(float).values for f in sorted(dd.fam.unique())[1:]]
    pr = vc.partial_r(dd.ret.values.astype(float), dd.cs_avg.values.astype(float), fdum)
    npos = sum(1 for v in per.values() if v > 0)
    print(f"  {name:32s} per-family r(adapt,ret)={per}  positive {npos}/6  family-partialed r={pr:+.2f}")

print()
print("=" * 70)
print("A3. KL vs F_DELTA WITHIN FAMILY (which wins, under which pool?)")
print("=" * 70)
ce = vc.load_ce()
m = df.merge(ce[["run", "kl"]], on="run", how="left")
m = m[np.isfinite(m.kl)]
print(f"CE join on frozen pool: n={len(m)}")


def kl_vs_f(d, label):
    wins = 0
    out = []
    for f, s in d.groupby("fam"):
        r2kl, *_ = vc.ols_r2([s.kl.values.astype(float)], s.ret.values.astype(float))
        # also log KL variant (fairer: 05 used raw KL? check both)
        lkl = np.log10(np.clip(s.kl.values.astype(float), 1e-6, None))
        r2lkl, *_ = vc.ols_r2([lkl], s.ret.values.astype(float))
        r2f, *_ = vc.ols_r2([s.logfd.values.astype(float)], s.ret.values.astype(float))
        best_kl = max(r2kl, r2lkl)
        w = best_kl > r2f
        wins += int(w)
        out.append(f"    {f}: R2(KL)={r2kl:.3f} R2(logKL)={r2lkl:.3f} R2(logF)={r2f:.3f}"
                   f"  KL-beats-F={w}")
    print(f"  [{label}] KL beats F_delta in {wins}/6 families")
    print("\n".join(out))


kl_vs_f(m, "frozen pool, quarantine INCLUDED (correlations convention)")
kl_vs_f(m[~m.quar], "quarantine EXCLUDED (pre-freeze doc-05 convention)")

print()
print("=" * 70)
print("B8. OBSERVATORY SPOT-CHECKS")
print("=" * 70)
geo = vc.load_geo()
g = df.merge(geo, on="run", how="inner")
print(f"geometry join on frozen pool: n={len(g)}")

# (a) spec_max === dw_sv_max
ok = np.isfinite(g.lspec) & np.isfinite(np.log10(g.dw_sv_max))
r = np.corrcoef(g.lspec[ok], np.log10(g.dw_sv_max[ok]))[0, 1]
print(f"(a) r(log spec_max, log dw_sv_max) = {r:.5f} on n={int(ok.sum())} (claim 1.0000, n=1034)")

# (b) stable-rank partial per family
print("(b) partial r(stable_rank, ret | log F_delta) per family "
      "(claim: -0.32..-0.67 Llama; qwsw -0.004, qwswm +0.073):")
for f, s in g.groupby("fam"):
    s = s[np.isfinite(s.stable_rank_w)]
    pr = vc.partial_r(s.ret.values.astype(float), s.stable_rank_w.values.astype(float),
                      [s.logfd.values.astype(float)])
    raw = np.corrcoef(s.stable_rank_w, s.ret)[0, 1]
    print(f"    {f}: partial={pr:+.3f} raw={raw:+.3f} (n={len(s)})")

# (c) matched-F_delta spread, lowest bins (binw=0.5, method>=2 runs, >=3 methods)
print("(c) matched-F_delta per-method spread in lowest kept bins "
      "(claim: lrswm@-1.0 0.63, lrsw@-1.0 0.99, qwswm@-1.5 0.91, qwsw@-1.0 1.78):")
d = df.copy()
d["fd_bin"] = (np.floor(d.logfd / 0.5) * 0.5).round(2)
gg = (d.groupby(["fam", "fd_bin", "method"])["ret"]
        .agg(n="count", mean="mean").reset_index())
gg = gg[gg.n >= 2]
keep = gg.groupby(["fam", "fd_bin"])["method"].nunique()
keep = keep[keep >= 3].reset_index()[["fam", "fd_bin"]]
gg = gg.merge(keep, on=["fam", "fd_bin"])
sp = (gg.groupby(["fam", "fd_bin"])["mean"]
        .agg(lambda s: s.max() - s.min()).reset_index(name="spread"))
for f in vc.FAMS:
    s = sp[sp.fam == f].sort_values("fd_bin")
    if len(s):
        lo = s.iloc[0]
        print(f"    {f}: lowest kept bin {lo.fd_bin:+.1f} spread={lo.spread:.2f} pp"
              f"   (all bins: {[(row.fd_bin, round(row.spread, 2)) for row in s.itertuples()]})")
