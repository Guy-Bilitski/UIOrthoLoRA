"""01 — Reproduce the frozen anchors before extending anything.

Anchors (key_numbers.md §18.1 / §19.1):
  (a) pooled r(ret, log10 F_delta) = -0.847, n=1035, per-family n/r to 3 dp
  (b) ladder A (frozen pool ∩ geometry, n=1034, family FE):
      R2 0.390 (FE) -> +0.395 (log F_delta) -> +0.017 (geometry: e_top,
      log spec_max, stable_rank) -> +0.006 (method dummies)
Writes anchors_reproduction.md.
"""
import numpy as np
import pandas as pd
import corr_common as cc

df, pre_txt = cc.build(dedupe=True)

lines = ["# Reproduction of frozen anchors", "", "```", pre_txt, "```", ""]

# ---- ladder A: frozen pool ∩ geometry --------------------------------------
lad = df.dropna(subset=["e_top_w", "lspec", "stable_rank_w"]).copy()
lines.append(f"Ladder pool: frozen(deduped) ∩ geometry, n={len(lad)} (target 1034)")

y = lad["ret"].values.astype(float)


def fit(terms, with_method=False):
    X, names = cc.design_fe(lad, terms)
    if with_method:
        meths = sorted(lad["method"].unique())
        for m in meths[1:]:
            X = np.column_stack([X, (lad["method"] == m).astype(float).values])
            names.append(f"m:{m}")
    _, _, r2, _ = cc.ols_fit(X, y)
    return r2


steps = [
    ("M0 family FE", fit([])),
    ("M1 + log10 F_delta", fit(["logfd"])),
    ("M2 + geometry (e_top, log spec_max, stable_rank)",
     fit(["logfd", "e_top_w", "lspec", "stable_rank_w"])),
    ("M3 + method dummies",
     fit(["logfd", "e_top_w", "lspec", "stable_rank_w"], with_method=True)),
]
target = [0.390, 0.785, 0.802, 0.808]
tgt_d = [0.390, 0.395, 0.017, 0.006]

lines += ["", "| step | R2 (mine) | dR2 (mine) | R2 (frozen) | dR2 (frozen) | match |",
          "|---|---|---|---|---|---|"]
prev = 0.0
ok_all = True
for (name, r2), tr2, td in zip(steps, target, tgt_d):
    d = r2 - prev
    ok = abs(r2 - tr2) < 0.0015 and abs(d - td) < 0.0015
    ok_all &= ok
    lines.append(f"| {name} | {r2:.3f} | {d:+.3f} | {tr2:.3f} | {td:+.3f} | {'OK' if ok else 'MISMATCH'} |")
    prev = r2

lines.append("")
lines.append(f"Ladder reproduction: {'ALL STEPS MATCH §19.1 to 3 decimals' if ok_all else 'MISMATCH — investigate'}")

# ---- commonality anchor (06 §5): shape-only split 0.296/0.016/0.099 ---------
shape = ["e_top_w", "stable_rank_w"]
sub = lad.dropna(subset=shape + ["logfd"])
yv = sub["ret"].values.astype(float)


def r2_of(terms):
    X, _ = cc.design_fe(sub, terms)
    return cc.ols_fit(X, yv)[2]


r_fe = r2_of([])
r_m = r2_of(["logfd"]) - r_fe
r_g = r2_of(shape) - r_fe
r_mg = r2_of(["logfd"] + shape) - r_fe
uniq_m, uniq_g = r_mg - r_g, r_mg - r_m
shared = r_m + r_g - r_mg
lines += ["", "Commonality anchor (06 §5, shape-only geometry = e_top + stable_rank):",
          f"  unique(magnitude) = {uniq_m:+.3f}  (frozen +0.296)",
          f"  unique(shape-geo) = {uniq_g:+.3f}  (frozen +0.016)",
          f"  shared            = {shared:+.3f}  (frozen +0.099)",
          f"  match: {'OK' if abs(uniq_m-0.296)<0.0015 and abs(uniq_g-0.016)<0.0015 and abs(shared-0.099)<0.0015 else 'MISMATCH'}"]

# ---- join coverage ----------------------------------------------------------
lines += ["", "## Join coverage (deduped pool, n=%d)" % len(df), "",
          "| family | n | geometry | CE/KL | ret_broad | adaptation |", "|---|---|---|---|---|---|"]
for fam in cc.FAMS:
    s = df[df.fam == fam]
    lines.append("| %s | %d | %d (%.0f%%) | %d (%.0f%%) | %d | %d |" % (
        fam, len(s), s.lspec.notna().sum(), 100 * s.lspec.notna().mean(),
        s.kl.notna().sum(), 100 * s.kl.notna().mean(),
        s.ret_broad.notna().sum(), s.adapt.notna().sum()))
s = df
lines.append("| ALL | %d | %d (%.0f%%) | %d (%.0f%%) | %d | %d |" % (
    len(s), s.lspec.notna().sum(), 100 * s.lspec.notna().mean(),
    s.kl.notna().sum(), 100 * s.kl.notna().mean(),
    s.ret_broad.notna().sum(), s.adapt.notna().sum()))
lines.append("")
lines.append("Duplicate run dropped after preflight: " + cc.DUPLICATE)

out = "\n".join(lines)
print(out)
open(cc.OUT + "/anchors_reproduction.md", "w").write(out + "\n")
