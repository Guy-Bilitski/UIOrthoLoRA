"""B6: adjudication claims — SC-LoRA Qwen-math exception, op points, seed SDs,
LR band, head-to-head count, Pareto frontier membership.

Adjudication pool convention (adjpool.py): current pool (stragglers IN),
quarantine EXCLUDED from op stats, _reeval dropped, qwswm _ep6_ excluded,
frm c256 primary, math retention = headline.bbh.
"""
import numpy as np
import pandas as pd
from scipy import stats

import verify_common as vc

raw = vc.load_raw(drop_corda=True, drop_stragglers=False)
raw = raw[raw.run != vc.DUPLICATE]
quar = vc.load_quarantine()
raw["quar"] = raw.run.isin(quar)

d = raw[~raw.quar].copy()
d = d[~((d.fam == "qwswm") & d.run.str.contains("_ep6_"))]
# frm: keep only c256 (primary context)
frm_c = d[(d.fam == "frm")]
d = d[~((d.fam == "frm") & ~d.run.str.contains("_c256"))]
# retention column per family: math arms use BBH
d["ret_adj"] = np.where(d.fam.isin(["lrswm", "frm", "qwswm"]), d.bbh, d.ret)
d = d[np.isfinite(d.cs_avg)]
print(f"adjudication-convention pool n={len(d)}")

FAM4 = {"llama_cs": "lrsw", "llama_math": "frm", "qwen_cs": "qwsw", "qwen_math": "qwswm"}
BASE = {"lrsw": 26.0, "frm": 33.1, "qwsw": 44.35, "qwswm": 47.93}

print("\n--- best-adapt operating points per (family, method): mean+-sd over seeds,")
print("    best mean adaptation, cells with >=2 seeds preferred ---")
for label, fam in FAM4.items():
    s = d[d.fam == fam]
    rows = []
    for (meth, cell), g in s.groupby(["method", "cell"]):
        rows.append(dict(method=meth, cell=cell, n=len(g),
                         adapt=g.cs_avg.mean(), adapt_sd=g.cs_avg.std(),
                         ret=g.ret_adj.mean(), ret_sd=g.ret_adj.std(),
                         fdelta=g.fdelta.mean()))
    t = pd.DataFrame(rows)
    picks = []
    for meth, g in t.groupby("method"):
        multi = g[g.n >= 2]
        pool_ = multi if len(multi) else g
        picks.append(pool_.sort_values("adapt").iloc[-1])
    tt = pd.DataFrame(picks).sort_values("adapt", ascending=False)
    print(f"\n[{label} / {fam}] base ret = {BASE[fam]}")
    for r in tt.itertuples():
        print(f"  {r.method:10s} {r.cell:42s} n={r.n} adapt={r.adapt:.2f}"
              f"±{0 if np.isnan(r.adapt_sd) else r.adapt_sd:.2f} "
              f"ret={r.ret:.2f}±{0 if np.isnan(r.ret_sd) else r.ret_sd:.2f} F={r.fdelta:.3f}")

print("\n--- SC-LoRA Qwen-math exception ---")
s = d[(d.fam == "qwswm")]
sc = s[(s.method == "sclora") & (s.cell == "qwswm_sclora_r32_lr5e5")]
lw = s[(s.method == "lorawd") & (s.lr == 3e-4)]
print("SC-LoRA qwswm cells (all):")
for cell, g in s[s.method == "sclora"].groupby("cell"):
    print(f"  {cell}: n={len(g)} adapt={g.cs_avg.mean():.2f}±{g.cs_avg.std():.2f} "
          f"bbh={g.bbh.mean():.2f}±{g.bbh.std():.2f} F_delta={g.fdelta.mean():.3f} "
          f"(per-seed F: {list(g.fdelta.round(3))})")
print("LoRA+wd qwswm cells:")
for cell, g in s[s.method == "lorawd"].groupby("cell"):
    print(f"  {cell}: n={len(g)} adapt={g.cs_avg.mean():.2f}±{g.cs_avg.std():.2f} "
          f"bbh={g.bbh.mean():.2f}±{g.bbh.std():.2f}")
# paired t on GSM8K between sclora@5e-5 and lorawd@3e-4 by seed
best_lw_cell = None
lwc = s[s.method == "lorawd"].groupby("cell").agg(n=("run", "count"), a=("cs_avg", "mean"))
lwc_multi = lwc[lwc.n >= 2]
best_lw_cell = lwc_multi.a.idxmax()
a = s[(s.cell == "qwswm_sclora_r32_lr5e5")].set_index("seed").cs_avg
b = s[(s.cell == best_lw_cell)].set_index("seed").cs_avg
common = sorted(set(a.index) & set(b.index))
print(f"\npaired seeds {common}: sclora {list(a[common].round(2))} vs lorawd[{best_lw_cell}] {list(b[common].round(2))}")
if len(common) >= 2:
    t, pv = stats.ttest_rel(a[common], b[common])
    print(f"delta = {a[common].mean() - b[common].mean():+.2f} pp, paired t = {t:.2f} (p={pv:.3f})")
rb = s[(s.cell == "qwswm_sclora_r32_lr5e5")].bbh
rl = s[(s.cell == best_lw_cell)].bbh
print(f"BBH: sclora {rb.mean():.2f}±{rb.std():.2f} vs lorawd {rl.mean():.2f}±{rl.std():.2f} (base 47.93)")

print("\n--- seed SD of retention (median within-cell, cells >=2 seeds) ---")
for meth in ["lorawd", "sclora", "lora", "clora", "milora", "dora", "lora_null"]:
    sds = []
    for (fam, cell), g in d[d.method == meth].groupby(["fam", "cell"]):
        if fam not in BASE:
            continue
        if len(g) >= 2:
            sds.append(g.ret_adj.std())
    if sds:
        print(f"  {meth:10s} median within-cell ret SD = {np.median(sds):.2f} (n_cells={len(sds)})")
sc_qwsw = [g.ret_adj.std() for c, g in d[(d.method == "sclora") & (d.fam == "qwsw")].groupby("cell") if len(g) >= 2]
print(f"  sclora on qwsw: within-cell ret SDs = {[round(v,2) for v in sc_qwsw]} (claim median/large 7.08)")

print("\n--- safe-LR band (cell-mean retention >= base-2pp; LR <= 1e-3) ---")
band = {}
for fam in ["lrsw", "frm", "qwsw", "qwswm"]:
    s = d[(d.fam == fam) & np.isfinite(d.lr) & (d.lr <= 1e-3)]
    base = BASE[fam]
    # qwen-cs relative band: adjudication uses family-relative there; compute abs anyway
    for meth, g in s.groupby("method"):
        per_lr = g.groupby("lr").ret_adj.mean()
        # attempted also counts all-diverged LRs -> from raw incl quarantined
        raw_s = raw[(raw.fam == fam) & (raw.method == meth) & np.isfinite(raw.lr) & (raw.lr <= 1e-3)]
        att = raw_s.lr.nunique()
        safe = int((per_lr >= base - 2).sum())
        band.setdefault(meth, []).append((fam, safe, att))
print("method: [(family, safe_2pp, attempted)] and totals:")
for meth, v in sorted(band.items()):
    tot_s = sum(x[1] for x in v)
    tot_a = sum(x[2] for x in v)
    print(f"  {meth:10s} {v}  total {tot_s}/{tot_a}")
print("(claim: LoRA+wd 26/29 with qwsw scored on RELATIVE band; SC-LoRA 6/25; abs-band totals here will differ on qwsw)")

print("\n--- head-to-head table check ---")
h2h = pd.read_csv(vc.os.path.join(vc.ROOT, "paper/writing/acl_analysis/adjudication/tables/head2head.csv"))
print(h2h.to_string(index=False))

print("\n--- Pareto frontier membership (cell means, >=2 seeds cells + n=1 flagged) ---")
for label, fam in FAM4.items():
    s = d[d.fam == fam]
    cm = s.groupby(["method", "cell"]).agg(adapt=("cs_avg", "mean"), ret=("ret_adj", "mean"),
                                           n=("run", "count")).reset_index()
    pts = cm[["adapt", "ret"]].values
    nd = []
    for i, r in cm.iterrows():
        dominated = ((cm.adapt > r.adapt) & (cm.ret >= r.ret) | (cm.adapt >= r.adapt) & (cm.ret > r.ret)).any()
        if not dominated:
            nd.append(r)
    nd = pd.DataFrame(nd)
    meths = sorted(nd.method.unique())
    print(f"  {label}: non-dominated cells -> methods {meths}")
    for r in nd.sort_values("adapt", ascending=False).itertuples():
        print(f"      {r.method:10s} {r.cell:40s} n={r.n} adapt={r.adapt:.2f} ret={r.ret:.2f}")
