"""01_dose_response.py — "all knobs are magnitude knobs": knob -> F_delta -> retention.

Three orthogonal knobs, all inside the frozen frc grid (Llama-2 CS, matched recipe):
  (1) weight decay  wd in {0, .1, .2, .3, .5}    (lorawd, LR grid 2e-5..1e-3)
  (2) CLoRA null-space k in {128,256,512,1024,2048} (lr 3e-4 fixed)
  (3) LoRA rank r in {8,16,32}                     (lr 3e-4 fixed)
Question: does each knob act on retention THROUGH F_delta (mediation), i.e. is the
knob's effect gone once you condition on the magnitude it produced?

Stats honesty: inference at CELL level (seed-averaged; within-cell ICC~0.78 makes
run-level SEs anticonservative). Run-level shown descriptively.
Outputs: dose_response_table.csv/.md, fig_dose_response.png/.pdf
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
df = pd.read_csv(os.path.join(HERE, "pool.csv"))

frc = df[df.fam == "frc"].copy()


def cellavg(d, keys):
    return d.groupby("cell", as_index=False).agg(
        {**{k: "mean" for k in keys}, "method": "first", "fam": "first"})


def partial_r(y, x, z):
    """r(y, x | z) via residualization (z = 2D array)."""
    z = np.column_stack([np.ones(len(y))] + list(z))
    ry = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    rx = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    r = np.corrcoef(ry, rx)[0, 1]
    n = len(y)
    dof = n - 2 - (z.shape[1] - 1)
    t = r * np.sqrt(dof / max(1e-12, 1 - r * r))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, t, p, n


lines = ["# Dose-response: knob -> F_delta -> retention (frc grid, Llama-2 CS)", ""]
rows_out = []

# ---------- (1) weight decay ----------
wd = frc[(frc.method == "lorawd") & frc.wd.notna()].copy()
wdc = cellavg(wd, ["wd", "lr", "logfd", "ret", "adapt"])
lines.append("## Knob 1: weight decay (lorawd cells, n_cell=%d, n_run=%d)" % (len(wdc), len(wd)))
# stage 1: knob -> F_delta, controlling LR (cells)
r1, t1, p1, n1 = partial_r(wdc.logfd.values, wdc.wd.values, [np.log10(wdc.lr.values)])
lines.append(f"- Stage 1 (cell level): partial r(log10 F_delta, wd | log LR) = {r1:.3f} (t={t1:.1f}, p={p1:.1e})")
# stage 2: knob -> retention beyond F_delta
r2, t2, p2, n2 = partial_r(wdc.ret.values, wdc.wd.values, [wdc.logfd.values])
r2l, _, _, _ = partial_r(wdc.ret.values, wdc.wd.values, [wdc.logfd.values, np.log10(wdc.lr.values)])
r0 = np.corrcoef(wdc.wd, wdc.ret)[0, 1]
lines.append(f"- Raw r(wd, retention) = {r0:.3f}; partial r(wd, ret | log F_delta) = {r2:.3f} (t={t2:.1f}, p={p2:.2f}); | logF + logLR = {r2l:.3f}")
# per-LR monotonicity table
tab = wdc.pivot_table(index="wd", columns="lr", values=["logfd", "ret"], aggfunc="mean")
lines.append("")
lines.append("wd -> mean F_delta (per LR column) [cells]:")
fdtab = wdc.pivot_table(index="wd", columns="lr", values="logfd")
lines.append("```")
lines.append((10 ** fdtab).round(3).to_string())
lines.append("```")
rettab = wdc.pivot_table(index="wd", columns="lr", values="ret")
lines.append("wd -> retention (per LR column) [cells]:")
lines.append("```")
lines.append(rettab.round(2).to_string())
lines.append("```")
for _, rr in wdc.iterrows():
    rows_out.append(dict(knob="wd", value=rr.wd, lr=rr.lr, fd=10**rr.logfd, ret=rr.ret, adapt=rr.adapt, cell=rr.cell))

# ---------- (2) CLoRA k ----------
ck = frc[(frc.method == "clora") & frc.clora_k.notna()].copy()
ckc = cellavg(ck, ["clora_k", "lr", "logfd", "ret", "adapt"])
ckc = ckc.sort_values("clora_k")
lines.append("")
lines.append("## Knob 2: CLoRA null-space dimension k (lr 3e-4 fixed, n_cell=%d, n_run=%d)" % (len(ckc), len(ck)))
rho1 = stats.spearmanr(np.log2(ckc.clora_k), ckc.logfd)
rho2 = stats.spearmanr(np.log2(ckc.clora_k), ckc.ret)
lines.append(f"- Stage 1: Spearman rho(log2 k, log F_delta) = {rho1.statistic:.3f} (p={rho1.pvalue:.3f}, n={len(ckc)})")
lines.append(f"- Stage 2: Spearman rho(log2 k, retention) = {rho2.statistic:.3f} (p={rho2.pvalue:.3f})")
lines.append("```")
lines.append(ckc[["clora_k", "logfd", "ret", "adapt"]].assign(fd=lambda d: 10**d.logfd)
             [["clora_k", "fd", "ret", "adapt"]].round(3).to_string(index=False))
lines.append("```")
# run-level: does k add anything beyond F_delta? (pool clora runs incl. seeds)
if len(ck) >= 8:
    r2k, t2k, p2k, _ = partial_r(ck.ret.values, np.log2(ck.clora_k.values), [ck.logfd.values])
    lines.append(f"- Run-level partial r(log2 k, ret | log F_delta) = {r2k:.3f} (t={t2k:.1f}, p={p2k:.2f}, n={len(ck)}) [descriptive; seeds not independent]")
for _, rr in ckc.iterrows():
    rows_out.append(dict(knob="clora_k", value=rr.clora_k, lr=rr.lr, fd=10**rr.logfd, ret=rr.ret, adapt=rr.adapt, cell=rr.cell))

# ---------- (3) rank ----------
rk = frc[(frc.method == "lora") & frc["rank"].notna() & (frc.lr == 3e-4)].copy()
rkc = cellavg(rk, ["rank", "lr", "logfd", "ret", "adapt"]).sort_values("rank")
lines.append("")
lines.append("## Knob 3: LoRA rank (lr 3e-4 fixed, n_cell=%d, n_run=%d)" % (len(rkc), len(rk)))
lines.append("```")
lines.append(rkc[["rank", "logfd", "ret", "adapt"]].assign(fd=lambda d: 10**d.logfd)
             [["rank", "fd", "ret", "adapt"]].round(3).to_string(index=False))
lines.append("```")
for _, rr in rkc.iterrows():
    rows_out.append(dict(knob="rank", value=rr["rank"], lr=rr.lr, fd=10**rr.logfd, ret=rr.ret, adapt=rr.adapt, cell=rr.cell))

# ---------- pooled mediation: all knob cells on the family curve ----------
lines.append("")
lines.append("## Pooled mediation check (all frc cells, cell level)")
frcc = cellavg(frc, ["logfd", "ret", "adapt", "lr"])
slope, icpt, r, p, se = stats.linregress(frcc.logfd, frcc.ret)
lines.append(f"- frc family curve (cells, n={len(frcc)}): ret = {icpt:.2f} + {slope:.2f}*log10 F_delta, r={r:.3f}")
frcc["resid"] = frcc.ret - (icpt + slope * frcc.logfd)
for name, sub in [("wd cells", frcc[frcc.cell.isin(wdc.cell)]),
                  ("clora-k cells", frcc[frcc.cell.isin(ckc.cell)]),
                  ("rank cells", frcc[frcc.cell.isin(rkc.cell)])]:
    lines.append(f"- {name}: mean on-curve residual = {sub.resid.mean():+.2f} pp (SD {sub.resid.std():.2f}, n={len(sub)})")

pd.DataFrame(rows_out).to_csv(os.path.join(HERE, "dose_response_table.csv"), index=False)
open(os.path.join(HERE, "dose_response_table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------- figure ----------
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.join(HERE, "..", ".."))
try:
    import figstyle  # noqa: F401  (repo style)
except Exception:
    pass
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
axA, axB, axC = axes

# panel A: knob -> F_delta (wd, one line per LR)
for lr_, sub in wdc.groupby("lr"):
    sub = sub.sort_values("wd")
    axA.plot(sub.wd, 10 ** sub.logfd, "o-", label=f"lr={lr_:.0e}", ms=4)
axA.set_yscale("log"); axA.set_xlabel("weight decay"); axA.set_ylabel(r"$F_\Delta$")
axA.set_title("A  wd shrinks the dose", loc="left")
axA.legend(fontsize=6, ncol=2)

# panel B: knob -> F_delta (clora k + rank)
axB.plot(ckc.clora_k, 10 ** ckc.logfd, "s-", color="C3", label="CLoRA k (lr3e-4)")
axB.set_xscale("log", base=2)
axB2 = axB.twiny()
axB2.plot(rkc["rank"], 10 ** rkc.logfd, "^--", color="C0", label="LoRA rank (lr3e-4)")
axB2.set_xscale("log", base=2)
axB.set_yscale("log")
axB.set_xlabel("CLoRA k (red)"); axB2.set_xlabel("LoRA rank (blue)")
axB.set_ylabel(r"$F_\Delta$")
axB.set_title("B  k and rank set the dose", loc="left")

# panel C: everything collapses on the family curve
axC.scatter(frcc.logfd, frcc.ret, s=8, color="0.8", label="all frc cells")
xs = np.linspace(frcc.logfd.min(), frcc.logfd.max(), 50)
axC.plot(xs, icpt + slope * xs, "-", color="0.4", lw=1)
sub = frcc[frcc.cell.isin(wdc.cell)]
axC.scatter(sub.logfd, sub.ret, s=22, color="C2", label="wd sweep", zorder=3)
sub = frcc[frcc.cell.isin(ckc.cell)]
axC.scatter(sub.logfd, sub.ret, s=30, color="C3", marker="s", label="CLoRA k sweep", zorder=3)
sub = frcc[frcc.cell.isin(rkc.cell)]
axC.scatter(sub.logfd, sub.ret, s=34, color="C0", marker="^", label="rank sweep", zorder=3)
axC.set_xlabel(r"$\log_{10} F_\Delta$"); axC.set_ylabel("retention (core)")
axC.set_title("C  ...and every dose lands on ONE curve", loc="left")
axC.legend(fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_dose_response.png"), dpi=200)
fig.savefig(os.path.join(HERE, "fig_dose_response.pdf"))
print("figure saved")
