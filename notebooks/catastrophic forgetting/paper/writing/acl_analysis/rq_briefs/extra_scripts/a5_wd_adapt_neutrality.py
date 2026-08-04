"""a5_wd_adapt_neutrality.py — direction (e): is weight decay adaptation-
neutral at matched dose, where CLoRA-k is not?

Insight 3 (findings.md) showed at fixed lr3e-4 raising k to 2048 costs
adaptation (cs_avg 76.8->69.4). Complement: does wd carry any adaptation
penalty beyond its dose effect? frc grid, cells; adapt ~ logfd + logfd^2 +
knob, CR1 @ cell on runs; same model for clora_k for contrast.
"""
import numpy as np
import pandas as pd
from extras_common import load_pool, ols, cr1_se

df = load_pool()


def knob_test(sub, knob_vals, label):
    x = sub.logfd.values
    X = np.column_stack([x, x ** 2, np.asarray(knob_vals, float)])
    beta, resid, _, _, Xf = ols(X, sub.adapt.values)
    se = cr1_se(Xf, resid, sub.cell.values)
    # and for retention
    br, rr_, _, _, Xr = ols(X, sub.ret.values)
    ser = cr1_se(Xr, rr_, sub.cell.values)
    print(f"{label}: n={len(sub)}  adapt: knob coef {beta[3]:+.2f} (t={beta[3]/se[3]:+.2f})"
          f" | ret: knob coef {br[3]:+.2f} (t={br[3]/ser[3]:+.2f})")


print("== frc grid: adaptation/retention residual effect of the knob beyond dose ==")
sub = df[(df.fam == "frc") & df.wd.notna() & df.adapt.notna()]
knob_test(sub, sub.wd, "wd (0..0.5), all lorawd arms")
sub2 = df[(df.fam == "frc") & df.clora_k.notna() & df.adapt.notna()]
knob_test(sub2, np.log2(sub2.clora_k), "log2 CLoRA k, clora arms")
sub3 = df[(df.fam == "frc") & df["rank"].notna() & df.adapt.notna() & df.method.isin(["lora", "lorawd", "lorawdr16"])]
knob_test(sub3, np.log2(sub3["rank"]), "log2 rank (8/16/32), lora(+wd) arms")

print("\n== same at fixed LR=3e-4 (the insight-3 operating point), cell means ==")
for label, mask, kv in [
    ("wd@3e-4", (df.fam == "frc") & (df.lr == 3e-4) & df.wd.notna(), "wd"),
    ("k@3e-4", (df.fam == "frc") & (df.lr == 3e-4) & df.clora_k.notna(), "clora_k"),
]:
    sub = df[mask]
    cm = sub.groupby(kv).agg(adapt=("adapt", "mean"), ret=("ret", "mean"),
                             logfd=("logfd", "mean"), n=("ret", "size"))
    print(f"-- {label} --"); print(cm.round(2).to_string())
