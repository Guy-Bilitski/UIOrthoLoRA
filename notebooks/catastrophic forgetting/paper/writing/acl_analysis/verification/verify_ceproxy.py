"""B5: CE-proxy claims — hardest verification.
Claims (correlations findings.md + ce_proxy.md):
  (i)  per-family KL->retention leave-cells-out RMSE 1.31-1.95 pp / MAE 0.83-1.19
       on the four Llama families; Qwen RMSE 5-6 pp (MAE 2.5-3.7).
  (ii) damage-detection AUC >= 0.976 in all six families
       (damaged = ret < family 90th-pct ceiling - 5pp, score = KL).
  (iii) KL knee ~ 0.26-0.29 nats in lrsw/lrswm/qwsw/qwswm; frc 0.40; frm no flat
        region (below-knee slope -8.3 pp/decade).
  (iv) KL-knee calibration beats the best log F_delta calibration in 6/6 families.

Independent implementation: own hinge (different knee grid), own CV folds
(different RNG seeds; 5- and 10-fold), own Mann-Whitney AUC.
"""
import numpy as np
import pandas as pd

import verify_common as vc

df = vc.frozen_pool()
assert vc.check_18_1(df, verbose=False)
df = df[df.run != vc.DUPLICATE]
ce = vc.load_ce()
p = df.merge(ce[["run", "kl"]], on="run", how="inner")
p = p[np.isfinite(p.kl)].copy()
p["logkl"] = np.log10(np.clip(p.kl, 1e-6, None))
print(f"pool with KL: n={len(p)}")


def cv(sub, cols, hinge_col=None, nfold=10, seed=12345):
    """Leave-cells-out CV; own fold assignment."""
    cells = sub.cell.unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cells))
    fold_of = {c: perm[i] % min(nfold, len(cells)) for i, c in enumerate(cells)}
    y = sub.ret.values.astype(float)
    pred = np.full(len(sub), np.nan)
    fold_ids = sub.cell.map(fold_of).values
    for f in np.unique(fold_ids):
        te = fold_ids == f
        tr = ~te
        if hinge_col:
            kk, beta = vc.hinge_fit(sub.loc[tr, hinge_col].values, y[tr])
            pred[te] = vc.hinge_pred(sub.loc[te, hinge_col].values, kk, beta)
        else:
            Xtr = np.column_stack([np.ones(tr.sum())] + [sub.loc[tr, c].values for c in cols])
            Xte = np.column_stack([np.ones(te.sum())] + [sub.loc[te, c].values for c in cols])
            beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
            pred[te] = Xte @ beta
    err = y - pred
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


print("\n(i)+(iv) leave-cells-out CV RMSE/MAE per family, several fold seeds:")
LLAMA = {"lrsw", "lrswm", "frc", "frm"}
summary = []
for fam in vc.FAMS:
    s = p[p.fam == fam].reset_index(drop=True)
    res = {}
    for label, kw in [("KL-knee", dict(cols=None, hinge_col="logkl")),
                      ("KL-lin", dict(cols=["kl"])),
                      ("KL-log", dict(cols=["logkl"])),
                      ("logF-lin", dict(cols=["logfd"])),
                      ("logF-knee", dict(cols=None, hinge_col="logfd"))]:
        rmses = []
        maes = []
        for seed in (1, 7, 99):
            for nf in (5, 10):
                r, a = cv(s, **kw, nfold=nf, seed=seed)
                rmses.append(r)
                maes.append(a)
        res[label] = (np.median(rmses), np.median(maes), min(rmses), max(rmses))
    best_kl = min(["KL-knee", "KL-lin", "KL-log"], key=lambda k: res[k][0])
    best_f = min(["logF-lin", "logF-knee"], key=lambda k: res[k][0])
    kl_beats_f = res[best_kl][0] < res[best_f][0]
    summary.append((fam, best_kl, res[best_kl], best_f, res[best_f], kl_beats_f))
    print(f"  {fam}: best-KL={best_kl} RMSE={res[best_kl][0]:.2f} "
          f"[{res[best_kl][2]:.2f},{res[best_kl][3]:.2f}] MAE={res[best_kl][1]:.2f} | "
          f"best-F={best_f} RMSE={res[best_f][0]:.2f} "
          f"[{res[best_f][2]:.2f},{res[best_f][3]:.2f}] -> KL beats F: {kl_beats_f}")

n_kl_wins = sum(1 for *_, w in summary if w)
print(f"  KL beats best-F calibration in {n_kl_wins}/6 families (claim 6/6)")
llama_rmse = [r[2][0] for r in summary if r[0] in LLAMA]
qwen_rmse = [r[2][0] for r in summary if r[0] not in LLAMA]
print(f"  Llama RMSE range {min(llama_rmse):.2f}-{max(llama_rmse):.2f} (claim 1.31-1.95)")
print(f"  Qwen RMSE range {min(qwen_rmse):.2f}-{max(qwen_rmse):.2f} (claim 5.0-6.0)")

print("\n(ii) damage-detection AUC (damaged = ret < 90th-pct - 5pp, score = KL):")
for fam in vc.FAMS:
    s = p[p.fam == fam]
    ceiling = np.quantile(s.ret, 0.90)
    dmg = (s.ret < ceiling - 5).values
    a_kl = vc.mw_auc(s.kl.values, dmg)
    a_f = vc.mw_auc(s.logfd.values, dmg)
    print(f"  {fam}: n_damaged={dmg.sum()}/{len(s)}  AUC(KL)={a_kl:.3f}  AUC(logF)={a_f:.3f}")

print("\n(iii) KL knee per family (hinge on log10 KL, full-family fit; own grid):")
for fam in vc.FAMS:
    s = p[p.fam == fam]
    kk, beta = vc.hinge_fit(s.logkl.values, s.ret.values)
    knee_nats = 10 ** kk
    print(f"  {fam}: knee = {knee_nats:.3f} nats (log10 {kk:+.3f}); "
          f"below-slope {beta[1]:+.2f} pp/decade, above-slope {beta[1] + beta[2]:+.2f}")
    # robustness: finer grid
    kk2, beta2 = vc.hinge_fit(s.logkl.values, s.ret.values, qgrid=np.linspace(0.02, 0.98, 97))
    print(f"        fine-grid knee = {10 ** kk2:.3f} nats; below {beta2[1]:+.2f} above {beta2[1] + beta2[2]:+.2f}")
