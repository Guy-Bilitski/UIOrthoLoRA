"""qa_recheck.py — INDEPENDENT second code path for every candidate insight of
the extra-insights pass (a1..a5). Different estimators on purpose:

  A  adaptation optimum: (i) run-level quadratic vertex with CR1 concavity t,
     (ii) binned-argmax + two-sided Spearman split at the frozen knee
     (no quadratic anywhere in this path).
  B1 slope moderators (rank/wd): cell-bootstrap CI of the interaction
     coefficient (no CR1).
  B2 SC-LoRA slope: residual-from-family-hinge path — fit hinge (knee fixed at
     section 18.2, slopes free) on NON-sclora runs, regress sclora cell
     residuals on dose. Plus cell bootstrap of the interaction from a2b.
  C  benchmark micro-structure: dose-decile MATCHING in raw pp (no regression,
     no ceiling normalization): each non-lorawd cell paired to lorawd cells
     within +/-0.15 dec of dose in the same family; paired diffs on bbh and
     mmlu_pro; sign census + Wilcoxon.
  D  seed lotteries: scipy Mann-Whitney AUC + p on dist-to-knee; threshold
     sensitivity (SD>2/3/4); logistic-style check via rank-biserial.

Both pool conventions (frozen / quarantine-excluded) where the claim needs it.
"""
import numpy as np
import pandas as pd
from scipy import stats
from extras_common import load_pool, cells_of, KNEE_182, FAMS, cr1_se

rng = np.random.default_rng(7)
df = load_pool()
cells = cells_of(df)

print("\n############ A. adaptation optimum, path 2 ############")
print("-- (i) run-level quadratic, CR1@cell concavity + vertex --")
for fam in FAMS:
    sub = df[(df.fam == fam) & df.adapt.notna()]
    x = sub.logfd.values
    X = np.column_stack([np.ones(len(x)), x, x * x])
    beta, *_ = np.linalg.lstsq(X, sub.adapt.values, rcond=None)
    resid = sub.adapt.values - X @ beta
    se = cr1_se(X, resid, sub.cell.values)
    t_conc = beta[2] / se[2]
    v = -beta[1] / (2 * beta[2]) if beta[2] < 0 else np.nan
    print(f"  {fam}: quad coef {beta[2]:+.2f} (t={t_conc:+.2f}) vertex {v:+.2f} "
          f"knee {KNEE_182[fam]:+.2f} diff {v - KNEE_182[fam]:+.2f} (runs n={len(sub)})")

print("-- (ii) quadratic-free: spearman(dose, adapt) below vs above frozen knee, cells --")
for fam in FAMS:
    c = cells[(cells.fam == fam) & cells.adapt.notna()]
    k = KNEE_182[fam]
    lo, hi = c[c.logfd <= k], c[c.logfd > k]
    r_lo = stats.spearmanr(lo.logfd, lo.adapt) if len(lo) > 3 else (np.nan, np.nan)
    r_hi = stats.spearmanr(hi.logfd, hi.adapt) if len(hi) > 3 else (np.nan, np.nan)
    # binned argmax (5 quantile bins)
    b = pd.qcut(c.logfd, 5, duplicates="drop")
    bm = c.groupby(b, observed=True).adapt.mean()
    top_bin = bm.idxmax()
    print(f"  {fam}: rho below knee {r_lo[0]:+.2f} (p={r_lo[1]:.3f}, n={len(lo)}); "
          f"above {r_hi[0]:+.2f} (p={r_hi[1]:.3f}, n={len(hi)}); "
          f"argmax dose-bin {top_bin}")

print("\n############ B1. rank/wd slope moderation, path 2 (cell bootstrap) ############")
def boot_interaction(sub, mod, nb=4000):
    cl = sub.cell.values
    ucells = pd.unique(cl)
    x, y, m = sub.logfd.values, sub.ret.values, np.asarray(mod, float)
    idx_by_cell = {c: np.where(cl == c)[0] for c in ucells}
    coefs = []
    for _ in range(nb):
        pick = rng.choice(ucells, len(ucells), replace=True)
        ii = np.concatenate([idx_by_cell[c] for c in pick])
        X = np.column_stack([np.ones(len(ii)), x[ii], m[ii], x[ii] * m[ii]])
        b, *_ = np.linalg.lstsq(X, y[ii], rcond=None)
        coefs.append(b[3])
    return np.percentile(coefs, [2.5, 97.5])

sub = df[(df.fam == "frc") & df["rank"].isin([8, 16, 32])]
lo, hi = boot_interaction(sub, np.log2(sub["rank"]))
print(f"  rank x dose interaction 95% cell-boot CI: [{lo:+.2f}, {hi:+.2f}]  (0 inside: {lo <= 0 <= hi})")
sub = df[(df.fam == "frc") & df.wd.notna()]
lo, hi = boot_interaction(sub, sub.wd)
print(f"  wd   x dose interaction 95% cell-boot CI: [{lo:+.2f}, {hi:+.2f}]  (0 inside: {lo <= 0 <= hi})")

print("\n############ B2. SC-LoRA slope, path 2 (residuals off family hinge) ############")
def sclora_resid_path(pool, label):
    print(f"  -- {label} --")
    for fam in FAMS:
        sub = pool[pool.fam == fam]
        base = sub[sub.method != "sclora"]
        sc = sub[sub.method == "sclora"]
        if sc.cell.nunique() < 4:
            print(f"    {fam}: <4 sclora cells, skip")
            continue
        k = KNEE_182[fam]
        Xb = np.column_stack([np.ones(len(base)), base.logfd, np.maximum(0, base.logfd - k)])
        b, *_ = np.linalg.lstsq(Xb, base.ret.values, rcond=None)
        # sclora CELL residuals vs dose
        scc = sc.groupby("cell").agg(logfd=("logfd", "mean"), ret=("ret", "mean")).reset_index()
        pred = b[0] + b[1] * scc.logfd + b[2] * np.maximum(0, scc.logfd - k)
        res = scc.ret - pred
        rho, p = stats.spearmanr(scc.logfd, res)
        sl = np.polyfit(scc.logfd, res, 1)[0]
        print(f"    {fam}: resid-vs-dose slope {sl:+.1f} pp/dec, spearman {rho:+.2f} "
              f"(p={p:.3f}, cells={len(scc)}); mean resid {res.mean():+.1f}")
sclora_resid_path(df, "frozen")
sclora_resid_path(df[~df.quarantined], "quarantine-excluded")

print("\n############ C. benchmark micro-structure, path 2 (dose-matched raw pp) ############")
def matched_diffs(pool, label, tol=0.15):
    print(f"  -- {label} (tol +/-{tol} dec) --")
    cm = pool.groupby(["fam", "cell"]).agg(
        method=("method", "first"), logfd=("logfd", "mean"),
        bbh=("bbh", "mean"), mmlu_pro=("mmlu_pro", "mean")).reset_index()
    census = {}
    for meth in sorted(cm.method.unique()):
        if meth in ("lorawd", "lorawdr16", "milorawd"):
            continue
        dif_pro_minus_bbh, per_fam = [], {}
        for fam in FAMS:
            ref = cm[(cm.fam == fam) & (cm.method == "lorawd")]
            mm = cm[(cm.fam == fam) & (cm.method == meth)]
            dd = []
            for _, r in mm.iterrows():
                near = ref[np.abs(ref.logfd - r.logfd) <= tol]
                if len(near) == 0:
                    continue
                d_bbh = r.bbh - near.bbh.mean()
                d_pro = r.mmlu_pro - near.mmlu_pro.mean()
                dd.append(d_pro - d_bbh)
            if dd:
                per_fam[fam] = np.mean(dd)
                dif_pro_minus_bbh += dd
        if len(per_fam) < 2:
            continue
        vals = np.array(list(per_fam.values()))
        neg = (vals < 0).sum()
        w = stats.wilcoxon(dif_pro_minus_bbh) if len(dif_pro_minus_bbh) >= 6 else (np.nan, np.nan)
        census[meth] = (neg, len(vals))
        print(f"    {meth:>10}: fam means " + ", ".join(f"{f}={v:+.1f}" for f, v in per_fam.items())
              + f" | neg {neg}/{len(vals)} | matched-cell wilcoxon p={w[1]:.4f} (pairs={len(dif_pro_minus_bbh)})")
    tot = sum(n for n, _ in census.values()); den = sum(d for _, d in census.values())
    print(f"    SIGN CENSUS (raw pp, matched): {tot}/{den} family-method pairs negative (Pro hit harder)")
matched_diffs(df, "frozen")
matched_diffs(df[~df.quarantined], "quarantine-excluded")

print("\n############ D. seed lotteries, path 2 ############")
def lottery_path2(pool, label):
    g = pool.groupby("cell")
    c = pd.DataFrame(dict(fam=g.fam.first(), n=g.size(), mean_lfd=g.logfd.mean(),
                          sd_lfd=g.logfd.std(), sd_ret=g.ret.std(),
                          sd_kl=g.forgetting_kl.std())).reset_index()
    c = c[c.n >= 3].copy()
    c["dist"] = c.mean_lfd - c.fam.map(KNEE_182)
    print(f"  -- {label}: cells={len(c)} --")
    for thr in (2.0, 3.0, 4.0):
        lot = c.sd_ret > thr
        if lot.sum() < 3:
            continue
        u, p = stats.mannwhitneyu(c.dist[lot], c.dist[~lot], alternative="greater")
        auc = u / (lot.sum() * (~lot).sum())
        # llama-only KL
        cl = c[c.fam.str.startswith(("lr", "fr"))].dropna(subset=["sd_kl"])
        lotl = cl.sd_ret > thr
        if lotl.sum() >= 3:
            ukl, pkl = stats.mannwhitneyu(cl.sd_kl[lotl], cl.sd_kl[~lotl], alternative="greater")
            auckl = ukl / (lotl.sum() * (~lotl).sum())
            kltxt = f"; SD(KL) llama AUC={auckl:.3f} p={pkl:.1e} (pos={int(lotl.sum())}, n={len(cl)})"
        else:
            kltxt = ""
        print(f"    SD>{thr:.0f}: n_lottery={int(lot.sum())}; dist-to-knee AUC={auc:.3f} "
              f"MW p={p:.1e}; median dist lot {c.dist[lot].median():+.2f} vs {c.dist[~lot].median():+.2f}{kltxt}")
    rho, p = stats.spearmanr(c.dist, c.sd_ret)
    print(f"    continuous: spearman(dist-to-knee, SD ret) = {rho:+.3f} (p={p:.1e}, n={len(c)})")
lottery_path2(df, "frozen")
lottery_path2(df[~df.quarantined], "quarantine-excluded")

print("\n############ E. wd adaptation-neutrality sanity (drop check) ############")
sub = df[(df.fam == "frc") & (df.lr == 3e-4) & df.wd.notna()]
cm = sub.groupby(["wd", "cell"]).adapt.mean().reset_index()
rho, p = stats.spearmanr(cm.wd, cm.adapt)
print(f"  wd@3e-4 spearman(wd, adapt) at cell level: {rho:+.2f} (p={p:.3f}, cells={len(cm)})")
