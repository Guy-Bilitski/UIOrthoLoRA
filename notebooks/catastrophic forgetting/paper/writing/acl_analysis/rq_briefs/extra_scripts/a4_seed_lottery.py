"""a4_seed_lottery.py — can train-time quantities predict which cells are
seed lotteries (within-cell retention SD > 3 pp)?

Unit of analysis = cell (n>=3 seeds), so this is inherently cluster-aware.
Predictors available WITHOUT running the retention eval:
  - cell mean log F_delta and its position vs the family knee (weight-space)
  - within-cell SD of log F_delta across seeds (weight-space)
  - within-cell SD of KL drift (behavior-space; Llama only, Qwen CE barred)
  - lr, method identity
AUC via rank statistic; logistic-free (no sklearn). Per-family consistency
check; quarantine-excluded sensitivity.
"""
import numpy as np
import pandas as pd
from extras_common import load_pool, KNEE_182, FAMS

df = load_pool()


def auc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, l = s[ok], l[ok]
    if l.sum() == 0 or (~l).sum() == 0:
        return np.nan, int(l.sum()), len(l)
    r = pd.Series(s).rank().values
    a = (r[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum())
    return a, int(l.sum()), len(l)


def build_cells(pool):
    g = pool.groupby("cell")
    c = pd.DataFrame(dict(
        fam=g.fam.first(), method=g.method.first(), n=g.size(),
        mean_lfd=g.logfd.mean(), sd_lfd=g.logfd.std(), sd_ret=g.ret.std(),
        mean_ret=g.ret.mean(), lr=g.lr.first(),
        sd_kl=g.forgetting_kl.std(), n_kl=g.forgetting_kl.count(),
        any_quar=g.quarantined.any())).reset_index()
    c = c[c.n >= 3].copy()
    c["dist_knee"] = c.mean_lfd - c.fam.map(KNEE_182)
    c["lottery"] = c.sd_ret > 3.0
    return c


def report(c, label):
    print(f"\n==== {label}: cells={len(c)}, lotteries={int(c.lottery.sum())} ====")
    preds = {
        "mean logF_delta": c.mean_lfd,
        "dist to family knee": c.dist_knee,
        "within-cell SD(logF_delta)": c.sd_lfd,
        "log10 lr": np.log10(c.lr),
        "SD(KL) [llama only]": c.sd_kl.where(c.fam.str.startswith(("lr", "fr"))),
    }
    for name, s in preds.items():
        a, npos, ntot = auc(s, c.lottery.values)
        print(f"  {name:<28} AUC={a:.3f} (pos={npos}, n={ntot})")
    # continuous version: spearman(sd_lfd, sd_ret), pooled and per family
    from scipy.stats import spearmanr
    rho, p = spearmanr(c.sd_lfd, c.sd_ret, nan_policy="omit")
    print(f"  spearman(SD logF_delta, SD ret) pooled: rho={rho:.3f} (p={p:.1e})")
    for fam in FAMS:
        cf = c[c.fam == fam]
        rho_f, p_f = spearmanr(cf.sd_lfd, cf.sd_ret, nan_policy="omit")
        a_f, npos_f, _ = auc(cf.dist_knee, cf.lottery.values)
        a_s, _, _ = auc(cf.sd_lfd, cf.lottery.values)
        print(f"    {fam}: rho={rho_f:+.2f} (n={len(cf)}, p={p_f:.3f}); "
              f"AUC dist_knee={a_f if np.isfinite(a_f) else float('nan'):.3f} "
              f"AUC sd_lfd={a_s if np.isfinite(a_s) else float('nan'):.3f} "
              f"(lotteries={npos_f})")
    # joint: is sd_lfd informative BEYOND dose position? rank-partial
    from scipy.stats import rankdata
    ok = np.isfinite(c.sd_lfd) & np.isfinite(c.dist_knee) & np.isfinite(c.sd_ret)
    rs = rankdata(c.sd_lfd[ok]); rd = rankdata(c.dist_knee[ok]); rr = rankdata(c.sd_ret[ok])
    def presid(a, b):
        b1 = np.polyfit(b, a, 1); return a - np.polyval(b1, b)
    pr = np.corrcoef(presid(rs, rd), presid(rr, rd))[0, 1]
    print(f"  rank-partial r(SD lfd, SD ret | dist_knee) = {pr:.3f} (n={ok.sum()})")
    # where are the lotteries relative to the knee?
    lot = c[c.lottery]
    print(f"  lottery cells dist_knee: median {lot.dist_knee.median():+.2f}, "
          f"IQR [{lot.dist_knee.quantile(.25):+.2f}, {lot.dist_knee.quantile(.75):+.2f}]; "
          f"non-lottery median {c[~c.lottery].dist_knee.median():+.2f}")
    print(f"  fraction of lottery cells above knee: {(lot.dist_knee>0).mean():.2f} "
          f"vs non-lottery {(c[~c.lottery].dist_knee>0).mean():.2f}")
    # simple two-variable rule: above knee OR sd_lfd high
    thr = c.sd_lfd.quantile(0.9)
    rule = (c.dist_knee > 0) | (c.sd_lfd > thr)
    tp = (rule & c.lottery).sum(); fn = ((~rule) & c.lottery).sum()
    fp = (rule & ~c.lottery).sum(); tn = ((~rule) & ~c.lottery).sum()
    print(f"  rule [above knee OR sd_lfd>q90]: sens={tp/(tp+fn):.2f} spec={tn/(tn+fp):.2f} "
          f"flagged={int(tp+fp)}/{len(c)}")
    # method composition of lotteries
    print("  lottery method counts:", lot.method.value_counts().to_dict())


c_frozen = build_cells(df)
report(c_frozen, "frozen pool")
c_qx = build_cells(df[~df.quarantined])
report(c_qx, "quarantine-excluded")
