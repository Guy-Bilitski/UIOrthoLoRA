"""Shared helpers for the RQ1 statistics pass (2026-07-30).

RQ1: when retention-aware adapters are compared under one protocol, swept over
tasks, learning rates, and seeds, are any significantly different?

This layer adds the three pieces the frozen analyses lack:
  01  multiple-comparison correction (Holm) + exact p / CI on the 26 paired
      head-to-heads vs LoRA+wd (adjudication/03_head2head.py conventions);
  02  equivalence testing (TOST) for method offsets at matched magnitude
      (corr_common.py cluster conventions, deduped n=1034 pool);
  03  minimum detectable effects per family (power notes: why Qwen "n.s."
      is not evidence of no effect).

Data contract: identical to the frozen layers. Every script preflights
key_numbers.md section 18.1 before emitting anything. No frozen file is
modified; outputs land in this directory only.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACL = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ACL, "adjudication"))
sys.path.insert(0, os.path.join(ACL, "correlations"))

OUT = HERE

try:
    from scipy import stats as sps
except Exception:  # pragma: no cover
    sps = None


def t_sf(t, df):
    """Two-sided p from a t statistic."""
    if not np.isfinite(t) or df <= 0:
        return np.nan
    return 2.0 * sps.t.sf(abs(t), df)


def t_crit(alpha_two_sided, df):
    if df <= 0:
        return np.inf
    return sps.t.ppf(1.0 - alpha_two_sided / 2.0, df)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values. NaNs stay NaN and are not counted."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    m = mask.sum()
    if m == 0:
        return adj
    order = np.argsort(p[mask])
    idx = np.where(mask)[0][order]
    running = 0.0
    for k, i in enumerate(idx):
        val = (m - k) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def paired_stats(deltas):
    """mean, se, t, df, p, ci95 for a vector of paired per-seed deltas."""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 2:
        return dict(n=n, mean=d.mean() if n else np.nan, se=np.nan, t=np.nan,
                    df=0, p=np.nan, lo=np.nan, hi=np.nan)
    se = d.std(ddof=1) / math.sqrt(n)
    df = n - 1
    t = d.mean() / se if se > 0 else np.inf * np.sign(d.mean() or 1)
    p = t_sf(t, df)
    tc = t_crit(0.05, df)
    return dict(n=n, mean=d.mean(), se=se, t=t, df=df, p=p,
                lo=d.mean() - tc * se, hi=d.mean() + tc * se)


def welch_stats(a, b):
    """Welch two-sample stats with Welch-Satterthwaite df.

    If either group has n<2 its variance is unobservable, so the comparison is
    NOT TESTABLE: the delta is reported, everything inferential is NaN.
    (The frozen 03_head2head.py treated the singleton as noiseless; for exact
    p-values that would overstate certainty, so we refuse instead.)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    dm = a.mean() - b.mean()
    if na < 2 or nb < 2:
        return dict(n=min(na, nb), mean=dm, se=np.nan, t=np.nan, df=0,
                    p=np.nan, lo=np.nan, hi=np.nan)
    va = a.var(ddof=1) / na
    vb = b.var(ddof=1) / nb
    se = math.sqrt(va + vb)
    if se == 0:
        return dict(n=min(na, nb), mean=dm, se=np.nan, t=np.nan, df=0,
                    p=np.nan, lo=np.nan, hi=np.nan)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    t = dm / se
    p = t_sf(t, df)
    tc = t_crit(0.05, df)
    return dict(n=min(na, nb), mean=dm, se=se, t=t, df=df, p=p,
                lo=dm - tc * se, hi=dm + tc * se)


def mde_paired(sd_diff, n, alpha=0.05, power=0.8):
    """Minimum detectable |mean delta| for a two-sided paired t-test.

    Exact: solves for the noncentrality parameter of the noncentral t at which
    P(|T| > t_crit) = power (the usual (t_a + t_b) shortcut undershoots badly
    at df=2)."""
    if n < 2 or not np.isfinite(sd_diff) or sd_diff <= 0:
        return np.nan
    from scipy.optimize import brentq
    df = n - 1
    tc = sps.t.ppf(1 - alpha / 2, df)
    # Exact power of the two-sided t-test at noncentrality ncp, by integrating
    # the normal tail over the chi-square denominator (scipy's nct returns NaN
    # for the large ncp values that df=1 needs, so we avoid it). Quantile
    # transform v = F^-1(u) absorbs the df=1 pdf singularity at 0.
    u = np.linspace(5e-8, 1 - 5e-8, 8000)
    s = np.sqrt(sps.chi2.ppf(u, df) / df)

    def pw(ncp):
        tail = sps.norm.sf(tc * s - ncp) + sps.norm.cdf(-tc * s - ncp)
        return np.trapezoid(tail, u) - power

    ncp = brentq(pw, 0.0, 1000.0)
    return ncp * sd_diff / math.sqrt(n)
