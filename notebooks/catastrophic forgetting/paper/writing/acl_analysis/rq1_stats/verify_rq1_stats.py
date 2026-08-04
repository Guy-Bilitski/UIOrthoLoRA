"""verify — independent adversarial recompute of the rq1_stats outputs.

Follows the acl_analysis/verification/ pattern: own implementations, different
code paths, compare against the committed CSVs.

  V1  head2head_corrected.csv: rebuild every paired comparison straight from
      results/<run>/summary.json (own loader, no adjpool cell machinery for
      the stats; adjpool used only to identify the best cells), p via
      scipy.stats.ttest_rel, own Holm via a different algorithm
      (step-down max scan), CIs via scipy.

  V2  tost_offsets.csv pooled model: re-fit with a different design build
      (pandas get_dummies, different column order), CR1 via einsum, and
      re-check every equivalence flag from the stored lo90/hi90.

  V3  power_notes.csv: Monte-Carlo power check of the analytic MDE for three
      representative (sd, n) rows — simulated power at the analytic MDE must
      be 0.8 +/- 0.03.

Run: /home/guyb/UIOrthoLoRA/.venv/bin/python verify_rq1_stats.py
Writes verify_log.md; exits nonzero on any FAIL.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

from rq1_common import OUT, mde_paired
from adjpool import (FAMILIES, DISPLAY, WITHHELD, RES, load_pool,
                     preflight_18_1, family_rows, cell_table, best_cell)
import corr_common as cc

LOG = []
FAILS = []


def check(name, ok, detail=""):
    line = f"{'OK  ' if ok else 'FAIL'} {name}" + (f" -- {detail}" if detail else "")
    LOG.append(line)
    if not ok:
        FAILS.append(line)


def holm_alt(p):
    """Independent Holm: explicit step-down with cummax on sorted array."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    k = m.sum()
    if k == 0:
        return out
    idx = np.where(m)[0]
    srt = idx[np.argsort(p[idx])]
    vals = np.minimum.accumulate(np.ones(k))  # placeholder shape
    vals = np.maximum.accumulate([(k - i) * p[j] for i, j in enumerate(srt)])
    out[srt] = np.minimum(vals, 1.0)
    return out


def raw_seed_values(run_prefix_rows, ret_field):
    """{seed: (adapt, ret)} read directly from summary.json files."""
    out = {}
    for _, r in run_prefix_rows.iterrows():
        p = os.path.join(RES, r.run, "summary.json")
        h = json.load(open(p)).get("headline", {})
        ret = h.get("retention_mean") if ret_field == "ret_core" else h.get(ret_field)
        if r.seed is not None and h.get("cs_avg") is not None and ret is not None:
            out[int(r.seed)] = (float(h["cs_avg"]), float(ret))
    return out


def v1():
    got = pd.read_csv(os.path.join(OUT, "head2head_corrected.csv"))
    df = load_pool()
    preflight_18_1(df)
    recomputed = []
    for fk, spec in FAMILIES.items():
        fr = family_rows(df, fk)
        cells = cell_table(fr, spec["ret_field"])
        ref = best_cell(cells, "lorawd")
        sub_w = fr[(fr.mkey == "lorawd") & (fr.lr == ref.lr)]
        if ref.k:
            sub_w = sub_w[sub_w.run.str.contains(f"_{ref.k}_")]
        vw = raw_seed_values(sub_w, spec["ret_field"])
        for mkey, _ in spec["specs"]:
            if mkey == "lorawd" or mkey in WITHHELD:
                continue
            bc = best_cell(cells, mkey)
            if bc is None:
                continue
            sub_m = fr[(fr.mkey == mkey) & (fr.lr == bc.lr)]
            if bc.k:
                sub_m = sub_m[sub_m.run.str.contains(f"_{bc.k}_")]
            vm = raw_seed_values(sub_m, spec["ret_field"])
            common = sorted(set(vm) & set(vw))
            row = dict(family=fk, method=DISPLAY[mkey])
            if len(common) >= 2:
                dr = [vm[s][1] - vw[s][1] for s in common]
                da = [vm[s][0] - vw[s][0] for s in common]
                tr = sps.ttest_rel([vm[s][1] for s in common], [vw[s][1] for s in common])
                ta = sps.ttest_rel([vm[s][0] for s in common], [vw[s][0] for s in common])
                row.update(d_ret=np.mean(dr), p_ret=tr.pvalue,
                           d_adapt=np.mean(da), p_adapt=ta.pvalue)
            else:
                a = [v[1] for v in vm.values()]
                b = [v[1] for v in vw.values()]
                w = sps.ttest_ind(a, b, equal_var=False)
                aa = [v[0] for v in vm.values()]
                ab = [v[0] for v in vw.values()]
                wa = sps.ttest_ind(aa, ab, equal_var=False)
                row.update(d_ret=np.mean(a) - np.mean(b),
                           p_ret=w.pvalue if np.isfinite(w.pvalue) else np.nan,
                           d_adapt=np.mean(aa) - np.mean(ab),
                           p_adapt=wa.pvalue if np.isfinite(wa.pvalue) else np.nan)
            recomputed.append(row)
    rec = pd.DataFrame(recomputed)
    mrg = got.merge(rec, on=["family", "method"], suffixes=("", "_v"))
    check("V1 row count matches", len(mrg) == len(got) == len(rec),
          f"{len(got)} committed vs {len(rec)} recomputed")
    for col in ("d_ret", "p_ret", "d_adapt", "p_adapt"):
        a, b = mrg[col].values, mrg[f"{col}_v"].values
        m = np.isfinite(a) & np.isfinite(b)
        bad = np.abs(a[m] - b[m]) > 5e-3
        check(f"V1 {col} agrees (|diff|<=5e-3)", not bad.any(),
              f"max diff {np.max(np.abs(a[m]-b[m])):.2e}, n={m.sum()}")
        both_nan = (~np.isfinite(a)) == (~np.isfinite(b))
        check(f"V1 {col} NaN pattern matches", both_nan.all())
    # Holm re-check with the alternative implementation, on the verifier's own
    # (unrounded) recomputed p-values; tolerance covers the committed CSV's
    # 4-dp rounding amplified by the (m-k) Holm factor.
    for ax in ("ret", "adapt"):
        alt = holm_alt(mrg[f"p_{ax}_v"].values)
        a = mrg[f"p_{ax}_holm_all"].values
        m = np.isfinite(a) & np.isfinite(alt)
        check(f"V1 Holm(all,{ax}) agrees", np.max(np.abs(a[m] - alt[m])) <= 2e-3,
              f"max diff {np.max(np.abs(a[m]-alt[m])):.2e}")
        check(f"V1 Holm(all,{ax}) NaN pattern matches",
              (np.isfinite(a) == np.isfinite(alt)).all())
    # headline claims
    n_better = ((mrg.ret_verdict_holm == "BETTER")).sum()
    n_worse = ((mrg.ret_verdict_holm == "WORSE")).sum()
    re_better = ((mrg.p_ret_holm_all < 0.05) & (mrg.d_ret > 0)).sum()
    re_worse = ((mrg.p_ret_holm_all < 0.05) & (mrg.d_ret < 0)).sum()
    check("V1 verdict counts consistent",
          n_better == re_better and n_worse == re_worse,
          f"better {n_better}={re_better}, worse {n_worse}={re_worse}")
    sc = mrg[(mrg.family == "qwen_math") & (mrg.method == "SC-LoRA")].iloc[0]
    check("V1 SC-LoRA qwen-math n.s. after Holm",
          sc.p_adapt_holm_all >= 0.05 and sc.p_adapt < 0.05,
          f"raw {sc.p_adapt:.4f}, holm-all {sc.p_adapt_holm_all:.4f}")


def v2():
    got = pd.read_csv(os.path.join(OUT, "tost_offsets.csv"))
    pooled = got[got.scope.str.startswith("pooled")].set_index("method")
    df, _ = cc.build(dedupe=True, verbose=False)
    sub = df.dropna(subset=["logfd", "ret"]).copy()
    D = pd.get_dummies(sub[["fam", "method"]], drop_first=False, dtype=float)
    fam_cols = sorted(c for c in D if c.startswith("fam_"))[1:]
    meth_cols = sorted(c for c in D if c.startswith("method_") and c != "method_lorawd")
    X = np.column_stack([np.ones(len(sub)), sub.logfd.values]
                        + [D[c].values for c in fam_cols]
                        + [D[c].values for c in meth_cols])
    names = ["const", "logfd"] + fam_cols + meth_cols
    beta, *_ = np.linalg.lstsq(X, sub.ret.values, rcond=None)
    resid = sub.ret.values - X @ beta
    # CR1 via einsum (different path from corr_common's loop)
    ids = sub.cell.astype(str).values
    uniq, inv = np.unique(ids, return_inverse=True)
    G, n, p = len(uniq), *X.shape
    S = np.zeros((G, p))
    np.add.at(S, inv, X * resid[:, None])
    meat = S.T @ S
    bread = np.linalg.pinv(X.T @ X)
    V = (G / (G - 1)) * ((n - 1) / (n - p)) * bread @ meat @ bread
    se = np.sqrt(np.diag(V))
    tc90 = sps.t.ppf(0.95, G - 1)
    ok_all = True
    details = []
    for c in meth_cols:
        m = c[len("method_"):]
        i = names.index(c)
        if m not in pooled.index:
            ok_all = False
            details.append(f"{m} missing from committed pooled table")
            continue
        row = pooled.loc[m]
        db = abs(beta[i] - row.beta)
        dse = abs(se[i] - row.se)
        if db > 5e-3 or dse > 5e-3:
            ok_all = False
            details.append(f"{m}: beta diff {db:.2e}, se diff {dse:.2e}")
        for marg in (1.0, 2.0, 3.0):
            eq = (beta[i] - tc90 * se[i] > -marg) and (beta[i] + tc90 * se[i] < marg)
            if bool(row[f"equiv_{marg:g}pp"]) != eq:
                ok_all = False
                details.append(f"{m}: equiv flag mismatch at {marg:g}pp")
    check("V2 pooled TOST model reproduces (beta, se, flags)", ok_all,
          "; ".join(details) if details else f"{len(meth_cols)} methods, G={G}")
    # flags internally consistent with stored CIs, all scopes
    for marg in (1.0, 2.0, 3.0):
        implied = (got.lo90 > -marg) & (got.hi90 < marg)
        check(f"V2 equiv_{marg:g}pp flags match stored 90% CIs",
              (implied == got[f"equiv_{marg:g}pp"]).all())


def v3():
    got = pd.read_csv(os.path.join(OUT, "power_notes.csv"))
    rows = got.dropna(subset=["mde_pp", "sd_paired_diff"])
    rows = rows[rows.n_common_seeds >= 2]
    picks = rows.iloc[[0, len(rows) // 2, len(rows) - 1]]
    rng = np.random.default_rng(0)
    ok_all = True
    details = []
    for _, x in picks.iterrows():
        n, sd, mde = int(x.n_common_seeds), float(x.sd_paired_diff), float(x.mde_pp)
        analytic = mde_paired(sd, n)
        if abs(analytic - mde) > 1e-2:
            ok_all = False
            details.append(f"{x.family}/{x.method}: stored {mde} != analytic {analytic:.3f}")
            continue
        sims = rng.normal(mde, sd, size=(20000, n))
        tcrit = sps.t.ppf(0.975, n - 1)
        tstat = sims.mean(1) / (sims.std(1, ddof=1) / np.sqrt(n))
        power = (np.abs(tstat) > tcrit).mean()
        if abs(power - 0.8) > 0.03:
            ok_all = False
            details.append(f"{x.family}/{x.method}: simulated power {power:.3f}")
        else:
            details.append(f"{x.family}/{x.method}: power {power:.3f}")
    check("V3 MDE analytic == stored, simulated power ~0.8", ok_all,
          "; ".join(details))


def main():
    for fn in (v1, v2, v3):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(f"{fn.__name__} raised", False, repr(e))
    hdr = ["# rq1_stats verification log", "",
           f"verdict: {'ALL OK' if not FAILS else f'{len(FAILS)} FAILURE(S)'}", ""]
    with open(os.path.join(OUT, "verify_log.md"), "w") as fh:
        fh.write("\n".join(hdr + LOG) + "\n")
    print("\n".join(hdr + LOG))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
