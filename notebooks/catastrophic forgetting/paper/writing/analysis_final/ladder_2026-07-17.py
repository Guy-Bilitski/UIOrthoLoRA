"""Pooled nested-model ΔR² ladder (2026-07-17, post-freeze addendum §19).

Question: how much marginal variance in retention does each axis explain, in ONE
nested-regression table — magnitude (log10 F_Δ), CE drift (KL to base), geometry
(e_top_w, log10 spec_max, stable_rank_w), method identity — beyond family effects?
This is the single-exhibit version of the "magnitude first-order, geometry
second-order" claim whose evidence was previously scattered across partials
(dyn4_geometry G3, 05_adapter_dynamics).

Freeze compliance:
- Loader is byte-identical in convention to analyze_full_2026-07-16.py:
  results/*/summary.json, drop SMOKE/smoke/corda substrings, require finite
  headline.fdelta>0 and finite headline.retention_mean, families
  lrsw/lrswm/qwsw/qwswm/frc/frm by run_name prefix.
- SEVEN runs synced 2026-07-17 12:49 — AFTER the §18 freeze was cut — are named
  in STRAGGLERS below and excluded from the PRIMARY pool. Preflight hard-asserts
  the primary pool reproduces key_numbers.md §18.1 (n=1035, pooled r=-0.847,
  per-family n and r to 3 decimals) and ABORTS on mismatch. The current pool
  (n=1042, stragglers included) is reported as a sensitivity variant only.
- Inputs: results/*/summary.json, results/forgetting_merged.jsonl (key run_name),
  results/geo_drift/adapter_metrics_merged.jsonl (key run),
  results/quarantine_diverged.txt. Pure stdlib. No numbers sourced from
  campaign_summary.jsonl or results_book/.

Primary ladder convention: run-level rows, family fixed effects, quarantine
INCLUDED (freeze convention: finite-value filter only; 32 quarantined runs are
legitimate far-end points inside n=1035 — 01_law_final.md §1.1). Variants:
quarantine-excluded, seed-averaged cells, no-family-FE pooled, current pool.
CE regressor = forgetting_kl (CE − base entropy; comparable across base models).
"""
import json, glob, math, os, re
from collections import defaultdict

RES = "results"

STRAGGLERS = {
    # synced 2026-07-17 12:49, post-freeze; verified: excluding exactly these
    # reproduces §18.1 per-family n and r to 3 decimals.
    "lrsw_clora_k1024_lr3e4_s45",
    "qwsw_lora_null_r16_lr5e4_s43",
    "qwsw_milora_r32_lr5e5_s44",
    "qwswm_clora_k1024_lr2e5_s44",
    "qwswm_dora_r16_lr2e4_s43",
    "qwswm_lora_r16_lr1e4_s44",
    "qwswm_sclora_r32_lr3e4_s44",
}

FROZEN = {  # §18.1 / 01_law_final.md §1.1 (n, r) per family
    "lrsw": (180, -0.886), "lrswm": (120, -0.865), "qwsw": (151, -0.840),
    "qwswm": (164, -0.830), "frc": (276, -0.928), "frm": (144, -0.929),
}
FROZEN_POOLED_R = -0.847
FAMS = ["lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"]


def method_of(rn):
    body = rn.split("_", 1)[1]
    if body.startswith("lora_null"):
        return "lora_null"
    return body.split("_")[0]


def load_rows():
    rows = []
    for f in glob.glob(os.path.join(RES, "*", "summary.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rn = d.get("run_name") or os.path.basename(os.path.dirname(f))
        if "SMOKE" in rn or "smoke" in rn or "corda" in rn:
            continue
        h = d.get("headline") or {}
        fd, ret = h.get("fdelta"), h.get("retention_mean")
        if not isinstance(fd, (int, float)) or not isinstance(ret, (int, float)):
            continue
        if not (math.isfinite(fd) and math.isfinite(ret)) or fd <= 0:
            continue
        m = re.match(r"^([a-z0-9]+)_", rn)
        fam = m.group(1) if m else "other"
        if fam not in FROZEN:
            continue
        sm = re.search(r"_s(4[2-9])$", rn)
        rows.append(dict(rn=rn, fam=fam, seed=sm.group(1) if sm else None,
                         cell=re.sub(r"_s4[2-9]$", "", rn),
                         method=method_of(rn), logfd=math.log10(fd), ret=ret))
    return rows


def load_ce():
    ce = {}
    with open(os.path.join(RES, "forgetting_merged.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            k = d.get("run_name")
            kl = d.get("forgetting_kl")
            if k and isinstance(kl, (int, float)) and math.isfinite(kl):
                ce[k] = kl
    return ce


def load_geo():
    geo = {}
    with open(os.path.join(RES, "geo_drift", "adapter_metrics_merged.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            k = d.get("run")
            sp, et, sr = d.get("spec_max"), d.get("e_top_w"), d.get("stable_rank_w")
            if k and all(isinstance(v, (int, float)) and math.isfinite(v) for v in (sp, et, sr)) and sp > 0:
                geo[k] = dict(lspec=math.log10(sp), e_top=et, srank=sr)
    return geo


def load_quarantine():
    q = set()
    with open(os.path.join(RES, "quarantine_diverged.txt")) as fh:
        for line in fh:
            name = line.split("\t")[0].strip()
            if name:
                q.add(name)
    return q


def pearson(pairs):
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n < 3:
        return float("nan"), n
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return float("nan"), n
    return cov / math.sqrt(vx * vy), n


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            raise ValueError("singular design matrix")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [vr - f * vc for vr, vc in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def ols(X, y):
    n, p = len(X), len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    beta = solve(XtX, Xty)
    ybar = sum(y) / n
    sst = sum((v - ybar) ** 2 for v in y)
    ssr = sum((y[i] - sum(beta[a] * X[i][a] for a in range(p))) ** 2 for i in range(n))
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else float("nan")
    return dict(beta=beta, ssr=ssr, sst=sst, r2=r2, adj=adj, n=n, p=p)


def design(rows, terms, methods=None, fams_present=None):
    fams_present = fams_present or sorted({r["fam"] for r in rows})
    fam_d = fams_present[1:]
    meth_d = (methods or [])[1:]
    X, names = [], (["const"] + [f"fam:{f}" for f in fam_d]
                    + [t for t in terms]
                    + [f"m:{m}" for m in meth_d])
    for r in rows:
        row = [1.0] + [1.0 if r["fam"] == f else 0.0 for f in fam_d]
        row += [r[t] for t in terms]
        row += [1.0 if r["method"] == m else 0.0 for m in meth_d]
        X.append(row)
    return X, names


def ftest(m_restricted, m_full):
    q = m_full["p"] - m_restricted["p"]
    if q <= 0 or m_full["n"] <= m_full["p"]:
        return float("nan")
    return ((m_restricted["ssr"] - m_full["ssr"]) / q) / (m_full["ssr"] / (m_full["n"] - m_full["p"]))


def run_ladder(rows, label, with_ce, no_fam_fe=False):
    methods = sorted({r["method"] for r in rows})
    fams_present = sorted({r["fam"] for r in rows}) if not no_fam_fe else ["_pooled_"]
    if no_fam_fe:
        for r in rows:
            r = r  # fam dummies suppressed via fams_present of length 1
    y = [r["ret"] for r in rows]

    steps = [("M0 family FE only", [], None)]
    steps.append(("M1 + log10 F_delta", ["logfd"], None))
    if with_ce:
        steps.append(("M2 + KL (CE drift)", ["logfd", "kl"], None))
        steps.append(("M3 + geometry (e_top, log spec_max, stable_rank)",
                      ["logfd", "kl", "e_top", "lspec", "srank"], None))
        steps.append(("M4 + method dummies",
                      ["logfd", "kl", "e_top", "lspec", "srank"], methods))
    else:
        steps.append(("M2 + geometry (e_top, log spec_max, stable_rank)",
                      ["logfd", "e_top", "lspec", "srank"], None))
        steps.append(("M3 + method dummies",
                      ["logfd", "e_top", "lspec", "srank"], methods))

    print(f"\n--- LADDER: {label} (n={len(rows)}, {len(methods)} methods: {','.join(methods)}) ---")
    print(f"{'step':<55} {'R2':>7} {'adjR2':>7} {'dR2':>7} {'F(step)':>8}")
    prev = None
    fits = []
    for name, terms, meths in steps:
        X, _ = design(rows, terms, methods=meths, fams_present=fams_present)
        m = ols(X, y)
        d = m["r2"] - prev["r2"] if prev else m["r2"]
        F = ftest(prev, m) if prev else float("nan")
        print(f"{name:<55} {m['r2']:>7.3f} {m['adj']:>7.3f} {d:>7.3f} {F:>8.1f}")
        fits.append((name, terms, meths, m))
        prev = m

    # alternate insertion: geometry directly after magnitude (skipping CE)
    if with_ce:
        X, _ = design(rows, ["logfd", "e_top", "lspec", "srank"], fams_present=fams_present)
        m_geo = ols(X, y)
        m1 = fits[1][3]
        print(f"{'[alt] M1 + geometry (no CE)':<55} {m_geo['r2']:>7.3f} {m_geo['adj']:>7.3f} "
              f"{m_geo['r2']-m1['r2']:>7.3f} {ftest(m1, m_geo):>8.1f}")
        X, _ = design(rows, ["kl"], fams_present=fams_present)
        m_kl = ols(X, y)
        m0 = fits[0][3]
        print(f"{'[alt] M0 + KL only (no magnitude)':<55} {m_kl['r2']:>7.3f} {m_kl['adj']:>7.3f} "
              f"{m_kl['r2']-m0['r2']:>7.3f} {ftest(m0, m_kl):>8.1f}")

    # standardized coefficients of the final pre-method model
    terms = fits[-2][1]
    mu = {t: sum(r[t] for r in rows) / len(rows) for t in terms + ["ret"]}
    sd = {t: math.sqrt(sum((r[t] - mu[t]) ** 2 for r in rows) / len(rows)) for t in terms}
    sd["ret"] = math.sqrt(sum((r["ret"] - mu["ret"]) ** 2 for r in rows) / len(rows))
    X, names = design(rows, terms, fams_present=fams_present)
    m = ols(X, y)
    print("standardized betas (final pre-method model, |beta| = axis strength):")
    for nm, b in zip(names, m["beta"]):
        if nm in terms:
            print(f"    {nm:<8} beta_std = {b * sd[nm] / sd['ret']:+.3f}")
    return fits


def main():
    rows_all = load_rows()
    primary = [r for r in rows_all if r["rn"] not in STRAGGLERS]
    current = rows_all

    print("=" * 78)
    print("POOLED NESTED dR2 LADDER — post-freeze addendum, 2026-07-17")
    print("=" * 78)

    # ---------- PREFLIGHT: reproduce §18.1 exactly or abort ----------
    print("\n[PREFLIGHT] frozen-pool reproduction of key_numbers.md §18.1")
    assert len(primary) == 1035, f"pool n={len(primary)} != 1035"
    for fam, (wn, wr) in FROZEN.items():
        sub = [(r["logfd"], r["ret"]) for r in primary if r["fam"] == fam]
        r, n = pearson(sub)
        assert n == wn, f"{fam}: n={n} != {wn}"
        assert abs(r - wr) < 0.0005, f"{fam}: r={r:.4f} != {wr}"
        print(f"  {fam}: n={n} r={r:.3f}  OK")
    rp, _ = pearson([(r["logfd"], r["ret"]) for r in primary])
    assert abs(rp - FROZEN_POOLED_R) < 0.0005, f"pooled r={rp:.4f}"
    print(f"  pooled: n=1035 r={rp:.3f}  OK — §18.1 reproduced, proceeding")
    print(f"  stragglers excluded from primary pool (synced post-freeze 07-17 12:49): {len(STRAGGLERS)}")
    for s in sorted(STRAGGLERS):
        print(f"    {s}")

    ce, geo, quar = load_ce(), load_geo(), load_quarantine()

    # ---------- join coverage ----------
    print("\n[JOIN COVERAGE] per family (primary pool)")
    print(f"{'family':<8} {'n':>5} {'CE':>5} {'CE%':>6} {'geo':>5} {'geo%':>6}")
    for fam in FAMS:
        sub = [r for r in primary if r["fam"] == fam]
        nce = sum(1 for r in sub if r["rn"] in ce)
        ng = sum(1 for r in sub if r["rn"] in geo)
        print(f"{fam:<8} {len(sub):>5} {nce:>5} {100*nce/len(sub):>5.1f}% {ng:>5} {100*ng/len(sub):>5.1f}%")

    for r in primary:
        if r["rn"] in ce:
            r["kl"] = ce[r["rn"]]
        if r["rn"] in geo:
            r.update(geo[r["rn"]])

    geo_pool = [r for r in primary if "lspec" in r]
    ce_geo_pool = [r for r in geo_pool if "kl" in r]

    # ---------- primary ladders ----------
    run_ladder(geo_pool, "PRIMARY A — frozen pool ∩ geometry (no CE step)", with_ce=False)
    run_ladder(ce_geo_pool, "PRIMARY B — frozen pool ∩ geometry ∩ CE", with_ce=True)

    # ---------- variants ----------
    print("\n" + "=" * 78 + "\nVARIANTS (headline dR2 lines only)\n" + "=" * 78)

    # V1: quarantine excluded
    v1 = [r for r in geo_pool if r["rn"] not in quar]
    run_ladder(v1, "V1 quarantine-excluded (sensitivity; freeze keeps them)", with_ce=False)

    # V2: current pool incl. 7 stragglers
    cur = [dict(r) for r in current]
    for r in cur:
        if r["rn"] in ce:
            r["kl"] = ce[r["rn"]]
        if r["rn"] in geo:
            r.update(geo[r["rn"]])
    run_ladder([r for r in cur if "lspec" in r], "V2 current pool (n=1042 incl. post-freeze stragglers)", with_ce=False)

    # V3: seed-averaged cells
    cells = defaultdict(list)
    for r in geo_pool:
        cells[(r["fam"], r["cell"], r["method"])].append(r)
    cellrows = []
    for (fam, cell, meth), rs in cells.items():
        cellrows.append(dict(rn=cell, fam=fam, method=meth,
                             logfd=sum(r["logfd"] for r in rs) / len(rs),
                             ret=sum(r["ret"] for r in rs) / len(rs),
                             e_top=sum(r["e_top"] for r in rs) / len(rs),
                             lspec=sum(r["lspec"] for r in rs) / len(rs),
                             srank=sum(r["srank"] for r in rs) / len(rs)))
    run_ladder(cellrows, "V3 seed-averaged recipe cells", with_ce=False)

    # V4: pooled, no family FE
    run_ladder(geo_pool, "V4 no family FE (fully pooled)", with_ce=False, no_fam_fe=True)

    # ---------- partial correlations (comparability with dyn4 G3) ----------
    print("\n[PARTIALS] r(x, ret | family FE + log10 F_delta) on frozen∩geometry pool")
    Xb, _ = design(geo_pool, ["logfd"])
    yb = [r["ret"] for r in geo_pool]
    mb = ols(Xb, yb)
    res_y = [yb[i] - sum(mb["beta"][a] * Xb[i][a] for a in range(len(Xb[0]))) for i in range(len(yb))]
    for t in ["e_top", "lspec", "srank"]:
        yt = [r[t] for r in geo_pool]
        mt = ols(Xb, yt)
        res_t = [yt[i] - sum(mt["beta"][a] * Xb[i][a] for a in range(len(Xb[0]))) for i in range(len(yt))]
        pr, n = pearson(list(zip(res_t, res_y)))
        print(f"  partial r({t}, ret | fam+logfd) = {pr:+.3f}  (n={n})")


if __name__ == "__main__":
    main()
