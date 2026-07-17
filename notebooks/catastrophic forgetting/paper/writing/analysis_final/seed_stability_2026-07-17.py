"""Seed-stability of the geometry partial effects (2026-07-17, addendum §19).

Question: the freeze reports geometry as a bounded second-order effect via pooled
partials (dyn4 G3). Is that effect seed-stable, or an artifact of particular seeds?
For each seed s (42-46), restrict the frozen pool to runs with that seed, demean
within family (family FE), and compute (a) partial r(x, ret | fam + log10 F_delta)
for x in {e_top_w, log10 spec_max, stable_rank_w}, and (b) the OLS coefficient of
each geometry term alongside log10 F_delta. Report per-seed values, mean±sd across
seeds, and sign consistency. log10 F_delta's own partial is the reference scale.

Same loader/freeze convention as ladder_2026-07-17.py (STRAGGLERS excluded;
preflight asserts pool n=1035). Inputs: results/*/summary.json +
results/geo_drift/adapter_metrics_merged.jsonl. Pure stdlib.
"""
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ladder", os.path.join(HERE, "ladder_2026-07-17.py"))
ladder = importlib.util.module_from_spec(spec)
sys.modules["ladder"] = ladder
_main, ladder.main = None, None  # placeholder; we only want the helpers
spec.loader.exec_module(ladder)

SEEDS = ["42", "43", "44", "45", "46"]
GEO_TERMS = ["e_top", "lspec", "srank"]


def partial_r(rows, term):
    Xb, _ = ladder.design(rows, ["logfd"])
    y = [r["ret"] for r in rows]
    mb = ladder.ols(Xb, y)
    res_y = [y[i] - sum(mb["beta"][a] * Xb[i][a] for a in range(len(Xb[0]))) for i in range(len(y))]
    yt = [r[term] for r in rows]
    mt = ladder.ols(Xb, yt)
    res_t = [yt[i] - sum(mt["beta"][a] * Xb[i][a] for a in range(len(Xb[0]))) for i in range(len(yt))]
    return ladder.pearson(list(zip(res_t, res_y)))


def partial_r_fd(rows):
    Xb, _ = ladder.design(rows, [])
    y = [r["ret"] for r in rows]
    mb = ladder.ols(Xb, y)
    res_y = [y[i] - sum(mb["beta"][a] * Xb[i][a] for a in range(len(Xb[0]))) for i in range(len(y))]
    yt = [r["logfd"] for r in rows]
    mt = ladder.ols(Xb, yt)
    res_t = [yt[i] - sum(mt["beta"][a] * Xb[i][a] for a in range(len(Xb[0]))) for i in range(len(yt))]
    return ladder.pearson(list(zip(res_t, res_y)))


def main():
    rows = [r for r in ladder.load_rows() if r["rn"] not in ladder.STRAGGLERS]
    assert len(rows) == 1035, f"frozen pool n={len(rows)} != 1035"
    geo = ladder.load_geo()
    for r in rows:
        if r["rn"] in geo:
            r.update(geo[r["rn"]])
    pool = [r for r in rows if "lspec" in r]
    print("=" * 78)
    print("SEED-STABILITY OF GEOMETRY PARTIALS — addendum, 2026-07-17")
    print(f"frozen pool ∩ geometry: n={len(pool)}")
    print("=" * 78)

    stats = {t: [] for t in GEO_TERMS + ["logfd"]}
    print(f"\n{'seed':<6} {'n':>5} {'pr(logfd)':>10} {'pr(e_top)':>10} {'pr(lspec)':>10} {'pr(srank)':>10}")
    for s in SEEDS:
        sub = [r for r in pool if r["seed"] == s]
        if len(sub) < 30:
            print(f"s{s:<5} {len(sub):>5}  (skipped, n<30)")
            continue
        prfd, _ = partial_r_fd(sub)
        line = f"s{s:<5} {len(sub):>5} {prfd:>+10.3f}"
        stats["logfd"].append(prfd)
        for t in GEO_TERMS:
            pr, _ = partial_r(sub, t)
            stats[t].append(pr)
            line += f" {pr:>+10.3f}"
        print(line)

    print("\nacross-seed summary (partial r):")
    for t in ["logfd"] + GEO_TERMS:
        v = stats[t]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
        signs = sum(1 for x in v if (x > 0) == (m > 0))
        print(f"  {t:<8} mean={m:+.3f}  sd={sd:.3f}  sign-consistent {signs}/{len(v)} seeds")

    print("\nOLS coefficients (ret ~ fam FE + logfd + e_top + lspec + srank), per seed:")
    print(f"{'seed':<6} {'n':>5} {'b(logfd)':>10} {'b(e_top)':>10} {'b(lspec)':>10} {'b(srank)':>10}")
    coefs = {t: [] for t in ["logfd"] + GEO_TERMS}
    for s in SEEDS:
        sub = [r for r in pool if r["seed"] == s]
        if len(sub) < 30:
            continue
        X, names = ladder.design(sub, ["logfd"] + GEO_TERMS)
        m = ladder.ols(X, [r["ret"] for r in sub])
        bmap = dict(zip(names, m["beta"]))
        line = f"s{s:<5} {len(sub):>5}"
        for t in ["logfd"] + GEO_TERMS:
            coefs[t].append(bmap[t])
            line += f" {bmap[t]:>+10.3f}"
        print(line)
    print("\nacross-seed summary (OLS coefficient):")
    for t in ["logfd"] + GEO_TERMS:
        v = coefs[t]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
        signs = sum(1 for x in v if (x > 0) == (m > 0))
        print(f"  {t:<8} mean={m:+.3f}  sd={sd:.3f}  sign-consistent {signs}/{len(v)} seeds")


if __name__ == "__main__":
    main()
