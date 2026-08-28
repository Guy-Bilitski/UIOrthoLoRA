"""Magnitude law from the frozen pool, then residuals for the intruder-slice configs.

Design (agreed 2026-08-28). The frozen pool has ~60 adapters with (F_delta, retention)
but NO intruder measurements -- its checkpoints were lost in the 2026-07 evacuation,
which is why the slice is being retrained. So the analysis is two-stage:

  1. Fit retention ~ f(log F_delta) on the POOL alone, per model family (the pool is
     large, so this is the robust part).
  2. For each slice config, compute the residual  e_i = R_i - f(log F_i).
  3. Ask whether intruder ENERGY SHARE explains those residuals: at a given update
     magnitude, do configs with more intruder energy retain more or less than the
     magnitude law predicts?

With ~7 configs this is descriptive, not a formal regression claim -- the causal
intervention (arms B/C/D/E) carries the argument; this is corroboration.

Usage: python magnitude_residuals.py
"""
import os, re, json, math, collections

HERE = os.path.dirname(os.path.abspath(__file__))

def pool_points():
    pts = collections.defaultdict(list)
    with open(os.path.join(HERE, "results", "campaign_summary.jsonl")) as f:
        for line in f:
            d = json.loads(line); rn = d.get("run_name", "")
            fam = "llama" if rn.startswith("frc_") else ("qwen" if rn.startswith("qwsw_") else None)
            if not fam or rn.startswith("qwswm"): continue
            F, R = d.get("fdelta"), d.get("retention_mean")
            if F and R is not None and F > 0:
                pts[fam].append((math.log(F), R))
    return pts

def fit(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    den = sum((x-mx)**2 for x in xs)
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/den if den else 0.0
    a = my - b*mx
    ss_t = sum((y-my)**2 for y in ys)
    ss_r = sum((y-(a+b*x))**2 for x, y in zip(xs, ys))
    return a, b, (1 - ss_r/ss_t if ss_t else float("nan"))

def pearson(x, y):
    n = len(x)
    if n < 3: return None
    mx, my = sum(x)/n, sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    if sx == 0 or sy == 0: return None
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/(sx*sy)

def main():
    pool = pool_points()
    laws = {}
    print("=== 1. Magnitude law fitted on the FROZEN POOL (retention ~ a + b*log F) ===")
    for fam, pts in sorted(pool.items()):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        a, b, r2 = fit(xs, ys); laws[fam] = (a, b)
        print(f"  {fam:6s} n={len(pts):3d}   a={a:7.3f}  b={b:7.3f}   R^2={r2:.3f}")

    print("\n=== 2. Slice configs: residual against that law, vs intruder energy ===")
    intr_dir = os.path.join(HERE, "results", "intruder")
    rows = []
    for fn in sorted(os.listdir(intr_dir)):
        if not fn.endswith(".json"): continue
        run = fn[:-5]
        if "__" in run: continue
        agg = json.load(open(os.path.join(intr_dir, fn)))["aggregate"]
        sp = os.path.join(HERE, "results", run + "__rl50", "summary.json")
        if not os.path.exists(sp): sp = os.path.join(HERE, "results", run, "summary.json")
        if not os.path.exists(sp): continue
        h = json.load(open(sp))["headline"]
        fam = "llama" if "_frc_" in run else "qwen"
        if fam not in laws: continue
        F, R = h.get("fdelta"), h.get("retention_mean")
        if not F or R is None: continue
        a, b = laws[fam]
        pred = a + b*math.log(F)
        nm = agg["n_matrices"]
        rows.append(dict(run=run, fam=fam, F=F, R=R, pred=pred, resid=R-pred,
                         Ien=agg["mean_energy_share_baseAll_t0.5"],
                         Icnt=agg["total_intruders_k10_baseAll_t0.5"]/(nm*10),
                         adapt=h.get("cs_avg")))
    if not rows:
        print("  (no scored configs yet)"); return
    print(f"  {'config':34s} {'F':>6s} {'R':>6s} {'pred':>6s} {'resid':>7s} {'I_energy':>9s} {'I_count':>8s} {'task':>6s}")
    for r in sorted(rows, key=lambda r: r["F"]):
        print(f"  {r['run']:34s} {r['F']:6.3f} {r['R']:6.2f} {r['pred']:6.2f} {r['resid']:+7.2f} "
              f"{r['Ien']:9.3f} {r['Icnt']:8.3f} {r['adapt'] if r['adapt'] else 0:6.2f}")
    if len(rows) >= 4:
        e = [r["resid"] for r in rows]
        print(f"\n  corr(residual, intruder energy) = {pearson(e,[r['Ien'] for r in rows])}")
        print(f"  corr(residual, intruder count)  = {pearson(e,[r['Icnt'] for r in rows])}")
        print("  (near zero => intruder structure adds little beyond update magnitude)")
    else:
        print(f"\n  [{len(rows)} configs; correlation reported once >=4 are scored]")

if __name__ == "__main__":
    main()
