"""
Phase-1 verdict: D1 (direction vs magnitude, causal) + D2-killer (magnitude alone reproduces CLoRA).
Runs incrementally — prints whatever results have landed so far.

  python analyze_d1_d2.py
"""
import os, glob, json, math
import run_lib
RES = os.path.join(run_lib.HERE, "results")

def load(pat):
    out = {}
    for f in glob.glob(os.path.join(RES, pat, "summary.json")):
        d = json.load(open(f)); rn = d.get("run_name") or os.path.basename(os.path.dirname(f))
        h = d.get("headline", {}) or {}; lk = d.get("leakage", {}) or {}; fx = d.get("forensics", {}) or {}
        c = d.get("config", {}) or {}; fd = d.get("fdelta", {}) or {}
        out[rn] = dict(cs=h.get("cs_avg"), ret=h.get("retention_mean"), muE=lk.get("mu_E"),
                       presF=lk.get("preserved_F"), out05=fx.get("out_top_0.5"), sig=fx.get("sigma_resp"),
                       dwF=fx.get("dw_F"), svmax=h.get("dw_sv_max") or fd.get("dw_sv_max"),
                       use_de=c.get("use_de"), lr=c.get("learning_rate"), lamE=c.get("lambda_E"))
    return out

def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    n = len(pts)
    if n < 3: return float('nan'), n
    mx = sum(p[0] for p in pts)/n; my = sum(p[1] for p in pts)/n
    cov = sum((x-mx)*(y-my) for x, y in pts); vx = sum((x-mx)**2 for x, _ in pts); vy = sum((y-my)**2 for _, y in pts)
    return (cov/math.sqrt(vx*vy) if vx > 0 and vy > 0 else float('nan')), n

def f(x, n=2): return f"{x:.{n}f}" if isinstance(x, (int, float)) else "-"

# ---------- D1: controlled clean vs leaky x magnitude ----------
d1 = load("d1_*")
print("="*78); print("D1 — corrected instrument: direction (use_de) vs magnitude (LR ladder)")
print(f"{'run':18s} {'dE':>3s} {'lr':>6s} {'CS':>5s} {'ret':>5s} {'muE':>6s} {'out.5':>6s} {'dwF':>6s} {'svmax':>6s}")
for rn in sorted(d1, key=lambda r: (d1[r]['use_de'] or 0, d1[r]['lr'] or 0)):
    r = d1[rn]
    print(f"{rn:18s} {f(r['use_de'],0):>3s} {f(r['lr'],4):>6s} {f(r['cs'],1):>5s} {f(r['ret'],1):>5s} {f(r['muE'],3):>6s} {f(r['out05'],3):>6s} {f(r['dwF'],1):>6s} {f(r['svmax'],1):>6s}")
if len(d1) >= 4:
    clean = [r for r in d1.values() if r['use_de'] == 0]; leaky = [r for r in d1.values() if r['use_de'] == 1]
    rc, nc = pearson([r['svmax'] for r in d1.values()], [r['ret'] for r in d1.values()])
    rm, nm = pearson([r['muE'] for r in d1.values()], [r['ret'] for r in d1.values()])
    print(f"\n  corr(retention, MAGNITUDE=svmax) = {rc:+.3f} (n={nc})")
    print(f"  corr(retention, DIRECTION=muE)   = {rm:+.3f} (n={nm})")
    print("  D1 read: if |corr(mag)| >> |corr(muE)| AND matched-magnitude clean~leaky retention -> magnitude drives forgetting.")
    print("  Matched-magnitude check (find clean & leaky with similar svmax, compare ret):")
    for cl in sorted(clean, key=lambda r: r['svmax'] or 0):
        if cl['svmax'] is None: continue
        best = min((l for l in leaky if l['svmax']), key=lambda l: abs((l['svmax'] or 0)-cl['svmax']), default=None)
        if best: print(f"    clean svmax={f(cl['svmax'],1)} ret={f(cl['ret'],1)} (muE~0)  vs  leaky svmax={f(best['svmax'],1)} ret={f(best['ret'],1)} (muE={f(best['muE'],2)})")

# ---------- D2-killer: magnitude alone reproduces CLoRA ----------
print("\n" + "="*78); print("D2-killer — LoRA+weight_decay vs CLoRA on the (magnitude -> retention) plane")
lora = load("lora_wd*"); clora = load("clora_*_fast")
print(f"{'run':18s} {'method':>7s} {'ret':>5s} {'dwF':>6s} {'svmax':>6s} {'out.5':>6s} {'sig':>6s}")
for rn in sorted(lora) + sorted(clora):
    src = lora.get(rn) or clora.get(rn); m = "LoRA+wd" if rn in lora else "CLoRA"
    print(f"{rn:18s} {m:>7s} {f(src['ret'],1):>5s} {f(src['dwF'],1):>6s} {f(src['svmax'],1):>6s} {f(src['out05'],3):>6s} {f(src['sig'],3):>6s}")
allpts = list(lora.values()) + list(clora.values())
if len(allpts) >= 4:
    r, n = pearson([p['svmax'] for p in allpts], [p['ret'] for p in allpts])
    print(f"\n  pooled corr(retention, magnitude=svmax) = {r:+.3f} (n={n})")
    print("  D2 read: if LoRA+wd and CLoRA points fall on ONE (magnitude,retention) curve -> CLoRA's")
    print("  random-subspace apparatus adds nothing beyond magnitude control.")
