"""
Join cross-method spectral forensics with measured forgetting and ask the decider:
does a SINGLE weight-space metric predict retention across LoRA + CLoRA (+ UIO)?

  - If `sigma_resp` (sigma^2-weighted top-spectral response = a weights-only proxy for
    output disruption) tracks retention monotonically across ALL methods on ONE curve
    -> unifying law (hypothesis A).
  - If CLoRA's `out_top_0.5` stays ~0.5 (spectrally neutral) regardless of k while
    retention varies with magnitude (`dw_F`) -> its orthogonality mechanism is
    mis-attributed; magnitude is the operative variable (hypothesis B).
  - If location adds NOTHING beyond magnitude -> no discovery (just re-derives ||dW||/F-delta).

    python analyze_forensics.py
"""
import os, json, glob, math
import run_lib
HERE = run_lib.HERE
RES = os.path.join(HERE, "results")

# FULL-retention (consistent scale) for the saved checkpoints; base=26.0 reference.
RET_FULL = {
    "lora_cs_r32_repro": 21.66, "lora": 21.66,
    "clora_k128": 22.48, "clora_k256": 22.39, "clora_k512": 23.06,
    "clora_k1024": 24.82, "clora_k2048": 25.65,
}
DWMAX = {  # ||dW||_max from the campaign (capacity/magnitude proxy)
    "lora": 55.4, "clora_k128": 32.49, "clora_k256": 29.83, "clora_k512": 25.89,
    "clora_k1024": 21.96, "clora_k2048": 16.41,
}

def pearson(xs, ys):
    n=len(xs)
    if n<2: return float('nan')
    mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    return cov/math.sqrt(vx*vy) if vx>0 and vy>0 else float('nan')

rows=[]
for f in sorted(glob.glob(os.path.join(RES, "forensics_*.json"))):
    d=json.load(open(f)); a=d["agg_energy_weighted"]; rn=d["run_name"]
    ret=RET_FULL.get(rn)
    rows.append(dict(run=rn, ret=ret, n=d["n_matrices"],
                     out05=a.get("out_top_0.5"), out01=a.get("out_top_0.1"), out005=a.get("out_top_0.05"),
                     in05=a.get("in_top_0.5"), sigma_resp=a.get("sigma_resp"),
                     in_com=a.get("in_com"), dw_F=a.get("dw_F"), sv_max=a.get("sv_max_dw")))

print(f"{'run':18s} {'ret':>5s} {'nMod':>4s} {'out@.5':>6s} {'out@.1':>6s} {'out@.05':>7s} {'sigResp':>7s} {'in_com':>6s} {'dw_F':>7s} {'sv_max':>7s}")
for r in sorted(rows, key=lambda r: (r['ret'] or 0)):
    def fm(x,n=3): return f"{x:.{n}f}" if isinstance(x,(int,float)) else "-"
    print(f"{r['run']:18s} {fm(r['ret'],1):>5s} {r['n']:>4d} {fm(r['out05']):>6s} {fm(r['out01']):>6s} {fm(r['out005']):>7s} {fm(r['sigma_resp']):>7s} {fm(r['in_com']):>6s} {fm(r['dw_F'],1):>7s} {fm(r['sv_max'],1):>7s}")

have=[r for r in rows if r['ret'] is not None]
print(f"\n--- corr(retention, metric) over {len(have)} checkpoints (more negative => higher metric means MORE forgetting) ---")
for m in ['out05','out01','out005','sigma_resp','in_com','dw_F','sv_max']:
    xs=[r[m] for r in have if isinstance(r[m],(int,float))]; ys=[r['ret'] for r in have if isinstance(r[m],(int,float))]
    print(f"  retention vs {m:10s}: r = {pearson(xs,ys):+.3f}")
# does CLoRA avoid the top subspace, and does avoidance scale with k?
clora=[r for r in have if r['run'].startswith('clora')]
print(f"\n--- CLoRA out_top_0.5 by k (0.5 = spectrally neutral; <<0.5 = avoids top) ---")
for r in sorted(clora, key=lambda r:r['ret'] or 0):
    print(f"  {r['run']:14s} ret={r['ret']}  out@.5={r['out05']:.3f}  sigma_resp={r['sigma_resp']:.3f}  dw_F={r['dw_F']:.1f}")
