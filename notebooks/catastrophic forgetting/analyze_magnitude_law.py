"""
Insight miner across ALL adapter runs: is retention a (method-independent) function of
UPDATE MAGNITUDE, and does DIRECTION add anything at matched magnitude? Plus the CS-vs-retention
Pareto frontier and adaptation-efficiency (CS per unit magnitude).

Confounds handled/flagged:
  - SCALE: retention mixes full vs fast eval -> flagged per row; correlations computed within the
    largest consistent groups too.
  - LEGACY major term: for drop_major!=1 UIO runs, dw_sv_max (a TAIL max-σ) UNDER-reports the
    preserved-band Frobenius mass from the S1=1 term -> those rows marked (legacy) and split out.

  python analyze_magnitude_law.py
"""
import json, glob, os, math
import run_lib
RES = os.path.join(run_lib.HERE, "results")

def pear(xy):
    xy=[(x,y) for x,y in xy if isinstance(x,(int,float)) and isinstance(y,(int,float))
        and math.isfinite(x) and math.isfinite(y)]  # drop NaN/inf (diverged cells) — they poison the sum
    n=len(xy)
    if n<3: return float('nan'),n
    mx=sum(p[0] for p in xy)/n; my=sum(p[1] for p in xy)/n
    cov=sum((x-mx)*(y-my) for x,y in xy); vx=sum((x-mx)**2 for x,_ in xy); vy=sum((y-my)**2 for _,y in xy)
    return (cov/math.sqrt(vx*vy) if vx>0 and vy>0 else float('nan')), n

def _norm(s):  # reconcile forensics run_name (clora_k128 / lora_wd0) with summary run_name (clora_cs_k128 / lora_wd0_fast)
    for t in ["_fast","_repro","_cs","cs_"]: s=s.replace(t,"")
    return s
rows=[]
fxext={}
for f in glob.glob(os.path.join(RES,"forensics_*.json")):
    try:d=json.load(open(f));fxext[_norm(d["run_name"])]=d.get("agg_energy_weighted",{})
    except:pass
for f in glob.glob(os.path.join(RES,"*","summary.json")):
    try:d=json.load(open(f))
    except:continue
    rn=d.get("run_name") or os.path.basename(os.path.dirname(f))
    h=d.get("headline",{}) or {}; c=d.get("config",{}) or {}; fd=d.get("fdelta",{}) or {}; fx=d.get("forensics",{}) or fxext.get(_norm(rn),{}) or {}
    ret=h.get("retention_mean"); cs=h.get("cs_avg")
    if ret is None and cs is None: continue
    dm=c.get("drop_major"); meth=d.get("method","")
    legacy = (meth=="uiortholora" and dm!=True)
    rows.append(dict(rn=rn, meth=meth, de=c.get("use_de"), dm=dm, legacy=legacy,
        mag=h.get("dw_sv_max") or fd.get("dw_sv_max"), dwF=fx.get("dw_F"),
        cs=cs, ret=ret, o5=fx.get("out_top_0.5"), sg=fx.get("sigma_resp")))

withret=[r for r in rows if r['ret'] is not None and r['mag'] is not None]

# ---------- (A) magnitude -> retention ----------
print("="*70,"\n(A) MAGNITUDE -> RETENTION (dw_sv_max)")
clean=[r for r in withret if not r['legacy']]      # no major-term confound
legc =[r for r in withret if r['legacy']]
for label,grp in [("ALL",withret),("no-major-term (CLoRA/LoRA/corrected-UIO)",clean),("legacy-UIO (mag under-reported)",legc)]:
    r,n=pear([(x['mag'],x['ret']) for x in grp])
    print(f"  corr(ret, dw_sv_max) [{label}] = {r:+.3f}  (n={n})")

# ---------- (B) is there a MAGNITUDE BUDGET (threshold)? ----------
print("="*70,"\n(B) MAGNITUDE BUDGET — mean retention by magnitude bin (no-major-term set)")
bins=[(0,5),(5,10),(10,20),(20,40),(40,999)]
for lo,hi in bins:
    g=[r for r in clean if lo<=r['mag']<hi]
    if g: print(f"  ||dW||max in [{lo:>3},{hi:>3}): mean ret={sum(x['ret'] for x in g)/len(g):5.1f}  (n={len(g)})  ret range {min(x['ret'] for x in g):.1f}-{max(x['ret'] for x in g):.1f}")

# ---------- (C) DIRECTION at matched magnitude: within a bin, does use_de/method change ret? ----------
print("="*70,"\n(C) DIRECTION @ matched magnitude (does ret vary by use_de within a mag bin?)")
for lo,hi in [(5,15),(15,40)]:
    g=[r for r in clean if lo<=r['mag']<hi]
    by_de={}
    for r in g: by_de.setdefault(r['de'],[]).append(r['ret'])
    s=" | ".join(f"use_de={k}: ret~{sum(v)/len(v):.1f}(n{len(v)})" for k,v in sorted(by_de.items(),key=lambda x:str(x[0])))
    print(f"  mag[{lo},{hi}): {s}")

# ---------- (D) PARETO frontier (maximize CS and RET) ----------
print("="*70,"\n(D) CS-vs-RETENTION PARETO FRONTIER (non-dominated; fast+full mixed -> indicative)")
pts=[r for r in withret if r['cs'] is not None]
front=[r for r in pts if not any((o['cs']>=r['cs'] and o['ret']>=r['ret'] and (o['cs']>r['cs'] or o['ret']>r['ret'])) for o in pts)]
for r in sorted(front,key=lambda r:-r['cs']):
    tag="CLoRA" if "clora" in r['rn'] else ("LoRA" if r['meth'] in("lora","LORA") else ("UIO-corr" if r['dm']==True else "UIO-leg"))
    print(f"  CS={r['cs']:5.1f} ret={r['ret']:5.1f} mag={r['mag']:6.1f} [{tag:8}] {r['rn']}")

# ---------- (F) THE UNIFYING PREDICTOR: retention vs PRESERVED-SUBSPACE MAGNITUDE ----------
# preserved-magnitude proxies (need forensics): sqrt(out_top_0.5)*dwF (top-half Frob) and sigma_resp*dwF
print("="*70,"\n(F) UNIFYING LAW test — retention vs PRESERVED-SUBSPACE magnitude (needs forensics)")
fxrows=[r for r in withret if r['dwF'] and r['o5'] is not None and r['sg'] is not None]
if len(fxrows)>=3:
    for name,fn in [("total ||dW||F", lambda r:r['dwF']),
                    ("preserved-Frob (sqrt(out.5)*dwF)", lambda r:math.sqrt(r['o5'])*r['dwF']),
                    ("sigma_resp*dwF (σ-weighted)", lambda r:r['sg']*r['dwF'])]:
        rr,nn=pear([(fn(r),r['ret']) for r in fxrows])
        print(f"  corr(ret, {name:34}) = {rr:+.3f} (n={nn})")
    print("  -> if preserved/σ-weighted predictor's |corr| >> total ||dW||F across methods, the")
    print("     'magnitude-in-important-subspace' law holds (method-independent). [Phase-2 fills this in]")
else:
    print(f"  only {len(fxrows)} runs have forensics yet (need Phase-2 / grid+controls to log preserved_F).")

# ---------- (E) adaptation EFFICIENCY (CS per unit magnitude) ----------
print("="*70,"\n(E) ADAPTATION EFFICIENCY — top CS-per-||dW||max (high adapt at low magnitude)")
eff=[r for r in pts if r['mag'] and r['mag']>0.5 and r['cs']]
for r in sorted(eff,key=lambda r:-(r['cs']/r['mag']))[:8]:
    tag="UIO-corr" if r['dm']==True else ("UIO-leg" if r['meth']=="uiortholora" else r['meth'])
    print(f"  CS/mag={r['cs']/r['mag']:5.2f}  CS={r['cs']:5.1f} ret={r['ret']} mag={r['mag']:5.1f} k_vec-rot? [{tag}] {r['rn']}")
