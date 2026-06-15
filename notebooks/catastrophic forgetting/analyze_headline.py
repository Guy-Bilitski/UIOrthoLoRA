"""
Headline readouts vs the pre-reg table (09): #2 directional-vs-raw norm, #1 scale-unified Frobenius,
#4 drop_major x rotation, #5 rank sweep (partial). Reads landed results.
"""
import json, glob, os, math
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def pear(xy):
    xy=[(x,y) for x,y in xy if isinstance(x,(int,float)) and isinstance(y,(int,float))];n=len(xy)
    if n<3: return float('nan'),n
    mx=sum(p[0] for p in xy)/n;my=sum(p[1] for p in xy)/n
    c=sum((a-mx)*(b-my) for a,b in xy);vx=sum((a-mx)**2 for a,_ in xy);vy=sum((b-my)**2 for _,b in xy)
    return (c/math.sqrt(vx*vy) if vx>0 and vy>0 else float('nan')),n
def _norm(s):
    for t in ["_fast","_repro","_cs","cs_"]: s=s.replace(t,"")
    return s

# forensics (raw Frobenius) and databasis (directional) keyed by normalized run
fro={}
for f in glob.glob(os.path.join(RES,"forensics_*.json")):
    try:d=json.load(open(f));fro[_norm(d["run_name"])]=d.get("agg_energy_weighted",{})
    except:pass
db={}
for f in glob.glob(os.path.join(RES,"databasis_*_retain.json")):
    try:d=json.load(open(f));db[_norm(d["run_name"])]=d.get("agg",{})
    except:pass

rows=[]
for f in glob.glob(os.path.join(RES,"*","summary.json")):
    try:d=json.load(open(f))
    except:continue
    rn=d.get("run_name") or os.path.basename(os.path.dirname(f));h=d.get("headline",{}) or {};c=d.get("config",{}) or {}
    nk=_norm(rn)
    g=d.get("forensics",{}) or fro.get(nk,{}) or {}; dbx=db.get(nk,{})
    arch="CLoRA" if "clora" in rn else ("LoRA" if d.get("method","") in("lora","LORA") else "UIO")
    rows.append(dict(rn=rn,arch=arch,ret=h.get("retention_mean"),cs=h.get("cs_avg"),
        spec=h.get("dw_sv_max"),frob=g.get("dw_F") or dbx.get("dwF"),
        dresp=dbx.get("data_resp"),d_inTop=dbx.get("d_inTop"),c_inTop=dbx.get("c_inTop"),w_inTop=dbx.get("w_inTop"),
        kval=c.get("k_val"),kvec=c.get("k_vec"),de=c.get("use_de"),dm=c.get("drop_major"),lr=c.get("learning_rate")))

# ---------- #2 HEADLINE: directional vs raw norm (runs with databasis C_retain) ----------
print("="*72,"\n#2 HEADLINE — does DIRECTIONAL norm ||dW.C_retain^1/2|| beat RAW ||dW||_F at predicting retention?")
H=[r for r in rows if r['dresp'] is not None and r['frob'] and r['ret'] is not None]
print(f"  runs with C_retain databasis + retention + Frobenius: {len(H)}")
if H:
    rr_raw,n=pear([(r['frob'],r['ret']) for r in H])
    rr_dir,_=pear([(math.sqrt(r['dresp']),r['ret']) for r in H])
    rr_dtop,_=pear([(r['d_inTop'],r['ret']) for r in H])
    rr_ctop,_=pear([(r['c_inTop'],r['ret']) for r in H])
    print(f"  corr(ret, RAW ||dW||_F)                 = {rr_raw:+.3f}")
    print(f"  corr(ret, DIRECTIONAL ||dW.C^1/2||)     = {rr_dir:+.3f}   <-- beats raw? {'YES' if abs(rr_dir)>abs(rr_raw) else 'no'}")
    print(f"  corr(ret, data-basis inTop frac)        = {rr_dtop:+.3f}")
    print(f"  corr(ret, CorDA-basis inTop frac)       = {rr_ctop:+.3f}  (n={n})")
    for r in sorted(H,key=lambda r:-(r['frob'] or 0)):
        print(f"    {r['arch']:6} {r['rn'][:22]:22} ret={r['ret']:5} rawF={r['frob']:6.1f} dirN={math.sqrt(r['dresp']):6.2f} d_inTop={r['d_inTop']:.3f}")

# ---------- #1 scale-unified cross-arch Frobenius (FAST scale only) ----------
print("="*72,"\n#1 SCALE-UNIFIED cross-arch Frobenius (fast-scale runs only)")
fast=[r for r in rows if r['frob'] and r['ret'] is not None and ('fast' in r['rn'] or r['arch']!='CLoRA')]
for a in ["CLoRA","UIO","LoRA","ALL"]:
    sub=[(r['frob'],r['ret']) for r in fast if a=="ALL" or r['arch']==a]
    rr,n=pear(sub); print(f"  {a:6}: corr(ret,||dW||_F)={rr:+.3f} (n={n})")

# ---------- #4 drop_major x rotation (grid k410 ladder) ----------
print("="*72,"\n#4 drop_major x ROTATION (grid, corrected drop_major=1)")
g=[r for r in rows if r['rn'].startswith('grid_') and r['cs'] is not None]
print(f"  {'run':26} {'kval':>4} {'kvec':>4} {'dE':>2} {'lr':>6} {'CS':>5} {'ret':>5} {'spec':>5}")
for r in sorted(g,key=lambda r:(r['kval'] or 0,r['kvec'] or 0)):
    print(f"  {r['rn'][:26]:26} {str(r['kval']):>4} {str(r['kvec']):>4} {str(int(r['de']) if r['de'] is not None else '?'):>2} {str(r['lr']):>6} {r['cs']:>5} {str(r['ret']):>5} {str(r['spec']):>5}")
print("  legacy uioT_k410(v410,dE1)=CS72.7/ret25.0 ; corrected lr1e2 was CS48 -> does lr2e2 recover CS? rotation effect?")

# ---------- #5 rank sweep (partial) ----------
print("="*72,"\n#5 RANK sweep (LoRA, partial)")
rk=[r for r in rows if r['rn'].startswith('lora_r') and r['rn'][6:].isdigit()]
print(f"  {'rank':>5} {'CS':>5} {'ret':>5} {'rawF':>6} {'spec(σ1)':>8}")
for r in sorted(rk,key=lambda r:int(r['rn'][6:])):
    print(f"  {r['rn'][6:]:>5} {str(r['cs']):>5} {str(r['ret']):>5} {str(round(r['frob'],1) if r['frob'] else '-'):>6} {str(r['spec']):>8}")
print("  HYP-A: ret~||dW||_F regardless of rank (one curve) | HYP-B: rank retains better at matched norm (diffusion: σ1 down)")
