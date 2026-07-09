"""
THE BOLD-CLAIM TEST: do ALL adapters (different geometries) collapse onto ONE retention-vs-‖ΔW‖_F curve?
- collapse  -> magnitude governs, geometry irrelevant (the bold claim)
- separate (methods differ at matched ‖ΔW‖_F) -> geometry matters; bold claim FALSE/nuanced
Also: CS-vs-‖ΔW‖_F (adaptation efficiency) — a SEPARATE axis where methods are ALLOWED to differ.
Fast-scale preview over existing runs.
"""
import json, glob, os, math
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
def norm(s):
    for t in ["_fast","_repro","_cs","cs_"]: s=s.replace(t,"")
    return s
fro={}
for f in glob.glob(os.path.join(RES,"forensics_*.json")):
    d=json.load(open(f));fro[norm(d["run_name"])]=d.get("agg_energy_weighted",{})
def family(rn,meth,cfg):
    if rn.startswith("lora_wd") and rn!="lora_wd0_fast": return "LoRA+wd"
    if rn.startswith("lora_r") or rn=="lora_wd0_fast": return "LoRA(rank)"
    if "clora" in rn: return "CLoRA"
    if rn.startswith("dora"): return "DoRA"
    if meth=="uiortholora": return "UIO"+("-corr" if cfg.get("drop_major")==True else "-leg")
    return meth
R=[]
for f in glob.glob(os.path.join(RES,"*","summary.json")):
    try:d=json.load(open(f))
    except:continue
    rn=d.get("run_name") or os.path.basename(os.path.dirname(f));h=d.get("headline",{}) or {};c=d.get("config",{}) or {}
    g=d.get("forensics",{}) or fro.get(norm(rn),{}) or {}
    F=g.get("dw_F"); ret=h.get("retention_mean"); cs=h.get("cs_avg")
    if F is None or ret is None: continue
    if "clora_cs_" in rn: continue   # dedupe: keep clora fast not full
    R.append((family(rn,d.get("method",""),c),F,ret,cs,rn))

fams=sorted(set(r[0] for r in R))
print(f"{len(R)} points across families: {fams}\n")
print("=== RETENTION vs ||dW||_F — do methods COLLAPSE (one curve) or SEPARATE? ===")
print("  binned by ||dW||_F: mean retention + which families present (separation = geometry matters)")
bins=[(0,5),(5,10),(10,20),(20,30),(30,50),(50,300)]
for lo,hi in bins:
    g=[r for r in R if lo<=r[1]<hi]
    if not g: continue
    byf={}
    for r in g: byf.setdefault(r[0],[]).append(r[2])
    rng=f"{min(r[2] for r in g):.1f}-{max(r[2] for r in g):.1f}"
    spread=max(r[2] for r in g)-min(r[2] for r in g)
    fstr=" ".join(f"{k}={sum(v)/len(v):.1f}" for k,v in sorted(byf.items()))
    print(f"  F[{lo:>3},{hi:>3}): ret {rng} (SPREAD {spread:.1f} at ~same F) | {fstr}")
print("  -> LARGE within-bin spread across families = geometry MATTERS (bold claim FALSE). Small = collapse (claim TRUE).")
print("\n=== matched-F head-to-head: LoRA+wd vs CLoRA (the key pair) ===")
for lo,hi in [(25,32),(15,22)]:
    lw=[r for r in R if r[0]=="LoRA+wd" and lo<=r[1]<hi]
    cl=[r for r in R if r[0]=="CLoRA" and lo<=r[1]<hi]
    if lw and cl:
        print(f"  F[{lo},{hi}]: LoRA+wd ret~{sum(r[2] for r in lw)/len(lw):.1f} (n{len(lw)}) vs CLoRA ret~{sum(r[2] for r in cl)/len(cl):.1f} (n{len(cl)})")
print("\n=== CS vs ||dW||_F (adaptation EFFICIENCY — methods ALLOWED to differ here) top per family ===")
for fam in fams:
    g=[r for r in R if r[0]==fam and r[3]]
    if g:
        best=max(g,key=lambda r:r[3]/max(r[1],1))
        print(f"  {fam:10}: best CS/F = CS {best[3]:.1f} @ F {best[1]:.1f} ({best[4]})")
