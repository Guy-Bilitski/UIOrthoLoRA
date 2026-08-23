import json, collections, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "legend.fontsize": 9, "legend.framealpha": 0.95, "legend.edgecolor": "0.8",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

SUMMARY="paper/writing/data/campaign_summary_clean.jsonl"  # clean 320-row registry (no CorDA, deduped)
OUT="paper/writing/figures/fig9_lr_artifact.png"

# ---- load + dedup by latest evaluated_at (CorDA old-wikitext vs new-nq_open dup rows) ----
rows=[json.loads(l) for l in open(SUMMARY)]
def parse(n):
    m=re.match(r"lrsw_(.+)_lr([0-9a-z]+)_s42$", n)
    return (m.group(1),m.group(2)) if m else None
best={}
for r in rows:
    n=r["run_name"]
    if not n.startswith("lrsw_") or not parse(n): continue
    if n not in best or r["evaluated_at"]>best[n]["evaluated_at"]: best[n]=r
lrorder=["2e5","5e5","1e4","2e4","3e4","5e4","1e3"]
LRVAL={"2e5":2e-5,"5e5":5e-5,"1e4":1e-4,"2e4":2e-4,"3e4":3e-4,"5e4":5e-4,"1e3":1e-3}
cov=collections.defaultdict(dict)
for n,r in best.items():
    arm,lr=parse(n); ad=r.get("cs_avg"); ret=r.get("retention_mean")
    if ad is None or ret is None or r.get("fdelta",0)>50: continue   # drop CorDA lr1e3 divergence
    cov[arm][lr]=dict(adapt=ad,ret=ret,fdelta=r.get("fdelta"),lr=lr)

lora=cov["lora_r16"]; lorawd=cov["lorawd_wd0p3"]

COL={"lora":"#0072B2","lorawd":"#56B4E9","milora":"#009E73","clora":"#E69F00",
     "dora":"#CC79A7","corda":"#D55E00","sclora":"#882255","lora_null":"#333333"}
MARK={"lora":"o","lorawd":"D","milora":"^","clora":"s",
      "dora":"v","corda":"P","sclora":"X","lora_null":"*"}
PRETTY={"dora_r16":"DoRA","corda_r16":"CorDA","milora_r32":"MiLoRA","sclora_r32":"SC-LoRA",
        "lora_null_r16":"LoRA-Null","clora_k1024":"CLoRA","lora_r16":"LoRA (plain)","lorawd_wd0p3":"LoRA + wd 0.3"}
def ck(arm): return arm.replace("_r16","").replace("_r32","").replace("_k1024","").replace("_wd0p3","")

# CorDA is EMBARGOED: excluded pending its nq_open re-run + calibration-fairness fix.
# It must not render as a series or a ringed operating point (matches the paper-local table).
fancy=["dora_r16","milora_r32","sclora_r32","lora_null_r16","clora_k1024"]

def pareto_ge(a,b): return a["adapt"]>=b["adapt"] and a["ret"]>=b["ret"] and (a["adapt"]>b["adapt"] or a["ret"]>b["ret"])
def pareto(pts):
    fr=[p for p in pts if not any(pareto_ge(q,p) for q in pts if q is not p)]
    return sorted(fr,key=lambda p:p["adapt"])
lorawd_fr=pareto([lorawd[l] for l in lrorder if l in lorawd])

BASE_CORE=(33.10+18.96)/2.0  # ~26.03

# ---- FIGURE ----
fig, ax = plt.subplots(figsize=(11.8, 8.2))

def traj(arm, lw, ms, z, edge, alpha=1.0, ls="-", line_alpha=None):
    k=ck(arm); pts=[cov[arm][l] for l in lrorder if l in cov[arm]]
    # order the connecting line by LR (traces the LR sweep, not by adapt)
    xs=[p["adapt"] for p in pts]; ys=[p["ret"] for p in pts]
    la=line_alpha if line_alpha is not None else alpha*0.6
    ax.plot(xs,ys,ls,color=COL[k],lw=lw,alpha=la,zorder=z)
    ax.scatter(xs,ys,s=ms,c=COL[k],marker=MARK[k],edgecolor=edge,linewidth=0.7,
               alpha=alpha,zorder=z+1)
    return pts

# fancy methods (thin, faint lines so crossings don't dominate)
for arm in fancy:
    traj(arm, lw=1.2, ms=62, z=3, edge="none", alpha=0.92, line_alpha=0.35)

# plain LoRA (medium, black edge)
traj("lora_r16", lw=1.8, ms=85, z=5, edge="k", line_alpha=0.5)

# LoRA+wd trajectory (bold)
traj("lorawd_wd0p3", lw=2.0, ms=100, z=6, edge="k", line_alpha=0.55)

# ---- highlight LoRA+wd Pareto frontier ----
fx=[p["adapt"] for p in lorawd_fr]; fy=[p["ret"] for p in lorawd_fr]
ax.plot(fx,fy,color=COL["lorawd"],lw=4.5,alpha=0.9,zorder=4,solid_capstyle="round",
        label="LoRA+wd swept frontier")
# shade the region dominated by the frontier (down-left of it)
import numpy as np
ax.fill_between(fx, [0]*len(fx), fy, color=COL["lorawd"], alpha=0.07, zorder=1)

# base retention line
ax.axhline(BASE_CORE, ls=":", color="green", lw=1.5, zorder=2)
ax.text(30, BASE_CORE+0.2, "base-model retention",
        color="green", fontsize=9, va="bottom", ha="center", fontweight="bold")

# ---- annotate the "if you only ran ONE LR" illusion points ----
# For each fancy method: its best single-LR pt (highest adapt with ret>=24, else best adapt)
annot=[]
for arm in fancy:
    m=cov[arm]
    cand=[m[l] for l in lrorder if l in m and m[l]["ret"]>=24.0] or [m[l] for l in lrorder if l in m]
    bp=max(cand,key=lambda p:p["adapt"])
    annot.append((arm,bp))

# ring the illusion points
for arm,bp in annot:
    k=ck(arm)
    ax.scatter([bp["adapt"]],[bp["ret"]],s=300,facecolors="none",edgecolors=COL[k],
               linewidths=2.0,zorder=8)

# a single explanatory callout
ax.annotate("Rings = each method's best-looking single-LR point.\n"
            "Every one sits on or below the LoRA+wd swept frontier\n"
            "(shaded region is on or below that frontier).\n"
            "Low points = same methods at high LR: magnitude blow-up → collapse.",
            xy=(0.015,0.025), xycoords="axes fraction", ha="left", va="bottom",
            fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fffbe6", ec="#c9a227", lw=1.0))

ax.set_xlabel(r"Adaptation: commonsense accuracy [%], higher is better")
ax.set_ylabel(r"Retention: mean of BBH and MMLU-Pro [%], higher is better")
ax.set_title("Claim 2 (diagnosis): structured adapters' 'wins' carry the ingredients of a learning-rate artifact\n"
             "sweep the LR and LoRA+wd's frontier sits on or above every method",
             fontsize=13.5, pad=12)
ax.grid(True, alpha=0.18)
ax.set_xlim(5, 88)
ax.set_ylim(0, 30)

# legend: methods
handles=[]
for arm in ["lorawd_wd0p3","lora_r16"]+fancy:
    k=ck(arm)
    handles.append(Line2D([0],[0],marker=MARK[k],color=COL[k],lw=0,ms=10,
                          markeredgecolor="k" if arm in("lora_r16","lorawd_wd0p3") else "none",
                          label=PRETTY[arm]))
handles.append(Line2D([0],[0],color=COL["lorawd"],lw=4.5,label="LoRA+wd swept frontier"))
handles.append(Line2D([0],[0],marker="o",color="none",markerfacecolor="none",
                      markeredgecolor="0.3",markeredgewidth=1.6,ms=13,lw=0,
                      label="best single-LR point (illusion)"))
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.005,0.42),
          ncol=1, fontsize=9, framealpha=0.96)

fig.text(0.5, 0.005,
    "Llama-2-7B, commonsense fine-tuning, seed 42, 7 LRs per method (2e-5…1e-3). "
    "CorDA withheld (calibration-fairness fix pending). Single seed; illustrative.",
    ha="center", va="bottom", fontsize=7.5, color="0.5", style="italic")

fig.tight_layout(rect=[0,0.02,1,1])
fig.savefig(OUT)
print("wrote", OUT)
