#!/usr/bin/env python3
"""Operating-point tables from final dataset (results/*/summary.json)."""
import json, os, re, statistics as st
from collections import defaultdict

ROOT = "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
RES = os.path.join(ROOT, "results")

quar = set()
with open(os.path.join(RES, "quarantine_diverged.txt")) as f:
    for line in f:
        line = line.strip()
        if line:
            quar.add(line.split("\t")[0].split()[0])

LRMAP = {"2e5":2e-5,"5e5":5e-5,"1e4":1e-4,"15e5":1.5e-4,"2e4":2e-4,"3e4":3e-4,
         "5e4":5e-4,"7e4":7e-4,"7e5":7e-5,"1e3":1e-3,"2e3":2e-3,"5e3":5e-3,"1e2":1e-2}
def fmt_lr(lr):
    if lr is None: return "??"
    for k,v in {"2e-5":2e-5,"5e-5":5e-5,"1e-4":1e-4,"1.5e-4":1.5e-4,"2e-4":2e-4,
                "3e-4":3e-4,"5e-4":5e-4,"7e-4":7e-4,"7e-5":7e-5,"1e-3":1e-3,"2e-3":2e-3,
                "5e-3":5e-3,"1e-2":1e-2}.items():
        if abs(lr-v) < 1e-9: return k
    return str(lr)

rows = []
for d in sorted(os.listdir(RES)):
    p = os.path.join(RES, d, "summary.json")
    if not os.path.isfile(p): continue
    try:
        s = json.load(open(p))
    except Exception as e:
        print("BADJSON", d, e); continue
    h = s.get("headline", {})
    m = re.search(r"_lr(\d+e\d)(?:_|$)", d)
    lr = LRMAP.get(m.group(1)) if m else None
    m2 = re.search(r"_s(\d+)(?:_|$)", d)
    seed = int(m2.group(1)) if m2 else None
    m3 = re.search(r"_c(\d\d\d+)(?:_|$)", d)
    ctx = int(m3.group(1)) if m3 else None
    rows.append(dict(run=d, quar=(d in quar), lr=lr, seed=seed, ctx=ctx,
        cs=h.get("cs_avg"), ret=h.get("retention_mean"), retb=h.get("retention_broad"),
        bbh=h.get("bbh"), fd=h.get("fdelta"), svmax=h.get("dw_sv_max"),
        adapt_task=h.get("adapt_task"), per=s.get("per_dataset")))

def sel(prefix, ctx=None):
    out = [r for r in rows if r["run"].startswith(prefix)]
    if ctx is not None: out = [r for r in out if r["ctx"] == ctx]
    return out

def mstd(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None, None, 0
    mu = st.mean(xs)
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    return mu, sd, len(xs)

def f2(x): return "None" if x is None else f"{x:.2f}"

def best_table(name, specs, acc="cs", retkey="ret", safe_thresh=24.0,
               safe_lrs=None, show_quar=True, ctx=None):
    print(f"\n===== {name} =====")
    for label, prefix in specs:
        rs = [r for r in sel(prefix, ctx) if not r["quar"]]
        qs = [r for r in sel(prefix, ctx) if r["quar"]]
        bylr = defaultdict(list)
        for r in rs:
            if r["lr"] is not None: bylr[r["lr"]].append(r)
        if not bylr:
            print(f"{label:20s} NO DATA (quarantined: {len(qs)})"); continue
        # best LR by mean accuracy
        def accmean(lr): return st.mean([x[acc] for x in bylr[lr] if x[acc] is not None])
        best = max(bylr, key=accmean)
        cell = sorted(bylr[best], key=lambda x: x["seed"])
        amu, asd, n = mstd([x[acc] for x in cell])
        rmu, rsd, _ = mstd([x[retkey] for x in cell])
        fds = [x["fd"] for x in cell if x["fd"] is not None]
        seeds = ",".join(str(x["seed"]) for x in cell)
        line = (f"{label:20s} bestLR={fmt_lr(best):7s} {acc}={amu:.2f}±{asd:.2f} "
                f"ret={rmu:.2f}±{rsd:.2f} F_D={min(fds):.3f}-{max(fds):.3f} n={n} seeds[{seeds}]")
        if safe_lrs:
            s42 = {r["lr"]: r for r in rs if r["seed"] == 42}
            band = sum(1 for L in safe_lrs if L in s42 and s42[L][retkey] is not None
                       and s42[L][retkey] >= safe_thresh)
            have = sum(1 for L in safe_lrs if L in s42)
            # quarantined s42 count as unsafe but note
            q42 = [q["run"] for q in qs if q["seed"] == 42]
            line += f" safe@s42={band}/{len(safe_lrs)} (evald {have}, quar_s42={q42})"
        print(line)
        # per-seed detail at best LR
        for x in cell:
            print(f"    s{x['seed']}: {acc}={f2(x[acc])} ret={f2(x[retkey])} fd={x['fd']:.4f} bbh={f2(x['bbh'])}")
        if show_quar and qs:
            print(f"    quarantined ({len(qs)}): {[q['run'] for q in qs]}")

# ---------------- 1. LLAMA CS ----------------
LRS7 = [2e-5,5e-5,1e-4,2e-4,3e-4,5e-4,1e-3]
llama_specs = [
    ("LoRA+wd(0.3)", "lrsw_lorawd_wd0p3_"),
    ("SC-LoRA", "lrsw_sclora_r32_"),
    ("LoRA", "lrsw_lora_r16_"),
    ("LoRA-Null", "lrsw_lora_null_r16_"),
    ("CLoRA-k1024", "lrsw_clora_k1024_"),
    ("DoRA", "lrsw_dora_r16_"),
    ("MiLoRA", "lrsw_milora_r32_"),
]
best_table("LLAMA CS (lrsw)", llama_specs, safe_lrs=LRS7)

# per-LR mean table for each lrsw adapter (to justify best-LR + safe band)
print("\n-- lrsw per-LR means (acc / ret / n) --")
for label, prefix in llama_specs:
    rs = [r for r in sel(prefix) if not r["quar"] and r["lr"] is not None]
    bylr = defaultdict(list)
    for r in rs: bylr[r["lr"]].append(r)
    parts = []
    for lr in sorted(bylr):
        a,_,n = mstd([x["cs"] for x in bylr[lr]]); rt,_,_ = mstd([x["ret"] for x in bylr[lr]])
        parts.append(f"{fmt_lr(lr)}:{a:.1f}/{rt:.1f}(n{n})")
    print(f"{label:14s} " + " ".join(parts))

# ---------------- 2. MATH ----------------
math_specs = [
    ("LoRA+wd(0.3)", "lrswm_lorawd_wd0p3_"),
    ("SC-LoRA", "lrswm_sclora_r32_"),
    ("LoRA", "lrswm_lora_r16_"),
    ("CLoRA-k1024", "lrswm_clora_k1024_"),
    ("DoRA", "lrswm_dora_r16_"),
    ("MiLoRA", "lrswm_milora_r32_"),
]
best_table("LLAMA MATH sweep (lrswm), acc=gsm8k(cs key), ret shown = bbh", math_specs, retkey="bbh")

frm_specs = [
    ("LoRA+wd0.3", "frm_lorawd_wd0p3_"),
    ("LoRA(wd0)", "frm_lorawd_wd0_"),
    ("MiLoRA", "frm_milora_"),
    ("CLoRA-k64", "frm_clora_k64_"),
    ("CLoRA-k128", "frm_clora_k128_"),
    ("CLoRA-k256", "frm_clora_k256_"),
    ("SC-LoRA", "frm_sclora_"),
    ("DoRA", "frm_dora_"),
    ("LoRA(r32,lr3e4)", "frm_lora_lr3e4"),
    ("LoRA-Null", "frm_lora_null_"),
    ("PiSSA", "frm_pissa_"),
    ("CorDA++", "frm_cordapp_"),
]
best_table("LLAMA MATH faithful-recipe (frm c256), acc=gsm8k, ret=bbh", frm_specs, retkey="bbh", ctx=256)

# frm lorawd wd0p3 lr2e4 3-seed explicit
print("\n-- frm_lorawd_wd0p3_lr2e4_c256 all seeds --")
for r in sorted(sel("frm_lorawd_wd0p3_lr2e4_c256"), key=lambda x: x["seed"]):
    print(f"  {r['run']}: gsm8k={f2(r['cs'])} bbh={f2(r['bbh'])} ret_core={f2(r['ret'])} fd={r['fd']:.3f} quar={r['quar']}")

# ---------------- 3. QWEN ----------------
qw_specs = [
    ("LoRA+wd(0.3)", "qwsw_lorawd_wd0p3_"),
    ("SC-LoRA", "qwsw_sclora_r32_"),
    ("LoRA", "qwsw_lora_r16_"),
    ("LoRA-Null", "qwsw_lora_null_r16_"),
    ("CLoRA-k1024", "qwsw_clora_k1024_"),
    ("DoRA", "qwsw_dora_r16_"),
    ("MiLoRA", "qwsw_milora_r32_"),
]
best_table("QWEN CS (qwsw)", qw_specs, safe_lrs=LRS7)

qwm_specs = [
    ("LoRA+wd(0.3)", "qwswm_lorawd_wd0p3_lr"),
    ("SC-LoRA", "qwswm_sclora_r32_"),
    ("LoRA-r32", "qwswm_lora_r32_lr"),
    ("LoRA-r16", "qwswm_lora_r16_"),
    ("LoRA-Null", "qwswm_lora_null_r16_"),
    ("CLoRA-k1024", "qwswm_clora_k1024_"),
    ("DoRA", "qwswm_dora_r16_"),
    ("MiLoRA", "qwswm_milora_r32_"),
]
# exclude ep6 variants from lr-parse (they match _lr2e4_ep6_ -> regex still parses lr; filter)
def qwm_filter(r): return "_ep6_" not in r["run"] and "SMOKE" not in r["run"]
rows_backup = rows
rows = [r for r in rows if qwm_filter(r)]
best_table("QWEN MATH (qwswm), acc=gsm8k, ret=bbh", qwm_specs, retkey="bbh")
rows = rows_backup

print("\n-- qwsw per-LR means (acc / ret / n) --")
for label, prefix in qw_specs:
    rs = [r for r in sel(prefix) if not r["quar"] and r["lr"] is not None]
    bylr = defaultdict(list)
    for r in rs: bylr[r["lr"]].append(r)
    parts = []
    for lr in sorted(bylr):
        a,_,n = mstd([x["cs"] for x in bylr[lr]]); rt,_,_ = mstd([x["ret"] for x in bylr[lr]])
        parts.append(f"{fmt_lr(lr)}:{a:.1f}/{rt:.1f}(n{n})")
    print(f"{label:14s} " + " ".join(parts))

# SC-LoRA qwsw seed story at each LR
print("\n-- qwsw_sclora per-LR per-seed (ret, fd) --")
bylr = defaultdict(list)
for r in sel("qwsw_sclora_r32_"):
    if r["lr"] is not None: bylr[r["lr"]].append(r)
for lr in sorted(bylr):
    xs = sorted(bylr[lr], key=lambda x: x["seed"])
    print(" ", fmt_lr(lr), [(f"s{x['seed']}", f2(x['cs']), f2(x['ret']), f2(x['fd']), "Q" if x["quar"] else "") for x in xs])

# ---------------- 4. COROLLARY CONTROLS ----------------
print("\n===== RANK LADDER frc_lora_r{8,16,32}_lr3e4 =====")
for pre in ["frc_lora_r8_lr3e4_c256", "frc_lora_r16_lr3e4_c256", "frc_lora_r32_lr3e4_c256"]:
    xs = sorted(sel(pre), key=lambda x: x["seed"])
    accs=[x["cs"] for x in xs if not x["quar"]]; rets=[x["ret"] for x in xs if not x["quar"]]
    fds=[x["fd"] for x in xs if not x["quar"]]
    a,asd,n = mstd(accs); r_,rsd,_ = mstd(rets); f_,fsd,_ = mstd(fds)
    print(f"{pre}: acc={a:.2f}±{asd:.2f} ret={r_:.2f}±{rsd:.2f} fd={f_:.3f}±{fsd:.3f} n={n}")
    for x in xs:
        print(f"    s{x['seed']}: acc={f2(x['cs'])} ret={f2(x['ret'])} fd={x['fd']:.3f} quar={x['quar']}")

print("\n===== r16 PARAM-MATCHED LoRA+wd (frc_lorawdr16) =====")
for r in sorted(sel("frc_lorawdr16"), key=lambda x: x["run"]):
    print(f"  {r['run']}: acc={f2(r['cs'])} ret={f2(r['ret'])} fd={r['fd']:.3f} quar={r['quar']}")

print("\n===== WD x LR GRID (frc_lorawd_wd*, lr3e4) monotonicity =====")
for wd in ["wd0","wd0p1","wd0p2","wd0p3","wd0p5"]:
    xs = [r for r in sel(f"frc_lorawd_{wd}_lr3e4_c256") if not r["quar"]]
    xs.sort(key=lambda x: x["seed"])
    a,asd,n = mstd([x["cs"] for x in xs]); r_,rsd,_ = mstd([x["ret"] for x in xs])
    f_,fsd,_ = mstd([x["fd"] for x in xs])
    seeds=",".join(str(x['seed']) for x in xs)
    print(f"  {wd:5s}: fd={f_:.3f}±{fsd:.3f} ret={r_:.2f}±{rsd:.2f} acc={a:.2f}±{asd:.2f} n={n} [{seeds}]")
# s42-only for direct comparison with 07-14 quote
print("  -- s42 only --")
for wd in ["wd0","wd0p1","wd0p2","wd0p3","wd0p5"]:
    xs = [r for r in sel(f"frc_lorawd_{wd}_lr3e4_c256") if r["seed"]==42 and not r["quar"]]
    for x in xs:
        print(f"  {wd:5s} s42: fd={x['fd']:.3f} ret={f2(x['ret'])} acc={f2(x['cs'])}")

# E6 milorawd / dorawd
print("\n===== E6 lrsw_milorawd / dorawd =====")
for r in sorted(sel("lrsw_milorawd") + sel("lrsw_dorawd") + sel("lrsw_milora_r32_lr2e4") + sel("lrsw_milora_r32_lr5e4"), key=lambda x: x["run"]):
    print(f"  {r['run']}: acc={f2(r['cs'])} ret={f2(r['ret'])} fd={r['fd'] if r['fd'] is None else round(r['fd'],3)} quar={r['quar']}")

# ---------------- 5. HIGH-K CLORA ----------------
print("\n===== CLORA K-GRID frc_clora_k* lr3e4 =====")
for k in ["k128","k256","k512","k1024","k2048"]:
    xs = sorted([r for r in sel(f"frc_clora_{k}_lr3e4_c256")], key=lambda x: x["seed"])
    ok = [x for x in xs if not x["quar"]]
    a,asd,n = mstd([x["cs"] for x in ok]); r_,rsd,_ = mstd([x["ret"] for x in ok])
    f_,fsd,_ = mstd([x["fd"] for x in ok])
    print(f"  {k:6s}: fd={f_:.3f}±{fsd:.3f} ret={r_:.2f}±{rsd:.2f} acc={a:.2f}±{asd:.2f} n={n}")
    for x in xs:
        print(f"      s{x['seed']}: acc={f2(x['cs'])} ret={f2(x['ret'])} fd={x['fd']:.3f} quar={x['quar']}")

# frm clora k grid too (math)
print("\n-- frm clora k grid (math) --")
for k in ["k64","k128","k256"]:
    xs = sorted([r for r in sel(f"frm_clora_{k}_lr3e4_c256")], key=lambda x: x["seed"])
    for x in xs:
        print(f"  {k} s{x['seed']}: gsm8k={f2(x['cs'])} bbh={f2(x['bbh'])} fd={x['fd']:.3f} quar={x['quar']}")

print("\nTOTAL usable rows:", len([r for r in rows if not r["quar"]]), "quarantined:", len([r for r in rows if r["quar"]]))

print("\n-- qwswm per-LR means (gsm8k / bbh / n) [ep6+SMOKE excluded] --")
rows2 = [r for r in rows if "_ep6_" not in r["run"] and "SMOKE" not in r["run"]]
def sel2(prefix): return [r for r in rows2 if r["run"].startswith(prefix)]
for label, prefix in qwm_specs:
    rs = [r for r in sel2(prefix) if not r["quar"] and r["lr"] is not None]
    bylr = defaultdict(list)
    for r in rs: bylr[r["lr"]].append(r)
    parts = []
    for lr in sorted(bylr):
        a,_,n = mstd([x["cs"] for x in bylr[lr]]); rt,_,_ = mstd([x["bbh"] for x in bylr[lr]])
        parts.append(f"{fmt_lr(lr)}:{a:.1f}/{rt:.1f}(n{n})")
    print(f"{label:14s} " + " ".join(parts))

print("\n-- frm lorawd c512 robustness cells --")
for r in sorted([r for r in rows if r["ctx"]==512], key=lambda x: x["run"]):
    print(f"  {r['run']}: gsm8k={f2(r['cs'])} bbh={f2(r['bbh'])} fd={r['fd']:.3f} quar={r['quar']}")

print("\n-- lrswm per-LR means (gsm8k / bbh / n) --")
for label, prefix in math_specs:
    rs = [r for r in sel(prefix) if not r["quar"] and r["lr"] is not None]
    bylr = defaultdict(list)
    for r in rs: bylr[r["lr"]].append(r)
    parts = []
    for lr in sorted(bylr):
        a,_,n = mstd([x["cs"] for x in bylr[lr]]); rt,_,_ = mstd([x["bbh"] for x in bylr[lr]])
        parts.append(f"{fmt_lr(lr)}:{a:.1f}/{rt:.1f}(n{n})")
    print(f"{label:14s} " + " ".join(parts))

print("\n-- frm per-LR means (gsm8k / bbh / n), c256 --")
for label, prefix in frm_specs:
    rs = [r for r in sel(prefix, 256) if not r["quar"] and r["lr"] is not None]
    bylr = defaultdict(list)
    for r in rs: bylr[r["lr"]].append(r)
    parts = []
    for lr in sorted(bylr):
        a,_,n = mstd([x["cs"] for x in bylr[lr]]); rt,_,_ = mstd([x["bbh"] for x in bylr[lr]])
        parts.append(f"{fmt_lr(lr)}:{a:.1f}/{rt:.1f}(n{n})")
    print(f"{label:14s} " + " ".join(parts))
