"""E-batch (adversarial-review experiments E1-E7) analysis over whatever has landed.
Rerunnable as results arrive; final run feeds key_numbers §18 and the assessment doc.

Blocks: E1 interventional (per-tier trained-vs-random, on-curve residuals vs the lrsw
observational fit), E2 full-FT ladder vs LoRA-family curve, E3 Qwen knee densification
(refit bottom-half r with new mid-LR cells), E4 SC-LoRA eval-matched ladder vs nq_open,
E5 replay, E6 wd-transfer, E7 bridging arms.
"""
import json, glob, math, os, re
import numpy as np

def load(pat):
    out = []
    for f in sorted(glob.glob(f"results/{pat}/summary.json")):
        try:
            h = json.load(open(f))["headline"]
        except Exception:
            continue
        rn = os.path.basename(os.path.dirname(f))
        fd, ret = h.get("fdelta"), h.get("retention_mean")
        if isinstance(fd, (int, float)) and isinstance(ret, (int, float)) and fd and fd > 0:
            out.append(dict(rn=rn, fd=fd, lfd=math.log10(fd), ret=ret,
                            broad=h.get("retention_broad"), adapt=h.get("cs_avg")))
    return out

def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b  # intercept, slope

# observational lrsw baseline fit (linear in log10 fd) for residual comparisons
obs = [r for r in load("lrsw_*") if not re.search(r"rep05|dorawd|milorawd", r["rn"])]
b_obs = linfit([r["lfd"] for r in obs], [r["ret"] for r in obs])
pred = lambda fd: b_obs[0] + b_obs[1] * math.log10(fd)
print(f"observational lrsw fit: ret = {b_obs[0]:.2f} + {b_obs[1]:.2f}*log10(fd)  (n={len(obs)})")

print("\n=== E1 INTERVENTIONAL (rescale + random-direction controls) ===")
e1 = load("e1_*")
print(f"{'run':28s} {'fd':>6s} {'ret':>6s} {'pred':>6s} {'resid':>6s} {'adapt':>6s}")
for r in sorted(e1, key=lambda r: (("rand" in r["rn"]), r["fd"])):
    res = r["ret"] - pred(r["fd"])
    print(f"{r['rn']:28s} {r['fd']:6.3f} {r['ret']:6.2f} {pred(r['fd']):6.2f} {res:+6.2f} {r['adapt']:6.2f}")
tr = [r for r in e1 if "randdir" not in r["rn"]]
rd = [r for r in e1 if "randdir" in r["rn"]]
if tr:
    res_t = [r["ret"] - pred(r["fd"]) for r in tr]
    print(f"trained rescales: n={len(tr)} mean on-curve residual {np.mean(res_t):+.2f} ± {np.std(res_t):.2f} pp")
    print(f"  r(ret, log fd) within rescales: {pearson([r['lfd'] for r in tr], [r['ret'] for r in tr]):+.3f}")
if rd:
    res_r = [r["ret"] - pred(r["fd"]) for r in rd]
    print(f"random controls:  n={len(rd)} mean residual {np.mean(res_r):+.2f} ± {np.std(res_r):.2f} pp "
          f"(direction penalty vs trained: {np.mean(res_r)-np.mean(res_t):+.2f} pp)")

print("\n=== E2 FULL-FT ANCHOR vs LoRA-family curve ===")
for r in sorted(load("fft_*"), key=lambda r: r["fd"]):
    print(f"{r['rn']:22s} fd={r['fd']:6.3f} ret={r['ret']:6.2f} (LoRA-curve pred {pred(r['fd']):5.2f}, "
          f"resid {r['ret']-pred(r['fd']):+.2f}) adapt={r['adapt']}")

print("\n=== E3 QWEN DENSIFICATION (does the mid-range fill the knee?) ===")
for fam, famlab in (("qwsw", "Qwen CS"), ("qwswm", "Qwen math")):
    allq = [r for r in load(f"{fam}_*") if re.match(rf"^{fam}_", r["rn"])]
    if not allq:
        continue
    lx = np.array([r["lfd"] for r in allq]); y = np.array([r["ret"] for r in allq])
    med = np.median(lx)
    newc = [r for r in allq if re.search(r"lr7e5|lr15e5", r["rn"])]
    print(f"{famlab}: n={len(allq)} (+{len(newc)} new mid-LR) pooled r={pearson(lx,y):+.3f} "
          f"bottom-half(median) r={pearson(lx[lx<=med], y[lx<=med]):+.3f}")
    for r in sorted(newc, key=lambda r: r["fd"]):
        print(f"   {r['rn']:34s} fd={r['fd']:7.4f} ret={r['ret']:6.2f} adapt={r['adapt']}")

print("\n=== E4 SC-LoRA EVAL-MATCHED CALIBRATION (b4 arm, full ladder) ===")
b4 = load("b4_sclora_*")
nq = [r for r in load("lrsw_sclora_*")]
for r in sorted(b4, key=lambda r: r["fd"]):
    print(f"{r['rn']:30s} fd={r['fd']:6.3f} ret={r['ret']:6.2f} resid_vs_lrsw_curve={r['ret']-pred(r['fd']):+.2f} adapt={r['adapt']}")
if b4:
    rb = [r["ret"] - pred(r["fd"]) for r in b4]
    rn_ = [r["ret"] - pred(r["fd"]) for r in nq]
    print(f"b4 (eval-matched) mean resid {np.mean(rb):+.2f} (n={len(rb)}) vs nq_open sclora {np.mean(rn_):+.2f} (n={len(rn_)})")

print("\n=== E5 REPLAY BASELINE ===")
for r in sorted(load("lrsw_lorarep05_*"), key=lambda r: r["rn"]):
    print(f"{r['rn']:34s} fd={r['fd']:6.3f} ret={r['ret']:6.2f} resid={r['ret']-pred(r['fd']):+.2f} adapt={r['adapt']}")

print("\n=== E6 WD ON OTHER METHODS ===")
for r in sorted(load("lrsw_dorawd_*") + load("lrsw_milorawd_*"), key=lambda r: r["rn"]):
    print(f"{r['rn']:36s} fd={r['fd']:6.3f} ret={r['ret']:6.2f} resid={r['ret']-pred(r['fd']):+.2f} adapt={r['adapt']}")

print("\n=== E7 BRIDGING ARMS (MedMCQA, attn-only) ===")
for fam in ("brl", "brq"):
    rs = load(f"{fam}_*")
    if rs:
        print(f"{fam}: r(ret, log fd) = {pearson([r['lfd'] for r in rs], [r['ret'] for r in rs]):+.3f} (n={len(rs)})")
        for r in sorted(rs, key=lambda r: r["fd"]):
            print(f"   {r['rn']:28s} fd={r['fd']:7.4f} ret={r['ret']:6.2f} medmcqa={r['adapt']}")

print("\n=== DSV4 (whatever has landed) ===")
ds = load("dsv4_*")
for r in sorted(ds, key=lambda r: r["rn"]):
    print(f"   {r['rn']:34s} fd={r['fd']!s:9s} ret={r['ret']} broad={r['broad']} medmcqa={r['adapt']}")
if len(ds) >= 5:
    print(f"dsv4: r(ret, log fd) = {pearson([r['lfd'] for r in ds], [r['ret'] for r in ds]):+.3f} (n={len(ds)})")
print("\ndone.")
