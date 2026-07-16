"""Full-picture recompute of the key law numbers (2026-07-16, n>=3 campaign data).

Mirrors key_numbers.md methodology: magnitude axis = headline.fdelta (CLoRA F_Delta),
retention = headline.retention_mean (core). Families by run_name prefix:
  lrsw_ = Llama-2 CS   | lrswm_ = Llama-2 math | qwsw_ = Qwen-2.5 CS | qwswm_ = Qwen-2.5 math
  frc_/frm_ = Llama operating-point grids (CS/math-395k)
Excluded: corda (contaminated), SMOKE/test runs, non-finite FDelta/retention (diverged).
Outputs: pooled + per-family + seed-averaged r(ret, log10 FDelta); seed variance;
merged-geometry spread correlations; CE cross-check.
"""
import json, glob, math, os, re
from collections import defaultdict

RES = "results"

def pearson(pairs):
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n < 3:
        return float("nan"), n
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return float("nan"), n
    return cov / math.sqrt(vx * vy), n

def spearman(pairs):
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n < 3:
        return float("nan"), n
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx = ranks([p[0] for p in pairs]); ry = ranks([p[1] for p in pairs])
    return pearson(list(zip(rx, ry)))

rows = []
for f in glob.glob(os.path.join(RES, "*", "summary.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rn = d.get("run_name") or os.path.basename(os.path.dirname(f))
    if "SMOKE" in rn or "smoke" in rn or "corda" in rn:
        continue
    h = d.get("headline") or {}
    fd, ret = h.get("fdelta"), h.get("retention_mean")
    if not isinstance(fd, (int, float)) or not isinstance(ret, (int, float)):
        continue
    if not (math.isfinite(fd) and math.isfinite(ret)) or fd <= 0:
        continue
    m = re.match(r"^([a-z0-9]+)_", rn)
    fam = m.group(1) if m else "other"
    sm = re.search(r"_s(4[2-9])$", rn)
    seed = sm.group(1) if sm else None
    cell = re.sub(r"_s4[2-9]$", "", rn)
    rows.append(dict(rn=rn, fam=fam, seed=seed, cell=cell, fd=fd, ret=ret,
                     cs=h.get("cs_avg"), dwmax=h.get("dw_sv_max")))

print(f"usable rows (finite FDelta & retention, non-corda/smoke): {len(rows)}")

FAMS = ["lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"]
LABEL = {"lrsw": "Llama-2 CS", "lrswm": "Llama-2 math", "qwsw": "Qwen-2.5 CS",
         "qwswm": "Qwen-2.5 math", "frc": "Llama CS grid (frc)", "frm": "Llama math-395k (frm)"}

print("\n=== (1) THE MAGNITUDE LAW: r(retention_core, log10 FDelta) — ALL SEEDS pooled ===")
allpairs = []
for fam in FAMS:
    pairs = [(math.log10(r["fd"]), r["ret"]) for r in rows if r["fam"] == fam]
    allpairs += pairs
    rP, n = pearson(pairs)
    rS, _ = spearman(pairs)
    seeds = sorted({r["seed"] for r in rows if r["fam"] == fam and r["seed"]})
    print(f"  {LABEL[fam]:22s} r={rP:+.3f} (rank {rS:+.3f})  n={n:4d}  seeds={','.join(seeds)}")
rP, n = pearson(allpairs); rS, _ = spearman(allpairs)
print(f"  {'ALL FAMILIES pooled':22s} r={rP:+.3f} (rank {rS:+.3f})  n={n}")

print("\n=== (2) SEED-AVERAGED law (cell = recipe averaged over its seeds) ===")
for fam in FAMS:
    cells = defaultdict(list)
    for r in rows:
        if r["fam"] == fam and r["seed"]:
            cells[r["cell"]].append(r)
    pairs, multi = [], 0
    for c, rs in cells.items():
        fdm = sum(x["fd"] for x in rs) / len(rs)
        retm = sum(x["ret"] for x in rs) / len(rs)
        pairs.append((math.log10(fdm), retm))
        if len(rs) >= 3:
            multi += 1
    rP, n = pearson(pairs)
    print(f"  {LABEL[fam]:22s} r={rP:+.3f}  cells={n:4d}  (cells with n>=3 seeds: {multi})")

print("\n=== (3) SEED VARIANCE (n>=3 cells): mean within-cell SD ===")
for fam in FAMS:
    cells = defaultdict(list)
    for r in rows:
        if r["fam"] == fam and r["seed"]:
            cells[r["cell"]].append(r)
    sds_ret, sds_fd = [], []
    for c, rs in cells.items():
        if len(rs) < 3:
            continue
        for key, acc in (("ret", sds_ret), ("fd", sds_fd)):
            v = [x[key] for x in rs]
            mu = sum(v) / len(v)
            acc.append(math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)))
    if sds_ret:
        print(f"  {LABEL[fam]:22s} cells(n>=3)={len(sds_ret):3d}  SD(ret)={sum(sds_ret)/len(sds_ret):.2f} pp  SD(FDelta)={sum(sds_fd)/len(sds_fd):.3f}")

print("\n=== (4) GEOMETRY (merged, n-full): spread metrics vs retention ===")
geo = {}
for line in open("results/geo_drift/adapter_metrics_merged.jsonl"):
    try:
        g = json.loads(line)
        geo[g.get("run")] = g
    except Exception:
        pass
byrn = {r["rn"]: r for r in rows}
for metric in ["stable_rank_w", "eff_rank_w", "spec_mean", "spec_max", "fro_total"]:
    pairs = []
    for rn, g in geo.items():
        r = byrn.get(rn)
        v = g.get(metric)
        if r and isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            pairs.append((math.log10(v) if metric in ("spec_mean", "spec_max", "fro_total") else v, r["ret"]))
    rP, n = pearson(pairs); rS, _ = spearman(pairs)
    print(f"  {metric:18s} r={rP:+.3f} (rank {rS:+.3f}) n={n}")

print("\n=== (5) CE cross-check (merged): delta-CE vs FDelta ===")
ce = {}
for line in open("results/forgetting_merged.jsonl"):
    try:
        c = json.loads(line)
        ce[c.get("run_name")] = c
    except Exception:
        pass
pairs = []
cekeys = set()
for rn, c in ce.items():
    r = byrn.get(rn)
    if not r:
        continue
    dce = c.get("delta_ce", c.get("dce", c.get("ce_delta")))
    if dce is None and "ce_adapter" in c and "ce_base" in c:
        dce = c["ce_adapter"] - c["ce_base"]
    cekeys |= set(c.keys())
    if isinstance(dce, (int, float)) and math.isfinite(dce):
        pairs.append((math.log10(r["fd"]), dce))
rP, n = pearson(pairs)
print(f"  r(log10 FDelta, delta-CE) = {rP:+.3f}  n={n}")
if n < 10:
    print(f"  [CE keys available: {sorted(cekeys)[:12]}]")
