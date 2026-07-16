"""Adversarial-review recompute (2026-07-16) — the 10 zero-GPU analyses from
paper/writing/adversarial_review_2026-07-16.md §3, run against the full campaign data.

Conventions match analyze_full_2026-07-16.py / key_numbers.md:
  magnitude = headline.fdelta (log10), retention_core = headline.retention_mean
  (= mean(bbh, mmlu_pro)), retention_broad = mean(bbh, mmlu_pro, mmlu, arc_c, truthfulqa).
  corda/cordapp/SMOKE and non-finite rows excluded. Families by run_name prefix.

Sections:
  A1 functional form (2-segment knee vs linear; robustness subsets; normalized slopes)
  A2 LR-proxy rescue (dummy-LR R2, fixed-LR strata, partials, grid decoupling)
  A3 direction as second-order (cell-level partial, per-method matched-FDelta offsets)
  A4 adaptation-efficiency ANCOVA
  A5 within-cell micro-test (seed-level fluctuations at fixed recipe)
  A6 retention_broad without ARC-c (adapt-suite contamination control)
  A7 format-collapse control (degenerate-output proxy; per-item parse rates not retained)
  A8 within-family CE corroboration + Qwen CE coverage holes (backfill list)
  A9 FDelta decomposition ||dW||_F x alignment; dw_sv comparison
  (A10 = geometry-key fix, applied directly to analyze_full_2026-07-16.py)
"""
import json, glob, math, os, re, sys
from collections import defaultdict
import numpy as np

RES = "results"
FAMS = ["lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"]
LABEL = {"lrsw": "Llama-2 CS", "lrswm": "Llama-2 math", "qwsw": "Qwen-2.5 CS",
         "qwswm": "Qwen-2.5 math", "frc": "Llama CS grid", "frm": "Llama math-395k"}
RET_TASKS = ["bbh", "mmlu_pro", "mmlu", "arc_c", "truthfulqa"]

# ---------- helpers ----------
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)

def ols(X, y):
    """OLS with intercept prepended. Returns beta, se, R2, n, resid."""
    X = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * s2, 0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, r2, n, resid

def partial_r(x, y, Z):
    """r(x, y | Z) via double residualization. Z: 2-D array of controls."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    _, _, _, _, rx = ols(Z, x)
    _, _, _, _, ry = ols(Z, y)
    r, n = pearson(rx, ry)
    t = r * math.sqrt(max(n - 2 - Z.shape[1], 1) / max(1e-12, 1 - r * r)) if math.isfinite(r) else float("nan")
    return r, n, t

def dummies(labels):
    """Full-rank dummy matrix (drop first level). Returns matrix, levels."""
    lv = sorted(set(labels))
    D = np.zeros((len(labels), len(lv) - 1))
    for i, l in enumerate(labels):
        j = lv.index(l)
        if j > 0:
            D[i, j - 1] = 1
    return D, lv

def seg2_fit(lx, y):
    """2-segment continuous piecewise-linear fit; knee grid = 20..80th pct.
    Returns knee, slopes (lo, hi), F-stat vs single line, p-ish note, sse pair."""
    lx, y = np.asarray(lx, float), np.asarray(y, float)
    order = np.argsort(lx)
    lx, y = lx[order], y[order]
    n = len(lx)
    b_lin, _, r2_lin, _, res_lin = ols(lx[:, None], y)
    sse_lin = float(res_lin @ res_lin)
    best = (None, None, None, np.inf)
    for q in np.linspace(0.20, 0.80, 25):
        knee = float(np.quantile(lx, q))
        hinge = np.maximum(lx - knee, 0)
        Xk = np.column_stack([lx, hinge])
        beta, _, _, _, res = ols(Xk, y)
        sse = float(res @ res)
        if sse < best[3]:
            best = (knee, float(beta[1]), float(beta[1] + beta[2]), sse)
    knee, slo, shi, sse2 = best
    df2 = n - 4  # intercept + slope + hinge + knee (knee counted as a param)
    F = ((sse_lin - sse2) / 2) / (sse2 / max(df2, 1)) if sse2 > 0 else float("nan")
    return knee, slo, shi, F, sse_lin, sse2, float(b_lin[1]), r2_lin

def parse_lr(rn):
    m = re.search(r"_lr([0-9]+)e([0-9]+)", rn)
    if not m:
        return None
    return int(m.group(1)) * 10 ** (-int(m.group(2)))

def parse_method(rn, fam):
    body = rn[len(fam) + 1:]
    for name in ["lora_null", "lorawd", "cordapp", "corda", "clora", "sclora",
                 "milora", "pissa", "dora", "lora_l2", "lora"]:
        if body.startswith(name):
            return name
    return body.split("_")[0]

# ---------- load ----------
rows = []
for f in glob.glob(os.path.join(RES, "*", "summary.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rn = d.get("run_name") or os.path.basename(os.path.dirname(f))
    if re.search(r"smoke|corda", rn, re.I):
        continue
    h = d.get("headline") or {}
    fd, ret = h.get("fdelta"), h.get("retention_mean")
    if not isinstance(fd, (int, float)) or not isinstance(ret, (int, float)):
        continue
    if not (math.isfinite(fd) and math.isfinite(ret)) or fd <= 0:
        continue
    m = re.match(r"^([a-z0-9]+)_", rn)
    fam = m.group(1) if m else "other"
    if fam not in FAMS:
        continue
    sm = re.search(r"_s(4[2-9])$", rn)
    rows.append(dict(
        rn=rn, fam=fam, seed=sm.group(1) if sm else None,
        cell=re.sub(r"_s4[2-9]$", "", rn), fd=fd, lfd=math.log10(fd), ret=ret,
        broad=h.get("retention_broad"), adapt=h.get("cs_avg"),
        lr=parse_lr(rn), method=parse_method(rn, fam),
        dwmax=h.get("dw_sv_max"), dwmean=h.get("dw_sv_mean"),
        tasks={t: h.get(t) for t in RET_TASKS}))

geo, ce = {}, {}
for line in open("results/geo_drift/adapter_metrics_merged.jsonl"):
    try:
        g = json.loads(line); geo[g.get("run")] = g
    except Exception:
        pass
for line in open("results/forgetting_merged.jsonl"):
    try:
        c = json.loads(line); ce[c.get("run_name")] = c
    except Exception:
        pass
byrn = {r["rn"]: r for r in rows}
print(f"usable rows: {len(rows)}   geometry rows: {len(geo)}   CE rows: {len(ce)}")

def fam_rows(fam):
    return [r for r in rows if r["fam"] == fam]

# ---------- A1 functional form ----------
print("\n=== A1. FUNCTIONAL FORM: linear vs 2-segment (knee), robustness subsets ===")
print(f"{'family':16s} {'r_pool':>7s} {'rank':>6s} {'slope':>7s} {'knee@':>6s} "
      f"{'s_lo':>6s} {'s_hi':>7s} {'F2seg':>6s} {'r_ret>=15':>9s} {'r_dropQ4':>10s} "
      f"{'r_bothalf-med':>13s} {'r_botrng':>8s}")
norm_slopes = {}
for fam in FAMS:
    rs = fam_rows(fam)
    lx = np.array([r["lfd"] for r in rs]); y = np.array([r["ret"] for r in rs])
    rP, n = pearson(lx, y); rS, _ = spearman(lx, y)
    knee, slo, shi, F, sse1, sse2, slope, r2lin = seg2_fit(lx, y)
    m15 = y >= 15
    r15, n15 = pearson(lx[m15], y[m15])
    q75 = np.quantile(lx, 0.75)
    rb75, nb75 = pearson(lx[lx <= q75], y[lx <= q75])     # = "drop top F_D quartile"
    med = np.median(lx)
    rbh, nbh = pearson(lx[lx <= med], y[lx <= med])        # bottom half (median split)
    mid = (lx.min() + lx.max()) / 2
    rbr, _ = pearson(lx[lx <= mid], y[lx <= mid])          # bottom half of log-fd RANGE
    norm_slopes[fam] = slope / (y.max() - y.min()) if y.max() > y.min() else float("nan")
    print(f"{LABEL[fam]:16s} {rP:+7.3f} {rS:+6.2f} {slope:7.2f} {knee:6.2f} "
          f"{slo:6.1f} {shi:7.1f} {F:6.1f} {r15:+9.3f} {rb75:+10.3f} {rbh:+13.3f} {rbr:+8.3f}")
print("  slope units: pp retention per decade of F_Delta. normalized slopes "
      "(slope / family retention range): " +
      ", ".join(f"{fam}={norm_slopes[fam]:+.2f}" for fam in FAMS))

# ---------- A2 LR-proxy rescue ----------
print("\n=== A2. LR RESCUE: R2(LR-dummies) vs R2(log F_D); fixed-LR strata; partials ===")
for fam in FAMS:
    rs = [r for r in fam_rows(fam) if r["lr"]]
    if len(rs) < 12:
        continue
    lx = np.array([r["lfd"] for r in rs]); y = np.array([r["ret"] for r in rs])
    llr = np.array([math.log10(r["lr"]) for r in rs])
    D, lv = dummies([f"{r['lr']:.0e}" for r in rs])
    _, _, r2_fd, _, _ = ols(lx[:, None], y)
    _, _, r2_lrc, _, _ = ols(llr[:, None], y)
    _, _, r2_lrd, _, _ = ols(D, y)
    p_fd, n1, t1 = partial_r(lx, y, D)          # r(F_D, ret | LR dummies)
    p_lr, n2, t2 = partial_r(llr, y, lx)        # r(LR, ret | F_D)
    r_fl, _ = pearson(lx, llr)
    strata = []
    for lrv in sorted(set(r["lr"] for r in rs)):
        sub = [r for r in rs if r["lr"] == lrv]
        if len(sub) >= 6:
            rr, nn = pearson([r["lfd"] for r in sub], [r["ret"] for r in sub])
            strata.append(f"lr{lrv:g}:{rr:+.2f}(n={nn})")
    print(f"  {LABEL[fam]:16s} R2: fd={r2_fd:.3f} lr_cont={r2_lrc:.3f} lr_dumm={r2_lrd:.3f} "
          f"| partial r(fd|LR)={p_fd:+.2f}(t={t1:.1f}) r(LR|fd)={p_lr:+.2f}(t={t2:.1f}) "
          f"| r(lfd,llr)={r_fl:+.2f}")
    if strata:
        print(f"      fixed-LR strata: {'  '.join(strata)}")

# ---------- A3 direction second-order ----------
print("\n=== A3. DIRECTION AS SECOND-ORDER ===")
for smetric in ("spec_max", "spec_mean"):
    # adapter level
    ax, ay, az = [], [], []
    for r in rows:
        g = geo.get(r["rn"])
        v = g.get(smetric) if g else None
        if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            ax.append(math.log10(v)); ay.append(r["ret"]); az.append(r["lfd"])
    pr_a, pn_a, pt_a = partial_r(ax, ay, np.array(az))
    # cell level (seed-averaged)
    cells = defaultdict(list)
    for r in rows:
        g = geo.get(r["rn"])
        v = g.get(smetric) if g else None
        if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            cells[(r["fam"], r["cell"])].append((r, v))
    cx, cy, cz = [], [], []
    for (fam, c), rvs in cells.items():
        cx.append(np.mean([math.log10(v) for _, v in rvs]))
        cy.append(np.mean([r["ret"] for r, _ in rvs]))
        cz.append(np.mean([r["lfd"] for r, _ in rvs]))
    pr_c, pn_c, pt_c = partial_r(cx, cy, np.array(cz))
    print(f"  partial r(log {smetric}, ret | log F_D): adapter-level {pr_a:+.3f} "
          f"(n={pn_a}, t={pt_a:.1f})  cell-level {pr_c:+.3f} (n={pn_c}, t={pt_c:.1f})")
print("  per-family method offsets at matched F_D (OLS ret ~ log F_D + method dummies; ref level = first alphabetically):")
for fam in FAMS:
    rs = [r for r in fam_rows(fam) if r["method"] not in ("cordapp", "corda")]
    if len(set(r["method"] for r in rs)) < 3:
        continue
    lx = np.array([r["lfd"] for r in rs]); y = np.array([r["ret"] for r in rs])
    D, lv = dummies([r["method"] for r in rs])
    X = np.column_stack([lx, D])
    beta, se, r2, n, _ = ols(X, y)
    offs = []
    for j, mname in enumerate(lv[1:]):
        b, s = beta[2 + j], se[2 + j]
        star = "*" if abs(b) > 1.96 * s else " "
        offs.append(f"{mname}:{b:+.1f}±{s:.1f}{star}")
    print(f"    {LABEL[fam]:16s} (ref={lv[0]}, n={n}, R2={r2:.2f}) " + "  ".join(offs))

# ---------- A4 adaptation-efficiency ANCOVA ----------
print("\n=== A4. ADAPTATION-EFFICIENCY ANCOVA (adapt ~ log F_D + method) vs (ret ~ log F_D + method) ===")
for fam in FAMS:
    rs = [r for r in fam_rows(fam)
          if isinstance(r["adapt"], (int, float)) and math.isfinite(r["adapt"])]
    if len(set(r["method"] for r in rs)) < 3 or len(rs) < 20:
        continue
    lx = np.array([r["lfd"] for r in rs])
    D, lv = dummies([r["method"] for r in rs])
    X = np.column_stack([lx, D])
    for tag, yv in (("adapt", np.array([r["adapt"] for r in rs])),
                    ("ret  ", np.array([r["ret"] for r in rs]))):
        _, _, r2_base, _, _ = ols(lx[:, None], yv)
        beta, se, r2_full, n, _ = ols(X, yv)
        spread = max(beta[2:]) - min(beta[2:]) if len(beta) > 2 else 0
        print(f"  {LABEL[fam]:16s} {tag}: R2 {r2_base:.2f} -> +method {r2_full:.2f} "
              f"(method-offset spread {spread:.1f} pp, n={n})")

# ---------- A5 within-cell micro-test ----------
print("\n=== A5. WITHIN-CELL MICRO-TEST (seed-level fluctuations at fixed recipe) ===")
for xkey, tag in (("lfd", "log10 F_D"), ("fd", "raw F_D")):
    dx, dy = [], []
    ncell = 0
    for (fam, c), rs in [(k, v) for k, v in
                         ((k2, [r for r in rows if (r["fam"], r["cell"]) == k2])
                          for k2 in {(r["fam"], r["cell"]) for r in rows if r["seed"]})]:
        rs = [r for r in rs if r["seed"]]
        if len(rs) < 3:
            continue
        mx = np.mean([r[xkey] for r in rs]); my = np.mean([r["ret"] for r in rs])
        for r in rs:
            dx.append(r[xkey] - mx); dy.append(r["ret"] - my)
        ncell += 1
    rr, nn = pearson(dx, dy)
    t = rr * math.sqrt((nn - 2) / max(1e-12, 1 - rr * rr))
    print(f"  demeaned {tag:10s}: pooled r={rr:+.3f}  n={nn} obs / {ncell} cells  t={t:.1f}")

# ---------- A6 retention_broad without ARC-c ----------
print("\n=== A6. BROAD RETENTION WITHOUT ARC-c (adapt-suite contamination control) ===")
for fam in FAMS:
    rs = [r for r in fam_rows(fam)
          if isinstance(r["broad"], (int, float)) and math.isfinite(r["broad"])
          and all(isinstance(r["tasks"][t], (int, float)) for t in RET_TASKS)]
    if len(rs) < 10:
        continue
    lx = [r["lfd"] for r in rs]
    rb, _ = pearson(lx, [r["broad"] for r in rs])
    noarc = [np.mean([r["tasks"][t] for t in RET_TASKS if t != "arc_c"]) for r in rs]
    rn_, nn = pearson(lx, noarc)
    print(f"  {LABEL[fam]:16s} r(broad)={rb:+.3f}  r(broad w/o ARC-c)={rn_:+.3f}  n={nn}")

# ---------- A7 format-collapse control (proxy) ----------
print("\n=== A7. FORMAT-COLLAPSE CONTROL (proxy: per-item parse rates not retained) ===")
print("  degenerate := adapt < 25 (below-random adapt) OR any retention task == 0.0")
for fam in FAMS:
    rs = fam_rows(fam)
    def degen(r):
        bad_adapt = isinstance(r["adapt"], (int, float)) and r["adapt"] < 25
        zero_task = any(r["tasks"][t] == 0.0 for t in RET_TASKS
                        if isinstance(r["tasks"][t], (int, float)))
        return bad_adapt or zero_task
    clean = [r for r in rs if not degen(r)]
    r_all, n_all = pearson([r["lfd"] for r in rs], [r["ret"] for r in rs])
    r_cl, n_cl = pearson([r["lfd"] for r in clean], [r["ret"] for r in clean])
    print(f"  {LABEL[fam]:16s} all r={r_all:+.3f}(n={n_all})  clean r={r_cl:+.3f}(n={n_cl})  "
          f"degenerate excluded: {n_all - n_cl}")

# ---------- A8 CE within-family + Qwen coverage ----------
print("\n=== A8. CE CORROBORATION, WITHIN FAMILY (mechanical r(F_D,CE) vs evidential r(CE,ret)) ===")
missing_qwen = []
for fam in FAMS:
    rs = fam_rows(fam)
    have = [r for r in rs if r["rn"] in ce
            and isinstance(ce[r["rn"]].get("forgetting_ce"), (int, float))]
    if fam.startswith("qw"):
        missing_qwen += [r["rn"] for r in rs if r["rn"] not in ce]
    if len(have) < 10:
        print(f"  {LABEL[fam]:16s} CE coverage {len(have)}/{len(rs)} — too sparse")
        continue
    cev = [ce[r["rn"]]["forgetting_ce"] for r in have]
    r1, _ = pearson([r["lfd"] for r in have], cev)
    r2_, _ = pearson(cev, [r["ret"] for r in have])
    print(f"  {LABEL[fam]:16s} coverage {len(have)}/{len(rs)}  r(logF_D,CE)={r1:+.3f}  r(CE,ret)={r2_:+.3f}")
if missing_qwen:
    out = "jobs/ce_backfill_qwen.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(sorted(missing_qwen)) + "\n")
    print(f"  -> {len(missing_qwen)} Qwen runs lack CE; run_names written to {out}")

# ---------- A9 FDelta decomposition ----------
print("\n=== A9. F_D DECOMPOSITION: alignment := F_D / ||dW||_F (adapt-distribution weighting) ===")
al_by_m = defaultdict(list)
pairs_fro, pairs_fd, pairs_dw = [], [], []
for r in rows:
    g = geo.get(r["rn"])
    if not g or not isinstance(g.get("fro_total"), (int, float)) or g["fro_total"] <= 0:
        continue
    pairs_fro.append((math.log10(g["fro_total"]), r["ret"]))
    pairs_fd.append((r["lfd"], r["ret"]))
    al_by_m[r["method"]].append(r["fd"] / g["fro_total"])
    if isinstance(r["dwmax"], (int, float)) and r["dwmax"] > 0:
        pairs_dw.append((math.log10(r["dwmax"]), r["ret"]))
r_fro, n_fro = pearson(*zip(*pairs_fro)); r_fd, _ = pearson(*zip(*pairs_fd))
r_dw, n_dw = pearson(*zip(*pairs_dw))
print(f"  pooled (n={n_fro}): R2(ret~log||dW||_F)={r_fro**2:.2f}  R2(ret~logF_D)={r_fd**2:.2f}  "
      f"R2(ret~log dw_sv_max)={r_dw**2:.2f} (n={n_dw})")
print("  alignment by method (mean±sd, x1e3):")
for mname in sorted(al_by_m, key=lambda m: -len(al_by_m[m])):
    v = np.array(al_by_m[mname]) * 1e3
    if len(v) >= 5:
        print(f"    {mname:10s} {v.mean():7.2f} ± {v.std():5.2f}  (n={len(v)})")
print("\ndone.")
