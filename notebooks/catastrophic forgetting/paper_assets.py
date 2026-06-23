"""
Reproducible paper assets (experiments section). Reads results/campaign_summary.jsonl
(the LIVE registry) + optional forensics_*.json (leakage) and regenerates, for each
domain, the main results table (LaTeX + text) and the figures. Re-run anytime: new data
points flow straight in. Implements the agent's blueprint (Table 2, Fig1 Pareto, Fig2
magnitude law, correlations; leakage Table4/Fig3 auto-activate once forensics lands).

  python paper_assets.py            # build everything from current data
Outputs -> paper/{table_main_<domain>.tex, table_main_<domain>.txt, fig1_pareto_<domain>.png,
                  fig2_magnitude.png, fig3_leakage.png(if forensics), summary.txt}
"""
import os, json, glob, math
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "paper")
os.makedirs(OUT, exist_ok=True)

SIMPLE = {"lora", "lorawd"}                       # the simple baselines (starred)
METHODS = ["lora", "lorawd", "clora", "dora", "corda", "milora", "sclora"]
PRETTY = {"lora": "LoRA", "lorawd": "LoRA+wd", "clora": "CLoRA", "dora": "DoRA",
          "corda": "CorDA", "milora": "MiLoRA", "sclora": "SC-LoRA"}
BASE_CORE = 26.0      # no-FT retention ceiling (BBH-AO 33.1 + MMLU-Pro 19.0)/2
ADAPT_FLOOR = {"cs": 70.0, "math": 5.0}           # min adaptation to count as a fair representative


def classify(run):
    domain = "math" if any(p in run for p in ("mtxm_", "lrswm_", "scl2m_")) else "cs"
    if run.startswith(("scl2_", "scl2m_")):
        method = "sclora"
    else:
        parts = run.split("_")
        method = parts[1] if len(parts) > 1 else parts[0]
    return domain, method


def config_label(run):
    """run minus the leading namespace and trailing _s<seed>, e.g. mtx_lorawd_wd0p3_s42 -> lorawd_wd0p3."""
    r = run
    for p in ("mtxm_", "mtx_", "lrswm_", "lrsw_", "scl2m_", "scl2_"):
        if r.startswith(p):
            r = r[len(p):]; break
    parts = r.split("_")
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def g(d, k):
    v = d.get(k)
    return v if isinstance(v, (int, float)) else None


def load():
    seen = {}
    for line in open(os.path.join(RES, "campaign_summary.jsonl")):
        try:
            d = json.loads(line)
        except Exception:
            continue
        rn = d.get("run_name", "")
        if not any(rn.startswith(p) for p in ("mtx_", "mtxm_", "lrsw_", "lrswm_", "scl2_", "scl2m_")):
            continue
        if rn.startswith(("mtx_sclora", "mtxm_sclora")):      # deprecated buggy SC-LoRA
            continue
        seen[rn] = d
    rows = []
    for rn, d in seen.items():
        dom, method = classify(rn)
        if method not in METHODS:
            continue
        rows.append(dict(run=rn, domain=dom, method=method, cfg=config_label(rn),
                         adapt=g(d, "cs_avg"), ret=g(d, "retention_mean"), broad=g(d, "retention_broad"),
                         F=g(d, "fdelta"), svmax=g(d, "dw_sv_max"),
                         bbh=g(d, "bbh"), mmlu_pro=g(d, "mmlu_pro"), mmlu=g(d, "mmlu"),
                         arc=g(d, "arc_c"), tqa=g(d, "truthfulqa")))
    return [r for r in rows if isinstance(r["ret"], (int, float)) and isinstance(r["F"], (int, float))]


def agg_cfg(rows):
    """group by (domain, method, cfg) -> mean/std over seeds."""
    grp = {}
    for r in rows:
        grp.setdefault((r["domain"], r["method"], r["cfg"]), []).append(r)
    cells = []
    for (dom, method, cfg), vs in grp.items():
        def ms(k):
            xs = [v[k] for v in vs if isinstance(v.get(k), (int, float))]
            return (round(st.mean(xs), 2), round(st.pstdev(xs), 2) if len(xs) > 1 else 0.0) if xs else (None, None)
        c = dict(domain=dom, method=method, cfg=cfg, n=len(vs))
        for k in ("adapt", "ret", "broad", "F", "svmax"):
            c[k], c[k + "_sd"] = ms(k)
        cells.append(c)
    return cells


def representative(cells, domain):
    """per method: highest core-retention config with adapt >= floor (high-retention corner)."""
    floor = ADAPT_FLOOR[domain]
    reps = {}
    for c in cells:
        if c["domain"] != domain or c["ret"] is None:
            continue
        ok = (c["adapt"] is not None and c["adapt"] >= floor)
        key = c["method"]
        cur = reps.get(key)
        # prefer adapt>=floor; among those max retention; else best retention overall as fallback
        score = (1 if ok else 0, c["ret"])
        if cur is None or score > cur[0]:
            reps[key] = (score, c)
    return {m: v[1] for m, v in reps.items()}


# ---------- TABLES ----------
def fmt(v, sd=None, p=1):
    if v is None:
        return "--"
    return f"{v:.{p}f}" + (f"$\\pm${sd:.1f}" if sd else "")


def build_table(cells, domain):
    reps = representative(cells, domain)
    order = [m for m in METHODS if m in reps]
    adapt_name = "GSM8K" if domain == "math" else "CS-8"
    # LaTeX
    L = [r"\begin{tabular}{l l c c c c c}", r"\toprule",
         f"Method & Config & {adapt_name} $\\uparrow$ & Ret-core $\\uparrow$ & Ret-broad $\\uparrow$ & $\\|\\Delta W\\|_F$ $\\downarrow$ & $\\sigma_{{\\max}}$ \\\\",
         r"\midrule",
         f"\\textit{{Base (no-FT)}} & -- & -- & \\textit{{{BASE_CORE:.1f}}} & -- & \\textit{{0}} & -- \\\\",
         r"\midrule"]
    txt = [f"=== MAIN TABLE [{domain}] (rep = max core-retention at {adapt_name}>={ADAPT_FLOOR[domain]:.0f}) ===",
           f'{"method":9s} {"config":14s} {adapt_name:>10s} {"ret_core":>11s} {"ret_broad":>9s} {"||dW||F":>8s} {"svmax":>7s} {"n":>2s}',
           f'{"BASE":9s} {"(no-FT)":14s} {"--":>10s} {BASE_CORE:>11.1f} {"--":>9s} {"0":>8s} {"--":>7s}']
    for m in order:
        c = reps[m]
        star = r"$\star$" if m in SIMPLE else ""
        nm = PRETTY[m] + star
        L.append(f"{nm} & {c['cfg'].replace('_',' ')} & {fmt(c['adapt'],c['adapt_sd'])} & "
                 f"{fmt(c['ret'],c['ret_sd'])} & {fmt(c['broad'])} & {fmt(c['F'],None,3)} & {fmt(c['svmax'],None,1)} \\\\")
        mk = "*" if m in SIMPLE else " "
        txt.append(f'{mk}{PRETTY[m]:8s} {c["cfg"][:14]:14s} '
                   f'{fmt(c["adapt"],c["adapt_sd"]):>10s} {fmt(c["ret"],c["ret_sd"]):>11s} '
                   f'{fmt(c["broad"]):>9s} {(("%.3f"%c["F"]) if c["F"] else "--"):>8s} '
                   f'{(("%.1f"%c["svmax"]) if c["svmax"] else "--"):>7s} {c["n"]:>2d}')
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, f"table_main_{domain}.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(OUT, f"table_main_{domain}.txt"), "w").write("\n".join(txt) + "\n")
    return "\n".join(txt)


# ---------- FIGURES ----------
def figures(cells, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"lora": "#1f77b4", "lorawd": "#d62728", "clora": "#2ca02c", "dora": "#9467bd",
           "corda": "#8c564b", "milora": "#e377c2", "sclora": "#ff7f0e"}
    for domain in ("cs", "math"):
        dc = [c for c in cells if c["domain"] == domain and c["adapt"] is not None]
        if not dc:
            continue
        # Fig1 Pareto: adapt vs ret, per-method config means
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for m in METHODS:
            pts = sorted([c for c in dc if c["method"] == m], key=lambda c: c["F"] or 0)
            if not pts:
                continue
            xs = [c["adapt"] for c in pts]; ys = [c["ret"] for c in pts]
            simple = m in SIMPLE
            ax.plot(xs, ys, "-o" if simple else "--s", color=COL[m], lw=2.4 if simple else 1.3,
                    ms=8 if simple else 5, alpha=0.95 if simple else 0.7,
                    label=PRETTY[m] + (" ★" if simple else ""), zorder=5 if simple else 3)
        ax.axhline(BASE_CORE, ls=":", color="gray", lw=1); ax.text(ax.get_xlim()[0], BASE_CORE + 0.1, "base ceiling", color="gray", fontsize=8)
        ax.set_xlabel(("GSM8K" if domain == "math" else "CS-8") + " adaptation acc. (%) →")
        ax.set_ylabel("retention (BBH+MMLU-Pro) →")
        ax.set_title(f"Adaptation–Retention frontier ({domain})")
        ax.legend(fontsize=7.5, ncol=2, loc="best"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig1_pareto_{domain}.png"), dpi=140); plt.close(fig)

    # Fig2 magnitude law: retention vs ||dW||_F (cs)
    dc = [c for c in cells if c["domain"] == "cs"]
    if dc:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for m in METHODS:
            pts = [c for c in dc if c["method"] == m and c["F"] is not None]
            if not pts:
                continue
            ax.scatter([c["F"] for c in pts], [c["ret"] for c in pts], color=COL[m],
                       s=70 if m in SIMPLE else 40, marker="o" if m in SIMPLE else "s",
                       edgecolor="k" if m in SIMPLE else "none", linewidth=0.6,
                       label=PRETTY[m], alpha=0.9, zorder=5 if m in SIMPLE else 3)
        # pooled correlation
        F = [c["F"] for c in dc]; R = [c["ret"] for c in dc]
        if len(F) > 3:
            r = _pearson(F, R)
            ax.set_title(f"Magnitude law: retention vs $\\|\\Delta W\\|_F$  (pooled r={r:.2f})")
        ax.axhline(BASE_CORE, ls=":", color="gray", lw=1)
        ax.set_xlabel("$\\|\\Delta W\\|_F$ →"); ax.set_ylabel("retention (BBH+MMLU-Pro) →")
        ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_magnitude.png"), dpi=140); plt.close(fig)


def _pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy + 1e-12)


def correlations(cells):
    lines = ["=== retention ~ ||dW||_F correlations ==="]
    for domain in ("cs", "math"):
        dc = [c for c in cells if c["domain"] == domain and c["F"] is not None]
        if len(dc) < 3:
            continue
        rp = _pearson([c["F"] for c in dc], [c["ret"] for c in dc])
        lines.append(f"[{domain}] POOLED r(ret, ||dW||_F) = {rp:.3f}  (n={len(dc)} configs)")
        for m in METHODS:
            pts = [c for c in dc if c["method"] == m]
            if len(pts) >= 3:
                lines.append(f"   {PRETTY[m]:9s} within-method r = {_pearson([c['F'] for c in pts],[c['ret'] for c in pts]):.3f} (n={len(pts)})")
    return "\n".join(lines)


def _parse_lr(cfg):
    """cfg like 'corda_r16_lr2e5' -> 2e-5 (encoding XeY means Xe-Y)."""
    for tok in cfg.split("_"):
        if tok.startswith("lr"):
            m = tok[2:]
            if "e" in m:
                a, b = m.split("e")
                try:
                    return float(f"{a}e-{b}")
                except Exception:
                    return None
    return None


def lr_sensitivity(rows):
    """THE LR-FAIRNESS EXHIBIT. From the per-method LR sweep (lrsw_/lrswm_ runs), show that
    each adapter's optimal LR differs -> fixing one LR (as prior work does) biases the comparison.
    Produces: (1) best-LR-per-method table (each adapter at its OWN adaptation-optimal LR -> the
    defensible 'best result' row), (2) full adapt-vs-LR sweep, (3) fig. Auto-fills once Phase 2 lands."""
    sw = {}   # (domain, method) -> [(lr, adapt, ret, F)]
    for r in rows:
        if not (r["run"].startswith("lrsw_") or r["run"].startswith("lrswm_")):
            continue
        lr = _parse_lr(r["cfg"])
        if lr is None or r["adapt"] is None:
            continue
        sw.setdefault((r["domain"], r["method"]), []).append((lr, r["adapt"], r["ret"], r["F"]))
    if not sw:
        return "[LR-fairness] pending Phase-2 LR sweep (lrsw_*). Table+fig auto-build when it lands."
    lines = ["=== LR-FAIRNESS: each adapter at its OWN best LR (defensible best-result table) ==="]
    for domain in ("cs", "math"):
        ms = [(m, sw[(d, m)]) for (d, m) in sw if d == domain]
        if not ms:
            continue
        lines.append(f"-- {domain} -- (best LR chosen by max adaptation per method)")
        lines.append(f'  {"method":9s} {"bestLR":>8s} {"adapt":>6s} {"ret":>6s} {"||dW||F":>8s}   sweep(LR:adapt)')
        best_lrs = []
        for m in METHODS:
            pts = next((v for mm, v in ms if mm == m), None)
            if not pts:
                continue
            pts = sorted(pts)
            lr_b, ad_b, rt_b, f_b = max(pts, key=lambda t: (t[1] if t[1] is not None else -1))  # max adaptation
            best_lrs.append(lr_b)
            sweepstr = " ".join(f"{lr:g}:{(ad if ad is not None else 0):.0f}" for lr, ad, _, _ in pts)
            lines.append(f'  {PRETTY[m]:9s} {lr_b:>8g} {ad_b or 0:6.1f} {rt_b or 0:6.1f} {(f_b or 0):8.3f}   {sweepstr}')
        if len(set(best_lrs)) > 1:
            lines.append(f"  => best LR is NOT the same across methods ({sorted(set(best_lrs))}) "
                         f"-> a single fixed LR biases the comparison (the thesis).")
    # figure: adaptation vs LR per method (cs)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        COL = {"lora": "#1f77b4", "lorawd": "#d62728", "clora": "#2ca02c", "dora": "#9467bd",
               "corda": "#8c564b", "milora": "#e377c2", "sclora": "#ff7f0e"}
        ms = [(m, sw[("cs", m)]) for (d, m) in sw if d == "cs"]
        if ms:
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            for m, pts in ms:
                pts = sorted(pts)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=COL.get(m, "k"),
                        lw=2.2 if m in SIMPLE else 1.3, label=PRETTY[m])
            ax.set_xscale("log"); ax.set_xlabel("learning rate (log) →"); ax.set_ylabel("adaptation acc (%)")
            ax.set_title("Per-method LR sensitivity (optimum differs → fixed-LR comparison is biased)")
            ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.25, which="both")
            fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_lr_sensitivity.png"), dpi=140); plt.close(fig)
            lines.append("  [figures] wrote paper/fig4_lr_sensitivity.png")
    except Exception as e:
        lines.append(f"  [fig4] skipped: {e}")
    return "\n".join(lines)


def leakage_analysis(rows):
    """Geometry/leakage: join forensics (weight-basis out_top = fraction of dW energy in W0's
    top singular subspace) + databasis (data_resp). Tests whether 'leakage into important
    directions' predicts retention BEYOND magnitude. Auto-activates once forensics_*.json exist."""
    fro, dat = {}, {}
    for f in glob.glob(os.path.join(RES, "forensics_*.json")):
        try:
            d = json.load(open(f)); fro[d.get("run_name", "")] = d
        except Exception:
            pass
    for f in glob.glob(os.path.join(RES, "databasis_*.json")):
        try:
            d = json.load(open(f)); dat[d.get("run_name", "")] = d
        except Exception:
            pass
    if not fro and not dat:
        return "[leakage] pending forensics pass (run_forensics.sh) — paper/fig3 will auto-build then."
    # build (method, F, ret, Lweight, Ldata) per run that has forensics
    pts = []
    for r in rows:
        rn = r["run"]
        fw = fro.get(rn, {}); fd = dat.get(rn, {})
        # weight-basis leakage: energy fraction in top output singular subspace (out_top_*), pick mid-k
        lw = next((fw[k] for k in fw if k.startswith("out_top_")), None)
        ld = fd.get("d_inTop") if isinstance(fd.get("d_inTop"), (int, float)) else None
        if r["F"] is not None and r["ret"] is not None and (lw is not None or ld is not None):
            pts.append(dict(method=r["method"], F=r["F"], ret=r["ret"], lw=lw, ld=ld))
    if not pts:
        return "[leakage] forensics files present but no joinable runs yet."
    lines = [f"=== LEAKAGE / GEOMETRY ({len(pts)} runs with forensics) ==="]
    # partial correlation retention ~ leakage controlling for ||dW||_F (residualize both on F)
    for name, key in (("weight-basis L_weight", "lw"), ("data-basis L_data", "ld")):
        sub = [p for p in pts if p[key] is not None]
        if len(sub) < 4:
            continue
        F = [p["F"] for p in sub]; R = [p["ret"] for p in sub]; L = [p[key] for p in sub]
        rRL = _pearson(R, L); rRF = _pearson(R, F)
        # residualize R and L on F, correlate residuals = partial corr
        def resid(y):
            b = _slope(F, y); a = (sum(y) - b * sum(F)) / len(y)
            return [yi - (a + b * fi) for yi, fi in zip(y, F)]
        rp = _pearson(resid(R), resid(L))
        lines.append(f"  {name}: r(ret,L)={rRL:.2f}  r(ret,||dW||F)={rRF:.2f}  "
                     f"PARTIAL r(ret,L | ||dW||F)={rp:.2f}  (n={len(sub)})")
    lines.append("  -> partial r near 0 = geometry adds nothing beyond magnitude (the thesis).")
    # Fig3: retention residual (after removing ||dW||_F) vs leakage
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sub = [p for p in pts if p["lw"] is not None]
        if len(sub) >= 4:
            F = [p["F"] for p in sub]; R = [p["ret"] for p in sub]
            b = _slope(F, R); a = (sum(R) - b * sum(F)) / len(R)
            res = [ri - (a + b * fi) for ri, fi in zip(R, F)]
            fig, ax = plt.subplots(figsize=(6.0, 4.4))
            COL = {"lora": "#1f77b4", "lorawd": "#d62728", "clora": "#2ca02c", "dora": "#9467bd",
                   "corda": "#8c564b", "milora": "#e377c2", "sclora": "#ff7f0e"}
            for m in METHODS:
                xs = [p["lw"] for p in sub if p["method"] == m]; ys = [r for p, r in zip(sub, res) if p["method"] == m]
                if xs:
                    ax.scatter(xs, ys, color=COL.get(m, "k"), s=55 if m in SIMPLE else 38,
                               marker="o" if m in SIMPLE else "s", label=PRETTY[m], alpha=0.85)
            ax.axhline(0, ls=":", color="gray")
            ax.set_xlabel("weight-basis leakage  $L_{weight}$ (energy in top-$k$ subspace) →")
            ax.set_ylabel("retention residual (after removing $\\|\\Delta W\\|_F$)")
            ax.set_title("Geometry adds nothing beyond magnitude (flat = thesis)")
            ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.25)
            fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_leakage.png"), dpi=140); plt.close(fig)
            lines.append("  [figures] wrote paper/fig3_leakage.png")
    except Exception as e:
        lines.append(f"  [fig3] skipped: {e}")
    return "\n".join(lines)


def _slope(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y)); den = sum((a - mx) ** 2 for a in x)
    return num / (den + 1e-12)


if __name__ == "__main__":
    rows = load()
    cells = agg_cfg(rows)
    out = [f"PAPER ASSETS — {len(rows)} runs, {len(cells)} configs"]
    for domain in ("cs", "math"):
        if any(c["domain"] == domain for c in cells):
            out.append("\n" + build_table(cells, domain))
    out.append("\n" + correlations(cells))
    out.append("\n" + lr_sensitivity(rows))
    out.append("\n" + leakage_analysis(rows))
    try:
        figures(cells, rows)
        out.append("\n[figures] wrote paper/fig1_pareto_*.png, paper/fig2_magnitude.png")
    except Exception as e:
        out.append(f"\n[figures] FAILED: {e}")
    report = "\n".join(out)
    open(os.path.join(OUT, "summary.txt"), "w").write(report + "\n")
    print(report)
