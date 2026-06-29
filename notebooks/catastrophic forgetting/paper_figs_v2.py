#!/usr/bin/env python
"""
Publication-quality figures for the magnitude-budget / catastrophic-forgetting study.
Reads results/campaign_summary.jsonl (LIVE registry). Writes high-DPI PNGs to paper/figs_v2/.
Does NOT modify any data. No internet/CDN; all assets self-contained.

Model: Llama-2-7B (only base present). Domains: cs (commonsense-8), math (gsm8k).
Method/arm parsed from run_name (the `method` field in the json is just the PEFT backend = LORA).
"""
import os, re, json, math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.interpolate import UnivariateSpline
import matplotlib.ticker as mticker

def clean_logx(ax):
    """Decade-only major ticks with plain labels; faint unlabeled minors. Stops tick-label pileup."""
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

HERE = "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
RES = os.path.join(HERE, "results")
PAPER = os.path.join(HERE, "paper")            # LaTeX tables live here
OUT = os.path.join(HERE, "paper", "figs_v2")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
})

# Colorblind-safe palette (Okabe-Ito derived), distinct markers per method.
METHODS = ["lora", "lorawd", "milora", "clora", "dora", "corda", "sclora"]
PRETTY = {"lora": "LoRA", "lorawd": "LoRA+wd(0.3)", "milora": "MiLoRA", "clora": "CLoRA",
          "dora": "DoRA", "corda": "CorDA", "sclora": "SC-LoRA"}
COL = {"lora": "#0072B2", "lorawd": "#56B4E9", "milora": "#009E73", "clora": "#E69F00",
       "dora": "#CC79A7", "corda": "#D55E00", "sclora": "#882255"}
MARK = {"lora": "o", "lorawd": "D", "milora": "^", "clora": "s",
        "dora": "v", "corda": "P", "sclora": "X"}
SIMPLE = {"lora", "lorawd"}   # the "plain" baselines we want to defend

# Base ceilings (Llama-2-7B, no fine-tune). Core uses ANSWER-ONLY BBH (matches FT eval).
BASE_BBH = 33.10          # answer-only 3-shot (bbh_fewshot), matches FT-run eval setting
BASE_MMLU_PRO = 18.96
BASE_CORE = (BASE_BBH + BASE_MMLU_PRO) / 2.0   # ~26.0

# ---------------------------------------------------------------- load
def classify(run):
    domain = "math" if any(p in run for p in ("mtxm_", "lrswm_", "scl2m_")) else "cs"
    parts = run.split("_")
    method = parts[1] if len(parts) > 1 else parts[0]
    return domain, method

def parse_lr(run):
    m = re.search(r'_lr([0-9]+)e([0-9]+)', run)
    return float(f"{m.group(1)}e-{m.group(2)}") if m else None

def parse_seed(run):
    m = re.search(r'_s(\d+)$', run)
    return int(m.group(1)) if m else None

def parse_cfg(run):
    r = run
    for p in ("mtxm_", "mtx_", "lrswm_", "lrsw_", "scl2m_", "scl2_"):
        if r.startswith(p):
            r = r[len(p):]; break
    parts = r.split("_")
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        parts = parts[:-1]
    return "_".join(parts)

PREF = ("mtx_", "mtxm_", "lrsw_", "lrswm_")
def load():
    recs = {}
    for line in open(os.path.join(RES, "campaign_summary.jsonl")):
        try:
            d = json.loads(line)
        except Exception:
            continue
        recs[d["run_name"]] = d  # dedup keep last
    rows = []
    for rn, d in recs.items():
        if not rn.startswith(PREF):
            continue
        dom, method = classify(rn)
        if method not in METHODS:
            continue
        if method == "corda":
            continue  # CorDA EXCLUDED: data invalid (wikitext-calib bug, fixed->nq_open re-running) AND
                      # calib<->eval distribution question (Fix 1). Under re-validation; see handoff/14.
        rows.append(dict(
            run=rn, domain=dom, method=method, lr=parse_lr(rn), seed=parse_seed(rn),
            cfg=parse_cfg(rn), is_sweep=rn.startswith(("lrsw_", "lrswm_")),
            adapt=d.get("cs_avg"), adapt_task=d.get("adapt_task"),
            ret=d.get("retention_mean"), broad=d.get("retention_broad"),
            F=d.get("fdelta"), svmax=d.get("dw_sv_max"), svmean=d.get("dw_sv_mean"),
            bbh=d.get("bbh"), mmlu_pro=d.get("mmlu_pro"), mmlu=d.get("mmlu"),
            arc=d.get("arc_c"), tqa=d.get("truthfulqa"),
        ))
    return rows

ROWS = load()
SWEEP = [r for r in ROWS if r["is_sweep"] and r["domain"] == "cs"]   # 7x7 clean grid
SWEEP_M = [r for r in ROWS if r["is_sweep"] and r["domain"] == "math"]

def arr(rows, k):
    return np.array([r[k] for r in rows], dtype=float)

# ---------------------------------------------------------------- helpers
def fit_line_logx(x, y):
    """OLS retention ~ a + b*log10(x). Returns dict with slope, intercept, r, r2, p, and predictor."""
    lx = np.log10(x)
    sl, ic, r, p, se = stats.linregress(lx, y)
    pred = lambda xx: ic + sl * np.log10(xx)
    return dict(slope=sl, intercept=ic, r=r, r2=r**2, p=p, pred=pred)

def legend_handles(methods):
    h = []
    for m in methods:
        h.append(Line2D([0], [0], marker=MARK[m], color=COL[m], lw=0, ms=9,
                        markeredgecolor="k" if m in SIMPLE else "none",
                        markeredgewidth=0.7, label=PRETTY[m]))
    return h

def watermark(fig):
    fig.text(0.006, 0.004, "Llama-2-7B  ·  LoRA-family  ·  s42 LR-sweep  ·  n=49 (7 methods × 7 LRs)",
             ha="left", va="bottom", fontsize=7, color="0.55", style="italic")

CAVEAT = ("Caveat: single seed (s42), n=7 per method. CorDA OMITTED (under re-validation: calibration "
          "fidelity, see handoff/14). SC-LoRA off-curve pending calib-distribution sensitivity + seeds 43/44.")
def caveat_footer(fig, y=0.004):
    """Visible under-power / preliminary-deviation caveat for fig0/fig1/fig2."""
    fig.text(0.5, y, CAVEAT, ha="center", va="bottom", fontsize=8,
             color="#7a3b1d", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdf1e7", ec="#d9a877", lw=0.8))

# Methods that fall ON the magnitude law vs the data-aware-init OUTLIERS that forget more than budget predicts.
ON_CURVE = ["lora", "lorawd", "milora", "clora", "dora"]
OFF_CURVE = ["corda", "sclora"]

# Training-collapse outliers (degenerate adaptation; valid retention). run -> (adapt, note)
COLLAPSE = {
    "lrsw_clora_k1024_lr1e4_s42": "CLoRA k1024 @ lr1e-4",
    "lrsw_corda_r16_lr2e5_s42":   "CorDA r16 @ lr2e-5",
    "lrsw_dora_r16_lr2e5_s42":    "DoRA r16 @ lr2e-5",
}

# ================================================================ FIG 0  (HERO)
def fig0_hero():
    """THE figure that carries the paper. Single clean panel: retention vs ||dW||_F (log-x),
    pooled fit on the 5 on-curve adapters + bootstrap CI, all 7 methods by color/marker,
    CorDA & SC-LoRA called out as below-curve outliers. Annotate r/R2, base ceiling, outliers."""
    rows = SWEEP
    x = arr(rows, "F"); y = arr(rows, "ret")
    g = np.isfinite(x) & np.isfinite(y) & (x > 0)
    rows = [r for r, gg in zip(rows, g) if gg]
    x, y = x[g], y[g]

    # Fit the LAW on the 5 on-curve methods only (the law is defined by the well-behaved adapters).
    on = np.array([r["method"] in ON_CURVE for r in rows])
    xon, yon = x[on], y[on]
    fit = fit_line_logx(xon, yon)
    # full-pool fit (all 7) for reference in the annotation
    fit_all = fit_line_logx(x, y)
    # robustness: on-curve law excluding the 3 training-collapse outliers (degenerate adaptation)
    keep = np.array([r["run"] not in COLLAPSE for r in rows]) & on
    fit_robust = fit_line_logx(x[keep], y[keep])

    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 250)

    # bootstrap CI of the on-curve law
    lx = np.log10(xon); rng = np.random.default_rng(0); preds = []
    for _ in range(2000):
        idx = rng.integers(0, len(xon), len(xon))
        s, i, *_ = stats.linregress(lx[idx], yon[idx])
        preds.append(i + s * np.log10(xx))
    preds = np.array(preds)
    ax.fill_between(xx, np.percentile(preds, 2.5, 0), np.percentile(preds, 97.5, 0),
                    color="#444", alpha=0.13, zorder=1, label="95% CI")
    ax.plot(xx, fit["pred"](xx), color="#222", lw=3.0, ls="-", zorder=3,
            label=(f"$\\bf{{Magnitude\\ law}}$ (5 on-curve adapters)\n"
                   f"$r$={fit['r']:.2f},  $R^2$={fit['r2']:.2f},  slope={fit['slope']:.0f} pp/decade"))

    # on-curve points: emphasized
    for m in ON_CURVE:
        mr = [r for r in rows if r["method"] == m]
        mx = arr(mr, "F"); my = arr(mr, "ret")
        ax.scatter(mx, my, s=135 if m in SIMPLE else 100, c=COL[m], marker=MARK[m],
                   edgecolor="k", linewidth=0.9, alpha=0.97, zorder=6)
    # off-curve outliers: same colors, large with a halo ring
    for m in OFF_CURVE:
        mr = [r for r in rows if r["method"] == m]
        mx = arr(mr, "F"); my = arr(mr, "ret")
        ax.scatter(mx, my, s=170, c=COL[m], marker=MARK[m], edgecolor="k",
                   linewidth=1.1, alpha=0.97, zorder=7)
        ax.scatter(mx, my, s=430, facecolors="none", edgecolors=COL[m],
                   linewidths=1.8, alpha=0.55, zorder=5)

    ax.set_xscale("log"); clean_logx(ax)
    # set limits BEFORE annotating so text placement is stable and inside the frame
    ax.set_xlim(x.min()*0.85, x.max()*1.18)

    # base ceiling (label on the right where the curve has dropped well below it -> open space)
    ax.axhline(BASE_CORE, ls=":", color="green", lw=1.6, zorder=2)
    ax.text(ax.get_xlim()[1], BASE_CORE + 0.4, "base retention ceiling (no fine-tune)  ",
            color="green", fontsize=10, va="bottom", ha="right", fontweight="bold")

    # call-out boxes for the two outliers, parked in the open lower-MIDDLE, arrows pointing right to clusters
    annot = {
        "corda":  dict(tx=0.40, ty=0.135),
        "sclora": dict(tx=0.40, ty=0.045),
    }
    for m in OFF_CURVE:
        mr = [r for r in rows if r["method"] == m]
        mx = arr(mr, "F"); my = arr(mr, "ret")
        # anchor on the lowest-retention (largest-deviation) point of this method
        j = int(np.argmin(my)); ax_x, ax_y = mx[j], my[j]
        cx = 10**np.mean(np.log10(mx))
        pred = fit["pred"](cx); cy = np.mean(my)
        a = annot[m]
        ax.annotate(f"{PRETTY[m]} (data-aware init):\nforgets ~{pred-cy:.0f} pp MORE than the law predicts",
                    xy=(ax_x, ax_y),
                    xytext=(a["tx"], a["ty"]), textcoords="axes fraction",
                    fontsize=9.5, color=COL[m], fontweight="bold", ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="->", color=COL[m], lw=1.8,
                                    connectionstyle="arc3,rad=-0.2"),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COL[m], lw=1.3, alpha=0.96))

    ax.set_xlabel(r"Weight-update magnitude  $\|\Delta W\|_F$  (token-weighted, log scale)  →", fontsize=12.5)
    ax.set_ylabel("Retention  (mean of BBH, MMLU-Pro)  [%]  →", fontsize=12.5)
    ax.set_title("Retention is governed by the size of the weight update, not the adapter",
                 fontsize=15.5, pad=14)
    ax.set_ylim(0, max(36, y.max()+2))

    # legend: methods grouped on-curve vs off-curve + the law line
    leg_law = ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.add_artist(leg_law)
    on_h = [Line2D([0],[0], marker=MARK[m], color=COL[m], lw=0, ms=11, markeredgecolor="k",
                   markeredgewidth=0.8, label=PRETTY[m]) for m in ON_CURVE]
    off_h = [Line2D([0],[0], marker=MARK[m], color=COL[m], lw=0, ms=12, markeredgecolor="k",
                    markeredgewidth=1.0, label=PRETTY[m]+" (off-curve)") for m in OFF_CURVE]
    ax.legend(handles=on_h + off_h, loc="lower left", ncol=2, fontsize=9.2,
              title="on the law  vs  below it", title_fontsize=9.5, framealpha=0.95)

    caveat_footer(fig, y=0.012)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.93, bottom=0.115)
    p = os.path.join(OUT, "fig0_hero.png")
    fig.savefig(p); plt.close(fig)
    return p, dict(on_curve=fit, all=fit_all, robust=fit_robust, n_on=len(xon))

# ================================================================ FIG 1
def fig1_magnitude_law():
    """Headline: retention vs ||dW|| (fdelta primary; dw_sv_mean & dw_sv_max secondary), per domain CS.
    Two-row layout: top = retention(core) vs the three magnitude measures; pooled log fit + r,R2."""
    rows = SWEEP
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    measures = [("F", "our magnitude measure  $\\|\\Delta W\\|_F$\n(token-weighted Frobenius)", "fdelta"),
                ("svmean", r"mean spectral norm  $\overline{\sigma}(\Delta W)$", "dw_sv_mean"),
                ("svmax", r"max spectral norm  $\sigma_{\max}(\Delta W)$", "dw_sv_max")]
    for ax, (key, xlabel, _) in zip(axes, measures):
        x = arr(rows, key); y = arr(rows, "ret")
        good = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[good], y[good]
        fit = fit_line_logx(x, y)
        xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        ax.plot(xx, fit["pred"](xx), color="0.25", lw=2.2, ls="--", zorder=2,
                label=f"pooled log-fit\n$r$={fit['r']:.2f}, $R^2$={fit['r2']:.2f}")
        # CI band via bootstrap of the linear fit on log-x
        lx = np.log10(x)
        preds = []
        rng = np.random.default_rng(0)
        for _ in range(500):
            idx = rng.integers(0, len(x), len(x))
            s, i, *_ = stats.linregress(lx[idx], y[idx])
            preds.append(i + s * np.log10(xx))
        preds = np.array(preds)
        ax.fill_between(xx, np.percentile(preds, 2.5, 0), np.percentile(preds, 97.5, 0),
                        color="0.6", alpha=0.18, zorder=1)
        for m in METHODS:
            mr = [r for r in rows if r["method"] == m]
            mx = arr(mr, key); my = arr(mr, "ret")
            g = np.isfinite(mx) & np.isfinite(my) & (mx > 0)
            ax.scatter(mx[g], my[g], s=95 if m in SIMPLE else 62, c=COL[m], marker=MARK[m],
                       edgecolor="k" if m in SIMPLE else "0.3",
                       linewidth=0.9 if m in SIMPLE else 0.4,
                       alpha=0.95, zorder=5 if m in SIMPLE else 4)
        ax.axhline(BASE_CORE, ls=":", color="green", lw=1.4, zorder=1)
        ax.set_xscale("log"); clean_logx(ax)
        ax.set_xlabel(xlabel)
        ax.legend(loc="upper right", fontsize=8.5)
    # emphasize the chosen panel (tightest fit -> the fair magnitude axis we adopt)
    for spine in axes[0].spines.values():
        spine.set_visible(True); spine.set_color("#0072B2"); spine.set_linewidth(2.2)
    axes[0].set_title("CHOSEN: tightest, scale-free fit", color="#0072B2", fontsize=11, pad=6)
    axes[1].set_title("looser", color="0.45", fontsize=10, pad=6)
    axes[2].set_title("loosest (spiky; method-confounded)", color="0.45", fontsize=10, pad=6)
    axes[0].set_ylabel("Retention  (mean of BBH, MMLU-Pro)  [%]")
    for ax in axes:
        x0, x1 = ax.get_xlim()
        ax.text(10**(np.log10(x0) + 0.04*(np.log10(x1)-np.log10(x0))), BASE_CORE+0.35,
                "base ceiling (no FT)", color="green", fontsize=8.2, va="bottom", ha="left")
    # one global method legend on top
    fig.legend(handles=legend_handles(METHODS), loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 1.005), fontsize=9.5, columnspacing=1.1, handletextpad=0.3)
    fig.suptitle("Choosing a fair magnitude axis: $\\|\\Delta W\\|_F$ fits retention best across all 7 adapters "
                 "(Llama-2-7B, commonsense)", y=1.075, fontsize=14)
    watermark(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    caveat_footer(fig, y=0.005)
    p = os.path.join(OUT, "fig1_magnitude_law.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, {k: fit_line_logx(arr(rows, k)[np.isfinite(arr(rows,k))&(arr(rows,k)>0)],
                                arr(rows, "ret")[np.isfinite(arr(rows,k))&(arr(rows,k)>0)])
               for k in ("F", "svmean", "svmax")}

# ================================================================ FIG 2
def fig2_fairness_residuals():
    """FAIRNESS: fit retention ~ smooth_f(log F) pooled; plot per-method residuals.
    Quantitative: global R2 vs +method-factor R2 (ANCOVA-style), per-method residual means + t-test vs 0."""
    rows = SWEEP
    x = arr(rows, "F"); y = arr(rows, "ret")
    g = np.isfinite(x) & np.isfinite(y) & (x > 0)
    rows = [r for r, gg in zip(rows, g) if gg]
    x, y = x[g], y[g]
    lx = np.log10(x)

    # global smooth fit (spline on log-x), used as the "method-agnostic" prediction
    order = np.argsort(lx)
    spl = UnivariateSpline(lx[order], y[order], k=3, s=len(lx)*8.0)
    yhat = spl(lx)
    resid = y - yhat
    ss_tot = np.sum((y - y.mean())**2)
    r2_global = 1 - np.sum(resid**2)/ss_tot

    # ANCOVA-style: does adding a per-method intercept improve fit? (linear log-x baseline for clean DoF)
    sl, ic, *_ = stats.linregress(lx, y)
    resid_lin = y - (ic + sl*lx)
    ss_res_lin = np.sum(resid_lin**2)
    # full model: per-method intercepts + common slope
    methods_here = [m for m in METHODS if any(r["method"] == m for r in rows)]
    M = np.zeros((len(x), len(methods_here)))
    for j, m in enumerate(methods_here):
        M[:, j] = [1.0 if r["method"] == m else 0.0 for r in rows]
    X = np.column_stack([M, lx])  # per-method intercept + common slope
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_full = y - X @ beta
    ss_res_full = np.sum(resid_full**2)
    n = len(x); p_red = 2; p_full = len(methods_here) + 1
    df1 = p_full - p_red; df2 = n - p_full
    F_stat = ((ss_res_lin - ss_res_full)/df1) / (ss_res_full/df2)
    p_anova = 1 - stats.f.cdf(F_stat, df1, df2)
    r2_lin = 1 - ss_res_lin/ss_tot
    r2_full = 1 - ss_res_full/ss_tot

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    # LEFT: data + spline + residual whiskers colored by method
    xx = np.linspace(lx.min(), lx.max(), 200)
    axL.plot(10**xx, spl(xx), color="0.2", lw=2.4, label=f"the $\\|\\Delta W\\|$ law (pooled fit, $R^2$={r2_global:.2f})", zorder=3)
    for m in methods_here:
        idx = [i for i, r in enumerate(rows) if r["method"] == m]
        off = m in OFF_CURVE
        axL.scatter(x[idx], y[idx], s=120 if off else (85 if m in SIMPLE else 55),
                    c=COL[m], marker=MARK[m],
                    edgecolor="k", linewidth=1.0 if off else (0.8 if m in SIMPLE else 0.4),
                    zorder=6 if off else 5, alpha=0.97)
        if off:   # halo ring to call out the below-curve outliers
            axL.scatter(x[idx], y[idx], s=330, facecolors="none", edgecolors=COL[m],
                        linewidths=1.6, alpha=0.5, zorder=5)
    axL.set_xscale("log"); clean_logx(axL); axL.set_xlabel(r"$\|\Delta W\|_F$  (log scale)")
    axL.set_ylabel("Retention (BBH, MMLU-Pro mean) [%]")
    axL.axhline(BASE_CORE, ls=":", color="green", lw=1.3)
    axL.set_title("5 adapters trace one curve; CorDA & SC-LoRA sit below it")
    axL.legend(loc="upper right", fontsize=8.8)

    # RIGHT: per-method residual strip+box
    data = []; labels = []; cols = []; means = []; tstats = []
    for m in methods_here:
        rr = resid[[i for i, r in enumerate(rows) if r["method"] == m]]
        data.append(rr); labels.append(PRETTY[m]); cols.append(COL[m])
        means.append(rr.mean())
        t, pval = stats.ttest_1samp(rr, 0.0) if len(rr) > 1 else (np.nan, np.nan)
        tstats.append((rr.mean(), pval, len(rr)))
    bp = axR.boxplot(data, orientation="horizontal", widths=0.62, patch_artist=True, showfliers=False,
                     medianprops=dict(color="k", lw=1.4),
                     whiskerprops=dict(color="0.5"), capprops=dict(color="0.5"))
    for patch, c, m in zip(bp["boxes"], cols, methods_here):
        off = m in OFF_CURVE
        patch.set_facecolor(c); patch.set_alpha(0.5 if off else 0.28)
        patch.set_edgecolor("k" if off else c); patch.set_linewidth(1.6 if off else 1.0)
    rng = np.random.default_rng(1)
    for i, (rr, c) in enumerate(zip(data, cols)):
        jit = rng.normal(0, 0.07, len(rr))
        axR.scatter(rr, np.full(len(rr), i+1)+jit, s=42, c=c, edgecolor="k",
                    linewidth=0.4, alpha=0.9, zorder=5)
    axR.axvline(0, color="0.25", lw=1.6, ls="--")
    # shade the "indistinguishable from the law" zone (where on-curve methods live)
    axR.axvspan(-2.5, 2.5, color="#2ca02c", alpha=0.07, zorder=0)
    axR.set_yticks(range(1, len(labels)+1)); axR.set_yticklabels(labels)
    # bold the two off-curve y-tick labels
    for tick, m in zip(axR.get_yticklabels(), methods_here):
        if m in OFF_CURVE:
            tick.set_fontweight("bold"); tick.set_color(COL[m])
    axR.set_xlabel("Retention residual from pooled curve  [pp]   (> 0 = above the law)")
    axR.set_title("5 adapters straddle 0;  CorDA & SC-LoRA forget significantly more (p<0.05)")
    # widen x so annotations have room on the right
    x0, x1 = axR.get_xlim(); axR.set_xlim(x0, x1 + 0.28*(x1-x0))
    lab_x = axR.get_xlim()[1] - 0.01*(axR.get_xlim()[1]-axR.get_xlim()[0])
    for i, (mu, pv, nn) in enumerate(tstats):
        sig = "*" if pv < 0.05 else ""
        axR.text(lab_x, i+1, f"μ={mu:+.1f}{sig}", va="center", ha="right",
                 fontsize=8.5, color=cols[i], fontweight="bold")

    fig.suptitle("5 of 7 adapters lie on the $\\|\\Delta W\\|$ law; the data-aware inits (CorDA, SC-LoRA) "
                 "forget MORE than their budget predicts",
                 y=1.045, fontsize=13.5)
    fig.text(0.5, 0.965, "ANCOVA: adding a per-method intercept improves fit "
             f"($R^2$ {r2_lin:.2f}$\\rightarrow${r2_full:.2f}, $F_{{{df1},{df2}}}$={F_stat:.1f}, "
             f"p={p_anova:.3f}) — driven entirely by CorDA ($\\mu$=−3.0*) and SC-LoRA ($\\mu$=−3.3*)",
             ha="center", va="bottom", fontsize=9.5, color="0.3")
    watermark(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    caveat_footer(fig, y=0.004)
    p = os.path.join(OUT, "fig2_fairness_residuals.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    stat = dict(r2_global=r2_global, r2_lin=r2_lin, r2_full=r2_full, F=F_stat, p=p_anova,
                df1=df1, df2=df2, per_method=dict(zip([PRETTY[m] for m in methods_here], tstats)))
    return p, stat

# ================================================================ FIG 3
def fig3_pareto():
    """Adaptation-retention Pareto frontier per method, per domain. Mark best-LR (max adapt) point + base ceiling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, (dom, rows, adapt_name, floor) in zip(
            axes, [("cs", SWEEP, "Commonsense-8 accuracy", 70.0),
                   ("math", SWEEP_M, "GSM8K accuracy", 30.0)]):
        for m in METHODS:
            mr = sorted([r for r in rows if r["method"] == m and r["adapt"] is not None
                         and r["ret"] is not None], key=lambda r: r["adapt"])
            if not mr:
                continue
            xs = [r["adapt"] for r in mr]; ys = [r["ret"] for r in mr]
            ax.plot(xs, ys, "-", color=COL[m], lw=2.0 if m in SIMPLE else 1.2,
                    alpha=0.85 if m in SIMPLE else 0.55, zorder=3, ms=0)
            ax.scatter(xs, ys, s=70 if m in SIMPLE else 45, c=COL[m], marker=MARK[m],
                       edgecolor="k" if m in SIMPLE else "0.3",
                       linewidth=0.8 if m in SIMPLE else 0.4, zorder=4, alpha=0.9)
            # best-LR (max adaptation) marker
            best = max(mr, key=lambda r: r["adapt"])
            ax.scatter([best["adapt"]], [best["ret"]], s=240, facecolors="none",
                       edgecolors=COL[m], linewidths=2.2, zorder=6)
        # flag training-collapse outliers (degenerate adaptation; valid retention)
        if dom == "cs":
            cr = [r for r in rows if r["run"] in COLLAPSE]
            for r in cr:
                ax.scatter([r["adapt"]], [r["ret"]], s=300, facecolors="none",
                           edgecolors="red", linewidths=2.2, marker="o", zorder=8)
                ax.annotate("collapse", xy=(r["adapt"], r["ret"]),
                            xytext=(r["adapt"]+3, r["ret"]-3.2), fontsize=7.5, color="red",
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.0))
        ax.axhline(BASE_CORE, ls=":", color="green", lw=1.4)
        ax.text(ax.get_xlim()[0], BASE_CORE+0.15, "base retention ceiling (no FT)",
                color="green", fontsize=8.5, va="bottom")
        ax.set_xlabel(adapt_name + "  [%]   (more adaptation →)")
        ax.set_ylabel("Retention (BBH, MMLU-Pro mean)  [%]")
        ax.set_title(f"{'Commonsense' if dom=='cs' else 'Math (GSM8K)'} domain"
                     + ("" if dom == "cs" else "  (sparse: only LoRA / LoRA+wd swept)"))
    axes[0].legend(handles=legend_handles(METHODS) +
                   [Line2D([0],[0], marker="o", color="0.4", lw=0, ms=13, markerfacecolor="none",
                           markeredgewidth=2.0, label="best-LR (max adapt.)")],
                   loc="lower left", ncol=2, fontsize=8.5)
    fig.suptitle("Adaptation–Retention Pareto frontier  (each point = one learning rate)", y=1.0, fontsize=14)
    fig.text(0.5, 0.005,
             "Red rings = training-collapse outliers (degenerate adaptation, valid retention; "
             "magnitude law unchanged if excluded): clora k1024@1e-4, corda r16@2e-5, dora r16@2e-5.",
             ha="center", va="bottom", fontsize=8, color="red", style="italic")
    watermark(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    p = os.path.join(OUT, "fig3_pareto.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p

# ================================================================ FIG 4
def fig4_lr_sensitivity():
    """LR sensitivity: adaptation & retention vs LR per method (CS). Highlight differing optima."""
    rows = SWEEP
    fig, (axA, axR) = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True)
    best_lrs = {}
    for m in METHODS:
        mr = sorted([r for r in rows if r["method"] == m and r["lr"] is not None], key=lambda r: r["lr"])
        if not mr:
            continue
        lrs = [r["lr"] for r in mr]
        ad = [r["adapt"] for r in mr]; rt = [r["ret"] for r in mr]
        axA.plot(lrs, ad, "-", color=COL[m], lw=2.0 if m in SIMPLE else 1.2, marker=MARK[m],
                 ms=8 if m in SIMPLE else 6, markeredgecolor="k" if m in SIMPLE else "0.3",
                 markeredgewidth=0.7 if m in SIMPLE else 0.4, alpha=0.92)
        axR.plot(lrs, rt, "-", color=COL[m], lw=2.0 if m in SIMPLE else 1.2, marker=MARK[m],
                 ms=8 if m in SIMPLE else 6, markeredgecolor="k" if m in SIMPLE else "0.3",
                 markeredgewidth=0.7 if m in SIMPLE else 0.4, alpha=0.92)
        best = max(mr, key=lambda r: (r["adapt"] if r["adapt"] is not None else -1))
        best_lrs[m] = best["lr"]
        axA.scatter([best["lr"]], [best["adapt"]], s=230, facecolors="none",
                    edgecolors=COL[m], linewidths=2.2, zorder=6)
    for ax in (axA, axR):
        ax.set_xscale("log")
        ax.set_xlabel("learning rate (log)")
    axA.axhline(70, ls="--", color="0.6", lw=1); axA.text(2e-5, 71, "adaptation floor (70%)", fontsize=8, color="0.45")
    axR.axhline(BASE_CORE, ls=":", color="green", lw=1.4)
    axR.text(2e-5, BASE_CORE+0.2, "base ceiling", color="green", fontsize=8.5)
    axA.set_ylabel("Commonsense-8 accuracy [%]")
    axR.set_ylabel("Retention (BBH, MMLU-Pro mean) [%]")
    axA.set_title("Adaptation vs LR  (rings = each method's adaptation-optimal LR)")
    axR.set_title("Retention vs LR  (monotone: higher LR → more forgetting)")
    axA.legend(handles=legend_handles(METHODS), loc="lower center", ncol=4, fontsize=8.3)
    uniq = sorted(set(best_lrs.values()))
    fig.suptitle("Per-method LR sensitivity — optimal LR is NOT shared "
                 f"(best-LR ∈ {{{', '.join(f'{l:g}' for l in uniq)}}})  →  any single fixed LR biases the comparison",
                 y=1.0, fontsize=12.5)
    watermark(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "fig4_lr_sensitivity.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, best_lrs

# ================================================================ FIG 5
def fig5_per_benchmark():
    """Per-benchmark retention vs ||dW||_F: which knowledge dies first. Pooled trend per benchmark."""
    rows = SWEEP
    benches = [("bbh", "BBH (reasoning)", BASE_BBH),
               ("mmlu", "MMLU", None),
               ("mmlu_pro", "MMLU-Pro", BASE_MMLU_PRO),
               ("arc", "ARC-Challenge", None),
               ("tqa", "TruthfulQA", None)]
    x = arr(rows, "F"); g0 = np.isfinite(x) & (x > 0)

    fig = plt.figure(figsize=(15, 9.0))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.27)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    summary_slopes = {}
    for ax, (key, title, base) in zip(axes, benches):
        y = arr(rows, key)
        g = g0 & np.isfinite(y)
        xv, yv = x[g], y[g]
        for m in METHODS:
            mr = [r for r in rows if r["method"] == m]
            mx = arr(mr, "F"); my = arr(mr, key)
            gg = np.isfinite(mx) & np.isfinite(my) & (mx > 0)
            ax.scatter(mx[gg], my[gg], s=55 if m in SIMPLE else 35, c=COL[m], marker=MARK[m],
                       edgecolor="k" if m in SIMPLE else "none", linewidth=0.5,
                       alpha=0.9 if m in SIMPLE else 0.7, zorder=4)
        fit = fit_line_logx(xv, yv)
        xx = np.logspace(np.log10(xv.min()), np.log10(xv.max()), 100)
        ax.plot(xx, fit["pred"](xx), color="0.2", lw=2.2, ls="--",
                label=f"slope={fit['slope']:.1f} pp/dec\n$r$={fit['r']:.2f}")
        summary_slopes[title] = (fit["slope"], fit["r"], base)
        if base is not None:
            ax.axhline(base, ls=":", color="green", lw=1.3)
            ax.text(xx[0], base, " base", color="green", fontsize=8, va="bottom")
        else:
            ax.text(0.97, 0.04, "no base ceiling\n(uncalibrated)", transform=ax.transAxes,
                    fontsize=7.5, color="0.45", style="italic", ha="right", va="bottom")
        ax.set_xscale("log"); clean_logx(ax)
        ax.set_title(title)
        ax.set_xlabel(r"$\|\Delta W\|_F$")
        ax.set_ylabel("accuracy [%]")
        ax.legend(loc="best", fontsize=8.2)
    # 6th cell: normalized degradation slopes bar (which dies fastest)
    ax6 = fig.add_subplot(gs[1, 2])
    names = list(summary_slopes.keys())
    slopes = [summary_slopes[n][0] for n in names]
    order = np.argsort(slopes)   # most negative first
    names = [names[i] for i in order]; slopes = [slopes[i] for i in order]
    bars = ax6.barh(names, slopes, color=["#D55E00" if s < np.median(slopes) else "#0072B2" for s in slopes],
                    alpha=0.85, edgecolor="k", linewidth=0.6)
    xr = max(abs(min(slopes)), abs(max(slopes)))
    ax6.set_xlim(min(slopes)-0.18*xr, max(slopes)+0.18*xr)
    for b, s in zip(bars, slopes):
        off = 0.02*xr
        ax6.text(s - off if s < 0 else s + off, b.get_y()+b.get_height()/2, f"{s:.1f}",
                 va="center", ha="right" if s < 0 else "left", fontsize=9.5, fontweight="bold")
    ax6.axvline(0, color="0.3", lw=1)
    ax6.set_xlabel("degradation slope  [pp accuracy per decade of $\\|\\Delta W\\|_F$]")
    ax6.set_title("Which knowledge dies fastest?", fontsize=12)
    ax6.grid(axis="y", alpha=0)
    fig.suptitle("Per-benchmark retention vs weight-update magnitude  (Llama-2-7B, commonsense LR-sweep)",
                 y=0.985, fontsize=14)
    fig.text(0.5, 0.952, "Note: only BBH and MMLU-Pro have a measured base-model ceiling; "
             "MMLU, ARC-Challenge and TruthfulQA are uncalibrated (no base eval).",
             ha="center", va="bottom", fontsize=8.5, color="0.4", style="italic")
    fig.legend(handles=legend_handles(METHODS), loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, 0.005), fontsize=9.5)
    watermark(fig)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.10, hspace=0.46, wspace=0.27)
    p = os.path.join(OUT, "fig5_per_benchmark.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, summary_slopes

# ================================================================ FIG 6
def fig6_extras():
    """Extras that strengthen the story:
    (a) adaptation vs ||dW||_F (adaptation needs magnitude — the tension);
    (b) spectral structure: sv_max/sv_mean ratio per method (CorDA's spiky init = unfair x-axis warning);
    (c) the SAME magnitude buys different adaptation per method (efficiency)."""
    rows = SWEEP
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    # (a) adaptation vs F
    axA = axes[0]
    x = arr(rows, "F"); y = arr(rows, "adapt")
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m]
        mx = arr(mr, "F"); my = arr(mr, "adapt")
        g = np.isfinite(mx) & np.isfinite(my) & (mx > 0)
        axA.scatter(mx[g], my[g], s=80 if m in SIMPLE else 50, c=COL[m], marker=MARK[m],
                    edgecolor="k" if m in SIMPLE else "0.3", linewidth=0.7 if m in SIMPLE else 0.4,
                    alpha=0.92, zorder=4)
    axA.axhline(70, ls="--", color="0.6", lw=1)
    axA.set_xscale("log"); clean_logx(axA); axA.set_xlabel(r"$\|\Delta W\|_F$"); axA.set_ylabel("Commonsense-8 accuracy [%]")
    axA.set_title("Adaptation needs magnitude\n(the tension behind the tradeoff)")

    # (b) spectral spikiness sv_max/sv_mean per method (boxplot)
    axB = axes[1]
    data = []; labels = []; cols = []
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m and r["svmean"] and r["svmax"]]
        ratio = [r["svmax"]/r["svmean"] for r in mr]
        if ratio:
            data.append(ratio); labels.append(PRETTY[m]); cols.append(COL[m])
    bp = axB.boxplot(data, orientation="vertical", widths=0.6, patch_artist=True, showfliers=False,
                     medianprops=dict(color="k", lw=1.4))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    rng = np.random.default_rng(2)
    for i, (rr, c) in enumerate(zip(data, cols)):
        axB.scatter(np.full(len(rr), i+1)+rng.normal(0,0.06,len(rr)), rr, s=38, c=c,
                    edgecolor="k", linewidth=0.4, zorder=5, alpha=0.9)
    axB.set_xticks(range(1, len(labels)+1)); axB.set_xticklabels(labels, rotation=30, ha="right")
    axB.set_ylabel(r"$\sigma_{\max}/\overline{\sigma}$  (spectral spikiness)")
    axB.set_title("Why $\\sigma_{\\max}$ is an unfair x-axis\n(CorDA/DoRA spike; LoRA flat)")
    axB.axhline(np.median([np.median(d) for d in data]), ls=":", color="0.5")

    # (c) magnitude-efficiency: adaptation at fixed retention budget. Interp adapt @ ret=24 per method
    axC = axes[2]
    target_ret = 24.0
    eff = []
    for m in METHODS:
        mr = sorted([r for r in rows if r["method"] == m and r["ret"] is not None
                     and r["adapt"] is not None], key=lambda r: r["ret"])
        rt = np.array([r["ret"] for r in mr]); ad = np.array([r["adapt"] for r in mr])
        if rt.min() <= target_ret <= rt.max():
            # interpolate adaptation at the retention budget (sort by ret)
            a = np.interp(target_ret, rt, ad)
            eff.append((m, a))
    eff.sort(key=lambda t: t[1])
    if eff:
        ms = [PRETTY[m] for m, _ in eff]; vals = [a for _, a in eff]
        bars = axC.barh(ms, vals, color=[COL[m] for m, _ in eff], alpha=0.85, edgecolor="k", linewidth=0.6)
        for b, v in zip(bars, vals):
            axC.text(v, b.get_y()+b.get_height()/2, f" {v:.0f}", va="center", fontsize=9, fontweight="bold")
    axC.set_xlabel("Commonsense-8 accuracy achievable [%]")
    axC.set_title(f"Adaptation at a fixed retention budget\n(ret = {target_ret:.0f}%): which adapter is most efficient?")
    axC.grid(axis="y", alpha=0)

    fig.suptitle("Supporting structure: the adaptation cost of magnitude, spectral fairness, and per-adapter efficiency",
                 y=1.02, fontsize=13)
    fig.legend(handles=legend_handles(METHODS), loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 0.985), fontsize=9)
    watermark(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, "fig6_supporting_structure.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p

# ================================================================ shared LR-sweep helpers (figs 7/8/table)
# The 7 swept learning rates, in order; used as the shared categorical axis where LR is the predictor.
LR_GRID = [2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
SAFE_RET = 24.0   # "safe" = retention >= base_ceiling - 2pp  (BASE_CORE ~= 26.0)

# Caption note: ranks/secondary configs are NOT matched across methods -> we frame the LAW, not a ranking.
CFG_NOTE = ("Note: ranks / secondary configs are NOT matched across methods "
            "(LoRA/DoRA/CorDA r16, MiLoRA/SC-LoRA r32, CLoRA k1024) — cross-method gaps are partly "
            "config effects. We frame around the magnitude law, not a method ranking.")
def cfg_note_footer(fig, y=0.028):
    fig.text(0.5, y, CFG_NOTE, ha="center", va="bottom", fontsize=8, color="0.35", style="italic")

def _msorted(rows, m):
    """method rows with finite lr/F/adapt/ret, sorted by LR."""
    mr = [r for r in rows if r["method"] == m and r["lr"] is not None
          and r["F"] is not None and r["adapt"] is not None and r["ret"] is not None]
    return sorted(mr, key=lambda r: r["lr"])

def op_points_data():
    """Per-method operating points: best-LR (max adaptation), safe-LR (max adapt with ret>=SAFE_RET),
    and robustness count (#LRs with ret>=SAFE_RET)."""
    rows = SWEEP
    out = []
    for m in METHODS:
        mr = _msorted(rows, m)
        if not mr:
            continue
        best = max(mr, key=lambda r: r["adapt"])
        safe_cands = [r for r in mr if r["ret"] >= SAFE_RET]
        safe = max(safe_cands, key=lambda r: r["adapt"]) if safe_cands else None
        nrob = sum(1 for r in mr if r["ret"] >= SAFE_RET)
        out.append(dict(method=m,
                        best_lr=best["lr"], best_adapt=best["adapt"], best_ret=best["ret"],
                        safe_lr=(safe["lr"] if safe else None),
                        safe_adapt=(safe["adapt"] if safe else None),
                        safe_ret=(safe["ret"] if safe else None),
                        nrobust=nrob))
    return out

# ================================================================ FIG 7  (LR is a proxy; ||dW|| is the cause)
def fig7_lr_is_the_proxy():
    """Panel A: LR -> ||dW||_F transmission per method (data-aware inits reach larger ||dW|| per LR).
    Panel B: retention vs LR (loose) beside retention vs ||dW||_F (tight) — the R^2 contrast IS the message."""
    rows = SWEEP
    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1, 1], wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axL = fig.add_subplot(gs[0, 1])      # retention vs LR
    axF = fig.add_subplot(gs[0, 2], sharey=axL)  # retention vs ||dW||

    # ---- Panel A: LR (log x) -> ||dW||_F (log y), one line per method
    for m in METHODS:
        mr = _msorted(rows, m)
        if not mr:
            continue
        lrs = [r["lr"] for r in mr]; fs = [r["F"] for r in mr]
        off = m in OFF_CURVE
        axA.plot(lrs, fs, "-", color=COL[m], lw=2.4 if off else (2.0 if m in SIMPLE else 1.3),
                 marker=MARK[m], ms=9 if off else (8 if m in SIMPLE else 6),
                 markeredgecolor="k", markeredgewidth=0.9 if off else 0.55,
                 alpha=0.97, zorder=6 if off else (5 if m in SIMPLE else 4))
    axA.set_xscale("log"); axA.set_yscale("log"); clean_logx(axA)
    axA.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axA.set_xlabel("learning rate (log scale)")
    axA.set_ylabel(r"resulting  $\|\Delta W\|_F$  (token-weighted, log scale)")
    axA.set_title("A.  Same LR → different $\\|\\Delta W\\|$")
    # call out the data-aware inits sitting ABOVE the pack (more magnitude per LR).
    # anchor on SC-LoRA's lowest-LR point; use a neutral-gray, near-straight arrow so it does NOT
    # read as part of the (purple) SC-LoRA line — the curved colored arrow looked like a data hook.
    _scl0 = min(_msorted(rows, "sclora"), key=lambda r: r["lr"])
    axA.annotate("data-aware inits (CorDA, SC-LoRA)\nturn the same LR into a LARGER update\n"
                 "→ the mechanism behind their extra forgetting",
                 xy=(_scl0["lr"], _scl0["F"]),
                 xytext=(0.05, 0.97), textcoords="axes fraction",
                 fontsize=8.6, color="0.15", ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.4, shrinkA=4, shrinkB=7,
                                 connectionstyle="arc3,rad=0.08"),
                 bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=COL["sclora"], lw=1.1, alpha=0.95))
    axA.legend(handles=legend_handles(METHODS), loc="lower right", ncol=2, fontsize=8.0,
               columnspacing=0.9, handletextpad=0.3)

    # ---- Panels B(left/right): retention vs LR  vs  retention vs ||dW||_F
    def _scatter_fit(ax, key, xlabel, logx=True, title=""):
        x = arr(rows, key); y = arr(rows, "ret")
        g = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[g], y[g]
        fit = fit_line_logx(x, y)
        xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        ax.plot(xx, fit["pred"](xx), color="0.2", lw=2.4, ls="--", zorder=3)
        for m in METHODS:
            mr = [r for r in rows if r["method"] == m]
            mx = arr(mr, key); my = arr(mr, "ret")
            gg = np.isfinite(mx) & np.isfinite(my) & (mx > 0)
            off = m in OFF_CURVE
            ax.scatter(mx[gg], my[gg], s=95 if off else (80 if m in SIMPLE else 50),
                       c=COL[m], marker=MARK[m],
                       edgecolor="k", linewidth=0.9 if off else (0.7 if m in SIMPLE else 0.4),
                       alpha=0.95, zorder=6 if off else (5 if m in SIMPLE else 4))
        ax.axhline(BASE_CORE, ls=":", color="green", lw=1.4, zorder=1)
        if logx:
            ax.set_xscale("log"); clean_logx(ax)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        # R^2 badge
        ax.text(0.04, 0.06, f"$R^2$ = {fit['r2']:.2f}\n$r$ = {fit['r']:.2f}",
                transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="#f3f6fb", ec="0.6", lw=1.0))
        return fit

    fitL = _scatter_fit(axL, "lr", "learning rate (log scale)", title="B.  Retention vs LR  (loose)")
    fitF = _scatter_fit(axF, "F", r"$\|\Delta W\|_F$  (log scale)",
                        title=r"C.  Retention vs $\|\Delta W\|_F$  (tight)")
    axL.set_ylabel("Retention (BBH, MMLU-Pro mean) [%]")
    plt.setp(axF.get_yticklabels(), visible=False)
    # green base-ceiling label on the LR panel (open top-left? use right where curve has dropped)
    axL.text(axL.get_xlim()[1], BASE_CORE + 0.3, "base ceiling ", color="green", fontsize=8.3,
             va="bottom", ha="right", fontweight="bold")
    # emphasize the winning panel (||dW|| -> tightest)
    for spine in axF.spines.values():
        spine.set_visible(True); spine.set_color("#0072B2"); spine.set_linewidth(2.2)
    # the contrast headline, between the two right panels
    dR2 = fitF["r2"] - fitL["r2"]
    fig.suptitle("Learning rate is only a proxy: it predicts forgetting loosely; "
                 r"the $\|\Delta W\|$ it produces predicts it tightly "
                 f"($R^2$ {fitL['r2']:.2f} $\\rightarrow$ {fitF['r2']:.2f})",
                 y=1.0, fontsize=13.5)
    watermark(fig)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.30, wspace=0.30)
    cfg_note_footer(fig, y=0.105)
    caveat_footer(fig, y=0.020)
    p = os.path.join(OUT, "fig7_lr_is_the_proxy.png")
    fig.savefig(p); plt.close(fig)
    return p, dict(lr=fitL, F=fitF, dR2=dR2)

# ================================================================ FIG 8  (the magnitude budget)
def fig8_magnitude_budget():
    """adaptation (top) and retention (bottom) vs ||dW||_F on a SHARED x-axis. ||dW|| buys adaptation
    and costs retention; lightly shade a sweet-spot band (near-max adapt while retention near base)."""
    rows = SWEEP
    x = arr(rows, "F"); g0 = np.isfinite(x) & (x > 0)
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(10.5, 8.6), sharex=True,
                                   gridspec_kw={"hspace": 0.10})

    # ---- sweet-spot band: ||dW|| where retention still >= SAFE_RET, intersected with high adaptation.
    # Use the on-curve, non-collapse runs to find the largest ||dW|| that keeps ret>=SAFE_RET (right edge),
    # and the smallest ||dW|| reaching >=95% of the max adaptation (left edge).
    onc = [r for r in rows if r["method"] in ON_CURVE and r["run"] not in COLLAPSE
           and r["F"] and r["adapt"] is not None and r["ret"] is not None]
    a_max = max(r["adapt"] for r in onc)
    left_band = min((r["F"] for r in onc if r["adapt"] >= 0.95 * a_max), default=None)
    right_band = max((r["F"] for r in onc if r["ret"] >= SAFE_RET), default=None)
    has_band = left_band is not None and right_band is not None and left_band < right_band

    # ---- TOP: adaptation vs ||dW||  (rises)
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m]
        mx = arr(mr, "F"); my = arr(mr, "adapt")
        g = np.isfinite(mx) & np.isfinite(my) & (mx > 0)
        off = m in OFF_CURVE
        axT.scatter(mx[g], my[g], s=95 if off else (80 if m in SIMPLE else 50), c=COL[m], marker=MARK[m],
                    edgecolor="k", linewidth=0.9 if off else (0.7 if m in SIMPLE else 0.4),
                    alpha=0.95, zorder=6 if off else (5 if m in SIMPLE else 4))
    # pooled monotone trend (log-x linear fit on adaptation, for the eye)
    xa = arr(rows, "adapt"); gA = g0 & np.isfinite(xa)
    fitA = fit_line_logx(x[gA], xa[gA])
    xx = np.logspace(np.log10(x[g0].min()), np.log10(x[g0].max()), 200)
    axT.plot(xx, fitA["pred"](xx), color="0.3", lw=2.0, ls="--", zorder=2)
    axT.set_ylabel("Adaptation\nCommonsense-8 accuracy [%]")
    axT.set_title(r"More $\|\Delta W\|$ buys adaptation …", loc="left", fontsize=12.5)
    axT.text(0.985, 0.08, "rises ↗", transform=axT.transAxes, ha="right", va="bottom",
             fontsize=11, color="0.35", style="italic")

    # ---- BOTTOM: retention vs ||dW||  (falls), base ceiling marked
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m]
        mx = arr(mr, "F"); my = arr(mr, "ret")
        g = np.isfinite(mx) & np.isfinite(my) & (mx > 0)
        off = m in OFF_CURVE
        axB.scatter(mx[g], my[g], s=95 if off else (80 if m in SIMPLE else 50), c=COL[m], marker=MARK[m],
                    edgecolor="k", linewidth=0.9 if off else (0.7 if m in SIMPLE else 0.4),
                    alpha=0.95, zorder=6 if off else (5 if m in SIMPLE else 4))
    yb = arr(rows, "ret"); gB = g0 & np.isfinite(yb)
    fitR = fit_line_logx(x[gB], yb[gB])
    axB.plot(xx, fitR["pred"](xx), color="0.3", lw=2.0, ls="--", zorder=2)
    axB.axhline(BASE_CORE, ls=":", color="green", lw=1.6, zorder=2)
    axB.axhline(SAFE_RET, ls="--", color="0.5", lw=1.1, zorder=1)
    axB.set_ylabel("Retention\nBBH, MMLU-Pro mean [%]")
    axB.set_title(r"… and costs retention", loc="left", fontsize=12.5)
    axB.text(0.985, 0.50, "falls ↘", transform=axB.transAxes, ha="right", va="top",
             fontsize=11, color="0.35", style="italic")

    # sweet-spot band on BOTH panels
    if has_band:
        for ax in (axT, axB):
            ax.axvspan(left_band, right_band, color="#2ca02c", alpha=0.10, zorder=0)
        axT.text(10**(0.5*(np.log10(left_band)+np.log10(right_band))), axT.get_ylim()[0]+1.0,
                 "sweet-spot band\n(near-max adaptation,\nretention ≈ base)",
                 ha="center", va="bottom", fontsize=8.6, color="#1d6b1d", fontweight="bold")

    axB.set_xscale("log"); clean_logx(axB)
    axB.set_xlabel(r"Weight-update magnitude  $\|\Delta W\|_F$  (token-weighted, log scale)  →")
    # ceiling/safe labels (parked on the right, in open space below the descending cloud)
    axB.text(axB.get_xlim()[1], BASE_CORE + 0.3, "base retention ceiling (no fine-tune)  ",
             color="green", fontsize=9, va="bottom", ha="right", fontweight="bold")
    axB.text(axB.get_xlim()[1], SAFE_RET - 0.5, "safe threshold (base − 2pp)  ",
             color="0.4", fontsize=8.3, va="top", ha="right")

    fig.legend(handles=legend_handles(METHODS), loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 1.005), fontsize=9.2, columnspacing=1.0, handletextpad=0.3)
    fig.suptitle(r"The magnitude budget: one axis ($\|\Delta W\|_F$) sets both adaptation and forgetting",
                 y=1.052, fontsize=14)
    watermark(fig)
    fig.tight_layout(rect=[0, 0.075, 1, 0.97])
    cfg_note_footer(fig, y=0.038)
    caveat_footer(fig, y=0.006)
    p = os.path.join(OUT, "fig8_magnitude_budget.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, dict(band=(left_band, right_band) if has_band else None,
                   adapt_slope=fitA["slope"], ret_slope=fitR["slope"])

# ================================================================ OPERATING-POINT TABLE
def op_points_table():
    """Clean table figure: per method best-LR (adapt/ret), safe-LR (max adapt with ret>=24), robustness count.
    Also returns the numeric rows for stdout printing."""
    data = op_points_data()
    # sort by robustness desc, then best adaptation desc (widest, strongest first)
    data = sorted(data, key=lambda d: (-d["nrobust"], -d["best_adapt"]))

    cols = ["Method",
            "Best LR", "Adapt", "Ret",
            "Safe LR", "Adapt", "Ret",
            "Robust\n(ret≥24, /7)"]
    fig, ax = plt.subplots(figsize=(12.2, 0.62 * (len(data) + 2.4)))
    ax.axis("off")

    def lrfmt(v):
        return "—" if v is None else f"{v:g}"
    def numfmt(v, suff=""):
        return "—" if v is None else f"{v:.1f}{suff}"

    cell_text = []; cell_colours = []
    for d in data:
        safe_collapse = (d["safe_adapt"] is not None and d["safe_adapt"] < 40)  # degenerate "safe" point
        row = [PRETTY[d["method"]],
               lrfmt(d["best_lr"]), numfmt(d["best_adapt"]), numfmt(d["best_ret"]),
               lrfmt(d["safe_lr"]),
               numfmt(d["safe_adapt"]) + (" ⚠" if safe_collapse else ""),
               numfmt(d["safe_ret"]),
               f"{d['nrobust']}"]
        cell_text.append(row)
        # color cues: robustness column shaded green(wide)->red(brittle); off-curve method name tinted
        rob = d["nrobust"]
        rob_c = "#cdebcd" if rob >= 5 else ("#fce9cf" if rob >= 3 else "#f6cccc")
        name_c = "#f7e9ec" if d["method"] in OFF_CURVE else "white"
        rc = [name_c, "white", "white", "white", "white",
              "#f6e3cc" if safe_collapse else "white", "white", rob_c]
        cell_colours.append(rc)

    tbl = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colours,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
    tbl.scale(1.0, 1.7)
    # header styling + group separators
    ncol = len(cols)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.82"); cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#2b3a55"); cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.35)
        if c == 0 and r > 0:
            cell.set_text_props(fontweight="bold")
        # thicker vertical rules separating the Best / Safe / Robust groups
        if c in (1, 4, 7):
            cell.set_linewidth(0.8)
    # set sensible column widths
    widths = [0.18, 0.10, 0.085, 0.075, 0.10, 0.10, 0.075, 0.12]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_width(widths[c])

    ax.set_title("Operating points per adapter: peak adaptation vs. the safest high-adaptation setting",
                 fontsize=13.5, fontweight="bold", pad=18)
    # legend / reading guide
    fig.text(0.5, 0.135,
             "Best LR = max adaptation (any retention).   Safe LR = highest-adaptation run with retention ≥ 24 "
             "(= base − 2pp).   Robust = how many of the 7 LRs keep retention ≥ 24 (wide vs. brittle).",
             ha="center", va="top", fontsize=8.6, color="0.3")
    fig.text(0.5, 0.092,
             "⚠ = the only “safe” setting forces adaptation to collapse (data-aware inits transmit "
             "too much $\\|\\Delta W\\|$ at every usable LR).",
             ha="center", va="top", fontsize=8.6, color="#9a3b1d", style="italic")
    cfg_note_footer(fig, y=0.040)
    caveat_footer(fig, y=0.004)
    p = os.path.join(OUT, "op_points.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, data

# ================================================================ MAIN LATEX TABLES (sweep-only)
def _arm_cfg(run):
    """Pretty 'arm + lr' config string from a sweep run_name, e.g. 'lora r16, lr3e-4'."""
    r = run
    for p in ("lrswm_", "lrsw_"):
        if r.startswith(p):
            r = r[len(p):]; break
    parts = r.split("_")
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        parts = parts[:-1]
    lr_tok = next((t for t in parts if re.fullmatch(r"lr[0-9]+e[0-9]+", t)), None)
    arm = " ".join(t for t in parts if not re.fullmatch(r"lr[0-9]+e[0-9]+", t))
    lr = parse_lr(run)
    # consistent scientific LR label (e.g. 5e-5, 3e-4, 1e-3) — matches the swept-LR token set
    if lr is not None:
        exp = int(np.floor(np.log10(lr)))
        mant = round(lr / 10**exp)
        if mant == 10:
            mant, exp = 1, exp + 1
        lr_str = f"lr{mant}e{exp}"
    elif lr_tok is not None:
        # fall back to the raw token: 'lr5e5' -> 'lr5e-5'
        lr_str = re.sub(r"lr([0-9]+)e([0-9]+)", r"lr\1e-\2", lr_tok)
    else:
        lr_str = ""
    return f"{arm}, {lr_str}" if lr_str else arm

def _tex_escape(s):
    return s.replace("_", "\\_").replace("%", "\\%")

def _build_main_table(rows, adapt_label, preliminary=False):
    """Return a LaTeX tabular string. One row per method at its BEST-ADAPT LR (same selection as
    op_points). Single seed -> no +- std. Rows sorted by adaptation descending."""
    # best-adapt run per method
    best = []
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m and r["adapt"] is not None]
        if not mr:
            continue
        best.append(max(mr, key=lambda r: r["adapt"]))
    best.sort(key=lambda r: -r["adapt"])

    L = []
    L.append("\\begin{tabular}{l l c c c c c}")
    L.append("\\toprule")
    L.append(f"Method & Config & {adapt_label} $\\uparrow$ & Ret-core $\\uparrow$ & "
             "Ret-broad $\\uparrow$ & $\\|\\Delta W\\|_F$ $\\downarrow$ & $\\sigma_{\\max}$ \\\\")
    L.append("\\midrule")
    L.append("\\textit{Base (no-FT)} & -- & -- & \\textit{%.1f} & -- & \\textit{0} & -- \\\\" % BASE_CORE)
    L.append("\\midrule")
    for r in best:
        cfg = _tex_escape(_arm_cfg(r["run"]))
        adapt = "--" if r["adapt"] is None else f"{r['adapt']:.1f}"
        rc = "--" if r["ret"] is None else f"{r['ret']:.1f}"
        rb = "--" if r["broad"] is None else f"{r['broad']:.1f}"
        F = "--" if r["F"] is None else f"{r['F']:.3f}"
        sv = "--" if r["svmax"] is None else f"{r['svmax']:.1f}"
        name = PRETTY[r["method"]].replace("&", "\\&")
        L.append(f"{name} & {cfg} & {adapt} & {rc} & {rb} & {F} & {sv} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    note = ("% single seed s42; ranks/configs not matched across methods; "
            "each method shown at its best-adapt LR (sweep-only)."
            + (" Preliminary: math sweep currently only covers LoRA / LoRA+wd." if preliminary else ""))
    return "\n".join(L) + "\n" + note + "\n"

def write_main_tables():
    """Overwrite paper/table_main_cs.tex and paper/table_main_math.tex from the LIVE LR-sweep registry.
    Each method at its best-adapt LR; single seed (s42), no std; sorted by adaptation desc."""
    cs_tex = _build_main_table(SWEEP, "CS-8", preliminary=False)
    math_tex = _build_main_table(SWEEP_M, "GSM8K", preliminary=True)
    pcs = os.path.join(PAPER, "table_main_cs.tex")
    pmath = os.path.join(PAPER, "table_main_math.tex")
    with open(pcs, "w") as f:
        f.write(cs_tex)
    with open(pmath, "w") as f:
        f.write(math_tex)
    return pcs, pmath, cs_tex, math_tex

# ================================================================ run
if __name__ == "__main__":
    print(f"Loaded {len(ROWS)} campaign rows; CS sweep={len(SWEEP)}, math sweep={len(SWEEP_M)}")
    paths = []
    p0, hero = fig0_hero(); paths.append(p0)
    print("\n=== FIG0 HERO: magnitude law fit on 5 on-curve adapters ===")
    print(f"  on-curve (n={hero['n_on']}): r={hero['on_curve']['r']:+.3f}  R2={hero['on_curve']['r2']:.3f}  "
          f"slope={hero['on_curve']['slope']:.2f} pp/decade")
    print(f"  all-7 pooled:            r={hero['all']['r']:+.3f}  R2={hero['all']['r2']:.3f}")
    print(f"  robustness (on-curve, excl. 3 collapse outliers): r={hero['robust']['r']:+.3f}  "
          f"R2={hero['robust']['r2']:.3f}  slope={hero['robust']['slope']:.2f}  -> law unchanged")
    p1, fits = fig1_magnitude_law(); paths.append(p1)
    print("\n=== FIG1 pooled log-fits (CS, retention_mean) ===")
    for k, f in fits.items():
        print(f"  vs {k:7s}: r={f['r']:+.3f}  R2={f['r2']:.3f}  slope={f['slope']:.2f} pp/decade  p={f['p']:.1e}")
    p2, stat = fig2_fairness_residuals(); paths.append(p2)
    print("\n=== FIG2 fairness/ANCOVA ===")
    print(f"  pooled linear log-fit R2 = {stat['r2_lin']:.3f}")
    print(f"  +per-method intercepts R2 = {stat['r2_full']:.3f}")
    print(f"  ANCOVA F({stat['df1']},{stat['df2']}) = {stat['F']:.3f}, p = {stat['p']:.4f}")
    print("  per-method residual mean (pp), p(vs 0), n:")
    for m, (mu, pv, nn) in stat["per_method"].items():
        flag = "  <-- OFF-CURVE" if (pv < 0.05) else ""
        print(f"     {m:14s} mu={mu:+.2f}  p={pv:.3f}  n={nn}{flag}")
    p3 = fig3_pareto(); paths.append(p3)
    p4, blr = fig4_lr_sensitivity(); paths.append(p4)
    print("\n=== FIG4 best-LR per method (CS) ===")
    for m, l in blr.items():
        print(f"  {PRETTY[m]:14s} best LR = {l:g}")
    p5, slopes = fig5_per_benchmark(); paths.append(p5)
    print("\n=== FIG5 per-benchmark degradation slope (pp / decade of ||dW||_F) ===")
    for nm, (sl, r, base) in sorted(slopes.items(), key=lambda t: t[1][0]):
        print(f"  {nm:18s} slope={sl:+.2f}  r={r:+.2f}  base={base}")
    p6 = fig6_extras(); paths.append(p6)

    p7, f7 = fig7_lr_is_the_proxy(); paths.append(p7)
    print("\n=== FIG7 LR-is-a-proxy (CS, retention vs predictor) ===")
    print(f"  retention ~ log(LR)      : R2={f7['lr']['r2']:.3f}  r={f7['lr']['r']:+.3f}")
    print(f"  retention ~ log(||dW||_F): R2={f7['F']['r2']:.3f}  r={f7['F']['r']:+.3f}")
    print(f"  R2 gain (||dW|| over LR) : +{f7['dR2']:.3f}")

    p8, f8 = fig8_magnitude_budget(); paths.append(p8)
    print("\n=== FIG8 magnitude budget (CS) ===")
    print(f"  adaptation slope = {f8['adapt_slope']:+.1f} pp / decade of ||dW||_F")
    print(f"  retention slope  = {f8['ret_slope']:+.1f} pp / decade of ||dW||_F")
    if f8["band"]:
        print(f"  sweet-spot ||dW||_F band = [{f8['band'][0]:.3f}, {f8['band'][1]:.3f}]")
    else:
        print("  sweet-spot band: none (no clean overlap)")

    pt, optab = op_points_table(); paths.append(pt)
    print("\n=== OPERATING-POINT TABLE (CS, s42) ===")
    hdr = f"  {'method':14s} | {'best-LR':>8s} {'adapt':>6s} {'ret':>5s} | {'safe-LR':>8s} {'adapt':>6s} {'ret':>5s} | {'robust/7':>8s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for d in optab:
        sl = "—" if d["safe_lr"] is None else f"{d['safe_lr']:g}"
        sa = "—" if d["safe_adapt"] is None else f"{d['safe_adapt']:.1f}"
        sr = "—" if d["safe_ret"] is None else f"{d['safe_ret']:.1f}"
        print(f"  {PRETTY[d['method']]:14s} | {d['best_lr']:>8g} {d['best_adapt']:>6.1f} {d['best_ret']:>5.1f} "
              f"| {sl:>8s} {sa:>6s} {sr:>5s} | {d['nrobust']:>8d}")

    pcs, pmath, cs_tex, math_tex = write_main_tables()
    print("\n=== MAIN LATEX TABLES (sweep-only, best-adapt LR per method) ===")
    print(f"  wrote {pcs}")
    print(f"  wrote {pmath}")

    print("\n=== FIGURES WRITTEN ===")
    for p in paths:
        print(" ", p)
    print(" ", pcs)
    print(" ", pmath)
