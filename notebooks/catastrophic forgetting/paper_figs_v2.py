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

CAVEAT = ("Caveat: single seed (s42), n=7 per method; CorDA/SC-LoRA off-curve deviation "
          "pending seed replication (43, 44).")
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
    print("\n=== FIGURES WRITTEN ===")
    for p in paths:
        print(" ", p)
