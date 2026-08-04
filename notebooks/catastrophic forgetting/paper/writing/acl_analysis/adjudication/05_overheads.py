"""05 — Overheads: the consolidated "cost of geometry" table.

- TRAIN wall-clock: results/train_registry.jsonl, median per method over the
  lrsw_ Commonsense sweep (fig_efficiency.py convention), normalized to LoRA;
  qwsw_ medians as a second-model replication.
- PEAK GPU MEMORY: peak_mem_train_gb is instrumented only on later runs and is
  BIMODAL within identical method x family cells (41 vs 63 GB on Llama — node/
  batching artifacts), so it is NOT adapter-comparable and is excluded from the
  cost table; the analytical resident comparison substitutes (INTERESTING_
  INSIGHTS.md section 7 reached the same conclusion).
- CLoRA k-memory: frozen-P resident memory = k x 1,753,088 bf16 floats
  ([EXTERNAL] constant, fig_efficiency.py / INTERESTING_INSIGHTS.md section 7).
- INIT tax: [EXTERNAL] measured counts (fleet_findings.md / INTERESTING_
  INSIGHTS.md section 7) — cited, not recomputed.
- PARAMS: trainable_params median per method (registry) + the residual-init
  deployment consideration: MiLoRA/PiSSA/CorDA modify the base weights at
  init, so the deployable checkpoint delta is rank-2r (init subtraction +
  learned adapter), ~2x the adapter bytes, unless the whole base is reshipped.

Outputs: tables/overheads.csv, tables/overheads.md, figures/fig_overheads.{png,pdf}
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 05_overheads.py
"""
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adjpool import ROOT, TABLES, FIGURES, DISPLAY, method_key

sys.path.insert(0, f"{ROOT}/paper/writing")
import figstyle as fs  # noqa: E402

fs.apply_rc()

# ---------------- [EXTERNAL] constants (cited, not recomputed) ----------------
CLORA_FLOATS_PER_K = 1_753_088          # fig_efficiency.py
BYTES_PER_FLOAT = 2                     # bf16
INIT_TAX = {                            # INTERESTING_INSIGHTS.md section 7
    "LoRA": ("none", 0),
    "LoRA+wd": ("none (free AdamW flag)", 0),
    "DoRA": ("none", 0),
    "CLoRA": ("k x d covariance/eigh on base weights (fast); frozen-P build", 1),
    "MiLoRA": ("160 base-weight SVDs (no forwards)", 1),
    "PiSSA": ("160 base-weight SVDs (no forwards)", 1),
    "LoRA-Null": ("256 calibration forwards + eigh", 2),
    "SC-LoRA": ("512 calibration forwards + eigh", 2),
    "CorDA": ("256 calibration forwards + inv/SVD", 2),
    "CorDA++": ("1280 forwards + 5x inv/SVD (~3.5e16 FLOPs, ~22.5 GB transient)", 3),
}
RESIDUAL_INIT = {"MiLoRA", "PiSSA", "CorDA", "CorDA++"}  # base weights modified at init


def mname(run):
    m = method_key(run)
    if m == "lorawd" and "_wd0_" in run:
        m = "lora"
    return DISPLAY.get(m, m)


def med(d):
    return {k: float(np.median(v)) for k, v in d.items() if v}


def run():
    reg = [json.loads(l) for l in
           open(f"{ROOT}/results/train_registry.jsonl") if l.strip()]
    rt, rt_q, params, peak = {}, {}, {}, {}
    for r in reg:
        rn = r["run_name"]
        m = mname(rn)
        if m is None:
            continue
        if rn.startswith("lrsw_") and r.get("train_runtime_s", 0) > 1000:
            rt.setdefault(m, []).append(r["train_runtime_s"])
            if r.get("trainable_params"):
                params.setdefault(m, []).append(r["trainable_params"])
        if rn.startswith("qwsw_") and r.get("train_runtime_s", 0) > 1000:
            rt_q.setdefault(m, []).append(r["train_runtime_s"])
        if r.get("peak_mem_train_gb") and rn.split("_")[0] in (
                "lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"):
            peak.setdefault(m, []).append(r["peak_mem_train_gb"])

    m_rt, m_rtq, m_par, m_peak = med(rt), med(rt_q), med(params), med(peak)
    base, base_q = m_rt["LoRA"], m_rtq.get("LoRA", np.nan)

    order = ["LoRA", "LoRA+wd", "CLoRA", "MiLoRA", "LoRA-Null", "SC-LoRA",
             "DoRA", "PiSSA", "CorDA", "CorDA++"]
    rows = []
    for m in order:
        rel = m_rt.get(m, np.nan) / base if m in m_rt else np.nan
        relq = m_rtq.get(m, np.nan) / base_q if (m in m_rtq and np.isfinite(base_q)) else np.nan
        p = m_par.get(m, np.nan)
        # CLoRA k-memory at the sweep's k (lrsw/qwsw use k1024)
        kmem = 1024 * CLORA_FLOATS_PER_K * BYTES_PER_FLOAT / 1024**3 if m == "CLoRA" else 0.0
        ck_delta = 2 if m in RESIDUAL_INIT else 1     # deployable delta rank factor
        rows.append(dict(
            method=m,
            train_rel_llama=round(rel, 2) if np.isfinite(rel) else None,
            train_rel_qwen=round(relq, 2) if np.isfinite(relq) else None,
            n_runs=len(rt.get(m, [])),
            trainable_params_M=round(p / 1e6, 1) if np.isfinite(p) else None,
            extra_resident_gb=round(kmem, 2),
            init_tax=INIT_TAX.get(m, ("?", 0))[0],
            init_class=INIT_TAX.get(m, ("?", 0))[1],
            deploy_delta_rank_factor=ck_delta,
        ))
    t = pd.DataFrame(rows)
    t.to_csv(f"{TABLES}/overheads.csv", index=False)

    md = ["# Overheads — the cost of geometry (adjudication)",
          "",
          "Wall-clock: live medians from `results/train_registry.jsonl` (lrsw_/qwsw_ runs,",
          "runtime > 1000 s), normalized to LoRA. Init tax + CLoRA per-k memory constant are",
          "[EXTERNAL] measured values (INTERESTING_INSIGHTS.md section 7, fig_efficiency.py) —",
          "cited, not recomputed. init_class: 0 = free, 1 = weight-only SVD/eigh,",
          "2 = needs a calibration forward pass, 3 = heavy multi-pass. deploy_delta_rank_factor:",
          "residual-init methods (MiLoRA/PiSSA/CorDA) modify base weights at init, so the",
          "deployable checkpoint delta is rank-2r (2x adapter bytes) unless the full base is",
          "reshipped. Script: `05_overheads.py`.", "",
          "| Method | train x (Llama) | train x (Qwen) | params (M) | "
          "extra resident GB | init tax | deploy delta |",
          "|---|---|---|---|---|---|---|"]
    for _, x in t.iterrows():
        md.append(f"| {x.method} | {x.train_rel_llama if x.train_rel_llama else '—'} | "
                  f"{x.train_rel_qwen if x.train_rel_qwen else '—'} | "
                  f"{x.trainable_params_M if x.trainable_params_M else '—'} | "
                  f"{x.extra_resident_gb if x.extra_resident_gb else '0'} | {x.init_tax} | "
                  f"{'rank-2r' if x.deploy_delta_rank_factor == 2 else 'rank-r'} |")
    md.append("")
    ks = np.array([128, 256, 512, 1024, 2048])
    mem = ks * CLORA_FLOATS_PER_K * BYTES_PER_FLOAT / 1024**3
    md.append("CLoRA frozen-P memory vs k: " +
              ", ".join(f"k{k}={g:.2f} GB" for k, g in zip(ks, mem)) +
              " (sweeps use k1024 = 3.34 GB; the frc k2048 boundary point pays 6.68 GB).")
    with open(f"{TABLES}/overheads.md", "w") as fh:
        fh.write("\n".join(md))

    # ------------------------------- figure -----------------------------------
    have = [m for m in order if m in m_rt]
    have = sorted(have, key=lambda m: m_rt[m] / base)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                                   gridspec_kw=dict(width_ratios=[1.4, 1.0], wspace=0.3))
    y = np.arange(len(have))
    axL.barh(y, [m_rt[m] / base for m in have],
             color=[fs.color(m) for m in have], edgecolor="white", height=0.62, zorder=3)
    axL.axvline(1.0, color=fs.MUTED, lw=1.2, ls=":", zorder=2)
    axL.set_yticks(y)
    axL.set_yticklabels(have)
    axL.invert_yaxis()
    xmax = max(m_rt[m] / base for m in have) * 1.55
    axL.set_xlim(0, xmax)
    for yi, m in zip(y, have):
        axL.text(m_rt[m] / base + 0.03, yi, f"{m_rt[m]/base:.2f}x", va="center",
                 fontsize=10, fontweight="bold", color=fs.INK)
        axL.text(xmax - 0.04, yi, INIT_TAX[m][0].split(" (")[0].split(";")[0],
                 va="center", ha="right", fontsize=8.4, color=fs.INK2, style="italic")
    axL.set_xlabel("train wall-clock (relative to LoRA, lrsw medians); italics = one-time init tax")
    axL.set_title("Train cost: DoRA pays 2x; the rest run at LoRA speed", loc="left")
    axL.grid(True, axis="x", alpha=0.5)

    axR.plot(ks, mem, color=fs.color("CLoRA"), lw=2.4, marker="v", ms=9,
             markeredgecolor="white", zorder=3)
    for k, g in zip(ks, mem):
        axR.annotate(f"{g:.2f}", (k, g), textcoords="offset points", xytext=(6, 7),
                     fontsize=9, color=fs.INK)
    axR.scatter([1024], [mem[3]], s=200, facecolor="none", edgecolor=fs.INK,
                linewidth=1.6, zorder=4)
    axR.annotate("sweep k=1024", (1024, mem[3]), textcoords="offset points",
                 xytext=(-4, -20), fontsize=9.5, color=fs.INK, fontweight="bold", ha="right")
    axR.set_xscale("log", base=2)
    axR.set_xticks(ks)
    axR.set_xticklabels([str(k) for k in ks])
    axR.set_xlabel("CLoRA null-space projector size k")
    axR.set_ylabel("extra resident memory (GB, bf16)")
    axR.set_title("CLoRA memory tax scales with k", loc="left")
    axR.grid(True, which="both", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{FIGURES}/fig_overheads.png")
    fig.savefig(f"{FIGURES}/fig_overheads.pdf")
    plt.close(fig)

    print(t.to_string(index=False))
    print(f"\nLoRA base wall-clock (lrsw median) = {base:.0f} s; qwsw = {base_q:.0f} s")
    print(f"wrote {TABLES}/overheads.csv/.md + {FIGURES}/fig_overheads.png/.pdf")


if __name__ == "__main__":
    run()
