#!/usr/bin/env python
"""FIGURE 3 - EFFICIENCY: fancy adapters pay strictly more for the same budget.

Left  : per-adapter TRAIN wall-clock relative to LoRA (read live from
        results/train_registry.jsonl, median over the Commonsense LR sweep) +
        each adapter's one-time data-aware INIT tax annotated (cited).
Right : CLoRA's frozen-P resident MEMORY tax vs k, computed from the
        per-k coefficient (k x 1,753,088 bf16 floats) -> 0.42 GB (k128) ..
        6.7 GB (k2048); default k512 highlighted, trainable-weights scale line.

Wall-clock is read live; the init-tax counts and CLoRA per-k coefficient are
[EXTERNAL] measured constants from the efficiency analysis (fleet_findings.md /
tasks/ae2310183ffb6dd65.output) and are labelled as such.
Supports paper section: Efficiency / cost taxonomy.
"""
import json
import re
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs
fs.apply_rc()

# ---------------------------------------------- LIVE: per-method wall-clock ----
import os
reg = [json.loads(l) for l in open(os.path.join(fs.ROOT, "results", "train_registry.jsonl")) if l.strip()]
runtime = {}
params = {}
for r in reg:
    rn = r["run_name"]
    if not rn.startswith("lrsw_"):
        continue
    m = fs.method_from_run(rn)
    if m is None:
        continue
    if r.get("train_runtime_s"):
        runtime.setdefault(m, []).append(r["train_runtime_s"])
    if r.get("trainable_params"):
        params.setdefault(m, []).append(r["trainable_params"])

med = {m: float(np.median(v)) for m, v in runtime.items()}
base_t = med["LoRA"]
rel = {m: med[m] / base_t for m in med}

# ------ [EXTERNAL] one-time init tax + CLoRA memory coefficient (cited) --------
# source: efficiency analysis, fleet_findings.md ("EFFICIENCY + MEMORY analyst")
INIT_TAX = {
    "LoRA": "no init",
    "LoRA+wd": "no init (free AdamW flag)",
    "DoRA": "no init",
    "MiLoRA": "160 base-W SVDs",
    "LoRA-Null": "256 calib fwd + eigh",
    "SC-LoRA": "512 calib fwd + eigh",
    "CorDA": "256 calib fwd + inv/SVD",
}
CLORA_FLOATS_PER_K = 1_753_088     # P_u(out x k)+P_v(in x k) summed over modules
BYTES_PER_FLOAT = 2                # bf16

order = ["LoRA", "LoRA+wd", "DoRA", "MiLoRA", "LoRA-Null", "SC-LoRA", "CorDA"]
order = [m for m in order if m in rel]
order = sorted(order, key=lambda m: rel[m])   # ascending wall-clock

# ------------------------------------------------------------------ figure -----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.4),
                               gridspec_kw=dict(width_ratios=[1.35, 1.0],
                                                wspace=0.32))

# ----- LEFT: wall-clock + init tax table -----
y = np.arange(len(order))
bars = axL.barh(y, [rel[m] for m in order],
                color=[fs.color(m) for m in order],
                edgecolor="white", linewidth=1.0, zorder=3, height=0.62)
axL.axvline(1.0, color=fs.MUTED, lw=1.2, ls=":", zorder=2)
axL.set_yticks(y)
axL.set_yticklabels(order)
axL.set_xlabel("train wall-clock  (relative to LoRA)")
axL.set_title("Train cost: DoRA pays 2x; the rest are LoRA-speed", loc="left")
axL.set_xlim(0, max(rel.values()) * 1.5)
axL.grid(True, axis="x", alpha=0.5)
axL.invert_yaxis()
for yi, m in zip(y, order):
    axL.text(rel[m] + 0.03, yi, f"{rel[m]:.2f}x", va="center", ha="left",
             fontsize=10, fontweight="bold", color=fs.INK)
    axL.text(max(rel.values()) * 1.5 - 0.05, yi, INIT_TAX[m], va="center",
             ha="right", fontsize=8.4, color=fs.INK2, style="italic")
axL.text(0.985, 0.985, "one-time init tax", transform=axL.transAxes,
         ha="right", va="top", fontsize=8.8, color=fs.INK2, style="italic",
         fontweight="bold")

# ----- RIGHT: CLoRA memory vs k -----
ks = np.array([128, 256, 512, 1024, 2048])
mem_gb = ks * CLORA_FLOATS_PER_K * BYTES_PER_FLOAT / (1024 ** 3)   # GiB
axR.plot(ks, mem_gb, color=fs.color("CLoRA"), lw=2.4, marker="v", ms=9,
         markeredgecolor="white", markeredgewidth=0.8, zorder=3)
for k, g in zip(ks, mem_gb):
    axR.annotate(f"{g:.2f} GB", (k, g), textcoords="offset points",
                 xytext=(6, 8), fontsize=9, color=fs.INK)
# default k highlight
axR.scatter([512], [mem_gb[2]], s=200, facecolor="none",
            edgecolor=fs.INK, linewidth=1.6, zorder=4)
axR.annotate("default k=512", (512, mem_gb[2]), textcoords="offset points",
             xytext=(10, -18), fontsize=9.5, color=fs.INK, fontweight="bold")
# trainable-adapter-weights scale reference
tw_gb = float(np.median(params.get("CLoRA", [56_098_816]))) * BYTES_PER_FLOAT / (1024 ** 3)
axR.axhline(tw_gb, color=fs.MUTED, lw=1.3, ls="--", zorder=2)
axR.text(ks.max(), tw_gb + 0.06, f"trainable adapter weights ≈ {tw_gb:.2f} GB",
         ha="right", va="bottom", fontsize=8.8, color=fs.INK2)
axR.set_xscale("log", base=2)
axR.set_xticks(ks)
axR.set_xticklabels([str(k) for k in ks])
axR.set_xlabel("CLoRA null-space projector size  k")
axR.set_ylabel("additional resident memory  (GB, bf16)")
axR.set_title("CLoRA memory tax scales with k", loc="left")
axR.grid(True, which="both", alpha=0.5)
axR.set_ylim(0, mem_gb.max() * 1.12)

pdf, png = fs.save(fig, "fig_efficiency")
plt.close(fig)

# --------------------------------------------------------------- report --------
print("FIGURE 3  efficiency")
print("  wall-clock relative to LoRA (live, median lrsw_):")
for m in order:
    print(f"     {m:10s} {rel[m]:.2f}x   (init: {INIT_TAX[m]})")
print("  CLoRA frozen-P memory vs k (GiB, from k x 1,753,088 bf16 floats):")
for k, g in zip(ks, mem_gb):
    print(f"     k={k:5d}  {g:.2f} GB")
print(f"  trainable-adapter-weights reference = {tw_gb:.2f} GB")
print(f"  wrote {pdf}\n        {png}")
