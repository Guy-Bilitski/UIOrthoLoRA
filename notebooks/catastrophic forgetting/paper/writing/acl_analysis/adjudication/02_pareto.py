"""02 — Pareto analysis per family: adaptation-vs-retention frontier.

- Cells = seed-averaged (method, LR[, k]) recipe cells (quarantine-excluded).
- Frontier = non-dominated cells (maximize adaptation AND retention).
- Bootstrap P(method on frontier): B=2000 replicates, resampling SEEDS WITHIN
  each cell (respects the within-cell seed correlation, ICC~0.78 — seeds are
  never treated as independent across cells; each replicate re-draws every
  cell's seed set and recomputes the frontier).
- CorDA/CorDA++ excluded from frontier (WITHHELD).

Outputs: tables/pareto_frontier.csv, tables/pareto_bootstrap.csv,
         figures/fig_pareto.{png,pdf}
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 02_pareto.py
"""
import math
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adjpool import (FAMILIES, DISPLAY, WITHHELD, TABLES, FIGURES, ROOT,
                     load_pool, preflight_18_1, family_rows, fmt_lr)

sys.path.insert(0, f"{ROOT}/paper/writing")
import figstyle as fs  # noqa: E402

fs.apply_rc()
rng = np.random.default_rng(20260718)
B = 2000


def cells_with_seeds(fr, ret_field):
    """(method, lr, k) -> dict(adapt array, ret array over seeds)."""
    out = {}
    for _, r in fr.iterrows():
        if r.lr is None or (isinstance(r.lr, float) and math.isnan(r.lr)):
            continue
        if r["adapt"] is None or r[ret_field] is None:
            continue
        mk = re.search(r"_(k\d+)_", r["run"])
        k = mk.group(1) if (r["mkey"] == "clora" and mk) else ""
        key = (r["mkey"], r.lr, k)
        out.setdefault(key, {"a": [], "r": []})
        out[key]["a"].append(float(r["adapt"]))
        out[key]["r"].append(float(r[ret_field]))
    return {k: {"a": np.array(v["a"]), "r": np.array(v["r"])} for k, v in out.items()}


def frontier_mask(adapt, ret):
    """Non-dominated mask: no other point has >= on both axes and > on one."""
    n = len(adapt)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dom = (adapt >= adapt[i]) & (ret >= ret[i]) & ((adapt > adapt[i]) | (ret > ret[i]))
        if dom.any():
            mask[i] = False
    return mask


def run():
    df = load_pool()
    n, r = preflight_18_1(df)
    print(f"PREFLIGHT OK: n={n}, pooled r={r:.3f}")

    frontier_rows, boot_rows = [], []
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4))
    axpos = {"llama_cs": (0, 0), "llama_math": (0, 1),
             "qwen_cs": (1, 0), "qwen_math": (1, 1)}

    for fk, spec in FAMILIES.items():
        fr = family_rows(df, fk)
        fr = fr[~fr.mkey.isin(WITHHELD)]
        cw = cells_with_seeds(fr, spec["ret_field"])
        keys = sorted(cw.keys())
        A = np.array([cw[k]["a"].mean() for k in keys])
        R = np.array([cw[k]["r"].mean() for k in keys])
        M = frontier_mask(A, R)
        methods = sorted({k[0] for k in keys})

        # observed frontier membership
        front_cells = [(keys[i], A[i], R[i]) for i in np.where(M)[0]]
        front_cells.sort(key=lambda x: -x[1])
        for (mk, lr, kk), a, rr in front_cells:
            frontier_rows.append(dict(family=fk, method=DISPLAY[mk], lr=fmt_lr(lr),
                                      k=kk, adapt=round(a, 2), ret=round(rr, 2),
                                      n_seeds=len(cw[(mk, lr, kk)]["a"])))

        # bootstrap over seeds-within-cells
        counts = {m: 0 for m in methods}
        for _ in range(B):
            Ab = np.empty(len(keys))
            Rb = np.empty(len(keys))
            for i, k in enumerate(keys):
                a, rr = cw[k]["a"], cw[k]["r"]
                idx = rng.integers(0, len(a), len(a))
                Ab[i], Rb[i] = a[idx].mean(), rr[idx].mean()
            mb = frontier_mask(Ab, Rb)
            on = {keys[i][0] for i in np.where(mb)[0]}
            for m in on:
                counts[m] += 1
        for m in methods:
            boot_rows.append(dict(family=fk, method=DISPLAY[m],
                                  p_on_frontier=counts[m] / B,
                                  n_cells=sum(1 for k in keys if k[0] == m)))

        # ---- panel ----
        ax = axes[axpos[fk]]
        order = [m for m in ["lora", "lorawd", "milora", "lora_null", "clora",
                             "sclora", "dora", "pissa", "lora_r32"] if m in methods]
        for m in order:
            sel = np.array([k[0] == m for k in keys])
            disp = DISPLAY[m]
            c = fs.color("LoRA" if m == "lora_r32" else disp)
            mk_ = fs.marker("LoRA" if m == "lora_r32" else disp)
            ax.scatter(A[sel], R[sel], s=46, color=c, marker=mk_,
                       edgecolor="white", linewidth=0.7, zorder=3, label=disp,
                       alpha=0.9 if m != "lora_r32" else 0.45)
        # frontier staircase
        fx = sorted(zip(A[M], R[M]), key=lambda t: t[0])
        ax.plot([p[0] for p in fx], [p[1] for p in fx], color=fs.INK, lw=1.4,
                ls="--", zorder=2, alpha=0.7)
        ax.axhline(spec["ret_base"], color=fs.CEILING_C, lw=1.2, ls=":", zorder=1)
        ax.text(ax.get_xlim()[0], spec["ret_base"], " base ceiling", va="bottom",
                ha="left", fontsize=8.5, color=fs.CEILING_C)
        # LoRA+wd best-adapt cell highlighted
        lwsel = [i for i, k in enumerate(keys) if k[0] == "lorawd"]
        if lwsel:
            ibest = max(lwsel, key=lambda i: A[i])
            ax.scatter([A[ibest]], [R[ibest]], s=210, facecolor="none",
                       edgecolor=fs.INK, linewidth=1.6, zorder=4)
        ax.set_xlabel(f"adaptation ({spec['adapt_name']})")
        ax.set_ylabel("retention (core)" if spec["ret_field"] == "ret_core"
                      else "retention (BBH)")
        ax.set_title(spec["title"].split(" (")[0], loc="left", fontsize=11.5)
        ax.grid(True, alpha=0.5)

    seen, H, L = set(), [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                H.append(h)
                L.append(l)
    fig.legend(H, L, loc="lower center", ncol=len(L), bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Adaptation vs retention: seed-averaged recipe cells and the Pareto frontier "
                 "(circled = LoRA+wd best-adaptation point)", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.975])
    fig.savefig(f"{FIGURES}/fig_pareto.png")
    fig.savefig(f"{FIGURES}/fig_pareto.pdf")
    plt.close(fig)

    fdf = pd.DataFrame(frontier_rows)
    bdf = pd.DataFrame(boot_rows).sort_values(["family", "p_on_frontier"],
                                              ascending=[True, False])
    fdf.to_csv(f"{TABLES}/pareto_frontier.csv", index=False)
    bdf.to_csv(f"{TABLES}/pareto_bootstrap.csv", index=False)
    print("\nOBSERVED FRONTIER CELLS")
    print(fdf.to_string(index=False))
    print(f"\nBOOTSTRAP P(on frontier), B={B}, seed-within-cell resampling")
    print(bdf.to_string(index=False))
    print(f"\nwrote {TABLES}/pareto_frontier.csv, {TABLES}/pareto_bootstrap.csv, "
          f"{FIGURES}/fig_pareto.png/.pdf")


if __name__ == "__main__":
    run()
