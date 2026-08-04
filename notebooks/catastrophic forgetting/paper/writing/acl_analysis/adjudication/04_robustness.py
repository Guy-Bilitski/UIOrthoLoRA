"""04 — Robustness trade-offs per method.

(a) SAFE LR BAND: number of sweep LRs whose cell-mean retention stays within
    2pp (and 1pp) of the base ceiling. An LR where ALL seeds diverged
    (quarantined) counts as attempted-and-unsafe. Denominator = LRs attempted.
(b) SEED VARIANCE: median within-cell seed SD of retention and adaptation
    (cells with n>=2).
(c) DIVERGENCE RATE: quarantined / attempted runs per method, from
    results/quarantine_diverged.txt, over the 4 adjudication families
    (spec prefixes) + the frc/frm grids for context.
(d) RETENTION-DEFINITION SENSITIVITY: method ranking (retention at the
    best-adaptation cell) under core vs broad vs BBH-only, CS families.

Outputs: tables/lr_band.csv, tables/seed_variance.csv, tables/divergence.csv,
         tables/ret_definition_sensitivity.csv, figures/fig_lr_band.{png,pdf}
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 04_robustness.py
"""
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adjpool import (FAMILIES, DISPLAY, WITHHELD, TABLES, FIGURES, ROOT,
                     load_pool, preflight_18_1, family_rows, cell_table,
                     best_cell, fmt_lr, method_key)

sys.path.insert(0, f"{ROOT}/paper/writing")
import figstyle as fs  # noqa: E402

fs.apply_rc()


def run():
    df = load_pool()
    n, r = preflight_18_1(df)
    print(f"PREFLIGHT OK: n={n}, r={r:.3f}")

    band_rows, var_rows, sens_rows = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    axpos = {"llama_cs": (0, 0), "llama_math": (0, 1),
             "qwen_cs": (1, 0), "qwen_math": (1, 1)}

    for fk, spec in FAMILIES.items():
        frq = family_rows(df, fk, include_quar=True)   # incl. quarantined
        fr = frq[~frq.quar]
        base = spec["ret_base"]
        rf = spec["ret_field"]
        ax = axes[axpos[fk]]
        diverged_marks = []

        # first pass: per-method per-LR cell means (quarantined-only LRs = NaN)
        fam_perlr = {}
        for mkey, _pref in spec["specs"]:
            if mkey in WITHHELD:
                continue
            sub_all = frq[frq.mkey == mkey]
            lrs = sorted({lr for lr in sub_all.lr.dropna().unique()
                          if lr <= 1e-3})           # sweep band; 2e-3/5e-3 excluded
            per_lr = []
            for lr in lrs:
                g = sub_all[sub_all.lr == lr]
                ok = g[~g.quar][rf].dropna()
                per_lr.append(dict(lr=lr, n=len(ok), nq=int(g.quar.sum()),
                                   ret=(ok.mean() if len(ok) else np.nan)))
            fam_perlr[mkey] = per_lr
        fam_top = max((p["ret"] for pl in fam_perlr.values() for p in pl
                       if np.isfinite(p["ret"])), default=np.nan)

        for mkey, per_lr in fam_perlr.items():
            att = len(per_lr)
            safe2 = sum(1 for p in per_lr if np.isfinite(p["ret"]) and p["ret"] >= base - 2)
            safe1 = sum(1 for p in per_lr if np.isfinite(p["ret"]) and p["ret"] >= base - 1)
            safer = sum(1 for p in per_lr if np.isfinite(p["ret"]) and p["ret"] >= fam_top - 2)
            band_rows.append(dict(family=fk, method=DISPLAY[mkey],
                                  lrs_attempted=att, safe_2pp=safe2, safe_1pp=safe1,
                                  safe_rel2pp=safer,
                                  frac_2pp=round(safe2 / att, 2) if att else np.nan,
                                  fam_top_ret=round(fam_top, 2)))
            # seed variance
            cells = fr[fr.mkey == mkey].groupby("lr")
            sds_r = [g[rf].std(ddof=1) for _, g in cells if len(g[rf].dropna()) >= 2]
            sds_a = [g["adapt"].std(ddof=1) for _, g in cells if len(g["adapt"].dropna()) >= 2]
            var_rows.append(dict(family=fk, method=DISPLAY[mkey],
                                 med_seed_sd_ret=round(float(np.median(sds_r)), 2) if sds_r else np.nan,
                                 max_seed_sd_ret=round(float(np.max(sds_r)), 2) if sds_r else np.nan,
                                 med_seed_sd_adapt=round(float(np.median(sds_a)), 2) if sds_a else np.nan,
                                 n_cells=len(sds_r)))
            # panel line
            xs = [p["lr"] for p in per_lr if np.isfinite(p["ret"])]
            ys = [p["ret"] for p in per_lr if np.isfinite(p["ret"])]
            disp = DISPLAY[mkey]
            cname = "LoRA" if mkey == "lora_r32" else disp
            ax.plot(xs, ys, color=fs.color(cname), marker=fs.marker(cname), ms=7,
                    lw=1.8, markeredgecolor="white", markeredgewidth=0.6,
                    label=disp, alpha=0.5 if mkey == "lora_r32" else 1.0, zorder=3)
            # collect fully-diverged LRs; drawn after ylim is final
            diverged_marks.extend((p["lr"], cname) for p in per_lr
                                  if not np.isfinite(p["ret"]) and p["nq"])

        y0 = ax.get_ylim()[0]
        for lr, cname in diverged_marks:
            ax.plot([lr], [y0], marker="x", ms=8, color=fs.color(cname),
                    zorder=4, clip_on=False)
        ax.axhline(base - 2, color=fs.CEILING_C, lw=1.3, ls="--", zorder=2)
        ax.axhline(base, color=fs.CEILING_C, lw=1.0, ls=":", zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("retention (core)" if rf == "ret_core" else "retention (BBH)")
        ax.set_title(spec["title"].split(" (")[0] + "  (dotted = base, dashed = base-2pp)",
                     loc="left", fontsize=10.5)
        ax.grid(True, which="both", alpha=0.45)

        # (d) retention-definition sensitivity at best-adapt cells
        cells_core = cell_table(fr, "ret_core") if rf == "ret_core" else None
        defs = ([("core", "ret_core"), ("broad", "ret_broad"), ("bbh_only", "bbh")]
                if rf == "ret_core" else [("bbh_only", "bbh"), ("core", "ret_core")])
        picks = {}
        for mkey, _pref in spec["specs"]:
            if mkey in WITHHELD:
                continue
            bc = best_cell(cell_table(fr, rf), mkey)
            if bc is None:
                continue
            g = fr[(fr.mkey == mkey) & (fr.lr == bc.lr)]
            if bc.k:
                g = g[g.run.str.contains(f"_{bc.k}_")]
            picks[mkey] = {dname: g[fld].mean() for dname, fld in defs}
        for dname, _f in defs:
            vals = {m: picks[m][dname] for m in picks if np.isfinite(picks[m][dname])}
            order = sorted(vals, key=lambda m: -vals[m])
            for rank, m in enumerate(order, 1):
                sens_rows.append(dict(family=fk, definition=dname, rank=rank,
                                      method=DISPLAY[m], ret=round(vals[m], 2)))

    seen, H, L = set(), [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                H.append(h)
                L.append(l)
    fig.legend(H, L, loc="lower center", ncol=min(len(L), 9), bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Retention vs learning rate: the safe-LR band per method "
                 "(x on the floor = all seeds diverged at that LR)", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    fig.savefig(f"{FIGURES}/fig_lr_band.png")
    fig.savefig(f"{FIGURES}/fig_lr_band.pdf")
    plt.close(fig)

    band = pd.DataFrame(band_rows)
    var = pd.DataFrame(var_rows)
    sens = pd.DataFrame(sens_rows)
    band.to_csv(f"{TABLES}/lr_band.csv", index=False)
    var.to_csv(f"{TABLES}/seed_variance.csv", index=False)
    sens.to_csv(f"{TABLES}/ret_definition_sensitivity.csv", index=False)

    # ---------------- (c) divergence rates ----------------
    div_rows = []
    fams_all = ["lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"]
    dfa = df[df.family.isin(fams_all)].copy()

    def mkey2(run):
        m = method_key(run)
        # frc/frm 'lorawd_wd0' arms are vanilla LoRA (wd=0), not LoRA+wd;
        # wd0.1/0.2/0.5 grid arms are LoRA+wd VARIANTS, not the adjudicated wd0.3
        if m == "lorawd":
            if "_wd0_" in run:
                return "lora"
            if any(t in run for t in ("_wd0p1_", "_wd0p2_", "_wd0p5_")):
                return "lorawd_var"
        return m
    dfa["mkey2"] = dfa.run.map(mkey2)
    for m, g in dfa.groupby("mkey2"):
        nq = int(g.quar.sum())
        gb = g[g.lr.notna() & (g.lr <= 1e-3)]      # shared sweep band
        nqb = int(gb.quar.sum())
        fam_break = ",".join(f"{f}:{int(gg.quar.sum())}/{len(gg)}"
                             for f, gg in g.groupby("family") if gg.quar.sum())
        div_rows.append(dict(method=DISPLAY.get(m, m), attempted=len(g),
                             quarantined=nq, rate_pct=round(100 * nq / len(g), 1),
                             attempted_band=len(gb), quarantined_band=nqb,
                             rate_band_pct=round(100 * nqb / len(gb), 1) if len(gb) else 0.0,
                             lrs_diverged=",".join(sorted({fmt_lr(x) for x in
                                                           g[g.quar].lr.dropna().unique()})),
                             families_diverged=fam_break))
    div = pd.DataFrame(div_rows).sort_values("rate_pct", ascending=False)
    div.to_csv(f"{TABLES}/divergence.csv", index=False)

    print("\nSAFE LR BAND (full table)")
    print(band.to_string(index=False))
    print("\nSAFE LR BAND pivot (cell-mean retention >= base-2pp)")
    print(band.pivot_table(index="method", columns="family", values="safe_2pp",
                           aggfunc="first").to_string())
    print("\nSEED VARIANCE (median within-cell seed SD of retention)")
    print(var.pivot_table(index="method", columns="family", values="med_seed_sd_ret",
                          aggfunc="first").to_string())
    print("\nDIVERGENCE RATES (6 families: lrsw/lrswm/qwsw/qwswm/frc/frm)")
    print(div.to_string(index=False))
    print("\nRETENTION-DEFINITION SENSITIVITY (rank at best-adapt cell)")
    piv = sens.pivot_table(index=["family", "method"], columns="definition",
                           values="rank", aggfunc="first")
    print(piv.to_string())
    # Kendall tau between definitions per family
    from scipy.stats import kendalltau
    print("\nKendall tau between definition rankings:")
    for fk in FAMILIES:
        sub = sens[sens.family == fk]
        ds = sub.definition.unique()
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a = sub[sub.definition == ds[i]].set_index("method")["rank"]
                b = sub[sub.definition == ds[j]].set_index("method")["rank"]
                common = a.index.intersection(b.index)
                tau = kendalltau(a[common], b[common]).statistic
                print(f"  {fk}: {ds[i]} vs {ds[j]}: tau={tau:.2f} (n={len(common)})")
    print(f"\nwrote lr_band/seed_variance/divergence/ret_definition_sensitivity CSVs + fig_lr_band")


if __name__ == "__main__":
    run()
