"""06 — VERDICT: one decision table.

Rows = methods; columns =
  ret@matched-adapt  : method's retention at its best-adaptation cell MINUS the
                       retention LoRA+wd achieves while matching that adaptation
                       (max ret over LoRA+wd cells with adapt >= method's adapt;
                       "beyond" if no LoRA+wd cell reaches it). Mean over families.
  adapt_ceiling      : method's best cell-mean adaptation minus LoRA+wd's, mean
                       over families.
  lr_band            : safe-LR band (retention within 2pp of base; qwen_cs uses
                       the family-relative band since no method is within 2pp of
                       base there) summed across families, "safe/attempted".
  seed_sd            : median within-cell seed SD of retention across families.
  divergence         : quarantined/attempted over the 6 sweep+grid families.
  train/init/memory  : from 05_overheads.
  beats LoRA+wd?     : head-to-head verdict count (03), families beaten/total.

Outputs: tables/verdict.csv, tables/verdict.md
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 06_verdict.py
"""
import numpy as np
import pandas as pd

from adjpool import (FAMILIES, DISPLAY, WITHHELD, TABLES, load_pool,
                     preflight_18_1, family_rows, cell_table, best_cell)


def run():
    df = load_pool()
    n, r = preflight_18_1(df)
    print(f"PREFLIGHT OK: n={n}, r={r:.3f}")

    # per-family matched-adapt retention gap + adapt ceiling gap
    gap_rows = []
    for fk, spec in FAMILIES.items():
        fr = family_rows(df, fk)
        cells = cell_table(fr, spec["ret_field"])
        wd = cells[cells.mkey == "lorawd"]
        wd_max_adapt = wd.adapt_mean.max()
        for mkey, _ in spec["specs"]:
            if mkey in WITHHELD or mkey == "lorawd":
                continue
            bc = best_cell(cells, mkey)
            if bc is None:
                continue
            match = wd[wd.adapt_mean >= bc.adapt_mean]
            if len(match):
                gap = bc.ret_mean - match.ret_mean.max()
                beyond = False
            else:
                # LoRA+wd never reaches this adaptation: compare vs its
                # best-adapt cell's retention and flag
                gap = bc.ret_mean - wd.sort_values("adapt_mean").iloc[-1].ret_mean
                beyond = True
            gap_rows.append(dict(family=fk, mkey=mkey, method=DISPLAY[mkey],
                                 ret_gap_matched=gap, beyond_wd_ceiling=beyond,
                                 adapt_gap=bc.adapt_mean - wd_max_adapt))
    gaps = pd.DataFrame(gap_rows)

    band = pd.read_csv(f"{TABLES}/lr_band.csv")
    var = pd.read_csv(f"{TABLES}/seed_variance.csv")
    div = pd.read_csv(f"{TABLES}/divergence.csv")
    h2h = pd.read_csv(f"{TABLES}/head2head.csv")
    ovh = pd.read_csv(f"{TABLES}/overheads.csv")

    # qwen_cs band: use relative band (base band is empty family-wide)
    band["band_eff"] = np.where(band.family == "qwen_cs", band.safe_rel2pp, band.safe_2pp)

    methods = ["LoRA+wd", "LoRA", "CLoRA", "MiLoRA", "LoRA-Null", "SC-LoRA",
               "DoRA", "PiSSA"]
    rows = []
    for m in methods:
        g = gaps[gaps.method == m]
        b = band[band.method == m]
        v = var[var.method == m]
        d = div[div.method == m]
        o = ovh[ovh.method == m]
        h = h2h[h2h.method == m]
        if m == "LoRA+wd":
            retg, adg, beats = "reference", "reference", "reference"
        else:
            marks = "".join("!" if x else "" for x in g.beyond_wd_ceiling)
            retg = f"{g.ret_gap_matched.mean():+.1f} pp" + (" (1 beyond)" if g.beyond_wd_ceiling.any() else "")
            adg = f"{g.adapt_gap.mean():+.1f} pp"
            nbeat = (h.beats_lorawd == "YES").sum()
            ndom = (h.beats_lorawd == "dominated").sum()
            beats = (f"{nbeat}/{len(h)} fam" + (f" (dominated in {ndom})" if ndom else ""))
        rows.append(dict(
            method=m,
            ret_at_matched_adapt_vs_wd=retg,
            adapt_ceiling_vs_wd=adg,
            lr_band=f"{int(b.band_eff.sum())}/{int(b.lrs_attempted.sum())}" if len(b) else "—",
            seed_sd_ret=round(float(v.med_seed_sd_ret.median()), 2) if len(v) else None,
            divergence=f"{int(d.quarantined_band.iloc[0])}/{int(d.attempted_band.iloc[0])} "
                       f"({d.rate_band_pct.iloc[0]}%)" if len(d) else "0",
            train_cost=f"{o.train_rel_llama.iloc[0]:.2f}x" if len(o) and np.isfinite(o.train_rel_llama.iloc[0]) else "—",
            init_cost=o.init_tax.iloc[0] if len(o) else "—",
            memory_gb=(f"+{o.extra_resident_gb.iloc[0]:.2f}" if len(o) and o.extra_resident_gb.iloc[0] else "0"),
            deploy_delta=("rank-2r" if len(o) and o.deploy_delta_rank_factor.iloc[0] == 2 else "rank-r"),
            beats_lorawd=beats,
        ))
    t = pd.DataFrame(rows)
    t.to_csv(f"{TABLES}/verdict.csv", index=False)

    md = ["# VERDICT decision table (adjudication, 2026-07-18)", "",
          "ret@matched-adapt = method retention at its best-adaptation cell minus the",
          "retention LoRA+wd delivers at >= that adaptation (its LR is re-chosen; 'beyond' =",
          "LoRA+wd's sweep never reaches that adaptation — only SC-LoRA on Qwen-math).",
          "lr_band = LRs with retention within 2pp of base (qwen_cs: family-relative band),",
          "summed over the 4 families. Divergence = quarantined/attempted runs at LR <= 1e-3",
          "(the shared sweep band) over the 6 sweep+grid families; 2e-3/5e-3 probe cells are",
          "excluded so methods are not penalized for having been probed at extreme LR.",
          "Qwen-math LoRA+wd ceiling note: its sweep-max adaptation cell is lr1e-3 with n=1",
          "(72.93; the two sibling seeds diverged) — SC-LoRA's 'beyond' flag and the qwen_math",
          "adapt gaps are measured against that cell; against the 3-seed rule (3e-4, 68.97)",
          "SC-LoRA's edge grows. Script: `06_verdict.py` (inputs: tables from 01-05).", "",
          "| Method | ret@matched-adapt | adapt ceiling | LR band | seed SD(ret) | "
          "divergence | train | init | memory | deploy | beats LoRA+wd? |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in t.iterrows():
        md.append(f"| {x.method} | {x.ret_at_matched_adapt_vs_wd} | {x.adapt_ceiling_vs_wd} | "
                  f"{x.lr_band} | {x.seed_sd_ret} | {x.divergence} | {x.train_cost} | "
                  f"{x.init_cost} | {x.memory_gb} | {x.deploy_delta} | {x.beats_lorawd} |")
    md += ["", "CorDA/CorDA++: WITHHELD (own port bug — divergence rows 28.6%/36.4% shown in",
           "tables/divergence.csv for completeness, never ranked).", "",
           "Per-family matched-adapt gaps:"]
    for _, x in gaps.iterrows():
        md.append(f"- {x.family} {x.method}: ret gap {x.ret_gap_matched:+.2f} pp"
                  + (" [beyond LoRA+wd adaptation ceiling]" if x.beyond_wd_ceiling else "")
                  + f", adapt gap {x.adapt_gap:+.2f} pp")
    with open(f"{TABLES}/verdict.md", "w") as fh:
        fh.write("\n".join(md))
    print(t.to_string(index=False))
    print("\nPer-family gaps:")
    print(gaps.to_string(index=False))
    print(f"\nwrote {TABLES}/verdict.csv/.md")


if __name__ == "__main__":
    run()
