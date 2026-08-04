"""01 — Operating-point tables per family (METHOD ADJUDICATION).

For every method in each of the 4 model x task families:
  (a) BEST-ADAPTATION cell: LR chosen per method by best MEAN adaptation over
      landed seeds (the mandated fair-sweep rule, 02_operating_points.md),
      preferring cells with n>=2 seeds when any exist;
  (b) MATCHED-RETENTION cut: best adaptation subject to cell-mean retention
      >= base ceiling - 1pp (and a -2pp secondary cut).

Outputs: tables/op_points_<family>.csv, tables/op_points.md
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 01_op_points.py
"""
import pandas as pd

from adjpool import (FAMILIES, DISPLAY, WITHHELD, TABLES, load_pool,
                     preflight_18_1, family_rows, cell_table, best_cell, fmt_lr)

pd.set_option("display.width", 220)


def run():
    df = load_pool()
    n, r = preflight_18_1(df)
    print(f"PREFLIGHT OK: n={n}, pooled r={r:.3f}")

    md = ["# Operating-point tables (adjudication, 2026-07-18)",
          "",
          "Source: `results/*/summary.json`, quarantine-excluded, `_reeval` duplicate dropped.",
          "Rule: best-mean-adaptation LR per method over landed seeds (n>=2 preferred).",
          "Matched-retention cut: best adaptation among cells with mean retention >= base-1pp",
          "(secondary cut at base-2pp). CorDA/CorDA++ WITHHELD (port bug) — shown, never ranked.",
          "`*` on a cut cell = answer-format-collapse seed(s) inside the cell (adaptation",
          "collapsed, retention intact — 02_operating_points.md section 1 note).",
          "Script: `01_op_points.py`.", ""]

    for fk, spec in FAMILIES.items():
        fr = family_rows(df, fk)
        cells = cell_table(fr, spec["ret_field"])
        base = spec["ret_base"]
        recs = []
        for mkey, _pref in spec["specs"]:
            bc = best_cell(cells, mkey)
            if bc is None:
                continue
            sub = cells[cells.mkey == mkey]
            # matched-retention cuts (same n>=2 preference as the best cell;
            # an n=1 fallback is flagged in the table via seeds count)
            def cut(th):
                ok = sub[sub.ret_mean >= th]
                if not len(ok):
                    return None
                multi = ok[ok.n >= 2]
                pick = (multi if len(multi) else ok)
                return pick.sort_values("adapt_mean", ascending=False).iloc[0]
            c1, c2 = cut(base - 1.0), cut(base - 2.0)
            recs.append(dict(
                method=DISPLAY[mkey], withheld=(mkey in WITHHELD),
                best_lr=fmt_lr(bc.lr), k=bc.k,
                adapt=f"{bc.adapt_mean:.2f} ± {bc.adapt_sd:.2f}",
                ret=f"{bc.ret_mean:.2f} ± {bc.ret_sd:.2f}",
                fdelta=f"{bc.fd_mean:.3f}", n_seeds=int(bc.n), seeds=bc.seeds,
                collapse_flag=bool(bc.collapse),
                cut1_lr=(fmt_lr(c1.lr) if c1 is not None else "—"),
                cut1_adapt=(f"{c1.adapt_mean:.2f} ± {c1.adapt_sd:.2f}"
                            + ("*" if c1.collapse else "") + (" (n=1)" if c1.n == 1 else "")
                            if c1 is not None else "none"),
                cut1_ret=(f"{c1.ret_mean:.2f}" if c1 is not None else "—"),
                cut2_lr=(fmt_lr(c2.lr) if c2 is not None else "—"),
                cut2_adapt=(f"{c2.adapt_mean:.2f} ± {c2.adapt_sd:.2f}"
                            + ("*" if c2.collapse else "") + (" (n=1)" if c2.n == 1 else "")
                            if c2 is not None else "none"),
                cut2_ret=(f"{c2.ret_mean:.2f}" if c2 is not None else "—"),
                _adapt_mean=bc.adapt_mean,
            ))
        t = pd.DataFrame(recs).sort_values("_adapt_mean", ascending=False)
        # withheld rows go last
        t = pd.concat([t[~t.withheld], t[t.withheld]])
        t = t.drop(columns=["_adapt_mean"])
        t.to_csv(f"{TABLES}/op_points_{fk}.csv", index=False)

        md.append(f"## {spec['title']}")
        md.append(f"Base retention ceiling = {base} ({spec['ret_field']}); "
                  f"adaptation = {spec['adapt_name']}. Cuts: ret >= {base-1:.2f} / {base-2:.2f}.")
        md.append("")
        md.append("| Method | best LR | " + spec['adapt_name'] + " | ret | F_D | n | flags | "
                  "cut(-1pp): LR / adapt / ret | cut(-2pp): LR / adapt / ret |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for _, x in t.iterrows():
            name = x["method"] + (" [WITHHELD]" if x.withheld else "")
            kk = f" ({x.k})" if x.k else ""
            fl = "collapse-seed" if x.collapse_flag else ""
            md.append(f"| {name} | {x.best_lr}{kk} | {x.adapt} | {x.ret} | {x.fdelta} | "
                      f"{x.n_seeds} | {fl} | {x.cut1_lr} / {x.cut1_adapt} / {x.cut1_ret} | "
                      f"{x.cut2_lr} / {x.cut2_adapt} / {x.cut2_ret} |")
        md.append("")
        print(f"\n===== {fk} =====")
        print(t.drop(columns=["seeds"]).to_string(index=False))

        # convention footnote cells (02_operating_points.md section 1 note):
        if fk == "llama_cs":
            md.append("Note (02_operating_points.md convention): under the s42-best-LR rule, "
                      "DoRA's retention-relevant point is 2e-4 and MiLoRA's is 3e-4 — the "
                      "mean-rule 5e-4 picks are the highest LR whose seeds all avoided the "
                      "answer-format-collapse basin, paying 3–5 pp retention. Alt rows:")
            for mk, alt_lr in [("dora", 2e-4), ("milora", 3e-4)]:
                c = cells[(cells.mkey == mk) & (cells.lr == alt_lr)]
                if len(c):
                    c = c.iloc[0]
                    md.append(f"- {DISPLAY[mk]} @ {fmt_lr(alt_lr)}: adapt {c.adapt_mean:.2f} ± "
                              f"{c.adapt_sd:.2f} (format-collapse seeds, retention intact), "
                              f"ret {c.ret_mean:.2f} ± {c.ret_sd:.2f}, n={int(c.n)}")
            md.append("")
        if fk in ("llama_cs", "llama_math"):
            md.append("Note (E4, section 18.3): SC-LoRA's Llama retention deficit at its "
                      "calibrated points is a calibration-set artifact — eval-matched "
                      "calibration puts it +0.92 pp ABOVE the family curve (n=20). Do not "
                      "read its below-LoRA+wd retention here as method geometry.")
            md.append("")

    with open(f"{TABLES}/op_points.md", "w") as fh:
        fh.write("\n".join(md))
    print(f"\nwrote {TABLES}/op_points.md and per-family CSVs")


if __name__ == "__main__":
    run()
