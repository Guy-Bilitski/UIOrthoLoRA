"""03 — Single-metric retention-prediction league table.

For each candidate metric alone: R2 of [family FE + metric] pooled, the dR2 over
family FE, cluster-robust t (cluster = recipe cell, 09_verification Q1), per-family
plain R2, and a same-sample ranking on the CE∩geometry pool (n=911) so KL/CE
compete fairly. LR is included as a predictor (task 7, extends §18.5).

Outputs: league_table.csv, league_table.md
"""
import numpy as np
import pandas as pd
import corr_common as cc

df, _ = cc.build(dedupe=True, verbose=False)
OUT = cc.OUT

METRICS = ["logfd", "lspec", "logfro", "kl", "ce", "loglr",
           "stable_rank_w", "eff_rank_w", "e_top_w", "e_bot_w", "amp_top_w"]
BLOCK = {"logfd": "magnitude", "lspec": "magnitude", "logfro": "magnitude",
         "kl": "CE drift", "ce": "CE drift", "loglr": "training knob",
         "stable_rank_w": "geometry", "eff_rank_w": "geometry",
         "e_top_w": "geometry", "e_bot_w": "geometry", "amp_top_w": "geometry"}

ce_pool = df.dropna(subset=["kl"] + cc.GEO_BLOCK + ["logfd", "loglr"]).copy()

rows = []
for m in METRICS:
    full = cc.fe_r2(df, [m])                    # pooled, family FE, full coverage
    same = cc.fe_r2(ce_pool, [m])               # same-sample n=911
    # seed-averaged cells (design-effect-free granularity)
    cells = df.dropna(subset=[m]).groupby(["fam", "cell"], as_index=False)[[m, "ret"]].mean()
    cellfit = cc.fe_r2(cells, [m], cluster="cell")
    fam_r2 = {}
    for fam in cc.FAMS:
        s = df[(df.fam == fam)].dropna(subset=[m])
        r = np.corrcoef(s[m], s.ret)[0, 1] if len(s) > 2 else np.nan
        fam_r2[fam] = r * r
    rows.append(dict(metric=m, label=cc.VLAB[m], block=BLOCK[m],
                     n=full["n"], dr2_fe=full["dr2"], r2_fe=full["r2"],
                     t_cluster=full["t"][m], n911_dr2=same["dr2"],
                     t911=same["t"][m], cell_dr2=cellfit["dr2"], **fam_r2))

L = pd.DataFrame(rows).sort_values("n911_dr2", ascending=False).reset_index(drop=True)
L.insert(0, "rank", np.arange(1, len(L) + 1))
L.to_csv(OUT + "/league_table.csv", index=False)

md = ["# Single-metric league table — predicting retention (core)", "",
      "Ranked by same-sample dR2 over family FE on the CE∩geometry pool (n=911) so",
      "CE/KL compete on the identical sample. `t (cluster)` = cluster-robust t of the",
      "metric in the pooled family-FE model (cluster = recipe cell, G≈340; per 09 Q1",
      "never quote naive F/t). Per-family columns are plain single-family R2.", "",
      "| rank | metric | block | dR2 (n=911) | t (cluster, n=911) | dR2 full pool (n) | cell-level dR2 | " +
      " | ".join(f"R2 {f}" for f in cc.FAMS) + " |",
      "|" + "---|" * (13)]
for _, r in L.iterrows():
    md.append("| %d | %s | %s | %+.3f | %+.1f | %+.3f (%d) | %+.3f | %s |" % (
        r["rank"], r.label, r.block, r.n911_dr2, r.t911, r.dr2_fe, r.n, r.cell_dr2,
        " | ".join("%.3f" % r[f] for f in cc.FAMS)))

md += ["", "Notes:",
       "- LR (log10) ranks below every magnitude measure — consistent with §18.5",
       "  (LR is a proxy; F_delta is the variable).",
       "- spec_max sits in the magnitude block (r=+0.93 with log F_delta; 06 §5, 09 Q1c).",
       "- raw CE vs KL nearly identical here because family FE absorbs base-entropy",
       "  offsets; cross-family comparability still requires KL (§18.6)."]
open(OUT + "/league_table.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
