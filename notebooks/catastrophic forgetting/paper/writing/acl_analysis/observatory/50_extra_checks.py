#!/usr/bin/env python
"""METRIC OBSERVATORY — supporting checks quoted in findings.md.

1. Cross-pipeline agreement: spec_max (geo_drift pipeline) vs dw_sv_max
   (summary.json headline) are the same measurement.
2. Method spread inside matched log10(F_Delta) bins for retention_core
   (max-min of per-method bin means) -> m1_matched_spread.{csv,md}.
3. Best-adaptation op-point KL range per family (from m4_op_points.csv).
"""
import os

import numpy as np
import pandas as pd

import obs_common as oc

OUT = oc.OUT


def main():
    df = oc.load_master()
    p = df[df.on_pool]

    ok = p.log10_spec_max.notna() & p.log10_dw_sv_max.notna()
    r = np.corrcoef(p.log10_spec_max[ok], p.log10_dw_sv_max[ok])[0, 1]
    print(f"[check1] r(log spec_max [geo pipeline], log dw_sv_max "
          f"[summary headline]) = {r:.4f} (n={int(ok.sum())}) — same "
          f"measurement, two pipelines")

    mb = pd.read_csv(os.path.join(OUT, "m1_matched_fdelta.csv"))
    sp = (mb.groupby(["family", "fd_bin"])
            .agg(n_methods=("method", "nunique"), n_runs=("n", "sum"),
                 spread_pp=("mean", lambda s: s.max() - s.min()))
            .reset_index())
    sp["spread_pp"] = sp["spread_pp"].round(2)
    sp.to_csv(os.path.join(OUT, "m1_matched_spread.csv"), index=False)
    oc.write_md(sp, os.path.join(OUT, "m1_matched_spread.md"),
                "M1 method spread (max-min of per-method mean retention_core)"
                " inside matched log10(F_Delta) bins",
                "Small spread = methods indistinguishable at matched "
                "magnitude. Above-knee bins (fd_bin >= 0.0, and qwswm -0.5) "
                "are unstable/collapse territory with few runs per method.")
    print("[check2]\n" + sp.to_string(index=False))

    opt = pd.read_csv(os.path.join(OUT, "m4_op_points.csv"))
    g = (opt.groupby("family")["forgetting_kl_mean"]
            .agg(["min", "max", "count"]).round(3))
    g["ratio"] = (g["max"] / g["min"]).round(1)
    print("[check3] op-point KL range per family (best-adapt cells):")
    print(g.to_string())


if __name__ == "__main__":
    main()
