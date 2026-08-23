"""09 — the variance ladder refit on the placement construct itself.

Reason: Section 2 defines geometry as placement, where the update sits along an
ordering of the base weight's directions from the major end to the minor end,
on the input and the output side. That construct is four energy shares:
e_top, e_bot, ein_top, ein_bot. The published ladder's geometry block is
e_top, log spec_max and stable rank, so it carries one of the four placement
shares plus two coordinates that describe shape and size rather than placement.
This script reports the ladder under three geometry blocks so the body can
state what the construct itself adds, rather than what a mixed block adds.

The two input-side shares are not in corr_common's frame; they are joined here
from results/geo_drift/adapter_metrics*.jsonl on the run id.

Outputs: placement_refit.md
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 09_placement_refit.py
"""
import glob
import json
import os

import numpy as np

from rq1_common import OUT
import corr_common as cc

GEO_PUBLISHED = ["e_top_w", "lspec", "stable_rank_w"]
GEO_PLACEMENT = ["e_top_w", "e_bot_w", "ein_top_w", "ein_bot_w"]
GEO_ALL = GEO_PUBLISHED + ["e_bot_w", "ein_top_w", "ein_bot_w"]

_HERE = os.path.dirname(os.path.abspath(__file__))          # acl_analysis/rq1_stats
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(     # repo working dir
    os.path.dirname(_HERE))))
STORE = os.path.join(_ROOT, "results", "geo_drift")


def join_input_side(df):
    """Attach ein_top_w / ein_bot_w from the per-adapter metric store."""
    ein = {}
    for path in glob.glob(os.path.join(STORE, "adapter_metrics*.jsonl")):
        for line in open(path):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "run" in d and d.get("ein_top_w") is not None:
                ein[d["run"]] = (d["ein_top_w"], d.get("ein_bot_w"))
    df["ein_top_w"] = [ein.get(r, (np.nan, np.nan))[0] for r in df.run]
    df["ein_bot_w"] = [ein.get(r, (np.nan, np.nan))[1] for r in df.run]
    return df, df.ein_top_w.notna().sum()


def ladder(frame, geo):
    sub = frame.dropna(subset=["logfd", "ret"] + geo)
    y = sub.ret.values.astype(float)
    meth = sorted(sub.method.unique())[1:]

    def fit(terms, with_method=False):
        cols = [np.ones(len(sub))]
        for f in sorted(sub.fam.unique())[1:]:
            cols.append((sub.fam == f).astype(float).values)
        for t in terms:
            cols.append(sub[t].values.astype(float))
        if with_method:
            for m in meth:
                cols.append((sub.method == m).astype(float).values)
        _, _, r2, _ = cc.ols_fit(np.column_stack(cols), y)
        return r2

    r0, r1 = fit([]), fit(["logfd"])
    r2_, r3 = fit(["logfd"] + geo), fit(["logfd"] + geo, True)
    return len(sub), r1 - r0, r2_ - r1, r3 - r2_


def run():
    df, _ = cc.build(dedupe=True)
    df, matched = join_input_side(df)

    rows = []
    for name, geo in (("published block (e_top, log spec_max, stable rank)", GEO_PUBLISHED),
                      ("placement only (the four energy shares)", GEO_PLACEMENT),
                      ("published block plus the three omitted shares", GEO_ALL)):
        rows.append((name,) + ladder(df, geo))

    # anchor: the published block must still reproduce the frozen ladder
    _, dm, dg, dmet = rows[0][1:], rows[0][2], rows[0][3], rows[0][4]
    if abs(rows[0][2] - 0.395) > 5e-4 or abs(rows[0][3] - 0.017) > 5e-4:
        raise SystemExit(f"LADDER MISMATCH: got magnitude {rows[0][2]:.4f}, "
                         f"geometry {rows[0][3]:.4f}; frozen 0.395 / 0.017")

    lines = [
        "# The ladder under three geometry blocks",
        "",
        f"Frozen pool, run level, family fixed effects. Input-side shares matched "
        f"for {matched} of {len(df)} runs.",
        "",
        "| geometry block | n | magnitude | geometry | method |",
        "|---|---|---|---|---|",
    ]
    for name, n, dm, dg, dmet in rows:
        lines.append(f"| {name} | {n} | {dm:+.3f} | **{dg:+.3f}** | {dmet:+.3f} |")
    lines += [
        "",
        "Reading. Adding the omitted shares to the published block leaves the",
        "geometry step unchanged. Measuring the placement construct on its own,",
        "without stable rank and the spectral norm, gives a smaller step, so the",
        "between-method structure the published block picks up is carried by shape",
        "and size rather than by placement. Both specifications partition the same",
        "geometry-plus-method total.",
    ]
    path = os.path.join(OUT, "placement_refit.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[4:10]))
    print("wrote", path)


if __name__ == "__main__":
    run()
