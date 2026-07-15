"""Flag diverged / degenerate adapter runs so the final publication dataset can exclude them.

A cell is quarantined if ANY of:
  - dw_sv_max (magnitude) is NaN/inf                     -> training blew up
  - dw_sv_max > 1000 (sane LoRA updates are ~0.1-100)    -> exploded-but-finite weights
  - fdelta   > 50                                        -> exploded update
  - retention_mean < 3                                   -> catastrophic collapse (no knowledge left)
  - forgetting_ce is NaN/inf                             -> CE on a broken adapter

Writes results/quarantine_diverged.txt (one run_name + reason per line). Idempotent; re-run any time.
These are REAL training divergences (concentrated in the high-LR arms: lr1e3 ~25%, 2e3, 7e4), not
pipeline bugs — they are excluded from analysis, not deleted.
  python flag_diverged.py
"""
import json, glob, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def num(x):
    return isinstance(x, (int, float))


def bad(x):
    return num(x) and (math.isnan(x) or math.isinf(x))


def main():
    flagged = []
    for f in glob.glob(os.path.join(RES, "*", "summary.json")):
        rn = os.path.basename(os.path.dirname(f))
        try:
            h = (json.load(open(f)).get("headline", {}) or {})
        except Exception:
            continue
        mag = h.get("dw_sv_max"); fd = h.get("fdelta"); ret = h.get("retention_mean")
        ce = None
        fp = os.path.join(RES, rn, "forgetting.json")
        if os.path.exists(fp):
            try:
                ce = json.load(open(fp)).get("forgetting_ce")
            except Exception:
                pass
        reasons = []
        if bad(mag):
            reasons.append("nan_magnitude")
        elif num(mag) and mag > 1000:
            reasons.append(f"exploded_magnitude({mag:.0f})")
        if num(fd) and fd > 50:
            reasons.append(f"exploded_fdelta({fd:.0f})")
        if num(ret) and ret < 3:
            reasons.append(f"collapsed_retention({ret:.1f})")
        if bad(ce):
            reasons.append("nan_ce")
        if reasons:
            flagged.append((rn, ",".join(reasons)))
    flagged.sort()
    out = os.path.join(RES, "quarantine_diverged.txt")
    with open(out, "w") as fh:
        for rn, why in flagged:
            fh.write(f"{rn}\t{why}\n")
    print(f"[quarantine] {len(flagged)} diverged/degenerate runs -> {out}")
    for rn, why in flagged[:15]:
        print(f"  {rn}\t{why}")


if __name__ == "__main__":
    main()
