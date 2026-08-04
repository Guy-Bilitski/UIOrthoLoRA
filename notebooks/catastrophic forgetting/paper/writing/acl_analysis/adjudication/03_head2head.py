"""03 — Head-to-head vs LoRA+wd: each method's best-adaptation cell vs
LoRA+wd's best-adaptation cell, per family, on BOTH axes.

Noise model: seeds are shared across cells (42..46) and within-cell seed
correlation is high (ICC~0.78, 09_verification Q1) — so where >=2 common seeds
exist we use PAIRED per-seed differences (mean +/- sd of the paired deltas,
t = mean/se); otherwise Welch on the two seed sets. "Outside noise" = |t| > 2
with >=2 (paired) df proxy — reported as win/tie/loss per axis, and a joint
verdict: BEATS LoRA+wd only if it wins one axis outside noise WITHOUT losing
the other outside noise.

Outputs: tables/head2head.csv, tables/head2head.md
Run: /home/guyb/UIOrthoLoRA/.venv/bin/python 03_head2head.py
"""
import math
import re

import numpy as np
import pandas as pd

from adjpool import (FAMILIES, DISPLAY, WITHHELD, TABLES, load_pool,
                     preflight_18_1, family_rows, cell_table, best_cell, fmt_lr)


def seed_map(fr, mkey, lr, k, ret_field):
    sub = fr[(fr.mkey == mkey) & (fr.lr == lr)]
    if k:
        sub = sub[sub.run.str.contains(f"_{k}_")]
    out = {}
    for _, r in sub.iterrows():
        if r.seed is not None and r["adapt"] is not None and r[ret_field] is not None:
            out[int(r.seed)] = (float(r["adapt"]), float(r[ret_field]))
    return out


def compare(vals_m, vals_w):
    """Return (delta, t, mode) for one axis given {seed: value} dicts."""
    common = sorted(set(vals_m) & set(vals_w))
    if len(common) >= 2:
        d = np.array([vals_m[s] - vals_w[s] for s in common])
        se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan
        t = d.mean() / se if se and se > 0 else np.inf * np.sign(d.mean() or 1)
        return d.mean(), t, f"paired(n={len(d)})"
    a = np.array(list(vals_m.values()))
    b = np.array(list(vals_w.values()))
    dm = a.mean() - b.mean()
    va = a.var(ddof=1) / len(a) if len(a) > 1 else 0.0
    vb = b.var(ddof=1) / len(b) if len(b) > 1 else 0.0
    se = math.sqrt(va + vb)
    t = dm / se if se > 0 else (np.inf * np.sign(dm) if dm else 0.0)
    return dm, t, f"welch({len(a)}v{len(b)})"


def verdict(t, thresh=2.0):
    if not np.isfinite(t):
        return "tie(n=1)"
    if t > thresh:
        return "WIN"
    if t < -thresh:
        return "LOSS"
    return "tie"


def run():
    df = load_pool()
    n, r = preflight_18_1(df)
    print(f"PREFLIGHT OK: n={n}, r={r:.3f}")

    rows = []
    for fk, spec in FAMILIES.items():
        fr = family_rows(df, fk)
        cells = cell_table(fr, spec["ret_field"])
        ref = best_cell(cells, "lorawd")
        vw = {"a": seed_map(fr, "lorawd", ref.lr, ref.k, spec["ret_field"])}
        vw_a = {s: v[0] for s, v in vw["a"].items()}
        vw_r = {s: v[1] for s, v in vw["a"].items()}
        for mkey, _ in spec["specs"]:
            if mkey == "lorawd" or mkey in WITHHELD:
                continue
            bc = best_cell(cells, mkey)
            if bc is None:
                continue
            vm = seed_map(fr, mkey, bc.lr, bc.k, spec["ret_field"])
            vm_a = {s: v[0] for s, v in vm.items()}
            vm_r = {s: v[1] for s, v in vm.items()}
            da, ta, mode_a = compare(vm_a, vw_a)
            dr, tr, mode_r = compare(vm_r, vw_r)
            va, vr = verdict(ta), verdict(tr)
            beats = (("WIN" in (va, vr)) and ("LOSS" not in (va, vr)))
            dominated = (va == "LOSS" and vr == "LOSS")
            rows.append(dict(
                family=fk, method=DISPLAY[mkey],
                method_cell=f"{fmt_lr(bc.lr)}{(' ' + bc.k) if bc.k else ''}",
                ref_cell=f"{fmt_lr(ref.lr)}",
                d_adapt=round(da, 2), t_adapt=round(ta, 2) if np.isfinite(ta) else None,
                adapt_verdict=va,
                d_ret=round(dr, 2), t_ret=round(tr, 2) if np.isfinite(tr) else None,
                ret_verdict=vr, mode=mode_a if mode_a == mode_r else f"{mode_a}/{mode_r}",
                beats_lorawd=("YES" if beats else ("dominated" if dominated else "no")),
            ))
    t = pd.DataFrame(rows)
    t.to_csv(f"{TABLES}/head2head.csv", index=False)

    md = ["# Head-to-head vs LoRA+wd (best-adaptation cells, per family)",
          "",
          "Delta = method - LoRA+wd; paired per-seed where >=2 common seeds (ICC-safe),",
          "Welch otherwise; outside-noise = |t| > 2. 'beats LoRA+wd' = wins >=1 axis",
          "outside noise without losing the other outside noise. Script: `03_head2head.py`.", ""]
    for fk in FAMILIES:
        sub = t[t.family == fk]
        md.append(f"## {FAMILIES[fk]['title']}")
        md.append("")
        md.append("| Method (cell) | dAdapt | t | verdict | dRet | t | verdict | test | beats LoRA+wd? |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for _, x in sub.iterrows():
            md.append(f"| {x.method} ({x.method_cell}) vs wd ({x.ref_cell}) | "
                      f"{x.d_adapt:+.2f} | {x.t_adapt if x.t_adapt is not None else '—'} | {x.adapt_verdict} | "
                      f"{x.d_ret:+.2f} | {x.t_ret if x.t_ret is not None else '—'} | {x.ret_verdict} | "
                      f"{x.mode} | {x.beats_lorawd} |")
        md.append("")
    # tally
    md.append("## Tally (both axes, outside noise)")
    tal = t.groupby("method").agg(
        adapt_wins=("adapt_verdict", lambda v: (v == "WIN").sum()),
        adapt_losses=("adapt_verdict", lambda v: (v == "LOSS").sum()),
        ret_wins=("ret_verdict", lambda v: (v == "WIN").sum()),
        ret_losses=("ret_verdict", lambda v: (v == "LOSS").sum()),
        beats=("beats_lorawd", lambda v: (v == "YES").sum()),
        dominated=("beats_lorawd", lambda v: (v == "dominated").sum()),
        families=("family", "count"))
    md.append("")
    md.append("| method | adapt W/L | ret W/L | beats | dominated | families |")
    md.append("|---|---|---|---|---|---|")
    for m, x in tal.iterrows():
        md.append(f"| {m} | {x.adapt_wins}/{x.adapt_losses} | {x.ret_wins}/{x.ret_losses} | "
                  f"{x.beats} | {x.dominated} | {x.families} |")
    with open(f"{TABLES}/head2head.md", "w") as fh:
        fh.write("\n".join(md))
    print(t.to_string(index=False))
    print("\nTALLY:\n", tal.to_string())
    print(f"\nwrote {TABLES}/head2head.csv, {TABLES}/head2head.md")


if __name__ == "__main__":
    run()
