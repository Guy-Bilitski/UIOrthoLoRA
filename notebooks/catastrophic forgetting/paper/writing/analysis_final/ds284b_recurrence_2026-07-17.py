"""DeepSeek-V4-Flash 284B arm: geometry recurrence + adapt-vs-magnitude
(2026-07-17, addendum §19; framing per handoff/41 TODO-2).

The 284B arm (7 methods x 3 seeds, r16/alpha32, MedMCQA, per-method fixed LR) lost
its retention/CE/F_delta evals in the 07-17 evacuation. What survives offline:
21/21 factor-only geometry rows (results/geo_drift/adapter_metrics_deepseek.jsonl)
and 20/21 MedMCQA adapt scores (results/dsv4_adapt_n1000_logscores.jsonl).
This is therefore a RECURRENCE analysis (does the per-method geometry fingerprint
seen at 7B recur at 284B/MLA?) plus an adapt-vs-magnitude-proxy relation.
It is explicitly NOT a retention slope: no retention exists for this arm and none
is inferred. Limitation: retention/CE/F_delta recoverable only by GPU re-eval of
the evacuated adapters (results/ds_adapters_evac/, integrity-verified 21/21).

Method: per-method mean (3 seeds) of the shared factor metrics at 284B
(stable_rank_w, eff_rank_w, log10 spec_max, log10 fro_total) vs per-method median
at 7B within each of the 6 frozen families; Spearman rank-correlation of the
7-method ordering across scales. Scale-dependent metrics (fro, spec) carry the
per-method-LR caveat at 284B (LR was fixed per method, not swept). Adapt relation:
medmcqa_acc vs log10 fro_total over the scored cells; the diverged run
dsv4_lora_null_r16_lr5e4_s44 (adapt ~ chance 25.7) is excluded from the primary
fit and reported in a sensitivity line. Pure stdlib.
"""
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ladder", os.path.join(HERE, "ladder_2026-07-17.py"))
ladder = importlib.util.module_from_spec(spec)
sys.modules["ladder"] = ladder
spec.loader.exec_module(ladder)

RES = "results"
METHODS = ["lora", "lorawd", "dora", "sclora", "clora", "milora", "lora_null"]
DIVERGED = "dsv4_lora_null_r16_lr5e4_s44"
METRICS = ["stable_rank_w", "eff_rank_w", "lspec", "lfro"]


def method_of_ds(rn):
    body = rn.split("_", 1)[1]
    return "lora_null" if body.startswith("lora_null") else body.split("_")[0]


def load_ds_geo():
    rows = []
    with open(os.path.join(RES, "geo_drift", "adapter_metrics_deepseek.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            rn = d.get("run") or d.get("run_name")
            rows.append(dict(rn=rn, method=method_of_ds(rn),
                             stable_rank_w=d["stable_rank_w"], eff_rank_w=d["eff_rank_w"],
                             lspec=math.log10(d["spec_max"]), lfro=math.log10(d["fro_total"])))
    return rows


def load_ds_adapt():
    a = {}
    with open(os.path.join(RES, "dsv4_adapt_n1000_logscores.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            a[d["run_name"]] = d["medmcqa_acc"]
    return a


def load_7b_geo_rows():
    rows = []
    with open(os.path.join(RES, "geo_drift", "adapter_metrics_merged.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            rn = d.get("run")
            if not rn:
                continue
            fam = rn.split("_")[0]
            if fam not in ladder.FROZEN:
                continue
            meth = method_of_ds(rn)
            if meth not in METHODS:
                continue
            if d.get("spec_max", 0) <= 0 or d.get("fro_total", 0) <= 0:
                continue
            rows.append(dict(fam=fam, method=meth,
                             stable_rank_w=d["stable_rank_w"], eff_rank_w=d["eff_rank_w"],
                             lspec=math.log10(d["spec_max"]), lfro=math.log10(d["fro_total"])))
    return rows


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def spearman_of_orderings(a, b):
    keys = [k for k in METHODS if k in a and k in b]
    pairs = [(a[k], b[k]) for k in keys]
    r, n = ladder.spearman(pairs) if hasattr(ladder, "spearman") else (float("nan"), 0)
    return r, len(keys)


# ladder.py has no spearman; implement here
def spearman(pairs):
    n = len(pairs)
    if n < 3:
        return float("nan"), n
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    return ladder.pearson(list(zip(rx, ry)))


def main():
    ds = load_ds_geo()
    adapt = load_ds_adapt()
    b7 = load_7b_geo_rows()
    print("=" * 78)
    print("DEEPSEEK-284B GEOMETRY RECURRENCE + ADAPT RELATION — addendum, 2026-07-17")
    print("=" * 78)
    print(f"284B geometry rows: {len(ds)}/21 | adapt scores: {len(adapt)}/21 "
          f"(missing: dsv4_lorawd_r16_lr5e4_s42) | diverged: {DIVERGED}")

    # ---- 284B per-method fingerprints (mean over seeds) ----
    ds_m = defaultdict(lambda: defaultdict(list))
    for r in ds:
        for t in METRICS:
            ds_m[r["method"]][t].append(r[t])
    print(f"\n[284B] per-method mean (3 seeds) — factor-only metrics")
    print(f"{'method':<10} {'stable_rank':>12} {'eff_rank':>10} {'log10 spec':>11} {'log10 fro':>10} {'adapt(mean)':>12}")
    ds_fp = {t: {} for t in METRICS}
    for m in METHODS:
        vals = {t: sum(ds_m[m][t]) / len(ds_m[m][t]) for t in METRICS}
        for t in METRICS:
            ds_fp[t][m] = vals[t]
        ad = [adapt[r["rn"]] for r in ds if r["method"] == m and r["rn"] in adapt and r["rn"] != DIVERGED]
        adm = sum(ad) / len(ad) if ad else float("nan")
        print(f"{m:<10} {vals['stable_rank_w']:>12.2f} {vals['eff_rank_w']:>10.2f} "
              f"{vals['lspec']:>11.2f} {vals['lfro']:>10.2f} {adm:>12.1f}")

    # ---- 7B per-family per-method medians + Spearman vs 284B ----
    b7_m = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in b7:
        for t in METRICS:
            b7_m[r["fam"]][t][r["method"]].append(r[t])
    print("\n[RECURRENCE] Spearman rank-corr of 7-method ordering, 284B vs each 7B family")
    print("(shape metrics stable_rank/eff_rank are the meaningful test; spec/fro at 284B")
    print(" carry the per-method fixed-LR caveat)")
    print(f"{'family':<8}" + "".join(f"{t:>15}" for t in METRICS))
    pooled = {t: defaultdict(list) for t in METRICS}
    for fam in ladder.FAMS:
        line = f"{fam:<8}"
        for t in METRICS:
            med = {m: median(v) for m, v in b7_m[fam][t].items() if len(v) >= 2}
            for m, x in med.items():
                pooled[t][m].append(x)
            keys = [m for m in METHODS if m in med]
            if len(keys) >= 5:
                r, n = spearman([(ds_fp[t][m], med[m]) for m in keys])
                line += f"{r:>+11.2f}(n{n})"
            else:
                line += f"{'—':>15}"
        print(line)
    line = f"{'POOLED':<8}"
    for t in METRICS:
        med = {m: median(v) for m, v in pooled[t].items() if v}
        keys = [m for m in METHODS if m in med]
        r, n = spearman([(ds_fp[t][m], med[m]) for m in keys])
        line += f"{r:>+11.2f}(n{n})"
    print(line)

    print("\n[7B fingerprints] pooled per-method median (median of family medians)")
    print(f"{'method':<10} {'stable_rank':>12} {'eff_rank':>10} {'log10 spec':>11} {'log10 fro':>10}")
    for m in METHODS:
        vals = []
        for t in METRICS:
            med = {mm: median(v) for mm, v in pooled[t].items() if v}
            vals.append(med.get(m, float("nan")))
        print(f"{m:<10} {vals[0]:>12.2f} {vals[1]:>10.2f} {vals[2]:>11.2f} {vals[3]:>10.2f}")

    # ---- adapt vs magnitude proxy ----
    print("\n[ADAPT] medmcqa_acc vs log10 fro_total (fro is the only magnitude proxy; F_delta lost)")
    pts = [(r["lfro"], adapt[r["rn"]], r["rn"]) for r in ds if r["rn"] in adapt]
    prim = [(x, y) for x, y, rn in pts if rn != DIVERGED]
    r_p, n_p = ladder.pearson(prim)
    r_s, _ = spearman(prim)
    print(f"  primary (diverged excluded): n={n_p}  pearson r={r_p:+.3f}  spearman={r_s:+.3f}")
    allp = [(x, y) for x, y, _ in pts]
    r_pa, n_pa = ladder.pearson(allp)
    r_sa, _ = spearman(allp)
    print(f"  sensitivity (incl diverged): n={n_pa}  pearson r={r_pa:+.3f}  spearman={r_sa:+.3f}")
    print(f"  adapt range (excl diverged): {min(y for _, y in prim):.1f}–{max(y for _, y in prim):.1f} "
          f"(diverged {DIVERGED}: {adapt.get(DIVERGED, float('nan')):.1f} ~ chance)")
    print("\nLIMITATION: no retention/CE/F_delta at 284B (lost in evacuation); this arm")
    print("contributes method-geometry recurrence and adaptation only. GPU re-eval of")
    print("results/ds_adapters_evac/ required for any 284B retention claim.")


if __name__ == "__main__":
    main()
