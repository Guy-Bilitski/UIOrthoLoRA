"""R1/R2/R3 readouts for the Tier A intruder slice (analysis-only, new file).

Joins results/intruder/<run>.json (geometry) with results/<run>/summary.json
(retention/adaptation/F_delta) and emits the three planned exhibits:

  R1  per-adapter intruder profile vs update magnitude, sorted by F_delta
      (the "ratio across adapters + trend" table)
  R2  retention vs intruder metrics, partial correlation given log F_delta
      (Spearman; reported per metric with n and the magnitude-only baseline)
  R3  design comparison at matched magnitude (residual of each geometry metric
      against the pooled log F_delta trend, grouped by method)

Ratios are per-slot: count / (n_matrices * k). k=64 window and the canonical
Shuttleworth k=10 window are both reported; energy share is the mean over
matrices of the intruder energy fraction within the top-k.

Usage:  python intruder_report.py [--csv results/intruder/r1_table.csv]
"""
import os
import csv
import json
import math
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
INTR = os.path.join(HERE, "results", "intruder")
RES = os.path.join(HERE, "results")
REF = "baseAll"          # criterion-exact reference (Shuttleworth Def. 3.1)
TAU = 0.5                # canonical threshold


def parse_run(rn):
    """run_name -> (model, method, lr, seed) for grouping."""
    fam = "Llama" if "_frc_" in rn else ("Qwen" if "_qwsw_" in rn else "?")
    meth = next((m for m in ("lorawd", "milora", "sclora", "clora", "lora")
                 if f"_{m}_" in rn or f"_{m}_wd" in rn or rn.endswith(m)), "?")
    for m in ("lorawd", "milora", "sclora", "clora"):
        if f"_{m}" in rn:
            meth = m
            break
    lr = next((p for p in rn.split("_") if p.startswith("lr")), "?")
    seed = next((p for p in rn.split("_") if p.startswith("s4")), "?")
    return fam, meth, lr, seed


def load():
    rows = []
    for fn in sorted(os.listdir(INTR)):
        if not fn.endswith(".json"):
            continue
        rn = fn[:-5]
        # intervention arms and protocol re-evals are NOT slice cells: they must
        # never enter the R1/R2/R3 correlations (they share a source adapter, so
        # they are not independent observations). They are read out by ablation().
        if "__abl" in rn or "__rl" in rn:
            continue
        agg = json.load(open(os.path.join(INTR, fn)))["aggregate"]
        # PROTOCOL CONSISTENCY: every slice cell must be read at the SAME eval
        # protocol or the R2 correlation mixes scales. Cells 2 and 6 were scored
        # on the full battery before the proxy protocol was adopted; their
        # <run>__rl50 re-eval is the comparable one, so prefer it when present.
        protocol = "rl50"
        summ_p = os.path.join(RES, rn + "__rl50", "summary.json")
        if not os.path.exists(summ_p):
            summ_p = os.path.join(RES, rn, "summary.json")
            protocol = "full/native"
        summ = json.load(open(summ_p)) if os.path.exists(summ_p) else {}
        head = summ.get("headline", {})
        nm, k = agg["n_matrices"], agg["topk"]
        fam, meth, lr, seed = parse_run(rn)
        rows.append({
            "run": rn, "model": fam, "method": meth, "lr": lr, "seed": seed,
            "protocol": protocol,
            "fdelta": head.get("fdelta", agg.get("fdelta")),
            "ret": head.get("retention_mean"), "adapt": head.get("cs_avg"),
            "ret_broad": head.get("retention_broad"),
            "sv_max": head.get("dw_sv_max"),
            "n_k64": agg[f"total_intruders_{REF}_t{TAU}"],
            "r_k64": agg[f"total_intruders_{REF}_t{TAU}"] / (nm * k),
            "n_k10": agg[f"total_intruders_k10_{REF}_t{TAU}"],
            "r_k10": agg[f"total_intruders_k10_{REF}_t{TAU}"] / (nm * 10),
            "energy": agg[f"mean_energy_share_{REF}_t{TAU}"],
            "r_k64_t07": agg[f"total_intruders_{REF}_t0.7"] / (nm * k),
            "margin": agg["max_margin_s1"],
            "perp_frac": agg.get("n_matrices_perp_over_base_sk", 0) / nm,
        })
    return [r for r in rows if r["fdelta"] is not None]


# ------------------------------------------------------------------ statistics

def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):                      # average ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            rk[order[t]] = avg
        i = j + 1
    return rk


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    return pearson(rank(x), rank(y))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    if None in (rxy, rxz, ryz):
        return None
    den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return None if den == 0 else (rxy - rxz * ryz) / den


def fmt(v, w=6, p=3):
    return " " * w if v is None else f"{v:{w}.{p}f}"


# ---------------------------------------------------------------------- report

def r1(rows):
    print("\n=== R1: intruder profile vs update magnitude "
          f"(criterion {REF}, tau={TAU}; ratio = intruders / slots) ===")
    print(f"{'run':38s} {'model':5s} {'method':7s} {'F_d':>6s} {'Ret':>6s} "
          f"{'r_k10':>6s} {'r_k64':>6s} {'energy':>6s} {'r@0.7':>6s} {'margin':>7s} {'proto':>10s}")
    for r in sorted(rows, key=lambda r: r["fdelta"]):
        print(f"{r['run']:38s} {r['model']:5s} {r['method']:7s} "
              f"{r['fdelta']:6.3f} {fmt(r['ret'])} {r['r_k10']:6.3f} {r['r_k64']:6.3f} "
              f"{r['energy']:6.3f} {r['r_k64_t07']:6.3f} {r['margin']:7.2f} {r['protocol']:>10s}")
    if len(rows) >= 3:
        lf = [math.log(r["fdelta"]) for r in rows]
        print(f"\n  trend vs log F_delta (Spearman, n={len(rows)}):")
        for key, lab in (("r_k10", "count ratio k10"), ("r_k64", "count ratio k64"),
                         ("energy", "energy share"), ("margin", "spike margin")):
            print(f"    {lab:18s} rho = {fmt(spearman([r[key] for r in rows], lf), 6, 3)}")


def r2(rows):
    have = [r for r in rows if r["ret"] is not None]
    protos = {r["protocol"] for r in have}
    if len(protos) > 1:
        print(f"  WARNING: mixed eval protocols {protos} — retention values are not "
              f"on one scale; correlations below are provisional until all cells "
              f"share a protocol.")
    print(f"\n=== R2: retention vs geometry, controlling for magnitude (n={len(have)}) ===")
    if len(have) < 5:
        print(f"  [need >=5 cells with evals; have {len(have)}] — deferred")
        return
    ret = [r["ret"] for r in have]
    lf = [math.log(r["fdelta"]) for r in have]
    print(f"  baseline: retention ~ log F_delta        rho = {fmt(spearman(ret, lf), 6, 3)}")
    print(f"  {'metric':18s} {'rho(ret,X)':>10s} {'rho(X,logF)':>11s} {'PARTIAL':>8s}")
    for key, lab in (("r_k10", "count ratio k10"), ("r_k64", "count ratio k64"),
                     ("energy", "energy share"), ("r_k64_t07", "count ratio@0.7"),
                     ("margin", "spike margin"), ("perp_frac", "perp>sigma_k frac")):
        x = [r[key] for r in have]
        print(f"  {lab:18s} {fmt(spearman(ret, x), 10, 3)} {fmt(spearman(x, lf), 11, 3)} "
              f"{fmt(partial_spearman(ret, x, lf), 8, 3)}")
    print("  (partial |rho| that stays large => geometry adds signal beyond magnitude)")


def r3(rows):
    print("\n=== R3: design comparison at matched magnitude "
          "(residual vs pooled log-F_delta trend) ===")
    if len(rows) < 6:
        print(f"  [need >=6 cells spanning >=2 methods; have {len(rows)}] — deferred")
        return
    lf = [math.log(r["fdelta"]) for r in rows]
    mlf = sum(lf) / len(lf)
    for key, lab in (("r_k10", "count ratio k10"), ("energy", "energy share")):
        y = [r[key] for r in rows]
        my = sum(y) / len(y)
        den = sum((a - mlf) ** 2 for a in lf)
        b = sum((a - mlf) * (c - my) for a, c in zip(lf, y)) / den if den else 0.0
        by_m = {}
        for r, a, c in zip(rows, lf, y):
            by_m.setdefault(r["method"], []).append(c - (my + b * (a - mlf)))
        print(f"  {lab}:")
        for m, res in sorted(by_m.items()):
            print(f"    {m:8s} n={len(res)}  mean residual = {sum(res)/len(res):+.4f}")


def ablation():
    """Causal read-out. Two magnitude-matched contrasts per source adapter:
         B vs C      : same ||dW|| (B removed the top intruder direction per
                       matrix; C shrank the whole update uniformly instead)
         D vs source : same ||dW|| as the source (D removed the intruder, then
                       rescaled back up)
    Retention higher for the intruder-removed member => geometry carries
    forgetting BEYOND update size. Equal => geometry is a passenger of size.
    Both arms of a pair are scored on IDENTICAL documents, so this is paired."""
    print("\n=== CAUSAL: intruder ablation at matched magnitude ===")
    srcs = ["tia1_frc_lorawd_wd0p3_lr5e4_s43", "tia1_frc_milora_lr1e3_s43"]

    def get(run):
        p = os.path.join(RES, run, "summary.json")
        if not os.path.exists(p):
            return None
        h = json.load(open(p)).get("headline", {})
        return {"ret": h.get("retention_mean"), "adapt": h.get("cs_avg"),
                "fd": h.get("fdelta"), "broad": h.get("retention_broad")}

    any_row = False
    for s in srcs:
        arms = {a: get(f"{s}__abl{a}") for a in ("B", "C", "D")}
        arms["source"] = get(f"{s}__rl50") or get(f"{s}__rl100")
        print(f"\n  {s}")
        print(f"    {'arm':8s} {'F_delta':>8s} {'Ret':>7s} {'Adapt':>7s}  what")
        WHAT = {"B": "intruder removed", "C": "uniform shrink (magnitude control)",
                "D": "intruder removed, renormed", "source": "unmodified"}
        for a in ("source", "D", "B", "C"):
            v = arms.get(a)
            if not v:
                print(f"    {a:8s} {'--':>8s} {'--':>7s} {'--':>7s}  {WHAT[a]} [pending]")
                continue
            any_row = True
            print(f"    {a:8s} {v['fd'] if v['fd'] else 0:8.3f} {v['ret'] if v['ret'] else 0:7.2f} "
                  f"{v['adapt'] if v['adapt'] else 0:7.2f}  {WHAT[a]}")
        if arms["B"] and arms["C"]:
            d = arms["B"]["ret"] - arms["C"]["ret"]
            fdgap = abs((arms["B"]["fd"] or 0) - (arms["C"]["fd"] or 0))
            print(f"    -> B-C retention delta = {d:+.2f} pp   (F_delta gap {fdgap:.4f}; "
                  f"must be ~0 for the control to be valid)")
        if arms["D"] and arms["source"]:
            d = arms["D"]["ret"] - arms["source"]["ret"]
            fdgap = abs((arms["D"]["fd"] or 0) - (arms["source"]["fd"] or 0))
            print(f"    -> D-source retention delta = {d:+.2f} pp   (F_delta gap {fdgap:.4f})")
    if not any_row:
        print("  [no ablation evals scored yet]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    rows = load()
    ablation()
    if not rows:
        print("\nno slice cells scored yet")
        return
    r1(rows)
    r2(rows)
    r3(rows)
    print(f"\n[{len(rows)} adapters scored]")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
