#!/usr/bin/env python
"""Frontier status: all finalized runs (CS / retention / leakage) vs the CLoRA bar.
Reads results/*/summary.json. UIO retention is FAST scale (~full + 0.9)."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "results", "*", "summary.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    h = d.get("headline", {})
    if h.get("cs_avg") is None and h.get("retention_mean") is None:
        continue
    c = d.get("config", {}) or {}
    rows.append({
        "run": d.get("run_name", os.path.basename(os.path.dirname(f))),
        "method": d.get("method", "?"),
        "kval": c.get("k_val"), "kvec": c.get("k_vec"), "dE": c.get("use_de"),
        "cs": h.get("cs_avg"), "ret": h.get("retention_mean"),
        "bbh": h.get("bbh"), "mmlu": h.get("mmlu_pro"),
        "fd": h.get("fdelta"), "muE": h.get("mu_E"), "nuD": h.get("nu_D"),
    })

def fnum(x, w=5, p=2):
    return ("%*.*f" % (w, p, x)) if isinstance(x, (int, float)) else (" " * (w - 1) + "-")

rows.sort(key=lambda r: (-(r["cs"] or 0)))
print("%-26s %-11s %-13s %6s %6s %6s %6s %6s %6s %6s" %
      ("run", "method", "k(val/vec/dE)", "CS", "ret", "bbh", "mmlu", "Fd", "muE", "nuD"))
print("-" * 108)
for r in rows:
    kk = "%s/%s/%s" % (r["kval"], r["kvec"], int(r["dE"]) if isinstance(r["dE"], bool) else r["dE"])
    print("%-26s %-11s %-13s %s %s %s %s %s %s %s" % (
        r["run"][:26], (r["method"] or "")[:11], kk,
        fnum(r["cs"]), fnum(r["ret"]), fnum(r["bbh"]), fnum(r["mmlu"]),
        fnum(r["fd"], 6, 3), fnum(r["muE"], 6, 3), fnum(r["nuD"], 6, 3)))
print("-" * 108)
print("BAR: LoRA CS78.1/ret21.7 | CLoRA-k1024 CS79.9/ret24.8 | CLoRA-k2048 CS65.4/ret25.7 | base ret26.0 (FULL scale)")
print("WIN: ret>24.8 @ CS>=80  OR  ret~25-26 @ CS>65  OR  better frontier area. UIO ret is FAST (~FULL+0.9).")
