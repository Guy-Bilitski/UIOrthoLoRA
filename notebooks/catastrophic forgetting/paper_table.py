"""THE table for the paper. Auto-fills from results/ as each arm lands; pending rows
are shown as "--" so the same command works at any point in the campaign.

Per configuration (model x method at its Pareto operating point):
  intruder profile   : intruders among the top-10 directions, and their energy share
  six arms A-F       : update size F_delta, task accuracy, retention
  key contrasts      : B-C, D-A, B-E, B-F

Usage:
  python paper_table.py                 # markdown
  python paper_table.py --latex         # LaTeX booktabs body
  python paper_table.py --csv out.csv
"""
import os, json, csv, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

CONFIGS = [
    ("Llama-2-7B", "LoRA+wd", "5e-4", "tia1_frc_lorawd_wd0p3_lr5e4_s43"),
    ("Llama-2-7B", "MiLoRA",  "3e-4", "tia1_frc_milora_lr3e4_s43"),
    ("Llama-2-7B", "CLoRA",   "3e-4", "tia1_frc_clora_k1024_lr3e4_s44"),
    ("Qwen2.5-7B", "LoRA+wd", "1e-4", "tia1_qwsw_lorawd_wd0p3_lr1e4_s43"),
    ("Qwen2.5-7B", "MiLoRA",  "1e-4", "tia1_qwsw_milora_lr1e4_s43"),
    ("Qwen2.5-7B", "SC-LoRA", "2e-5", "tia1_qwsw_sclora_lr2e5_s43"),
    ("Qwen2.5-7B", "CLoRA",   "2e-4", "tia1_qwsw_clora_k1024_lr2e4_s44"),
]
# extra (off-Pareto) rows kept for the magnitude contrast
EXTRA = [("Llama-2-7B", "MiLoRA (high-F)", "1e-3", "tia1_frc_milora_lr1e3_s43")]

ARMS = [("A", "source", "{r}__rl50", "{r}"),
        ("B", "intruders deleted", "{r}__k10allablB", None),
        ("C", "uniform shrink to B", "{r}__k10allablC", None),
        ("D", "B rescaled to A", "{r}__k10allablD", None),
        ("E", "non-intruder, magnitude-matched", "{r}__k10allablE", None),
        ("Ep", "non-intruder, energy-matched", "{r}__k10allablEp", None),
        ("F", "non-intruder, count-matched", "{r}__k10allablF1", None)]


def head(run):
    for cand in (run,):
        p = os.path.join(RES, cand, "summary.json")
        if os.path.exists(p):
            return json.load(open(p))["headline"]
    return None


def intruder(run):
    p = os.path.join(RES, "intruder", run + ".json")
    if not os.path.exists(p):
        return None
    a = json.load(open(p))["aggregate"]
    n = a["n_matrices"]
    return dict(frac=a["total_intruders_k10_baseAll_t0.5"] / (n * 10),
                energy=a["mean_energy_share_baseAll_t0.5"], n=n)


def rows():
    out = []
    for model, meth, lr, run in CONFIGS + EXTRA:
        ii = intruder(run)
        arms = {}
        for tag, _lbl, pat, alt in ARMS:
            h = head(pat.format(r=run)) or (head(alt.format(r=run)) if alt else None)
            arms[tag] = h
        out.append(dict(model=model, method=meth, lr=lr, run=run, intr=ii, arms=arms))
    return out


def fmt(v, p=2):
    return "--" if v is None else f"{v:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    R = rows()

    print("\n### Table 1 — intruder profile and causal interventions, per configuration\n")
    print("| config | intruders (top-10) | intruder energy | arm | F_delta | task | retention |")
    print("|---|---|---|---|---|---|---|")
    for r in R:
        cfg = f"{r['model'].split('-')[0]} {r['method']} {r['lr']}"
        ip = f"{100*r['intr']['frac']:.1f} %" if r['intr'] else "--"
        ie = fmt(r['intr']['energy'], 3) if r['intr'] else "--"
        first = True
        for tag, lbl, _p, _alt in ARMS:
            h = r["arms"].get(tag)
            if h is None and tag in ("E", "Ep", "F") and not any(r["arms"].get(t) for t in ("E","Ep","F")):
                pass
            c1 = cfg if first else ""
            c2 = ip if first else ""
            c3 = ie if first else ""
            print(f"| {c1} | {c2} | {c3} | **{tag}** {lbl} | "
                  f"{fmt(h['fdelta'],3) if h else '--'} | {fmt(h['cs_avg']) if h else '--'} | "
                  f"{fmt(h['retention_mean']) if h else '--'} |")
            first = False

    print("\n### Table 2 — key contrasts (retention pp / task pp)\n")
    print("| config | B-C (matched magnitude) | D-A (matched magnitude) | B-E (matched magnitude) | B-Ep (matched energy) | B-F (matched count) |")
    print("|---|---|---|---|---|---|")
    for r in R:
        A = r["arms"]
        def d(x, y):
            hx, hy = A.get(x), A.get(y)
            if not hx or not hy: return "--"
            return (f"{hx['retention_mean']-hy['retention_mean']:+.2f} / "
                    f"{hx['cs_avg']-hy['cs_avg']:+.2f}")
        cfg = f"{r['model'].split('-')[0]} {r['method']} {r['lr']}"
        print(f"| {cfg} | {d('B','C')} | {d('D','A')} | {d('B','E')} | {d('B','Ep')} | {d('B','F')} |")

    done = sum(1 for r in R for t,_,_,_ in ARMS if r["arms"].get(t))
    total = len(R) * len(ARMS)
    print(f"\n[{done}/{total} arm evaluations complete]")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model","method","lr","run","intruder_frac","intruder_energy",
                        "arm","fdelta","task","retention"])
            for r in R:
                for tag,_l,_p,_al in ARMS:
                    h = r["arms"].get(tag)
                    w.writerow([r["model"],r["method"],r["lr"],r["run"],
                                r["intr"]["frac"] if r["intr"] else "",
                                r["intr"]["energy"] if r["intr"] else "",
                                tag, h["fdelta"] if h else "", h["cs_avg"] if h else "",
                                h["retention_mean"] if h else ""])
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
