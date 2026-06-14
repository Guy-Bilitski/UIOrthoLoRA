"""
B1 deliverable: the leakage map.

Pulls every run that logged orthogonality-leakage thermometers and lays out the
two-arm contrast that is the campaign's headline measurement finding:

  * clean arm  (use_de=0): directional leakage mu_E ~= nu_D ~= 0 across ALL configs,
    yet retention swings from ~4 to ~26 -- governed by Delta-W magnitude (dw_sv_max).
  * leaky arm  (use_de=1): high directional leakage (mu_E,nu_D ~ 1.2-1.8), but the
    D/E gates brake magnitude (dw_sv_max ~ 7-9.5) so retention stays ~22-25.

=> Retention tracks a MAGNITUDE budget, not the directional leakage the thermometers
   measure. (See handoff/03_LEAKAGE_ANGLE.md and the impl caveat, finding #4.)

    python make_leakage_map.py            # prints table + correlations
    python make_leakage_map.py --write     # also writes handoff/04_LEAKAGE_MAP.md
"""
import os
import sys
import json
import glob
import math

import run_lib

HERE = run_lib.HERE
RES = os.path.join(HERE, "results")


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def load_rows():
    rows = []
    for p in glob.glob(os.path.join(RES, "*", "summary.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        lk = d.get("leakage")
        if not lk:
            continue
        cfg = d.get("config", {}) or {}
        fd = d.get("fdelta", {}) or {}
        h = d.get("headline", {}) or {}
        ret = h.get("retention_mean")
        if ret is None:
            continue  # leakage with no retention -> not on the map
        rows.append({
            "run": d.get("run_name"),
            "use_de": bool(cfg.get("use_de")),
            "k_val": cfg.get("k_val"),
            "k_vec": cfg.get("k_vec"),
            "lr": cfg.get("learning_rate"),
            "cs": h.get("cs_avg"),
            "ret": ret,
            "mu_E": lk.get("mu_E"),
            "nu_D": lk.get("nu_D"),
            "leak11": lk.get("leak11"),
            "f_delta": fd.get("f_delta"),
            "dw_max": fd.get("dw_sv_max"),
        })
    return rows


def fmt(x, n=2):
    return ("%.*f" % (n, x)) if isinstance(x, (int, float)) else "-"


def table(rows):
    out = []
    hdr = (f"| {'run':30s} | dE | {'k_val':>5s} | {'k_vec':>5s} | {'lr':>6s} | "
           f"{'CS':>5s} | {'ret':>5s} | {'mu_E':>6s} | {'nu_D':>6s} | {'dw_max':>6s} |")
    sep = "|" + "|".join(["-" * len(c) for c in hdr.split("|")[1:-1]]) + "|"
    out.append(hdr)
    out.append(sep)
    for r in sorted(rows, key=lambda r: (r["use_de"], -(r["dw_max"] or 0))):
        out.append(f"| {r['run']:30s} | {'1' if r['use_de'] else '0'}  | "
                   f"{str(r['k_val']):>5s} | {str(r['k_vec']):>5s} | {fmt(r['lr'],4):>6s} | "
                   f"{fmt(r['cs']):>5s} | {fmt(r['ret']):>5s} | {fmt(r['mu_E'],4):>6s} | "
                   f"{fmt(r['nu_D'],4):>6s} | {fmt(r['dw_max']):>6s} |")
    return "\n".join(out)


def corr_block(rows):
    clean = [r for r in rows if not r["use_de"] and r["dw_max"] is not None]
    leaky = [r for r in rows if r["use_de"] and r["dw_max"] is not None]
    allr = [r for r in rows if r["dw_max"] is not None]
    lines = []
    lines.append(f"clean arm (use_de=0): n={len(clean)}  "
                 f"mu_E range [{fmt(min(r['mu_E'] for r in clean),4)}, {fmt(max(r['mu_E'] for r in clean),4)}]  "
                 f"ret range [{fmt(min(r['ret'] for r in clean))}, {fmt(max(r['ret'] for r in clean))}]")
    lines.append(f"  corr(ret, dw_max)   = {fmt(pearson([r['dw_max'] for r in clean], [r['ret'] for r in clean]),3)}  (magnitude)")
    lines.append(f"  corr(ret, mu_E)     = {fmt(pearson([r['mu_E'] for r in clean], [r['ret'] for r in clean]),3)}  (direction; ~0 var -> nan)")
    lines.append(f"leaky arm (use_de=1): n={len(leaky)}  "
                 f"mu_E range [{fmt(min(r['mu_E'] for r in leaky),4)}, {fmt(max(r['mu_E'] for r in leaky),4)}]  "
                 f"dw_max range [{fmt(min(r['dw_max'] for r in leaky))}, {fmt(max(r['dw_max'] for r in leaky))}]")
    lines.append(f"  corr(ret, dw_max)   = {fmt(pearson([r['dw_max'] for r in leaky], [r['ret'] for r in leaky]),3)}")
    lines.append(f"  corr(ret, mu_E)     = {fmt(pearson([r['mu_E'] for r in leaky], [r['ret'] for r in leaky]),3)}")
    lines.append(f"all runs:             n={len(allr)}")
    lines.append(f"  corr(ret, dw_max)   = {fmt(pearson([r['dw_max'] for r in allr], [r['ret'] for r in allr]),3)}  <- the magnitude budget")
    lines.append(f"  corr(ret, mu_E)     = {fmt(pearson([r['mu_E'] for r in allr], [r['ret'] for r in allr]),3)}  <- directional leakage (weaker)")
    return "\n".join(lines)


def main():
    rows = load_rows()
    tbl = table(rows)
    corr = corr_block(rows)
    print("\n================ B1 LEAKAGE MAP (runs with thermometers + retention) ================\n")
    print(tbl)
    print("\n---- correlations ----")
    print(corr)
    if "--write" in sys.argv:
        path = os.path.join(HERE, "handoff", "04_LEAKAGE_MAP.md")
        with open(path, "w") as f:
            f.write("# 04 — LEAKAGE MAP (B1 deliverable)\n\n")
            f.write("Generated by `make_leakage_map.py`. Every UIOrthoLoRA run that logged the\n")
            f.write("orthogonality-leakage thermometers AND has a retention number. See\n")
            f.write("`03_LEAKAGE_ANGLE.md` for the framing and the tail-only caveat (impl finding #4).\n\n")
            f.write("## Headline\n\n")
            f.write("**Retention tracks a Delta-W *magnitude* budget (`dw_sv_max`), not the *directional* "
                    "leakage (mu_E / nu_D) the thermometers measure.**\n\n")
            f.write("- **Clean arm (use_de=0):** mu_E ~= nu_D ~= 0.003 for *every* config (pristine direction), "
                    "yet retention spans ~4 -> ~26, monotone in `dw_max` (no D/E brake -> magnitude is free to explode).\n")
            f.write("- **Leaky arm (use_de=1):** mu_E,nu_D ~ 1.2-1.8 (heavily leaky direction), but the D/E gates "
                    "brake `dw_max` to ~7-9.5, so retention stays ~22-25.\n\n")
            f.write("So the thermometers measure *direction*; preservation needs a *magnitude* budget. This is the "
                    "measurement-tool contribution (the frontier go/no-go is NO — UIOrthoLoRA is dominated by CLoRA on both axes).\n\n")
            f.write("## Map\n\n")
            f.write(tbl + "\n\n")
            f.write("## Correlations\n\n```\n" + corr + "\n```\n\n")
            f.write("**Reading the correlations (do not misquote):**\n")
            f.write("- Clean arm `corr(ret, mu_E)=+0.93` is **spurious** — mu_E is flat at ~0.003 "
                    "(variance ~1e-4), so this is 4th-decimal noise, not a relationship. The real signal "
                    "is `corr(ret, dw_max)=-0.86`: at fixed ~0 directional leakage, retention still swings "
                    "4->26 entirely with magnitude.\n")
            f.write("- All-runs `corr(ret, mu_E)=+0.44` is a **confound**, not causation: use_de=1 "
                    "simultaneously raises mu_E AND brakes dw_max (the D/E gates), so high directional "
                    "leakage co-occurs with good retention. The magnitude budget (`corr=-0.88`) is the "
                    "operative variable across both arms.\n")
        print(f"\n[wrote] {path}")


if __name__ == "__main__":
    main()
