#!/usr/bin/env python3
"""Results book generator.

Reads all raw result sources and writes results_book/ -- one readable
GitHub-flavored markdown table file per experiment, plus a README index
and a flat master table.  Idempotent; re-run any time new data lands.

Sources:
  results/*/summary.json            adaptation + retention + F_delta per run
  results/geo_drift/master_labeled.jsonl   per-run geometry (e_top/e_bot/ein_*/amp_top/ranks)
  results/geo_drift/adapter_metrics.jsonl  geometry fallback (fields suffixed _w)
  results/forgetting*.jsonl         CE-to-base per run (forgetting_ce)
  results/train_registry.jsonl      train_runtime_s, trainable_params
"""
import glob
import json
import math
import os
import re
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results_book")

MISSING = "·"  # middle dot

CS_BASE = 26.0    # Llama-2-7B base: mean(BBH 33.10, MMLU-Pro 18.96)
MATH_BASE = 33.1  # Llama-2-7B base BBH (answer-only)

CS_RET_CAP = ("Retention = mean(BBH, MMLU-Pro), answer-only; "
              f"Llama-2-7B base reference = {CS_BASE:.1f}.")
MATH_RET_CAP = ("Retention = BBH only (MMLU-Pro parser unreliable on math-tuned runs); "
                f"Llama-2-7B base BBH = {MATH_BASE:.1f}.")
QWEN_CS_RET_CAP = ("Retention = mean(BBH, MMLU-Pro), answer-only; base model Qwen2.5-7B "
                   "(no base reference evaluated in this tree yet).")
QWEN_MATH_RET_CAP = ("Retention = BBH only; base model Qwen2.5-7B "
                     "(no base reference evaluated in this tree yet).")

GEO_COLS = ["e_top", "e_bot", "ein_top", "ein_bot", "amp_top", "stable_rank", "eff_rank"]
GEO_HDRS = ["e_top", "e_bot", "ein_top", "ein_bot", "amp_top", "st.rank", "eff.rank"]


# ----------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------

def f2(x):
    if x is None:
        return MISSING
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return MISSING
    return f"{v:.2f}"


def flr(lr):
    """0.0003 -> 3e-4"""
    if lr is None:
        return MISSING
    s = f"{lr:.0e}"
    m, e = s.split("e")
    return f"{int(float(m))}e{int(e)}"


def fint(x):
    if x is None:
        return MISSING
    return f"{int(x):,}"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return MISSING
    if len(vals) == 1:
        return f2(vals[0])
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def spearman(xs, ys):
    """Spearman rank correlation (average ranks for ties)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    n = len(xs)
    if n < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


# ----------------------------------------------------------------------------
# run-name parsing
# ----------------------------------------------------------------------------

METHOD_TOKENS = [
    ("lora_null", "LoRA-Null"),
    ("lorawd", "LoRA+wd"),
    ("milora", "MiLoRA"),
    ("sclora", "SC-LoRA"),
    ("scl2", "SC-LoRA"),
    ("cordapp", "CorDA++"),
    ("corda", "CorDA"),
    ("clora", "CLoRA"),
    ("dora", "DoRA"),
    ("pissa", "PiSSA"),
    ("lora", "LoRA"),
    ("uio", "UIOrtho"),
]


def parse_name(run):
    p = {}
    m = re.search(r"_s(\d+)$", run)
    p["seed"] = int(m.group(1)) if m else 42
    m = re.search(r"_lr(\d+)e-?(\d+)", run)
    p["lr"] = int(m.group(1)) * 10.0 ** (-int(m.group(2))) if m else None
    m = re.search(r"wd(\d+p\d+|\d+)", run)
    p["wd"] = float(m.group(1).replace("p", ".")) if m else None
    m = re.search(r"_k(\d+)", run)
    p["k"] = int(m.group(1)) if m else None
    m = re.search(r"_c(\d+)(?:_|$)", run)
    p["cutoff"] = int(m.group(1)) if m else None
    m = re.search(r"_r(\d+)(?:_|$)", run)
    p["rank"] = int(m.group(1)) if m else None
    m = re.search(r"_b(\d+p\d+)", run)
    p["beta"] = float(m.group(1).replace("p", ".")) if m else None
    p["method"] = "?"
    for tok, name in METHOD_TOKENS:
        if tok in run:
            p["method"] = name
            break
    if run.startswith("lrswm_"):
        fam = "math_lr_sweep"
    elif run.startswith("lrsw_"):
        fam = "cs_lr_sweep"
    elif run.startswith("frm_"):
        fam = "math_faithful"
    elif run.startswith("frc_"):
        fam = "cs_faithful"
    elif run.startswith("mtxm_"):
        fam = "rank_wd_matrix_math"
    elif run.startswith("mtx_"):
        fam = "rank_wd_matrix"
    elif run.startswith(("qwsw_", "qwswm_")):
        fam = "qwen"
    elif run.startswith("b4_"):
        fam = "calibration"
    else:
        fam = "misc"
    p["family"] = fam
    # config = run name minus LR and seed tokens -> groups a sub-table
    cfg = re.sub(r"_s\d+$", "", run)
    cfg = re.sub(r"_lr\d+e-?\d+", "", cfg)
    p["config"] = cfg
    return p


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_geometry():
    geo = {}
    fp = os.path.join(RES, "geo_drift", "master_labeled.jsonl")
    if os.path.exists(fp):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            geo[d["run"]] = {
                "e_top": d.get("e_top"), "e_bot": d.get("e_bot"),
                "ein_top": d.get("ein_top"), "ein_bot": d.get("ein_bot"),
                "amp_top": d.get("amp_top"),
                "stable_rank": d.get("stable_rank_w"),
                "eff_rank": d.get("eff_rank_w"),
                "fro_total": d.get("fro_total"), "spec_max": d.get("spec_max"),
            }
    fp = os.path.join(RES, "geo_drift", "adapter_metrics.jsonl")
    if os.path.exists(fp):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["run"] in geo:
                continue
            geo[d["run"]] = {
                "e_top": d.get("e_top_w"), "e_bot": d.get("e_bot_w"),
                "ein_top": d.get("ein_top_w"), "ein_bot": d.get("ein_bot_w"),
                "amp_top": d.get("amp_top_w"),
                "stable_rank": d.get("stable_rank_w"),
                "eff_rank": d.get("eff_rank_w"),
                "fro_total": d.get("fro_total"), "spec_max": d.get("spec_max"),
            }
    return geo


def load_ce():
    ce = {}
    for fp in sorted(glob.glob(os.path.join(RES, "forgetting*.jsonl"))):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rn = d.get("run_name")
            if rn and d.get("forgetting_ce") is not None:
                ce[rn] = d["forgetting_ce"]  # dedup: last wins
    return ce


def load_registry():
    reg = {}
    fp = os.path.join(RES, "train_registry.jsonl")
    if os.path.exists(fp):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rn = d.get("run_name")
            if rn:
                reg[rn] = {"train_runtime_s": d.get("train_runtime_s"),
                           "trainable_params": d.get("trainable_params")}
    return reg


def load_runs():
    geo = load_geometry()
    ce = load_ce()
    reg = load_registry()
    runs = []
    for fp in sorted(glob.glob(os.path.join(RES, "*", "summary.json"))):
        run = os.path.basename(os.path.dirname(fp))
        try:
            d = json.load(open(fp))
        except json.JSONDecodeError:
            continue
        h = d.get("headline") or {}
        per = d.get("per_dataset") or {}
        fd = d.get("fdelta") or {}
        r = parse_name(run)
        r["run"] = run
        adapt_task = h.get("adapt_task", "")
        is_math = adapt_task in ("math_faithful", "gsm8k", "math") or "gsm8k" in per
        r["task"] = "math" if is_math else "cs"
        r["model"] = "Qwen2.5-7B" if run.startswith("qw") else "Llama-2-7B"
        r["bbh"] = h.get("bbh")
        r["mmlu_pro"] = h.get("mmlu_pro")
        r["mmlu"] = h.get("mmlu")
        r["arc_c"] = h.get("arc_c")
        r["truthfulqa"] = h.get("truthfulqa")
        if is_math:
            r["adapt"] = h.get("gsm8k", per.get("gsm8k"))
            r["math_acc"] = h.get("math", per.get("math"))
            r["retention"] = h.get("retention_bbh", h.get("bbh"))
        else:
            r["adapt"] = h.get("cs_avg")
            r["math_acc"] = None
            ret = h.get("retention_mean")
            if ret is None and r["bbh"] is not None and r["mmlu_pro"] is not None:
                ret = (r["bbh"] + r["mmlu_pro"]) / 2.0
            r["retention"] = ret
        r["fdelta"] = fd.get("fdelta_token_weighted", h.get("fdelta"))
        r["dw_sv_max"] = fd.get("dw_sv_max")
        r["dw_sv_mean"] = fd.get("dw_sv_mean")
        g = geo.get(run, {})
        for c in GEO_COLS + ["fro_total", "spec_max"]:
            r[c] = g.get(c)
        r["ce"] = ce.get(run)
        rg = reg.get(run, {})
        r["runtime_s"] = rg.get("train_runtime_s")
        r["params"] = rg.get("trainable_params")
        r["evaluated_at"] = (d.get("evaluated_at") or "")[:10]
        runs.append(r)
    return runs


# ----------------------------------------------------------------------------
# page builders
# ----------------------------------------------------------------------------

def sort_key(r):
    return (r["config"], r["lr"] if r["lr"] is not None else 1e9, r["seed"])


def config_label(r):
    bits = [r["method"]]
    if r.get("rank"):
        bits.append(f"r={r['rank']}")
    if r.get("wd") is not None:
        bits.append(f"wd={r['wd']}")
    if r.get("k"):
        bits.append(f"k={r['k']}")
    if r.get("beta") is not None:
        bits.append(f"beta={r['beta']}")
    if r.get("cutoff"):
        bits.append(f"cutoff={r['cutoff']}")
    return ", ".join(bits)


def sweep_subtables(runs, adapt_hdr, math_col=False):
    """Per-adapter sub-tables for an LR sweep. Bold row = best adaptation."""
    out = []
    configs = sorted({r["config"] for r in runs})
    for cfg in configs:
        rs = sorted([r for r in runs if r["config"] == cfg], key=sort_key)
        out.append(f"### `{cfg}` — {config_label(rs[0])}\n")
        best = None
        with_adapt = [r for r in rs if r["adapt"] is not None]
        if with_adapt:
            best = max(with_adapt, key=lambda r: r["adapt"])
        hdrs = ["LR", "seed", adapt_hdr]
        if math_col:
            hdrs.append("MATH")
        hdrs += ["Retention", "BBH", "MMLU-Pro", "F_Δ"] + GEO_HDRS + ["CE"]
        rows = []
        for r in rs:
            cells = [flr(r["lr"]), r["seed"], f2(r["adapt"])]
            if math_col:
                cells.append(f2(r["math_acc"]))
            cells += [f2(r["retention"]), f2(r["bbh"]), f2(r["mmlu_pro"]), f2(r["fdelta"])]
            cells += [f2(r[c]) for c in GEO_COLS]
            cells.append(f2(r["ce"]))
            if r is best:
                cells = [f"**{c}**" for c in cells]
            rows.append(cells)
        out.append(md_table(hdrs, rows))
        # seed mean +/- SD where multiple seeds share an LR
        by_lr = {}
        for r in rs:
            by_lr.setdefault(r["lr"], []).append(r)
        multi = {lr: v for lr, v in by_lr.items() if len(v) > 1}
        if multi:
            out.append("\nSeed mean ± SD (LRs with ≥ 2 seeds):\n")
            hdrs2 = ["LR", "n", adapt_hdr, "Retention", "F_Δ"]
            rows2 = []
            for lr in sorted(multi):
                v = multi[lr]
                rows2.append([flr(lr), len(v),
                              mean_sd([x["adapt"] for x in v]),
                              mean_sd([x["retention"] for x in v]),
                              mean_sd([x["fdelta"] for x in v])])
            out.append(md_table(hdrs2, rows2))
        out.append("")
    return "\n".join(out)


def page(title, caption, body, bold_note=False):
    note = (f"_Auto-generated by `results_book.py` — do not edit by hand. "
            f"Missing values shown as {MISSING}; numbers rounded to 2 decimals.")
    if bold_note:
        note += " **Bold row** = best-LR (highest adaptation) within a sub-table."
    note += "_"
    return f"# {title}\n\n> {caption}\n\n{note}\n\n{body}\n"


def build_01(runs):
    rs = [r for r in runs if r["family"] == "cs_lr_sweep"]
    body = sweep_subtables(rs, "CS avg")
    cap = (f"CS LR sweep (`lrsw_*`): commonsense-170k adaptation on Llama-2-7B, "
           f"7 LRs per adapter. {CS_RET_CAP}")
    return page("01 · CS LR sweep", cap, body, bold_note=True), len(rs)


def build_02(runs):
    frm = [r for r in runs if r["family"] == "math_faithful"]
    lrswm = [r for r in runs if r["family"] == "math_lr_sweep"]
    parts = []
    parts.append("## A. Faithful math repro (`frm_*`)\n")
    hdrs = (["Method", "wd", "k", "LR", "cutoff", "seed", "GSM8K", "MATH",
             "BBH (retention)", "F_Δ"] + GEO_HDRS + ["CE"])
    rows = []
    frm_sorted = sorted(frm, key=lambda r: (r["method"], r["wd"] if r["wd"] is not None else -1,
                                            r["k"] or 0, r["cutoff"] or 0,
                                            r["lr"] if r["lr"] is not None else 1e9, r["seed"]))
    for r in frm_sorted:
        rows.append([r["method"], f2(r["wd"]) if r["wd"] is not None else MISSING,
                     r["k"] or MISSING,
                     flr(r["lr"]), r["cutoff"] or MISSING, r["seed"],
                     f2(r["adapt"]), f2(r["math_acc"]), f2(r["retention"]),
                     f2(r["fdelta"])] + [f2(r[c]) for c in GEO_COLS] + [f2(r["ce"])])
    parts.append(md_table(hdrs, rows))
    parts.append("\n## B. Math LR sweep (`lrswm_*`)\n")
    parts.append(sweep_subtables(lrswm, "GSM8K", math_col=True))
    cap = (f"Math (MetaMath) fine-tuning on Llama-2-7B. Adaptation = GSM8K (+MATH). "
           f"{MATH_RET_CAP}")
    return page("02 · Math faithful repro + math LR sweep", cap, "\n".join(parts), bold_note=True), len(frm) + len(lrswm)


def build_03(runs):
    rs = [r for r in runs if r["family"] == "cs_faithful"]
    if not rs:
        body = ("_No `frc_*` runs have landed yet — this page fills in "
                "automatically as the faithful-CS spine lands._")
    else:
        hdrs = (["Method", "wd", "LR", "cutoff", "seed", "CS avg", "Retention",
                 "BBH", "MMLU-Pro", "F_Δ"] + GEO_HDRS + ["CE"])
        rows = []
        for r in sorted(rs, key=lambda r: (r["method"], r["wd"] if r["wd"] is not None else -1,
                                           r["lr"] if r["lr"] is not None else 1e9, r["seed"])):
            rows.append([r["method"], f2(r["wd"]) if r["wd"] is not None else MISSING,
                         flr(r["lr"]), r["cutoff"] or MISSING, r["seed"],
                         f2(r["adapt"]), f2(r["retention"]), f2(r["bbh"]),
                         f2(r["mmlu_pro"]), f2(r["fdelta"])]
                        + [f2(r[c]) for c in GEO_COLS] + [f2(r["ce"])])
        body = md_table(hdrs, rows)
    cap = f"Faithful CS repro (`frc_*`) on Llama-2-7B. {CS_RET_CAP}"
    return page("03 · CS faithful repro", cap, body), len(rs)


def matrix_section(rs, adapt_hdr):
    parts = []
    methods = sorted({r["method"] for r in rs})
    for meth in methods:
        mrs = [r for r in rs if r["method"] == meth]
        parts.append(f"### {meth}\n")
        hdrs = (["Config", "seed", adapt_hdr, "Retention", "BBH", "MMLU-Pro",
                 "F_Δ"] + GEO_HDRS + ["CE"])
        rows = []
        for cfg in sorted({r["config"] for r in mrs}):
            crs = sorted([r for r in mrs if r["config"] == cfg], key=lambda r: r["seed"])
            for r in crs:
                rows.append([f"`{cfg}`", r["seed"], f2(r["adapt"]), f2(r["retention"]),
                             f2(r["bbh"]), f2(r["mmlu_pro"]), f2(r["fdelta"])]
                            + [f2(r[c]) for c in GEO_COLS] + [f2(r["ce"])])
            if len(crs) > 1:
                rows.append([f"`{cfg}`", "**mean±SD**",
                             mean_sd([x["adapt"] for x in crs]),
                             mean_sd([x["retention"] for x in crs]),
                             mean_sd([x["bbh"] for x in crs]),
                             mean_sd([x["mmlu_pro"] for x in crs]),
                             mean_sd([x["fdelta"] for x in crs])]
                            + [mean_sd([x[c] for x in crs]) for c in GEO_COLS]
                            + [mean_sd([x["ce"] for x in crs])])
        parts.append(md_table(hdrs, rows))
        parts.append("")
    return "\n".join(parts)


def build_04(runs):
    mtx = [r for r in runs if r["family"] == "rank_wd_matrix"]
    mtxm = [r for r in runs if r["family"] == "rank_wd_matrix_math"]
    parts = ["## A. CS rank/wd matrix (`mtx_*`, 3 seeds)\n",
             matrix_section(mtx, "CS avg")]
    if mtxm:
        parts.append("\n## B. Math rank/wd matrix (`mtxm_*`)\n")
        parts.append(f"_{MATH_RET_CAP}_\n")
        parts.append(matrix_section(mtxm, "GSM8K"))
    cap = (f"Rank/weight-decay matrix on Llama-2-7B, 3 seeds per config "
           f"(42/43/44). CS arm: {CS_RET_CAP}")
    return page("04 · Rank/wd matrix", cap, "\n".join(parts)), len(mtx) + len(mtxm)


def build_05(runs):
    qw = [r for r in runs if r["family"] == "qwen"]
    cs = [r for r in qw if r["task"] == "cs"]
    mt = [r for r in qw if r["task"] == "math"]
    parts = [f"## A. Qwen CS LR sweep (`qwsw_*`)\n", f"_{QWEN_CS_RET_CAP}_\n",
             sweep_subtables(cs, "CS avg"),
             f"\n## B. Qwen math LR sweep (`qwswm_*`)\n", f"_{QWEN_MATH_RET_CAP}_\n",
             sweep_subtables(mt, "GSM8K", math_col=True)]
    cap = "Second-model replication on Qwen2.5-7B: CS arm + math arm."
    return page("05 · Qwen replication", cap, "\n".join(parts), bold_note=True), len(qw)


def build_06(runs):
    rs = sorted([r for r in runs if r["family"] == "calibration"], key=sort_key)
    hdrs = (["Run", "Method", "LR", "seed", "CS avg", "Retention", "BBH",
             "MMLU-Pro", "F_Δ"] + GEO_HDRS + ["CE"])
    rows = []
    for r in rs:
        rows.append([f"`{r['run']}`", r["method"], flr(r["lr"]), r["seed"],
                     f2(r["adapt"]), f2(r["retention"]), f2(r["bbh"]),
                     f2(r["mmlu_pro"]), f2(r["fdelta"])]
                    + [f2(r[c]) for c in GEO_COLS] + [f2(r["ce"])])
    body = md_table(hdrs, rows) if rows else "_No `b4_*` runs yet._"
    cap = f"Eval-matched calibration control (`b4_*`) on Llama-2-7B. {CS_RET_CAP}"
    return page("06 · Calibration control", cap, body), len(rs)


def build_07(runs):
    rs = [r for r in runs if r["ce"] is not None]
    rs.sort(key=lambda r: (r["fdelta"] is None, r["fdelta"] or 0))
    hdrs = ["Run", "Family", "Method", "Task", "F_Δ", "CE-to-base",
            "Retention", "Adaptation"]
    rows = []
    for r in rs:
        rows.append([f"`{r['run']}`", r["family"], r["method"], r["task"],
                     f2(r["fdelta"]), f2(r["ce"]), f2(r["retention"]), f2(r["adapt"])])
    body = md_table(hdrs, rows) if rows else "_No CE measurements yet._"
    pairs = [(r["fdelta"], r["ce"]) for r in rs if r["fdelta"] is not None]
    if len(pairs) >= 3:
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if rho is not None:
            body += (f"\n\n**Spearman ρ(CE, F_Δ) = {rho:.2f}** "
                     f"(n = {len(pairs)} runs with both metrics).")
    cap = ("CE-forgetting: cross-entropy of each adapted model against base-model "
           "next-token targets (lower = closer to base). Sorted by F_Δ. "
           "Retention per each run's own arm definition (CS: mean(BBH, MMLU-Pro), "
           "base 26.0; math: BBH-only, base 33.1).")
    return page("07 · CE forgetting", cap, body), len(rs)


def build_08(runs):
    rs = [r for r in runs if r["e_top"] is not None]
    parts = ["## Per-method geometry fingerprint (mean ± SD over all runs with geometry)\n"]
    hdrs = ["Method", "n runs"] + GEO_HDRS
    rows = []
    for meth in sorted({r["method"] for r in rs}):
        mrs = [r for r in rs if r["method"] == meth]
        rows.append([meth, len(mrs)] + [mean_sd([x[c] for x in mrs]) for c in GEO_COLS])
    parts.append(md_table(hdrs, rows))
    parts.append("\n## Appendix: per-run geometry\n")
    hdrs2 = ["Run", "Method", "Family", "F_Δ"] + GEO_HDRS + ["fro_total", "spec_max"]
    rows2 = []
    for r in sorted(rs, key=lambda r: (r["method"], r["run"])):
        rows2.append([f"`{r['run']}`", r["method"], r["family"], f2(r["fdelta"])]
                     + [f2(r[c]) for c in GEO_COLS]
                     + [f2(r["fro_total"]), f2(r["spec_max"])])
    parts.append(md_table(hdrs2, rows2))
    cap = ("Geometry of ΔW vs the base weights: e_top/e_bot = energy of ΔW in "
           "the top/bottom base singular subspaces; ein_top/ein_bot = energy of the "
           "input side; amp_top = MiLoRA Table-7 amplification of top base directions; "
           "stable/effective rank of ΔW. Token-weighted across 160 matrices.")
    return page("08 · Geometry fingerprint", cap, "\n".join(parts)), len(rs)


def build_99(runs):
    hdrs = (["Run", "Family", "Model", "Method", "Task", "LR", "wd", "seed",
             "Adapt", "MATH", "Retention", "BBH", "MMLU-Pro", "MMLU", "ARC-c",
             "TruthfulQA", "F_Δ", "sv_max"] + GEO_HDRS
            + ["CE", "train_s", "params", "eval date"])
    rows = []
    for r in sorted(runs, key=lambda r: (r["family"], sort_key(r))):
        rows.append([f"`{r['run']}`", r["family"], r["model"], r["method"], r["task"],
                     flr(r["lr"]), f2(r["wd"]) if r["wd"] is not None else MISSING,
                     r["seed"], f2(r["adapt"]), f2(r["math_acc"]), f2(r["retention"]),
                     f2(r["bbh"]), f2(r["mmlu_pro"]), f2(r["mmlu"]), f2(r["arc_c"]),
                     f2(r["truthfulqa"]), f2(r["fdelta"]), f2(r["dw_sv_max"])]
                    + [f2(r[c]) for c in GEO_COLS]
                    + [f2(r["ce"]),
                       fint(round(r["runtime_s"])) if r["runtime_s"] else MISSING,
                       fint(r["params"]) if r["params"] else MISSING,
                       r["evaluated_at"] or MISSING])
    cap = ("Every run with a summary.json, all metrics, flat. 'Adapt' = CS avg for "
           "CS runs, GSM8K for math runs. Retention: CS = mean(BBH, MMLU-Pro) "
           "(Llama-2 base 26.0); math = BBH-only (Llama-2 base 33.1); Qwen rows use "
           "the same definitions on Qwen2.5-7B (no base ref in tree).")
    return page("99 · All runs (master table)", cap, md_table(hdrs, rows)), len(runs)


def build_readme(counts, misc_n):
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        "# Results book",
        "",
        "One readable folder with every experiment's results — adaptation, "
        "retention, magnitude (F_Δ), geometry, and MiLoRA-style amplification "
        "metrics, per seed and per LR. Regenerated automatically as new runs land "
        "(`results_book.py`, auto-loop every 30 min).",
        "",
        f"_Last updated: {now}_",
        "",
        "**Retention definitions** — CS runs: mean(BBH, MMLU-Pro), Llama-2-7B "
        "base = 26.0. Math runs: BBH-only, Llama-2-7B base = 33.1. "
        f"Missing values are shown as {MISSING}.",
        "",
    ]
    idx = [
        ("01_cs_lr_sweep.md", "CS LR sweep on Llama-2-7B (lrsw_): 8 adapters × 7 LRs, per-adapter tables"),
        ("02_math_faithful.md", "Math arm (frm_ faithful repro + lrswm_ math LR sweep): GSM8K/MATH vs BBH retention"),
        ("03_cs_faithful.md", "Faithful CS repro (frc_) — fills as the spine lands"),
        ("04_rank_wd_matrix.md", "Rank/weight-decay matrix (mtx_/mtxm_), 3 seeds, per-config mean±SD"),
        ("05_qwen.md", "Qwen2.5-7B second-model replication: CS + math LR sweeps"),
        ("06_calibration_control.md", "Eval-matched calibration control (b4_)"),
        ("07_ce_forgetting.md", "CE-to-base forgetting for every measured run + Spearman(CE, F_Δ)"),
        ("08_geometry_fingerprint.md", "Per-method geometry fingerprint + per-run appendix"),
        ("99_all_runs.md", "Flat master table — every run, all metrics"),
    ]
    lines.append(md_table(["File", "What it holds", "Data rows"],
                          [[f"[{f}]({f})", d, counts.get(f, 0)] for f, d in idx]))
    lines.append("")
    lines.append(f"_{misc_n} additional exploratory/legacy runs (uio*, clora_*, scl2_*, "
                 "a5_*, grid_*, …) are classified `misc` and appear only in "
                 "[99_all_runs.md](99_all_runs.md)._")
    lines.append("")
    return "\n".join(lines)


def write_if_changed(path, content):
    """Write only when content differs, so the auto-update loop only commits
    (and README's timestamp only moves) when actual data changed."""
    try:
        old = open(path).read()
    except OSError:
        old = None
    if old == content:
        return False
    with open(path, "w") as f:
        f.write(content)
    return True


def main():
    runs = load_runs()
    os.makedirs(OUT, exist_ok=True)
    builders = [
        ("01_cs_lr_sweep.md", build_01),
        ("02_math_faithful.md", build_02),
        ("03_cs_faithful.md", build_03),
        ("04_rank_wd_matrix.md", build_04),
        ("05_qwen.md", build_05),
        ("06_calibration_control.md", build_06),
        ("07_ce_forgetting.md", build_07),
        ("08_geometry_fingerprint.md", build_08),
        ("99_all_runs.md", build_99),
    ]
    counts = {}
    changed = False
    for fname, fn in builders:
        content, n = fn(runs)
        counts[fname] = n
        changed |= write_if_changed(os.path.join(OUT, fname), content)
    misc_n = sum(1 for r in runs if r["family"] == "misc")
    readme_path = os.path.join(OUT, "README.md")
    if changed or not os.path.exists(readme_path):
        write_if_changed(readme_path, build_readme(counts, misc_n))
    print(f"results_book: {len(runs)} runs total"
          f" ({'updated' if changed else 'no change'})")
    for fname, n in counts.items():
        print(f"  {fname}: {n} rows")
    print(f"  misc runs (99 only): {misc_n}")


if __name__ == "__main__":
    main()
