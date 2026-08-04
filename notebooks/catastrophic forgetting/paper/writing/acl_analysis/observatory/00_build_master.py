#!/usr/bin/env python
"""METRIC OBSERVATORY step 0 — build the tidy master run table.

Loading logic replicated from paper/writing/analysis_final/ladder_2026-07-17.py
(the canonical frozen-pool loader): results/*/summary.json, families
lrsw/lrswm/qwsw/qwswm/frc/frm, drop SMOKE/smoke, finite fdelta>0 and finite
retention_mean, 7 post-freeze stragglers excluded from the primary (on_pool)
set. lora_null split convention = method from run_name (ladder method_of).
CorDA/CorDA++ rows are KEPT in the master but flagged withheld=True and
on_pool=False (own port bug; shown but not assessed).

Preflight hard-asserts key_numbers.md SS18.1: n=1035, pooled r=-0.847, and all
six per-family (n, r) cells to 3 decimals. Aborts on mismatch.

Output: master_runs.csv (one row per run) + build log to stdout.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "writing", "acl_analysis", "observatory")

STRAGGLERS = {  # ladder_2026-07-17.py — synced post-freeze 2026-07-17 12:49
    "lrsw_clora_k1024_lr3e4_s45",
    "qwsw_lora_null_r16_lr5e4_s43",
    "qwsw_milora_r32_lr5e5_s44",
    "qwswm_clora_k1024_lr2e5_s44",
    "qwswm_dora_r16_lr2e4_s43",
    "qwswm_lora_r16_lr1e4_s44",
    "qwswm_sclora_r32_lr3e4_s44",
}

FROZEN = {  # key_numbers.md SS18.1 per-family (n, r)
    "lrsw": (180, -0.886), "lrswm": (120, -0.865), "qwsw": (151, -0.840),
    "qwswm": (164, -0.830), "frc": (276, -0.928), "frm": (144, -0.929),
}
FROZEN_POOLED_R = -0.847

FAM_META = {  # family -> (model, task, recipe type, display label)
    "lrsw":  ("Llama-2-7B", "cs",   "sweep", "Llama-CS (lrsw)"),
    "lrswm": ("Llama-2-7B", "math", "sweep", "Llama-Math (lrswm)"),
    "qwsw":  ("Qwen-2.5-7B", "cs",   "sweep", "Qwen-CS (qwsw)"),
    "qwswm": ("Qwen-2.5-7B", "math", "sweep", "Qwen-Math (qwswm)"),
    "frc":   ("Llama-2-7B", "cs",   "faithful-CLoRA grid", "Llama-CS grid (frc)"),
    "frm":   ("Llama-2-7B", "math", "faithful-CLoRA grid", "Llama-Math grid (frm)"),
}

DISPLAY = {
    "lora": "LoRA", "lora_null": "LoRA-Null", "lorawd": "LoRA+wd",
    "lorawdr16": "LoRA+wd-r16", "milora": "MiLoRA", "milorawd": "MiLoRA+wd",
    "clora": "CLoRA", "dora": "DoRA", "sclora": "SC-LoRA", "pissa": "PiSSA",
    "corda": "CorDA (withheld)", "cordapp": "CorDA++ (withheld)",
}

HEADLINE_KEYS = ["cs_avg", "gsm8k", "adapt_task", "bbh", "mmlu_pro", "mmlu",
                 "arc_c", "truthfulqa", "retention_mean", "retention_broad",
                 "fdelta", "dw_sv_max", "dw_sv_mean"]
GEO_KEYS = ["fro_total", "spec_max", "spec_mean", "stable_rank_w",
            "eff_rank_w", "e_top_w", "e_bot_w", "amp_top_w"]
CE_KEYS = ["forgetting_ce", "base_entropy", "forgetting_kl"]


def method_of(rn):
    """ladder_2026-07-17.py convention, extended to keep corda/cordapp visible."""
    body = rn.split("_", 1)[1]
    if body.startswith("lora_null"):
        return "lora_null"
    return body.split("_")[0]


def parse_lr(rn):
    m = re.search(r"_lr([0-9]+)e([0-9]+)", rn)
    return float(f"{m.group(1)}e-{m.group(2)}") if m else np.nan


def main():
    # ---- summaries ----------------------------------------------------------
    recs = {}
    for f in sorted(glob.glob(os.path.join(RES, "*", "summary.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rn = d.get("run_name") or os.path.basename(os.path.dirname(f))
        if "SMOKE" in rn or "smoke" in rn:
            continue
        m = re.match(r"^([a-z0-9]+)_", rn)
        fam = m.group(1) if m else "other"
        if fam not in FAM_META:
            continue
        recs[rn] = d  # dict keyed by run_name = dedupe (keep last)
    print(f"[load] {len(recs)} unique runs in the 6 assessed families")

    # ---- joins (dict-keyed -> deduped, keep last) ---------------------------
    ce = {}
    for line in open(os.path.join(RES, "forgetting_merged.jsonl")):
        d = json.loads(line)
        if d.get("run_name"):
            ce[d["run_name"]] = {k: d.get(k) for k in CE_KEYS}
    geo = {}
    for line in open(os.path.join(RES, "geo_drift", "adapter_metrics_merged.jsonl")):
        d = json.loads(line)
        if d.get("run"):
            geo[d["run"]] = {k: d.get(k) for k in GEO_KEYS}
    quar = set()
    for line in open(os.path.join(RES, "quarantine_diverged.txt")):
        name = line.split("\t")[0].strip()
        if name:
            quar.add(name)
    print(f"[load] CE rows {len(ce)}, geo rows {len(geo)}, quarantine {len(quar)}")

    rows = []
    for rn, d in sorted(recs.items()):
        h = d.get("headline") or {}
        fam = rn.split("_", 1)[0]
        model, task, recipe, fam_label = FAM_META[fam]
        meth = method_of(rn)
        withheld = meth in ("corda", "cordapp")
        sm = re.search(r"_s(4[2-9])$", rn)
        seed = int(sm.group(1)) if sm else np.nan
        cell = re.sub(r"_s4[2-9]$", "", rn)
        fd, ret = h.get("fdelta"), h.get("retention_mean")
        finite = (isinstance(fd, (int, float)) and isinstance(ret, (int, float))
                  and math.isfinite(fd) and math.isfinite(ret) and fd > 0)
        on_pool = finite and (not withheld) and (rn not in STRAGGLERS)
        row = dict(
            run=rn, family=fam, family_label=fam_label, model=model, task=task,
            recipe=recipe, method=meth,
            method_display=DISPLAY.get(meth, meth),
            lr=parse_lr(rn), seed=seed, cell=cell,
            withheld=withheld, quarantined=rn in quar,
            straggler=rn in STRAGGLERS, on_pool=on_pool,
        )
        for k in HEADLINE_KEYS:
            row[k] = h.get(k)
        # adaptation = cs_avg for CS families; gsm8k (== cs_avg column) for math
        row["adapt"] = h.get("cs_avg") if task == "cs" else h.get("gsm8k", h.get("cs_avg"))
        g = geo.get(rn, {})
        for k in GEO_KEYS:
            row[k] = g.get(k)
        c = ce.get(rn, {})
        for k in CE_KEYS:
            row[k] = c.get(k)
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["fdelta", "spec_max", "fro_total", "dw_sv_max", "dw_sv_mean"]:
        v = pd.to_numeric(df[col], errors="coerce")
        df["log10_" + col] = np.where(v > 0, np.log10(v), np.nan)

    # ---- PREFLIGHT: reproduce key_numbers.md SS18.1 or abort ----------------
    pool = df[df.on_pool]
    n = len(pool)
    print(f"\n[preflight] on_pool n = {n} (frozen target 1035)")
    assert n == 1035, f"pool n={n} != 1035 — join broken, aborting"
    for fam, (wn, wr) in FROZEN.items():
        sub = pool[pool.family == fam]
        r = np.corrcoef(sub.log10_fdelta, sub.retention_mean)[0, 1]
        assert len(sub) == wn, f"{fam}: n={len(sub)} != {wn}"
        assert abs(r - wr) < 5e-4, f"{fam}: r={r:.4f} != {wr}"
        print(f"  {fam}: n={len(sub)} r={r:.3f}  OK")
    rp = np.corrcoef(pool.log10_fdelta, pool.retention_mean)[0, 1]
    assert abs(rp - FROZEN_POOLED_R) < 5e-4, f"pooled r={rp:.4f}"
    print(f"  pooled r = {rp:.3f}  OK — SS18.1 reproduced")

    # ---- census + coverage --------------------------------------------------
    print("\n[census] on-pool method x family (run counts):")
    print(pool.pivot_table(index="method", columns="family", values="run",
                           aggfunc="count", fill_value=0).to_string())
    print(f"\n[census] withheld CorDA/CorDA++ rows kept in master: "
          f"{int(df.withheld.sum())}")
    print(f"[census] quarantined rows inside on-pool "
          f"(freeze keeps them; finite-filter only): "
          f"{int(pool.quarantined.sum())} of {len(quar)} listed")
    cov = pool.groupby("family").agg(
        n=("run", "size"),
        geo=("stable_rank_w", lambda s: s.notna().sum()),
        ce=("forgetting_kl", lambda s: s.notna().sum()))
    print("\n[coverage] geometry / CE joins per family (on-pool):")
    print(cov.to_string())

    p = os.path.join(OUT, "master_runs.csv")
    df.to_csv(p, index=False)
    print(f"\n[write] {p}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    sys.exit(main())
