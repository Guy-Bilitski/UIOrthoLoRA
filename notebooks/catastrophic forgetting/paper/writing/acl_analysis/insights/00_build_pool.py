"""00_build_pool.py — canonical merged pool for the insight-prospecting pass.

Loader conventions are byte-identical to analysis_final/ladder_2026-07-17.py
(the §19 script): results/*/summary.json; drop SMOKE/smoke/corda substrings;
require finite headline.fdelta>0 and finite retention_mean; families
lrsw/lrswm/qwsw/qwswm/frc/frm by run_name prefix; the 7 post-freeze STRAGGLERS
are excluded from the primary pool. Preflight hard-asserts §18.1
(n=1035, pooled r=-0.847, per-family n/r to 3 decimals) before writing anything.

Extras kept per row (new for this pass): per-benchmark retention components
(bbh/mmlu_pro/mmlu/arc_c/truthfulqa), adapt (cs_avg), per-CS-dataset accuracies,
parsed knob values (lr, wd, clora-k, rank), geometry join, CE/KL join,
quarantine flag. Output: pool.csv (primary pool only) + pool_all.csv
(everything usable incl. stragglers + non-family arms like b4_/e1_, flagged).
"""
import json, glob, math, os, re, sys
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # .../catastrophic forgetting
RES = os.path.join(ROOT, "results")

STRAGGLERS = {
    "lrsw_clora_k1024_lr3e4_s45", "qwsw_lora_null_r16_lr5e4_s43",
    "qwsw_milora_r32_lr5e5_s44", "qwswm_clora_k1024_lr2e5_s44",
    "qwswm_dora_r16_lr2e4_s43", "qwswm_lora_r16_lr1e4_s44",
    "qwswm_sclora_r32_lr3e4_s44",
}
FROZEN = {"lrsw": (180, -0.886), "lrswm": (120, -0.865), "qwsw": (151, -0.840),
          "qwswm": (164, -0.830), "frc": (276, -0.928), "frm": (144, -0.929)}
FROZEN_POOLED_R = -0.847
FAMS = list(FROZEN)
CS_DS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
         "ARC-Easy", "ARC-Challenge", "openbookqa"]

LR_RE = re.compile(r"_lr(\d+)e(\d)")


def parse_lr(rn):
    m = LR_RE.search(rn)
    if not m:
        return np.nan
    mant, exp = m.group(1), int(m.group(2))
    # convention: lr3e4 = 3e-4, lr15e5 = 1.5e-4 (15e-5)
    return float(mant) * 10.0 ** (-exp)


def parse_wd(rn):
    m = re.search(r"_wd(\d+)(?:p(\d+))?_", rn)
    if not m:
        return np.nan
    return float(m.group(1) + ("." + m.group(2) if m.group(2) else ""))


def parse_k(rn):
    m = re.search(r"_k(\d+)_", rn)
    return float(m.group(1)) if m else np.nan


def parse_rank(rn):
    m = re.search(r"_r(\d+)_", rn)
    return float(m.group(1)) if m else np.nan


def method_of(rn):
    body = rn.split("_", 1)[1] if "_" in rn else rn
    if body.startswith("lora_null"):
        return "lora_null"
    return body.split("_")[0]


def load_quarantine():
    q = set()
    with open(os.path.join(RES, "quarantine_diverged.txt")) as fh:
        for line in fh:
            name = line.split("\t")[0].strip()
            if name:
                q.add(name)
    return q


def main():
    quar = load_quarantine()
    ce = {}
    with open(os.path.join(RES, "forgetting_merged.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            k = d.get("run_name")
            if k:
                ce[k] = (d.get("forgetting_ce"), d.get("base_entropy"), d.get("forgetting_kl"))
    geo = {}
    with open(os.path.join(RES, "geo_drift", "adapter_metrics_merged.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            k = d.get("run")
            if k:
                geo[k] = d

    rows = []
    for f in glob.glob(os.path.join(RES, "*", "summary.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rn = d.get("run_name") or os.path.basename(os.path.dirname(f))
        if "SMOKE" in rn or "smoke" in rn or "corda" in rn:
            continue
        h = d.get("headline") or {}
        fd, ret = h.get("fdelta"), h.get("retention_mean")
        if not isinstance(fd, (int, float)) or not isinstance(ret, (int, float)):
            continue
        if not (math.isfinite(fd) and math.isfinite(ret)) or fd <= 0:
            continue
        m = re.match(r"^([a-z0-9]+)_", rn)
        fam = m.group(1) if m else "other"
        sm = re.search(r"_s(4[2-9])", rn)
        row = dict(
            rn=rn, fam=fam, seed=int(sm.group(1)) if sm else np.nan,
            cell=re.sub(r"_s4[2-9]", "", rn), method=method_of(rn),
            fd=fd, logfd=math.log10(fd), ret=ret,
            ret_broad=h.get("retention_broad"), adapt=h.get("cs_avg"),
            adapt_task=h.get("adapt_task"),
            bbh=h.get("bbh"), mmlu_pro=h.get("mmlu_pro"), mmlu=h.get("mmlu"),
            arc_c=h.get("arc_c"), truthfulqa=h.get("truthfulqa"),
            dw_sv_max=h.get("dw_sv_max"), dw_sv_mean=h.get("dw_sv_mean"),
            lr=parse_lr(rn), wd=parse_wd(rn), clora_k=parse_k(rn), rank=parse_rank(rn),
            quarantined=rn in quar, straggler=rn in STRAGGLERS,
            in_family=fam in FAMS,
        )
        pd_ = d.get("per_dataset") or {}
        for ds in CS_DS:
            row["cs_" + ds.replace("-", "_")] = pd_.get(ds)
        if rn in ce:
            row["forgetting_ce"], row["base_entropy"], row["forgetting_kl"] = ce[rn]
        g = geo.get(rn)
        if g:
            for kk in ("fro_total", "spec_max", "spec_mean", "stable_rank_w",
                       "eff_rank_w", "e_top_w", "e_bot_w", "ein_top_w",
                       "ein_bot_w", "amp_top_w"):
                row[kk] = g.get(kk)
        rows.append(row)

    df = pd.DataFrame(rows)
    primary = df[df.in_family & ~df.straggler].copy()

    # ---- preflight: reproduce §18.1 exactly ----
    ok = True
    for fam, (n_ref, r_ref) in FROZEN.items():
        sub = primary[primary.fam == fam]
        r = np.corrcoef(sub.logfd, sub.ret)[0, 1]
        match = (len(sub) == n_ref) and (round(r, 3) == r_ref)
        print(f"preflight {fam}: n={len(sub)} (ref {n_ref})  r={r:.3f} (ref {r_ref})  {'OK' if match else 'MISMATCH'}")
        ok &= match
    r_all = np.corrcoef(primary.logfd, primary.ret)[0, 1]
    match = (len(primary) == 1035) and (round(r_all, 3) == FROZEN_POOLED_R)
    print(f"preflight pooled: n={len(primary)} r={r_all:.3f}  {'OK' if match else 'MISMATCH'}")
    ok &= match
    if not ok:
        sys.exit("PREFLIGHT FAILED — refusing to write pool")

    primary.to_csv(os.path.join(HERE, "pool.csv"), index=False)
    df.to_csv(os.path.join(HERE, "pool_all.csv"), index=False)
    print(f"wrote pool.csv (n={len(primary)}) and pool_all.csv (n={len(df)})")
    print("coverage: geometry", primary.spec_max.notna().sum(),
          "CE", primary.forgetting_ce.notna().sum() if "forgetting_ce" in primary else 0)


if __name__ == "__main__":
    main()
