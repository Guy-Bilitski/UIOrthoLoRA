"""Shared data pool for the METHOD ADJUDICATION analysis (2026-07-18).

Loads results/*/summary.json into a pandas DataFrame using the SAME conventions
as the frozen analysis layer (key_numbers.md section 18, analysis_final/
ladder_2026-07-17.py, op_points_2026-07-17.py):

- drop SMOKE/smoke runs;
- quarantine list results/quarantine_diverged.txt is EXCLUDED from operating
  point / Pareto statistics but KEPT as divergence-rate data;
- dedupe the known duplicate: any run whose name ends in "_reeval" with an
  existing parent directory is dropped (Q4, 09_verification_2026-07-18.md);
- lora_null split convention: "lora_null" matched BEFORE "lora"
  (make_figs_split_lora_null.py);
- qwswm: exclude "_ep6_" variants (op_points_2026-07-17.py convention);
- frm: primary context = c256 (02_operating_points.md);
- CorDA/CorDA++ = WITHHELD (own port bug, key_numbers section 8) — loaded,
  flagged, never ranked.

Adaptation = headline.cs_avg (CS-8 accuracy on CS arms, GSM8K EM on math arms).
Retention  = headline.retention_mean (CS arms, core = mean(BBH, MMLU-Pro)),
             headline.bbh (math arms — 02_operating_points.md convention).

PREFLIGHT: reproduces key_numbers.md section 18.1 pooled magnitude relation
(n=1035, r=-0.847) before any adjudication number is emitted.
"""
import json
import math
import os
import re

import numpy as np
import pandas as pd

ROOT = "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "writing", "acl_analysis", "adjudication")
TABLES = os.path.join(OUT, "tables")
FIGURES = os.path.join(OUT, "figures")

# post-freeze stragglers (ladder_2026-07-17.py) — excluded only from the
# section-18.1 preflight, INCLUDED in adjudication (current-pool convention,
# key_numbers section 19.4: they move nothing by >0.002).
STRAGGLERS = {
    "lrsw_clora_k1024_lr3e4_s45",
    "qwsw_lora_null_r16_lr5e4_s43",
    "qwsw_milora_r32_lr5e5_s44",
    "qwswm_clora_k1024_lr2e5_s44",
    "qwswm_dora_r16_lr2e4_s43",
    "qwswm_lora_r16_lr1e4_s44",
    "qwswm_sclora_r32_lr3e4_s44",
}

LRMAP = {"2e5": 2e-5, "5e5": 5e-5, "7e5": 7e-5, "1e4": 1e-4, "15e5": 1.5e-4,
         "2e4": 2e-4, "3e4": 3e-4, "5e4": 5e-4, "7e4": 7e-4, "1e3": 1e-3,
         "2e3": 2e-3, "5e3": 5e-3, "1e2": 1e-2}


def fmt_lr(lr):
    if lr is None or (isinstance(lr, float) and math.isnan(lr)):
        return "?"
    for k, v in [("2e-5", 2e-5), ("5e-5", 5e-5), ("7e-5", 7e-5), ("1e-4", 1e-4),
                 ("1.5e-4", 1.5e-4), ("2e-4", 2e-4), ("3e-4", 3e-4), ("5e-4", 5e-4),
                 ("7e-4", 7e-4), ("1e-3", 1e-3), ("2e-3", 2e-3), ("5e-3", 5e-3),
                 ("1e-2", 1e-2)]:
        if abs(lr - v) < 1e-9:
            return k
    return f"{lr:g}"


# canonical method key from run name (lora_null before lorawd before lora)
_METHOD_PATTERNS = [
    ("lora_null", "lora_null"), ("lorawdr16", "lorawd_r16"), ("lorawd", "lorawd"),
    ("milorawd", "milorawd"), ("dorawd", "dorawd"), ("milora", "milora"),
    ("sclora", "sclora"), ("clora", "clora"), ("cordapp", "cordapp"),
    ("corda", "corda"), ("pissa", "pissa"), ("dora", "dora"), ("lora", "lora"),
]

DISPLAY = {"lora": "LoRA", "lorawd": "LoRA+wd", "lora_null": "LoRA-Null",
           "milora": "MiLoRA", "clora": "CLoRA", "sclora": "SC-LoRA",
           "dora": "DoRA", "pissa": "PiSSA", "corda": "CorDA",
           "cordapp": "CorDA++", "lorawd_r16": "LoRA+wd(r16)"}


def method_key(run):
    body = "_" + run
    for pat, key in _METHOD_PATTERNS:
        if re.search(r"[_/]" + pat + r"[_0-9]", body):
            return key
    return None


def load_quarantine():
    q = set()
    with open(os.path.join(RES, "quarantine_diverged.txt")) as fh:
        for line in fh:
            name = line.split("\t")[0].strip()
            if name:
                q.add(name)
    return q


def load_pool():
    quar = load_quarantine()
    rows = []
    for d in sorted(os.listdir(RES)):
        p = os.path.join(RES, d, "summary.json")
        if not os.path.isfile(p):
            continue
        if "SMOKE" in d or "smoke" in d:
            continue
        # known duplicate (Q4, 09_verification): the _reeval row is byte-identical
        # to its parent and sits INSIDE the frozen n=1035 — keep for preflight,
        # flag for exclusion from every adjudication statistic.
        dup = d.endswith("_reeval") and os.path.isdir(os.path.join(RES, d[:-len("_reeval")]))
        try:
            s = json.load(open(p))
        except Exception:
            continue
        h = s.get("headline", {}) or {}
        fam = d.split("_", 1)[0]
        mlr = re.search(r"_lr(\d+e\d)(?:_|$)", d)
        msd = re.search(r"_s(\d+)(?:_|$)", d)
        mcx = re.search(r"_c(\d\d\d+)(?:_|$)", d)
        rows.append(dict(
            run=d, family=fam, method=method_key(d),
            lr=LRMAP.get(mlr.group(1)) if mlr else None,
            seed=int(msd.group(1)) if msd else None,
            ctx=int(mcx.group(1)) if mcx else None,
            ep6=("_ep6_" in d),
            adapt=h.get("cs_avg"), ret_core=h.get("retention_mean"),
            ret_broad=h.get("retention_broad"), bbh=h.get("bbh"),
            mmlu_pro=h.get("mmlu_pro"), fdelta=h.get("fdelta"),
            adapt_task=h.get("adapt_task"), quar=(d in quar), dup=dup,
        ))
    return pd.DataFrame(rows)


def preflight_18_1(df):
    """Reproduce key_numbers.md section 18.1 pooled r (n=1035, r=-0.847)."""
    fams = ["lrsw", "lrswm", "qwsw", "qwswm", "frc", "frm"]
    sub = df[df.family.isin(fams) & ~df.run.isin(STRAGGLERS)
             & df.run.map(lambda r: "corda" not in r)].copy()
    sub = sub[np.isfinite(sub.fdelta.astype(float)) & (sub.fdelta > 0)
              & np.isfinite(sub.ret_core.astype(float))]
    n = len(sub)
    r = np.corrcoef(np.log10(sub.fdelta.astype(float)), sub.ret_core.astype(float))[0, 1]
    assert n == 1035, f"preflight pool n={n} != 1035"
    assert abs(r - (-0.847)) < 0.0005, f"preflight pooled r={r:.4f} != -0.847"
    return n, r


# ------------------------------------------------------------------ families ---
# Each spec: method key -> run-name prefix (op_points_2026-07-17.py prior art).
LLAMA = dict(core=26.0, broad=35.26, bbh=33.1)
QWEN = dict(core=44.35, broad=None, bbh=47.93)
LRS7 = [2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]

FAMILIES = {
    "llama_cs": dict(
        title="Llama-2-7B x Commonsense-8 (lrsw sweep, r-matched grid)",
        prefix="lrsw", adapt_name="CS-8", ret_field="ret_core",
        base=LLAMA, ret_base=LLAMA["core"], grid=LRS7,
        specs=[("lorawd", "lrsw_lorawd_wd0p3_"),
               ("sclora", "lrsw_sclora_r32_"),
               ("lora", "lrsw_lora_r16_"),
               ("lora_null", "lrsw_lora_null_r16_"),
               ("clora", "lrsw_clora_k1024_"),
               ("dora", "lrsw_dora_r16_"),
               ("milora", "lrsw_milora_r32_")]),
    "llama_math": dict(
        title="Llama-2-7B x math/GSM8K (frm faithful CLoRA recipe, c256)",
        prefix="frm", adapt_name="GSM8K", ret_field="bbh",
        base=LLAMA, ret_base=LLAMA["bbh"], grid=None, ctx=256,
        specs=[("lorawd", "frm_lorawd_wd0p3_"),
               ("lora", "frm_lorawd_wd0_"),          # r32 wd=0 = vanilla LoRA arm
               ("milora", "frm_milora_"),
               ("clora", "frm_clora_"),              # k64/128/256; k tracked
               ("sclora", "frm_sclora_"),
               ("dora", "frm_dora_"),
               ("lora_null", "frm_lora_null_"),
               ("pissa", "frm_pissa_"),
               ("cordapp", "frm_cordapp_")]),        # WITHHELD
    "qwen_cs": dict(
        title="Qwen-2.5-7B x Commonsense-8 (qwsw sweep)",
        prefix="qwsw", adapt_name="CS-8", ret_field="ret_core",
        base=QWEN, ret_base=QWEN["core"], grid=LRS7,
        specs=[("lorawd", "qwsw_lorawd_wd0p3_"),
               ("sclora", "qwsw_sclora_r32_"),
               ("lora", "qwsw_lora_r16_"),
               ("lora_null", "qwsw_lora_null_r16_"),
               ("clora", "qwsw_clora_k1024_"),
               ("dora", "qwsw_dora_r16_"),
               ("milora", "qwsw_milora_r32_")]),
    "qwen_math": dict(
        title="Qwen-2.5-7B x math/GSM8K (qwswm sweep; ep6 variants excluded)",
        prefix="qwswm", adapt_name="GSM8K", ret_field="bbh",
        base=QWEN, ret_base=QWEN["bbh"], grid=LRS7, no_ep6=True,
        specs=[("lorawd", "qwswm_lorawd_wd0p3_lr"),
               ("sclora", "qwswm_sclora_r32_"),
               ("lora", "qwswm_lora_r16_"),
               ("lora_r32", "qwswm_lora_r32_lr"),
               ("lora_null", "qwswm_lora_null_r16_"),
               ("clora", "qwswm_clora_k1024_"),
               ("dora", "qwswm_dora_r16_"),
               ("milora", "qwswm_milora_r32_")]),
}
DISPLAY["lora_r32"] = "LoRA(r32)"
WITHHELD = {"corda", "cordapp"}


def family_rows(df, fam_key, include_quar=False):
    """Rows of one adjudication family, labeled with the spec method key."""
    spec = FAMILIES[fam_key]
    out = []
    for mkey, prefix in spec["specs"]:
        sub = df[df.run.str.startswith(prefix)].copy()
        if spec.get("ctx") is not None:
            sub = sub[sub.ctx == spec["ctx"]]
        if spec.get("no_ep6"):
            sub = sub[~sub.ep6]
        sub = sub[~sub.dup]
        if not include_quar:
            sub = sub[~sub.quar]
        sub["mkey"] = mkey
        sub["fam_key"] = fam_key
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def cell_table(fam_df, ret_field):
    """Seed-aggregated (method, LR[, k]) cells: mean/sd/n for adapt & retention."""
    fam_df = fam_df.copy()
    # CLoRA-k in frm: keep k in the cell id so k64/128/256 are separate cells
    def cell_id(r):
        mk = re.search(r"_(k\d+)_", r["run"])
        k = mk.group(1) if (r["mkey"] == "clora" and mk) else ""
        return (r["mkey"], r["lr"], k)
    fam_df["cell"] = fam_df.apply(cell_id, axis=1)
    rows = []
    for (mkey, lr, k), g in fam_df.groupby("cell"):
        if lr is None or (isinstance(lr, float) and math.isnan(lr)):
            continue
        a, rr = g["adapt"].dropna(), g[ret_field].dropna()
        if not len(a) or not len(rr):
            continue
        rows.append(dict(
            mkey=mkey, lr=lr, k=k, n=len(g),
            adapt_mean=a.mean(), adapt_sd=a.std(ddof=1) if len(a) > 1 else 0.0,
            ret_mean=rr.mean(), ret_sd=rr.std(ddof=1) if len(rr) > 1 else 0.0,
            fd_mean=g["fdelta"].mean(),
            seeds=",".join(str(s) for s in sorted(g["seed"].dropna().astype(int))),
            collapse=bool(len(a) > 1 and (a.std(ddof=1) > 5.0)),
        ))
    return pd.DataFrame(rows)


def best_cell(cells, mkey, min_n=2):
    """Best-mean-adaptation cell for a method; prefer cells with n>=min_n
    when the method has any (guards against single-seed argmax like the
    quarantine-orphaned qwswm lorawd lr1e-3 point, 02_operating_points.md 3b)."""
    sub = cells[cells.mkey == mkey]
    if not len(sub):
        return None
    multi = sub[sub.n >= min_n]
    pick = (multi if len(multi) else sub).sort_values("adapt_mean", ascending=False).iloc[0]
    return pick


if __name__ == "__main__":
    df = load_pool()
    n, r = preflight_18_1(df)
    print(f"PREFLIGHT OK: section 18.1 pooled n={n}, r={r:.3f}")
    for fk in FAMILIES:
        fr = family_rows(df, fk)
        print(f"{fk}: usable rows={len(fr)}, methods={sorted(fr.mkey.unique())}")
