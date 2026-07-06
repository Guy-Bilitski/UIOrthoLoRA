"""retfix_bbh_only_report.py — ZERO-GPU interim fix for the broken math retention axis.

MMLU-Pro's extraction regex (`answer is \\(?([ABCDEFGHIJ])\\)?`) cannot parse the answer
format MetaMathQA training installs ("The answer is: <value>"), so post-tune MMLU-Pro
measures extraction failure, not retention (6/7 faithful math cells <= random-10%,
base = 18.96). retention_mean = mean(BBH, MMLU-Pro) is therefore half-broken for every
math-trained cell.

This script recomputes a clean `retention_math` (= BBH answer-only, bbh_fewshot) column
for every completed math run, straight from the existing results/*/summary.json files.
It writes:
    results/retention_bbh_only.jsonl   (one row per math cell)
    results/retention_bbh_only.md      (human-readable table)
It does NOT touch campaign_summary.jsonl or any live script. Idempotent; re-run any time
(e.g. after more frm_* cells land) to refresh both outputs.

    python retfix_bbh_only_report.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Full-set base ceilings under the answer-only convention (results/base_l2-7b_bbhAO,
# 250/subtask) and MMLU-Pro (results/base_l2-7b). CLoRA's published refs: 34.91 / 18.56.
# These are LLAMA-2 ceilings; Qwen cells (qwswm_ prefix) get no %base (different base).
BASE_BBH_AO = 33.10
BASE_MMLU_PRO = 18.96


def is_llama2_run(name):
    return not name.startswith("qw")  # qwswm_/qwsw_ = Qwen2.5-7B cells

MATH_ADAPT_TASKS = {"math_faithful", "gsm8k_faithful", "gsm8k", "math"}


def collect():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*", "summary.json"))):
        try:
            s = json.load(open(p))
        except Exception:
            continue
        h = s.get("headline") or {}
        if h.get("adapt_task") not in MATH_ADAPT_TASKS:
            continue
        bbh = h.get("bbh")
        mp = h.get("mmlu_pro")
        run_name = s.get("run_name", os.path.basename(os.path.dirname(p)))
        l2 = is_llama2_run(run_name)
        rows.append({
            "run_name": run_name,
            "base_family": "llama2" if l2 else "qwen",
            "method": s.get("method", "unknown"),
            "adapt_task": h.get("adapt_task"),
            "gsm8k": h.get("gsm8k"),
            "math": h.get("math"),
            "bbh": bbh,
            "mmlu_pro": mp,
            "retention_mean_OLD": h.get("retention_mean"),
            # the fixed math retention axis: BBH answer-only, base ceiling 33.10
            "retention_math": bbh,
            "retention_math_frac_of_base": (round(bbh / BASE_BBH_AO, 4)
                                            if l2 and isinstance(bbh, (int, float)) else None),
            # flag: MMLU-Pro at/below the random-10% floor => extraction failure
            "mmlu_pro_extract_fail": (isinstance(mp, (int, float)) and mp <= 10.0),
            "fdelta": h.get("fdelta"),
            "dw_sv_max": h.get("dw_sv_max"),
            "git_commit": (s.get("git_commit") or "")[:12],
            "evaluated_at": s.get("evaluated_at"),
        })
    return rows


def main():
    rows = collect()
    rows.sort(key=lambda r: -(r["retention_math"] or 0))

    jl = os.path.join(RESULTS, "retention_bbh_only.jsonl")
    with open(jl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    md = os.path.join(RESULTS, "retention_bbh_only.md")
    lines = [
        "# Math retention, BBH-only (interim fix, zero-GPU re-report)",
        "",
        f"Base ceilings (full set): BBH answer-only = {BASE_BBH_AO}, MMLU-Pro = {BASE_MMLU_PRO}.",
        "`retention_math` = BBH answer-only (bbh_fewshot). MMLU-Pro is shown only to document",
        "the extraction failure (<=10 = at/below the random floor for 10-option MC).",
        "`retention_mean_OLD` = the broken mean(BBH, MMLU-Pro) — do not use for math cells.",
        "",
        "| run | gsm8k | math | retention_math (BBH) | %base | mmlu_pro | extract-fail | ret_mean OLD | fdelta |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pct = (f"{100 * r['retention_math_frac_of_base']:.1f}%"
               if r["retention_math_frac_of_base"] is not None else "-")
        lines.append(
            f"| {r['run_name']} | {r['gsm8k']} | {r['math']} | **{r['retention_math']}** "
            f"| {pct} | {r['mmlu_pro']} | {'YES' if r['mmlu_pro_extract_fail'] else 'no'} "
            f"| {r['retention_mean_OLD']} | {r['fdelta']} |")
    n_fail = sum(1 for r in rows if r["mmlu_pro_extract_fail"])
    lines += ["",
              f"{len(rows)} math cells; {n_fail} with MMLU-Pro at/below the random-10% floor.",
              "Source: retfix_bbh_only_report.py (reads results/*/summary.json only; "
              "campaign_summary.jsonl untouched)."]
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[retfix] {len(rows)} math cells -> {jl}")
    print(f"[retfix] table -> {md}")
    for r in rows:
        print(f"  {r['run_name']:42s} retention_math={r['retention_math']} "
              f"(old mean={r['retention_mean_OLD']}, mmlu_pro={r['mmlu_pro']}"
              f"{' EXTRACT-FAIL' if r['mmlu_pro_extract_fail'] else ''})")


if __name__ == "__main__":
    main()
