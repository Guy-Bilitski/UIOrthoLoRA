"""Regenerate the focused matched-control tables in neurips_2026.tex.

Reads only the frozen snapshot data/focused_norm/{runs.csv,selection_final.json}
(produced by extract_focused_runs.py from hash-verified ledger completions).
--check verifies the checked-in generated blocks; --write regenerates them.
A confirmation block is emitted only for task/condition cells with all three
declared seeds validated; otherwise it stays a placeholder comment.
"""
from pathlib import Path
import argparse
import csv
import json
import re
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "neurips_2026.tex"
CALIBRATION_SEED = 31415
CONFIRMATION_SEEDS = {42, 17, 123}
TASKS = [("rte", "RTE"), ("mrpc", "MRPC")]
CONDITIONS = [("P1_UNREG", "Unregularized"), ("P1_MIX", "Two-sided mixing"), ("P1_NORM", "Norm-matched")]


def load():
    with open(ROOT / "data/focused_norm/runs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    selection = json.loads((ROOT / "data/focused_norm/selection_final.json").read_text())["selection"]
    return rows, selection


def fmt_dose(value):
    return f"{float(value):g}"


def cell(row):
    cross = float(row["pooled_LT"]) + float(row["pooled_TL"])
    acc = 100 * float(row["fixed_accuracy"])
    return (
        f'${float(row["rho_pooled"]):.4f}$ & ${100 * cross:.1f}$ & '
        f'${100 * float(row["pooled_LL"]):.1f}$ & ${100 * float(row["pooled_TT"]):.1f}$ & '
        f'${float(row["probe_masked_ce"]):.2f}$ & ${acc:.1f}$'
    )


def matched_summary(rows, selection):
    lines = []
    for task, label in TASKS:
        sel = selection[task]
        chosen = fmt_dose(sel["selected_coefficient"])
        status = "matched" if sel["match_status"] == "matched" else "nearest, failed match"
        for condition, name in CONDITIONS:
            sub = [
                r
                for r in rows
                if r["stage"] == "calibration"
                and r["task"] == task
                and r["condition"] == condition
                and int(r["seed"]) == CALIBRATION_SEED
                and (condition != "P1_NORM" or fmt_dose(r["coefficient"]) == chosen)
            ]
            assert len(sub) == 1, (task, condition, chosen, len(sub))
            shown = name if condition != "P1_NORM" else f"Norm-matched ($\\beta={chosen}$, {status})"
            lines.append(f"{label} & {shown} & {cell(sub[0])} \\\\")
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    return lines


def frontier(rows, selection):
    lines = []
    for task, label in TASKS:
        target = selection[task]["target_pooled_relative_frobenius"]
        norm_rows = sorted(
            [r for r in rows if r["stage"] == "calibration" and r["task"] == task and r["condition"] == "P1_NORM"],
            key=lambda r: float(r["coefficient"]),
        )
        for row in norm_rows:
            rho = float(row["rho_pooled"])
            cross = float(row["pooled_LT"]) + float(row["pooled_TL"])
            error = abs(rho / target - 1)
            lines.append(
                f"{label} & ${fmt_dose(row['coefficient'])}$ & ${rho:.4f}$ & ${100 * error:.1f}$ & "
                f"${100 * cross:.1f}$ & ${100 * float(row['pooled_TT']):.1f}$ & "
                f"${float(row['probe_masked_ce']):.2f}$ \\\\"
            )
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    return lines


def confirmation_summary(rows, selection):
    lines, complete = [], True
    match_notes = []
    for task, label in TASKS:
        chosen = fmt_dose(selection[task]["selected_coefficient"])
        by_condition = {}
        for condition, _ in CONDITIONS:
            sub = [
                r
                for r in rows
                if r["stage"] == "confirmation"
                and r["task"] == task
                and r["condition"] == condition
                and (condition != "P1_NORM" or fmt_dose(r["coefficient"]) == chosen)
            ]
            if len(sub) != len(CONFIRMATION_SEEDS) or {int(r["seed"]) for r in sub} != CONFIRMATION_SEEDS:
                complete = False
            by_condition[condition] = {int(r["seed"]): r for r in sub}
        if not complete:
            continue
        # Matching status is derived per task/seed pair from the achieved norms
        # of THIS study's runs (e_s = |rho_NORM,s / rho_MIX,s - 1|), never
        # inherited from the calibration-seed label.
        errors = {
            seed: (
                float(by_condition["P1_NORM"][seed]["rho_pooled"])
                / float(by_condition["P1_MIX"][seed]["rho_pooled"])
                - 1.0
            )
            for seed in sorted(CONFIRMATION_SEEDS)
        }
        error_text = ", ".join(f"{100 * errors[s]:+.1f}" for s in sorted(CONFIRMATION_SEEDS))
        match_notes.append(f"{label}: ${error_text}$\\%")
        block = []
        for condition, name in CONDITIONS:
            sub = list(by_condition[condition].values())

            def ms(key, scale=1.0, digits=4):
                values = [scale * float(r[key]) for r in sub]
                return f"${mean(values):.{digits}f} \\pm {stdev(values):.{digits}f}$"

            cross = [100 * (float(r["pooled_LT"]) + float(r["pooled_TL"])) for r in sub]
            eqm_cross = [100 * (float(r["eqm_LT"]) + float(r["eqm_TL"])) for r in sub]
            if condition == "P1_NORM":
                shown = f"NORM ($\\beta={chosen}$)"
            else:
                shown = condition.removeprefix("P1_")
            block.append(
                f"{label} & {shown} & {ms('rho_pooled')} & ${mean(cross):.1f} \\pm {stdev(cross):.1f}$ & ${mean(eqm_cross):.1f} \\pm {stdev(eqm_cross):.1f}$ & "
                f"{ms('pooled_LL', 100, 1)} & {ms('pooled_TT', 100, 1)} & "
                f"{ms('probe_masked_ce', 1, 2)} & {ms('fixed_accuracy', 100, 1)} \\\\"
            )
        lines.extend(block)
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    if not complete:
        return ["% confirmation seeds incomplete; rerun --write once all 18 runs validate"]
    return (
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Confirmation of the calibration-selected norm control across seeds 17, 42, and 123",
            "(mean$\\pm$sample SD, $n=3$), fixed-step endpoints, common protocol of",
            "Section~\\ref{sec:matchedcontrol}. Cross P, LL, and TT are pooled energy percentages; Cross E weights modules equally.",
            "The probe is masked-token cross-entropy under the frozen original MLM head;",
            "accuracy is the inner-selection split at the fixed step, scaled by 100. Achieved errors",
            "$\\delta_s=\\rho_{\\mathrm{NORM},s}/\\rho_{\\mathrm{MIX},s}-1$ (signed; $e_s=|\\delta_s|$) against the declared",
            "$\\pm5\\%$ rule, in that seed order: " + "; ".join(match_notes) + ".",
            "N is the dimension-only reference, not a run or test of isotropy.}",
            "\\label{tab:matchedconfirmation}",
            "\\begin{adjustbox}{max width=\\linewidth}",
            "\\begin{tabular}{llccccccc}",
            "\\toprule",
            "Task & Condition & $\\rho_F$ & Cross P & Cross E & LL & TT & Probe CE & Acc \\\\",
            "\\midrule",
        ]
        + lines
        + [r"\midrule", r"--- & N (dimension only) & --- & $44.4$ & $44.4$ & $44.4$ & $11.1$ & --- & --- \\", "\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
    )


def paired_summary(rows, selection):
    from analyze_focused_confirmation import analyze
    result = analyze()
    lines = []
    contrasts = [("P1_MIX_minus_P1_NORM", "MIX--NORM"),
                 ("P1_MIX_minus_P1_UNREG", "MIX--UNREG"),
                 ("P1_NORM_minus_P1_UNREG", "NORM--UNREG")]
    for task, label in TASKS:
        for contrast, name in contrasts:
            for metric, shown in (("cross_pp", "Cross"), ("accuracy_pp", "Accuracy")):
                item = result["tasks"][task]["paired"][contrast][metric]
                values = " & ".join(f"${v:.2f}$" for v in item["values"])
                lo, hi = item["nominal_t95"]
                lines.append(f"{label} & {name} & {shown} & {values} & ${item['mean']:.2f} \\pm {item['sample_sd']:.2f}$ & $[{lo:.2f}, {hi:.2f}]$ \\\\")
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    return lines


def module_summary(rows, selection):
    from analyze_focused_confirmation import analyze
    result = analyze()
    lines = []
    for task, label in TASKS:
        task_result = result["tasks"][task]
        for match, pair in zip(task_result["matching"], task_result["module_pairs"]):
            lines.append(f"{label} & {pair['seed']} & ${match['signed_error_percent']:+.1f}$ & "
                         f"${pair['mix_median_rho']:.4f}$ & ${pair['norm_median_rho']:.4f}$ & "
                         f"{pair['modules_within_5pct']}/48 & {pair['rescaled_modules_within_5pct']}/48 & "
                         f"{pair['mix_modules_below_1e6']}/{pair['norm_modules_below_1e6']} \\\\")
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    return lines


def task_outcomes(rows, selection):
    lines = []
    for task, label in TASKS:
        for condition, _ in CONDITIONS:
            sub = sorted([r for r in rows if r['stage'] == 'confirmation' and r['task'] == task and r['condition'] == condition], key=lambda r: int(r['seed']))
            assert len(sub) == 3 and {int(r['seed']) for r in sub} == CONFIRMATION_SEEDS
            for row in sub:
                def score(key):
                    return f"${100*float(row[key]):.2f}$" if row[key] else "---"
                lines.append(f"{label} & {condition.removeprefix('P1_')} & {row['seed']} & "
                             f"{score('fixed_accuracy')} & {score('fixed_f1')} & {row['best_step']} & "
                             f"{score('best_accuracy')} & {score('best_f1')} \\\\")
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    return lines


BLOCKS = {
    "FOCUSED MATCHED SUMMARY": matched_summary,
    "FOCUSED FRONTIER": frontier,
    "FOCUSED CONFIRMATION": confirmation_summary,
    "FOCUSED PAIRED": paired_summary,
    "FOCUSED MODULES": module_summary,
    "FOCUSED TASK OUTCOMES": task_outcomes,
}


def render(name, rows, selection):
    body = BLOCKS[name](rows, selection)
    header = f"% Generated by scripts/build_focused_tables.py from data/focused_norm/runs.csv."
    return "\n".join([header] + body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows, selection = load()
    text = TEX.read_text()
    changed = text
    for name in BLOCKS:
        begin, end = f"% BEGIN GENERATED {name}\n", f"% END GENERATED {name}"
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
        if not pattern.search(changed):
            print(f"note: marker block absent in TeX, skipped: {name}")
            continue
        replacement = begin + render(name, rows, selection) + "\n" + end
        changed = pattern.sub(lambda _: replacement, changed)
    if args.write:
        TEX.write_text(changed)
        print("generated blocks written")
    elif args.check:
        if changed != text:
            raise SystemExit("FAIL: generated focused blocks differ from data/focused_norm snapshot")
        print("PASS: focused generated blocks match the data snapshot")
    else:
        raise SystemExit("pass --write or --check")


if __name__ == "__main__":
    main()
