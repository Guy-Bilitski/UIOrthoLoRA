"""Evidence export and descriptive analysis for the decoder subspace study.

Reads only validated, completed confirmation runs plus the frozen reference. It
reports BOTH registered primary outcomes side by side, never substituting one
for the other, and labels every paired interval as nominal and exploratory.

Primary location contrast: leading minus tail WITHIN each family. Middle
contrasts and the within-band rotation contrast complete the descriptive view.
Nothing here selects, drops or reweights an arm, and a null or reversed result
is exported exactly like any other.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from notebooks.iclr.campaign.artifacts import sha256, utc_now, write_json_new
from notebooks.iclr.campaign.protocol import Resources, owned_path

from . import subspace, subspace_plan

# Two-sided Student t for a paired n=3 difference (2 degrees of freedom).
T_CRITICAL_95_DF2 = 4.302652729911275
OUTCOMES = ("exact_match", "held_out_completion_nll")
PRIMARY_CONTRASTS = (("LEAD", "TAIL"), ("LEAD", "MID"), ("MID", "TAIL"))


def _mean_sd(values):
    values = list(values)
    return dict(
        n=len(values),
        mean=statistics.fmean(values) if values else None,
        sd=statistics.stdev(values) if len(values) > 1 else None,
        values=values,
    )


def paired_difference(first, second):
    """Per-seed paired difference with a NOMINAL, exploratory t interval. n is 3, not a population."""
    seeds = sorted(set(first) & set(second))
    if len(seeds) < 2:
        return dict(status="insufficient_paired_seeds", paired_seeds=seeds)
    differences = [first[seed] - second[seed] for seed in seeds]
    mean = statistics.fmean(differences)
    sd = statistics.stdev(differences)
    half = T_CRITICAL_95_DF2 * sd / math.sqrt(len(differences)) if len(differences) == 3 else None
    return dict(
        status="paired",
        paired_seeds=seeds,
        per_seed_difference={str(seed): first[seed] - second[seed] for seed in seeds},
        mean_difference=mean,
        sd_difference=sd,
        nominal_95_interval=None if half is None else [mean - half, mean + half],
        interval_note=(
            "Nominal and exploratory: a paired t interval on three seeds, not multiplicity adjusted. Seeds are "
            "not pooled with examples and the arms are not independent replicates. A wide interval around a "
            "small mean does not establish equivalence."
        ),
    )


def load_reference(rows):
    for row in rows:
        if row.get("stage") == "reference" and row.get("status") == "completed":
            return row
    return None


def summarize(rows, protocol):
    """Per-arm outcome tables, paired band contrasts and the diagnostics they must be read with."""
    record = protocol["design"]
    completed = [row for row in rows if row.get("status") == "completed" and row.get("stage") == "confirmation"]
    def unusable(value):
        """None, a non-number, or a nonfinite number. A NaN must not pass as a present outcome."""
        return value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)

    missing = [
        f"{row['arm']} seed {row['seed']}: " + ", ".join(
            name for name in ("exact_match", "held_out_completion_nll") if unusable(row.get(name))
        )
        for row in completed
        if any(unusable(row.get(name)) for name in ("exact_match", "held_out_completion_nll"))
    ]
    if missing:
        # Refuse rather than return an apparently complete study whose primary outcomes are empty or nonfinite.
        raise ValueError("Completed confirmations are missing a registered primary outcome: " + "; ".join(missing))
    by_arm = defaultdict(dict)
    diagnostics = defaultdict(dict)
    for row in completed:
        arm, seed = row["arm"], row["seed"]
        by_arm[arm][seed] = dict(
            exact_match=row.get("exact_match"),
            held_out_completion_nll=row.get("held_out_completion_nll"),
            selection_token_mean_nll=row.get("selection_token_mean_nll"),
            run_id=row["run_id"],
        )
        diagnostics[arm][seed] = dict(
            pooled_relative_frobenius=row.get("pooled_relative_frobenius"),
            max_off_band_fraction=row.get("max_off_band_fraction"),
            pooled_in_band_off_diagonal_fraction=row.get("pooled_in_band_off_diagonal_fraction"),
            any_rotation_active=row.get("any_rotation_active"),
            length_limit_rate=row.get("length_limit_rate"),
        )
    arms = dict()
    for arm in record["arms_included"]:
        seeds = by_arm.get(arm, {})
        band, family = subspace.parse_arm(arm)
        arms[arm] = dict(
            band=band,
            family=family,
            seeds_completed=sorted(seeds),
            missing_seeds=sorted(set(subspace_plan.CONFIRMATION_SEEDS) - set(seeds)),
            outcomes={
                outcome: _mean_sd([values[outcome] for _, values in sorted(seeds.items()) if values.get(outcome) is not None])
                for outcome in OUTCOMES
            },
            diagnostics={key: [diagnostics[arm][seed][key] for seed in sorted(seeds)] for key in ("pooled_relative_frobenius", "max_off_band_fraction", "pooled_in_band_off_diagonal_fraction", "any_rotation_active", "length_limit_rate")},
        )
    contrasts = {}
    for family in record["families"]:
        for high, low in PRIMARY_CONTRASTS:
            high_arm, low_arm = f"{high}_{family}", f"{low}_{family}"
            if high_arm not in by_arm or low_arm not in by_arm:
                continue
            for outcome in OUTCOMES:
                first = {seed: values[outcome] for seed, values in by_arm[high_arm].items() if values.get(outcome) is not None}
                second = {seed: values[outcome] for seed, values in by_arm[low_arm].items() if values.get(outcome) is not None}
                contrasts[f"{family}/{high}_minus_{low}/{outcome}"] = dict(
                    family=family, higher_band=high, lower_band=low, outcome=outcome,
                    primary=(high, low) == ("LEAD", "TAIL"), **paired_difference(first, second),
                )
    rotation = {}
    if "ROT128" in record["families"] and "DIAG" in record["families"]:
        for band in subspace.BAND_ORDER:
            rot_arm, diag_arm = f"{band}_ROT128", f"{band}_DIAG"
            if rot_arm not in by_arm or diag_arm not in by_arm:
                continue
            for outcome in OUTCOMES:
                first = {seed: values[outcome] for seed, values in by_arm[rot_arm].items() if values.get(outcome) is not None}
                second = {seed: values[outcome] for seed, values in by_arm[diag_arm].items() if values.get(outcome) is not None}
                rotation[f"{band}/ROT128_minus_DIAG/{outcome}"] = dict(
                    band=band, outcome=outcome,
                    rotation_active_all_seeds=all(diagnostics[rot_arm][seed]["any_rotation_active"] for seed in by_arm[rot_arm]),
                    **paired_difference(first, second),
                )
    reference = load_reference(rows)
    return dict(
        schema_version=1,
        purpose="decoder_subspace_analysis",
        created_utc=utc_now(),
        confirmation_protocol_sha256=protocol.get("_sha256"),
        primary_outcomes=list(subspace_plan.PRIMARY_OUTCOMES),
        outcome_policy=subspace_plan.OUTCOME_POLICY,
        arms=arms,
        band_contrasts=contrasts,
        rotation_contrasts=rotation,
        frozen_reference=None if reference is None else dict(
            run_id=reference["run_id"],
            exact_match=(reference.get("generation_summary") or {}).get("exact_match"),
            held_out_completion_nll=(reference.get("held_out_completion_nll") or {}).get("token_mean_nll"),
            note="Inference-only anchor for how much adaptation happened; not an arm of the comparison.",
        ),
        enforcement_checks=dict(
            max_off_band_fraction_over_all_runs=max((row.get("max_off_band_fraction") or 0.0) for row in completed) if completed else None,
            note=(
                "Off-band energy near zero is an enforcement check that the update stayed in its band, not a "
                "discovery. If a ROT128 arm never made a meaningful within-band off-diagonal update, the "
                "rotation contrast cannot support an empirical claim about effective rotation."
            ),
        ),
        interpretation_limits=record["not_established"],
        completeness=dict(
            expected_runs=len(record["arms_included"]) * len(subspace_plan.CONFIRMATION_SEEDS),
            completed_runs=len(completed),
            complete=all(not arms[arm]["missing_seeds"] for arm in arms),
            note="Incomplete paired location sets are reported as incomplete; no arm is dropped to balance a table.",
        ),
    )


def learning_curves(rows):
    """Per-run selection NLL trajectory at the registered evaluation steps, from the immutable observations."""
    curves = []
    for row in rows:
        if row.get("status") != "completed" or not row.get("observation_path"):
            continue
        engine = Path(row["observation_path"]).parent
        for observation in sorted(engine.glob("observation_*.json")):
            payload = json.loads(observation.read_text())
            curves.append(
                dict(
                    run_id=row["run_id"], arm=row["arm"], band=row.get("band"), family=row.get("family"),
                    seed=row["seed"], learning_rate=row.get("learning_rate"), step=payload["step"],
                    selection_token_mean_nll=payload["selection_metrics"]["token_mean_nll"],
                    examples=payload.get("examples"), tokens=payload.get("tokens"),
                    pooled_relative_frobenius=(payload.get("diagnostics") or {}).get("pooled", {}).get("pooled_relative_frobenius"),
                )
            )
    return sorted(curves, key=lambda row: (row["arm"], row["seed"], row["step"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True, help="registered confirmation protocol")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    directory = owned_path(resources.output_root, args.output_directory)
    ledger = Path(resources.output_root) / "run_ledger.jsonl"
    protocol = json.loads(args.protocol.read_text())
    protocol["_sha256"] = sha256(args.protocol)
    rows = subspace_plan.collect_runs(ledger, args.protocol, purpose=subspace_plan.CONFIRMATION_PURPOSE)
    directory.mkdir(parents=True, exist_ok=False)
    analysis = summarize(rows, protocol)
    write_json_new(directory / "analysis.json", analysis)
    write_json_new(directory / "run_rows.json", rows)
    curves = learning_curves(rows)
    with (directory / "learning_curves.csv").open("x", encoding="utf-8") as handle:
        columns = ["run_id", "arm", "band", "family", "seed", "learning_rate", "step", "selection_token_mean_nll", "examples", "tokens", "pooled_relative_frobenius"]
        handle.write(",".join(columns) + "\n")
        for row in curves:
            handle.write(",".join("" if row.get(column) is None else str(row.get(column)) for column in columns) + "\n")
    print(json.dumps(dict(output=str(directory), runs=len(rows), curve_rows=len(curves), complete=analysis["completeness"], contrasts=sorted(analysis["band_contrasts"])), indent=1, default=str))


if __name__ == "__main__":
    main()
