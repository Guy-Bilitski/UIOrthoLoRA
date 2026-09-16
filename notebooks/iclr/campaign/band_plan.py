"""Registration and admission for the band-by-flexibility study (2026-09-15 plan).

RTE first (stage B): 3 bands x {diagonal, rotated-64} x seeds {42,17,123} plus
three original-backbone head-only references, preceded by two timing pilots at
the calibration seed (tail band, both flexibility settings). Strict update
family Delta = U_B H V_B^T via band.BandLinear; no scalers, no leading core,
zero-coefficient/identity-rotation initialization. Task budgets, splits, inputs
and recipes are inherited unchanged from the registered focused protocol's
task_jobs; the study never modifies the focused study's artifacts.
"""

import argparse
import copy
import json
from pathlib import Path

from .artifacts import sha256, utc_now, write_json_new
from .band import BAND_SIZE, BAND_STARTS
from .engine import TrainSettings
from .protocol import CALIBRATION_SEED, CONFIRMATION_SEEDS, Resources, owned_path

BAND_PURPOSE = "band_flexibility_v1"
FLEX = {"DIAG": 0, "ROT64": 64}


def entries(task="rte"):
    rows = []
    for band, start in BAND_STARTS.items():
        for flex, rotation in FLEX.items():
            condition = f"BAND_{band.upper()}_{flex}"
            for seed in CONFIRMATION_SEEDS:
                rows.append(
                    dict(
                        entry_id=f"{task}/{condition}/seed_{seed}",
                        task=task,
                        condition=condition,
                        seed=seed,
                        purpose="confirmation",
                        band_config=dict(band_start=start, band_size=BAND_SIZE, rotation_size=rotation),
                    )
                )
    for seed in CONFIRMATION_SEEDS:
        rows.append(
            dict(
                entry_id=f"{task}/P1_HEAD_BASE/seed_{seed}",
                task=task,
                condition="P1_HEAD_BASE",
                seed=seed,
                purpose="confirmation",
                band_config=None,
            )
        )
    for flex, rotation in FLEX.items():
        rows.append(
            dict(
                entry_id=f"{task}/BAND_TAIL_{flex}/pilot_{CALIBRATION_SEED}",
                task=task,
                condition=f"BAND_TAIL_{flex}",
                seed=CALIBRATION_SEED,
                purpose="timing_pilot",
                band_config=dict(band_start=BAND_STARTS["tail"], band_size=BAND_SIZE, rotation_size=rotation),
            )
        )
    return rows


def materialize_entry(protocol, entry):
    job = copy.deepcopy(protocol["task_jobs"][entry["task"]])
    job["train_settings"]["seed"] = entry["seed"]
    resolved = {
        **job,
        "task": entry["task"],
        "condition": entry["condition"],
        "seed": entry["seed"],
        "head_seed": entry["seed"],
        "batch_seed": entry["seed"],
        "regularization_coefficient": 0.0,
        "band_config": copy.deepcopy(entry["band_config"]),
        "band_entry_id": entry["entry_id"],
        "stage": "confirmation",
        "confirmation_purpose": BAND_PURPOSE,
        "random_projector_seeds": {},
        "matching_status": "not_applicable_band_study",
    }
    return resolved


def validate_band_admission(job, protocol):
    if (
        protocol.get("purpose") != BAND_PURPOSE
        or protocol.get("registered") is not True
        or protocol.get("confirmation_authorized") is not True
        or protocol.get("entries") != entries(protocol.get("task", "rte"))
    ):
        raise ValueError("Require the registered, author-authorized band study protocol")
    bound = protocol.get("focused_protocol", {})
    if not bound.get("path") or sha256(bound["path"]) != bound.get("sha256"):
        raise ValueError("Band study evidence changed or is unbound: focused_protocol")
    if job.get("seed") not in CONFIRMATION_SEEDS + (CALIBRATION_SEED,):
        raise ValueError("Band runs must use a declared confirmation or pilot seed")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("band_entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown band study entry")
    expected = materialize_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Band job differs from its registered paired configuration")
    TrainSettings(**expected["train_settings"]).validate()


def register(focused_protocol_path, authorization, task="rte"):
    focused = json.loads(Path(focused_protocol_path).read_text())
    if focused.get("purpose") != "focused_norm_calibration" or focused.get("registered") is not True:
        raise ValueError("Band study inherits its recipe from the registered focused protocol")
    return dict(
        schema_version=1,
        purpose=BAND_PURPOSE,
        registered=True,
        registered_utc=utc_now(),
        confirmation_authorized=True,
        authorization_record=authorization,
        task=task,
        focused_protocol=dict(path=str(Path(focused_protocol_path).resolve()), sha256=sha256(focused_protocol_path)),
        task_jobs={task: copy.deepcopy(focused["task_jobs"][task])},
        entries=entries(task),
        scope=(
            "Band-by-flexibility study, one task per registration: three equal-thirds bands x "
            "{diagonal, rotated-64} x three seeds, three original-backbone head-only references, "
            "and two tail-band timing pilots at the calibration seed. Strict Delta = U_B H V_B^T "
            "family; unregularized; location comparisons are capacity-equal within a flexibility "
            "setting; rotations add capacity and are not capacity-matched claims."
        ),
        reporting_policy=(
            "Report task adaptation across the six conditions and versus the frozen-backbone "
            "references, with update norm and within-band off-diagonal energy; off-band energy "
            "is an enforcement check, not a finding. No band superiority claim from one task."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--focused-protocol", type=Path, required=True)
    parser.add_argument("--task", default="rte", choices=("rte", "mrpc"))
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    output = owned_path(resources.output_root, args.output)
    protocol = register(args.focused_protocol, args.authorization, task=args.task)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, protocol)
    print(json.dumps(dict(protocol=str(output), sha256=sha256(output), entries=len(protocol["entries"]))))


if __name__ == "__main__":
    main()
