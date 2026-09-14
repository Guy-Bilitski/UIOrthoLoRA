"""Write an immutable two-task timing protocol before launching its workers."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .artifacts import utc_now, write_json_new
from .engine import TrainSettings
from .protocol import RECIPE_REFERENCE, Resources, owned_path
from .spectral import SpectralConfig


def timing_protocol(steps=128):
    tasks = {}
    for task, recipe in RECIPE_REFERENCE.items():
        settings = TrainSettings(
            seed=31415,
            max_steps=steps,
            non_head_lr=recipe["adapter_lr"],
            head_lr=recipe["head_lr"],
            weight_decay=0.0,
            warmup_steps=0,
            accumulation_steps=4,
            eval_every_steps=32,
            max_gradient_norm=1.0,
            precision="float32",
            task=task,
        )
        settings.validate()
        tasks[task] = dict(
            condition="P1_MIX",
            seed=31415,
            head_seed=31415,
            batch_seed=31415,
            train_settings=asdict(settings),
            batch_size=8,
            eval_batch_size=16,
            probe_batch_size=8,
            spectral_config=asdict(
                SpectralConfig(tail_size=256, initial_scaler=recipe["scaler"], initial_coefficient=recipe["sigma"])
            ),
            regularization_coefficient=1e-3,
            random_projector_seeds={},
            diagnostic_device="cpu",
            diagnostic_workers=4,
            diagnostic_cutoffs=[16, 64, 128, 256, 512],
            orientation_seeds=[[17, 42], [123, 2021], [1054, 31415]],
            attention_implementation="eager",
            cost_exclude_initial_steps=16,
        )
    return dict(
        schema_version=1,
        purpose="throughput_only",
        selection_allowed=False,
        created_utc=utc_now(),
        tasks=tasks,
        concurrency="one independent worker per explicitly assigned GPU; four CPU diagnostic workers per run, two intraop threads",
        interpretation="These short seed-31415 pilots measure throughput, utilization, memory and overhead. They are not magnitude calibration, a frozen confirmation endpoint, or confirmatory outcomes.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    output = owned_path(resources.output_root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, timing_protocol())
    print(f"Immutable timing protocol: {output}")


if __name__ == "__main__":
    main()
