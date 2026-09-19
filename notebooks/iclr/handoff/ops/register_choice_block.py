"""Register block 3, the CommonsenseQA replication of the strict-band study.

Six strict-band arms by three seeds on the same sealed Qwen checkpoint, evaluated once on the
1,221-example public validation split, which this study calls the held-out validation split and
never calls a test split.

Every recipe value is carried or measured, none is tuned here:

  max_length 160      MEASURED. The longest prompt over all 10,962 CommonsenseQA rows is 126
                      tokens, so a full example needs 127 with its single scored answer token.
                      160 guarantees no example is ever truncated.
  optimizer_steps 842 block 1's budget unchanged, so the task is the only thing that changed.
                      At effective batch 16 that is 13,472 examples, about 1.54 epochs of the
                      8,767-example inner training split.
  batch 2 x accum 8   block 1's memory-revised recipe unchanged. This is deliberately
                      conservative rather than measured on this task: block 1 held a 16.3 GiB
                      peak at microbatch 2 with 640-token sequences, and these sequences are
                      capped at 160, so this configuration is strictly dominated on memory by a
                      run that already succeeded. The step time of the first run is reported as
                      the measured cost of the task before the remaining runs are judged.
  learning_rate 1e-3  the fixed common recipe, identical to both families in block 1.

Not established: one rate on one task. This is a replication of the location question on a
different task, not a tuned result, and the rate is not claimed optimal for any band.
"""
import argparse, json, shutil
from pathlib import Path

from notebooks.iclr.campaign.artifacts import sha256
from notebooks.iclr.decoder_pilot import choice_plan as cp

CHECKOUT = Path(__file__).resolve().parents[4]
BAND_ROOT = CHECKOUT / "campaign_outputs_decoder_subspace_v1"

PROVENANCE = (
    "Block 3 of the author-approved five-day plan (2026-09-19): 'Run one CommonsenseQA replication "
    "on the same Qwen checkpoint: the six strict-band conditions x three seeds, plus the frozen "
    "reference. Reserve the public validation split for final evaluation and label it accurately.' "
    "max_length is measured from the pinned dataset; every other value is block 1's, unchanged."
)


def build_design():
    return cp.design(
        projections=("q_proj", "o_proj"),
        band_size=512,
        rotation_size=128,
        arms_included=("LEAD_DIAG", "LEAD_ROT128", "MID_DIAG", "MID_ROT128", "TAIL_DIAG", "TAIL_ROT128"),
        optimizer_steps=842,
        max_length=160,
        batch_size=2,
        accumulation_steps=8,
        precision="bfloat16",
        gradient_checkpointing=False,
        learning_rate=1e-3,
        selection_eval_steps=(0, 211, 421, 632, 842),
        provenance=PROVENANCE,
    )


def write_prepared(root):
    """This block's own inputs record: the same model and SVD cache as block 1, and no GSM8K field.

    Copying block 1's record wholesale would carry a dataset path that this study never reads, which
    is exactly the kind of stale binding a sealed record exists to prevent.
    """
    band = json.loads((BAND_ROOT / "inputs/prepared.json").read_text())
    prepared = dict(schema_version=1, model=band["model"], model_source_sha256=band["model_source_sha256"],
                    svd_reference_cache=band["svd_reference_cache"], svd_reference_sha256=band["svd_reference_sha256"],
                    dataset=str((root / "inputs/commonsense_qa").resolve()),
                    carried_from=str((BAND_ROOT / "inputs/prepared.json").resolve()),
                    carried_from_sha256=sha256(BAND_ROOT / "inputs/prepared.json"))
    for key in ("model", "svd_reference_cache", "dataset"):
        if not Path(prepared[key]).exists():
            raise SystemExit(f"Sealed input is missing: {prepared[key]}")
    if sha256(prepared["svd_reference_cache"]) != prepared["svd_reference_sha256"]:
        raise SystemExit("The SVD reference cache no longer matches its recorded hash")
    path = root / "inputs/prepared.json"
    path.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if (root / "protocols/confirmation.json").exists():
        raise SystemExit(f"Block 3 is already registered at {root}; registration is not repeated")
    for sub in ("protocols", "logs", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    prepared_path = write_prepared(root)
    record = build_design()
    (root / "design.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    protocol = cp.register_confirmation(prepared_path, root / "inputs/commonsense_qa", record, args.authorization)
    (root / "protocols/confirmation.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (root / "run_ledger.jsonl").touch()
    print(json.dumps(dict(root=str(root), entries=len(protocol["entries"]),
                          reference=protocol["reference_entry"]["entry_id"],
                          held_out=record["held_out_split"], max_length=record["max_length"],
                          optimizer_steps=record["optimizer_steps"]), indent=2))


if __name__ == "__main__":
    main()
