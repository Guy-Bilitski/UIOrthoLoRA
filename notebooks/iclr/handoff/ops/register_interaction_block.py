"""Register block 2, the decoder interaction control study (UNREG / MIX / NORM).

Stage 1 only: the sealed design plus the norms-only NORM calibration grid. Confirmations are
registered later, by register_interaction_confirmation.py, from the frozen NORM decision record.
Every design value is carried from a record, not chosen here:

  tail_size 512        one equal third of Qwen's 1536 hidden size, the same fraction of the
                       spectrum as the encoder practical study's 256 of 768.
  mix_coefficient 1e-3 the registered practical dose of the completed encoder study
                       (focused_confirmation_rte_20260915.json, P1_MIX), a fixed design value.
  learning_rate 1e-3   the fixed common recipe of DECODER_SCOPE_REVIEW_20260919.md, identical to
                       the rate registered for both band families in block 1.
  batch 2 x accum 8    the memory-revised recipe of block 1; effective batch 16 unchanged.
  842 steps, 640 tokens, bfloat16, generation 640 new tokens at batch 16: block 1 exactly.
"""
import argparse, json, shutil
from pathlib import Path

from notebooks.iclr.decoder_pilot import interaction_plan as ip

CHECKOUT = Path(__file__).resolve().parents[4]
BAND_ROOT = CHECKOUT / "campaign_outputs_decoder_subspace_v1"

PROVENANCE = (
    "Block 2 of the author-approved five-day plan (2026-09-19). Design carried from the completed "
    "encoder practical study (mix dose, adapter form) and from the block 1 band study (recipe, "
    "budget, decode settings). No value here is a new tuned choice."
)


def build_design():
    return ip.design(
        projections=("q_proj", "o_proj"),
        tail_size=512,
        learning_rate=1e-3,
        mix_coefficient=1e-3,
        optimizer_steps=842,
        max_length=640,
        batch_size=2,
        accumulation_steps=8,
        precision="bfloat16",
        gradient_checkpointing=False,
        initial_scaler=0.01,
        initial_coefficient=0.01,
        generation_max_new_tokens=640,
        generation_batch_size=16,
        selection_eval_steps=(0, 211, 421, 632, 842),
        provenance=PROVENANCE,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if (root / "protocols/calibration.json").exists():
        raise SystemExit(f"Calibration is already registered at {root}; registration is not repeated")
    for sub in ("inputs", "protocols", "decisions", "logs", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    prepared_source = BAND_ROOT / "inputs/prepared.json"
    prepared_path = root / "inputs/prepared.json"
    shutil.copy2(prepared_source, prepared_path)
    prepared = json.loads(prepared_path.read_text())
    for key in ("model", "dataset", "svd_reference_cache"):
        if not Path(prepared[key]).exists():
            raise SystemExit(f"Sealed input is missing: {prepared[key]}")

    record = build_design()
    (root / "design.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    protocol = ip.register_calibration(prepared_path, record, args.authorization)
    (root / "protocols/calibration.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (root / "run_ledger.jsonl").touch()

    print(json.dumps(dict(root=str(root), entries=[e["entry_id"] for e in protocol["entries"]],
                          mix_coefficient=record["mix_coefficient"], tail_size=record["tail_size"],
                          learning_rate=record["learning_rate"], optimizer_steps=record["optimizer_steps"],
                          batch_size=record["batch_size"], accumulation_steps=record["accumulation_steps"]), indent=2))


if __name__ == "__main__":
    main()
