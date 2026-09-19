"""Run the exploratory NLL partition over every validated block 1 endpoint.

Nineteen states: the eighteen confirmation endpoints and the frozen full-test reference. The
partition is exploratory and is never reported as a registered outcome; it exists to say WHERE on
the sequence the band difference in completion loss sits, not to add an outcome. Each state is
bound to its registered endpoint by the partition module itself, so a run that was not validated
under the confirmation protocol cannot enter.

Already-written outputs are skipped, so an interrupted pass resumes without recomputing.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from notebooks.iclr.decoder_pilot import subspace_plan as sp

CHECKOUT = Path(__file__).resolve().parents[4]


def endpoints(root):
    """Every validated endpoint, as (name, run_directory, protocol_path)."""
    confirmation = root / "protocols/confirmation.json"
    ledger = root / "run_ledger.jsonl"
    rows = sp.collect_runs(ledger, confirmation, purpose=sp.CONFIRMATION_PURPOSE)
    found = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        found.append((row["entry_id"].replace("/", "_"), Path(row["job_path"]).parent, confirmation))
    reference = root / "protocols/reference_held_aside.json"
    for row in sp.collect_runs(ledger, reference, purpose=sp.REFERENCE_PURPOSE):
        if row.get("status") != "completed":
            continue
        found.append((row["entry_id"].replace("/", "_"), Path(row["job_path"]).parent, reference))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--expected", type=int, default=19)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    args = parser.parse_args()

    root = args.root.resolve()
    states = endpoints(root)
    if len(states) != args.expected:
        raise SystemExit(f"Expected {args.expected} validated endpoints, found {len(states)}; refusing a partial pass")
    out_dir = root / "loss_partition"
    out_dir.mkdir(exist_ok=True)
    done, ran = [], []
    for name, run_directory, protocol in states:
        output = out_dir / f"{name}.json"
        if output.exists():
            done.append(name)
            continue
        command = [str(CHECKOUT / ".venv/bin/python"), "-m", "notebooks.iclr.decoder_pilot.nll_partition",
                   "--resources", str(args.resources), "--gpu", str(args.gpu), "--protocol", str(protocol),
                   "--run-directory", str(run_directory), "--eval-batch-size", str(args.eval_batch_size),
                   "--output", str(output)]
        print(f"partition {name}", flush=True)
        result = subprocess.run(command, cwd=CHECKOUT, env={**__import__("os").environ, "PYTHONPATH": f"{CHECKOUT}:{CHECKOUT}/src"})
        if result.returncode != 0:
            raise SystemExit(f"Partition failed on {name}; the pass stops rather than reporting a partial set")
        ran.append(name)
    print(json.dumps(dict(states=len(states), computed=ran, already_present=done, output_directory=str(out_dir)), indent=2))


if __name__ == "__main__":
    main()
