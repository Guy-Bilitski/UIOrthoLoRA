"""A short CPU process for supervision tests, not a scientific training worker."""

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    assert job["synthetic_cpu_test"] is True and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    directory = args.job.parent
    print("Synthetic CPU supervisor fixture; no torch import and no training", flush=True)
    if job["fixture_action"] == "fail":
        (directory / "failed_fixture_artifact").write_text("keep me")
        return 7
    if job["fixture_action"] == "await_stop":
        deadline = time.monotonic() + 10  # Always independently bounded.
        while time.monotonic() < deadline:
            if (directory / "stop_request.json").exists():
                (directory / "boundary_state_fixture").write_text("safe synthetic boundary")
                return 0
            time.sleep(0.01)
        return 9
    if job["fixture_action"] == "ignore_stop":
        time.sleep(10)
        return 9
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
