"""Geometry battery, phase 2 — Qwen variant.

Reuses geo_drift_phase2.process_adapter with the reference subspaces swapped to the
Qwen base SVDs (results/geo_drift/base_svd_qwen, produced by geo_drift_phase1_qwen.py)
and processes ONLY adapters whose adapter_config base_model_name_or_path is Qwen.
Output: results/geo_drift/adapter_metrics_qwen.jsonl (+ permatrix_qwen/), kept separate
from the Llama battery so the two models' fingerprints are never pooled by accident.

CPU-only. Run after phase1_qwen finishes:
  setsid nice -n 10 .venv/bin/python geo_drift_phase2_qwen.py >> logs/geo_qwen.log 2>&1 &
"""
import glob
import json
import os
import time

import geo_drift_phase2 as g2

HERE = os.path.dirname(os.path.abspath(__file__))
g2.BASE_SVD = os.path.join(HERE, "results", "geo_drift", "base_svd_qwen")
g2.PM_DIR = os.path.join(HERE, "results", "geo_drift", "permatrix_qwen")
os.makedirs(g2.PM_DIR, exist_ok=True)
OUT = os.path.join(HERE, "results", "geo_drift", "adapter_metrics_qwen.jsonl")


def is_qwen_adapter(d):
    try:
        cfg = json.load(open(os.path.join(d, "adapter_config.json")))
        return "Qwen" in cfg.get("base_model_name_or_path", "")
    except Exception:
        return False


def main():
    t0 = time.time()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["run"])
            except Exception:
                pass
    n = skipped = 0
    with open(OUT, "a") as fout:
        for d in sorted(glob.glob(os.path.join(g2.SCRATCH, "*"))):
            run = os.path.basename(d)
            if run in done or "__" in run:
                continue
            if not is_qwen_adapter(d):
                skipped += 1
                continue
            try:
                agg = g2.process_adapter(run, d)
            except Exception as e:
                print(f"[geo2q] ERR {run}: {e}", flush=True)
                continue
            if agg:
                fout.write(json.dumps(agg) + "\n")
                fout.flush()
                n += 1
                if n % 10 == 0:
                    print(f"[geo2q] {n} adapters  {time.time()-t0:6.0f}s  last={run}", flush=True)
    print(f"[geo2q] DONE: {n} qwen adapters ({skipped} non-qwen skipped) in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
