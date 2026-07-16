"""Evacuation final-merge: rebuild the merged aggregates from per-run artifacts so the
git-pushed snapshot is self-contained even if the ad-hoc merged files were stale.

- results/*/geo.json        -> results/geo_drift/adapter_metrics_merged.jsonl (key: run)
- results/*/forgetting.json -> results/forgetting_merged.jsonl               (key: run_name)

Existing merged rows are kept when no per-run file exists (union, per-run file wins).
"""
import glob, json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)


def merge(per_run_name, merged_path, key):
    rows = {}
    if os.path.isfile(merged_path):
        for line in open(merged_path):
            try:
                d = json.loads(line)
                if d.get(key):
                    rows[d[key]] = d
            except Exception:
                pass
    n_old = len(rows)
    for f in glob.glob(os.path.join("results", "*", per_run_name)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        k = d.get(key) or os.path.basename(os.path.dirname(f))
        d.setdefault(key, k)
        rows[k] = d
    tmp = merged_path + ".tmp"
    os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
    with open(tmp, "w") as fh:
        for k in sorted(rows):
            fh.write(json.dumps(rows[k]) + "\n")
    os.replace(tmp, merged_path)
    print(f"[evac_merge] {merged_path}: {n_old} -> {len(rows)} rows")


merge("geo.json", "results/geo_drift/adapter_metrics_merged.jsonl", "run")
merge("forgetting.json", "results/forgetting_merged.jsonl", "run_name")
