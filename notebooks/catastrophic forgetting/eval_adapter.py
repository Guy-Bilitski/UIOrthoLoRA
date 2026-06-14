"""
Full eval of ONE trained adapter, then (optionally) delete its checkpoint to keep
disk bounded — we only need the result JSONs, not the weights. Runs all three legs
8-way sharded: in-domain CS (8 datasets), out-domain retention (BBH-AO + MMLU-Pro),
and the F-delta mechanism metric. Writes results/<run>/summary.json.

    python eval_adapter.py --adapter /scratch/cf_models/uio_kval2048_kvec410 --delete
"""
import os
import json
import argparse
import subprocess
import shutil

import run_lib
import run_cs_eval
import run_retention

HERE = run_lib.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--delete", action="store_true", help="rm the checkpoint after eval")
    ap.add_argument("--no_fdelta", action="store_true")
    args = ap.parse_args()

    run_name = os.path.basename(os.path.normpath(args.adapter))
    print(f"==== EVAL {run_name} ({args.adapter}) ====", flush=True)

    # 1) in-domain commonsense (8 datasets, one per GPU)
    subprocess.run(["python", "run_cs_eval.py", "--adapter", args.adapter, "--gpus", str(args.gpus),
                    "--base_model", args.base_model], cwd=HERE, check=False)
    # 2) out-domain retention (sharded)
    subprocess.run(["python", "run_retention.py", "--adapter", args.adapter, "--gpus", str(args.gpus),
                    "--base_model", args.base_model], cwd=HERE, check=False)
    # 3) F-delta mechanism metric (single GPU)
    if not args.no_fdelta:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
        with open(os.path.join(HERE, "logs", f"{run_name}__fdelta.log"), "w") as lf:
            subprocess.run(["python", "fdelta.py", "--adapter", args.adapter], cwd=HERE, env=env,
                           stdout=lf, stderr=subprocess.STDOUT, check=False)

    # collect into one summary
    rdir = os.path.join(HERE, "results", run_name)
    summary = {"run_name": run_name}
    for fname, key in [("cs_agg.json", "cs"), ("retention_agg.json", "retention"), ("fdelta.json", "fdelta")]:
        p = os.path.join(rdir, fname)
        if os.path.exists(p):
            summary[key] = json.load(open(p))
    cs = summary.get("cs", {}).get("cs_avg")
    ret = summary.get("retention", {}).get("scores", {})
    ret_mean = round((ret.get("bbh", 0) + ret.get("mmlu_pro", 0)) / 2, 2) if ret else None
    summary["headline"] = {"cs_avg": cs, "bbh": ret.get("bbh"), "mmlu_pro": ret.get("mmlu_pro"),
                           "retention_mean": ret_mean,
                           "fdelta": summary.get("fdelta", {}).get("fdelta_token_weighted"),
                           "dw_sv_max": summary.get("fdelta", {}).get("dw_sv_max")}
    run_lib.write_json(os.path.join(rdir, "summary.json"), summary)
    run_lib.append_registry("campaign_summary.jsonl", {"run_name": run_name, **summary["headline"],
                                                       "evaluated_at": run_lib.now_iso()})
    print(f"==== {run_name} HEADLINE: {summary['headline']} ====", flush=True)

    if args.delete:
        shutil.rmtree(args.adapter, ignore_errors=True)
        print(f"[eval_adapter] deleted checkpoint {args.adapter}", flush=True)


if __name__ == "__main__":
    main()
