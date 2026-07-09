"""
In-domain commonsense eval orchestrator: runs the 8 LLM-Adapters datasets for one
adapter across N GPUs (one dataset per GPU at a time) via eval_cs.py, then
aggregates into the 8-dataset average (the MiLoRA/DoRA protocol; gate: LoRA ~79.9).

    python run_cs_eval.py --adapter models/lora_cs_l2-7b_r32
"""
import os
import json
import time
import argparse
import subprocess
import threading
import queue

import run_lib

HERE = run_lib.HERE
CS_DATASETS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
               "ARC-Easy", "ARC-Challenge", "openbookqa"]


def run_one(gpu, jobq, adapter, base_model, results):
    while True:
        try:
            ds = jobq.get_nowait()
        except queue.Empty:
            return
        run_name = os.path.basename(os.path.normpath(adapter))
        logpath = os.path.join(HERE, "logs", f"{run_name}__cs_{ds}.log")
        cmd = ["python", "eval_cs.py", "--lora_weights", adapter, "--dataset", ds,
               "--base_model", base_model, "--batch_size", "32"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   HF_HUB_DISABLE_XET="1")
        print(f"[cs] GPU{gpu} START {ds}", flush=True)
        t0 = time.time()
        with open(logpath, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
        print(f"[cs] GPU{gpu} DONE {ds} rc={rc} {time.time()-t0:.0f}s", flush=True)
        results.append((ds, rc))


def aggregate(adapter):
    run_name = os.path.basename(os.path.normpath(adapter))
    accs = {}
    for ds in CS_DATASETS:
        p = os.path.join(HERE, "results", run_name, f"eval_{ds}.json")
        if os.path.exists(p):
            accs[ds] = round(100 * json.load(open(p))["accuracy"], 2)
    avg = round(sum(accs.values()) / len(accs), 2) if accs else None
    return accs, avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()

    run_name = os.path.basename(os.path.normpath(args.adapter))
    if not args.aggregate_only:
        jobq = queue.Queue()
        for ds in CS_DATASETS:
            jobq.put(ds)
        results = []
        threads = [threading.Thread(target=run_one, args=(g, jobq, args.adapter, args.base_model, results))
                   for g in range(args.gpus)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f"[cs] all done {time.time()-t0:.0f}s; failures={[r for r in results if r[1]!=0]}", flush=True)

    accs, avg = aggregate(args.adapter)
    summary = {"run_name": run_name, "adapter": args.adapter, "kind": "cs_agg",
               "per_dataset": accs, "cs_avg": avg, "git_commit": run_lib.git_commit(),
               "evaluated_at": run_lib.now_iso()}
    run_lib.write_json(os.path.join(HERE, "results", run_name, "cs_agg.json"), summary)
    run_lib.append_registry("eval_registry.jsonl", summary)
    print(f"\n[cs] {run_name} per_dataset={accs}\n[cs] {run_name} CS_AVG={avg}", flush=True)


if __name__ == "__main__":
    main()
