"""
Sharded retention eval: distributes the 27 BBH + 14 MMLU-Pro subtasks across N
GPUs (size-balanced LPT bin-packing), runs each shard via eval_retention.py, then
aggregates size-weighted exact_match per group — matching lm-eval's
`weight_by_size` aggregation for the `bbh` and `mmlu_pro` groups.

    python run_retention.py --run_name base_l2-7b               # base model
    python run_retention.py --adapter models/lora_cs_l2-7b_r32  # an adapter
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

BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
    "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]
# approx test sizes for LPT balancing (BBH ~250 each)
MMLU_PRO_SIZES = {
    "biology": 717, "business": 789, "chemistry": 1132, "computer_science": 410,
    "economics": 844, "engineering": 969, "health": 818, "history": 381, "law": 1101,
    "math": 1351, "other": 924, "philosophy": 499, "physics": 1299, "psychology": 798,
}


def build_subtask_sizes(bbh_prefix="bbh_cot_fewshot_", include_mmlu=True):
    sizes = {}
    for s in BBH_SUBTASKS:
        sizes[f"{bbh_prefix}{s}"] = 250
    if include_mmlu:
        for cat, n in MMLU_PRO_SIZES.items():
            sizes[f"mmlu_pro_{cat}"] = n
    return sizes


def lpt_pack(sizes, n_bins):
    bins = [[] for _ in range(n_bins)]
    loads = [0] * n_bins
    for task, sz in sorted(sizes.items(), key=lambda kv: -kv[1]):
        i = loads.index(min(loads))
        bins[i].append(task)
        loads[i] += sz
    return bins, loads


def run_shard(gpu, shardq, base_model, adapter, run_name, results):
    while True:
        try:
            idx, subtasks = shardq.get_nowait()
        except queue.Empty:
            return
        tasks_arg = ",".join(subtasks)
        shard_name = f"{run_name}__retshard{idx}"
        logpath = os.path.join(HERE, "logs", f"{shard_name}.log")
        cmd = ["python", "eval_retention.py", "--tasks", tasks_arg, "--batch_size", "auto",
               "--run_name", shard_name, "--base_model", base_model]
        if adapter:
            cmd += ["--adapter", adapter]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   HF_HUB_DISABLE_XET="1")
        print(f"[ret] GPU{gpu} START shard{idx} ({len(subtasks)} tasks)", flush=True)
        t0 = time.time()
        with open(logpath, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
        print(f"[ret] GPU{gpu} DONE shard{idx} rc={rc} {time.time()-t0:.0f}s", flush=True)
        results.append((idx, shard_name, rc))


def aggregate(run_name, sizes):
    """Read all shard result jsons, compute size-weighted exact_match per group."""
    groups = {"bbh": [], "mmlu_pro": []}
    per_subtask = {}
    n = 0
    while True:
        path = os.path.join(HERE, "results", f"{run_name}__retshard{n}", "retention.json")
        if not os.path.exists(path):
            break
        data = json.load(open(path))
        for tname, row in data["results"].items():
            em = next((v for k, v in row.items() if k.startswith("exact_match") and "stderr" not in k), None)
            if em is None:
                continue
            sz = sizes.get(tname, row.get("sample_len", 0))
            per_subtask[tname] = {"exact_match": float(em), "n": sz}
            grp = "bbh" if tname.startswith("bbh") else "mmlu_pro"
            groups[grp].append((float(em), sz))
        n += 1
    out = {}
    for grp, vals in groups.items():
        if vals:
            tot = sum(s for _, s in vals)
            out[grp] = round(100 * sum(em * s for em, s in vals) / tot, 2)
    return out, per_subtask, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu_ids", default="", help="comma list overriding --gpus, e.g. 5,6,7")
    ap.add_argument("--aggregate_only", action="store_true")
    # answer-only 3-shot BBH (bbh_fewshot) reproduces CLoRA base 34.91 (got 33.1);
    # CoT (bbh_cot_fewshot) runs ~39.5 and is ~15x slower. Canonical = answer-only.
    ap.add_argument("--bbh_prefix", default="bbh_fewshot_")
    ap.add_argument("--include_mmlu", type=int, default=1)
    args = ap.parse_args()

    run_name = args.run_name or (os.path.basename(os.path.normpath(args.adapter)) if args.adapter else "base_l2-7b")
    sizes = build_subtask_sizes(bbh_prefix=args.bbh_prefix, include_mmlu=bool(args.include_mmlu))

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else list(range(args.gpus))
    if not args.aggregate_only:
        bins, loads = lpt_pack(sizes, len(gpu_ids))
        print(f"[ret] {run_name}: {len(sizes)} subtasks -> GPUs {gpu_ids}; loads={loads}", flush=True)
        shardq = queue.Queue()
        for i, b in enumerate(bins):
            if b:
                shardq.put((i, b))
        results = []
        threads = [threading.Thread(target=run_shard,
                   args=(g, shardq, args.base_model, args.adapter, run_name, results))
                   for g in gpu_ids]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f"[ret] all shards done {time.time()-t0:.0f}s; failures={[r for r in results if r[2]!=0]}", flush=True)

    scores, per_subtask, nshards = aggregate(run_name, sizes)
    summary = {"run_name": run_name, "adapter": args.adapter or "(base)", "kind": "retention_agg",
               "scores": scores, "per_subtask": per_subtask, "n_shards": nshards,
               "git_commit": run_lib.git_commit(), "evaluated_at": run_lib.now_iso()}
    out_path = os.path.join(HERE, "results", run_name, "retention_agg.json")
    run_lib.write_json(out_path, summary)
    run_lib.append_registry("eval_registry.jsonl",
                            {k: summary[k] for k in ("run_name", "kind", "adapter", "scores",
                                                     "git_commit", "evaluated_at")})
    print(f"\n[ret] {run_name} AGG scores={scores}  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
