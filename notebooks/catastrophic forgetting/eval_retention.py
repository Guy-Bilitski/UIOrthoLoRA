"""
Out-domain retention eval (BBH + MMLU-Pro) via EleutherAI lm-eval, in the modern
env. Used for the reproduction-campaign retention leg. Reproduces the CLoRA-paper
base numbers (gate: base LLaMA-2-7B BBH 34.91 / MMLU-Pro 18.56) before any
conclusion about an adapter is drawn.

Defaults: `bbh` (group = bbh_cot_fewshot, 3-shot CoT) and `mmlu_pro` (5-shot CoT) —
lm-eval's standard task defaults. Pass --adapter to evaluate a trained PEFT adapter
(any registered type incl. UIOrthoLoRA); omit it for the base model.
"""
import os
import json
import time
import argparse

import torch
import run_lib

HERE = run_lib.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter", default="", help="path to PEFT adapter dir; empty = base model")
    ap.add_argument("--tasks", default="bbh,mmlu_pro")
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--limit", type=float, default=0, help="0=full; >0 fraction or count for a quick check")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tasks = args.tasks.split(",")
    model_args = dict(pretrained=args.base_model, dtype=torch.bfloat16, batch_size=args.batch_size)
    if args.adapter:
        model_args["peft"] = args.adapter
    lm = HFLM(**model_args)

    run_name = args.run_name or (os.path.basename(os.path.normpath(args.adapter)) if args.adapter else "base_l2-7b")
    print(f"[retention] run={run_name} tasks={tasks} adapter={args.adapter or '(base)'}", flush=True)

    t0 = time.time()
    res = simple_evaluate(model=lm, tasks=tasks,
                          limit=(args.limit if args.limit > 0 else None),
                          bootstrap_iters=0)
    dt = time.time() - t0

    # pull the group-level exact_match for each requested task
    scores = {}
    for t in tasks:
        row = res["results"].get(t, {})
        # group rows expose 'exact_match,<filter>' keys; grab the first exact_match*
        em = next((v for k, v in row.items() if k.startswith("exact_match") and "stderr" not in k), None)
        scores[t] = em
    print(f"[retention] run={run_name} scores={scores} ({dt:.0f}s)", flush=True)

    out_path = args.out or os.path.join(HERE, "results", run_name, "retention.json")
    summary = {"run_name": run_name, "adapter": args.adapter or "(base)", "base_model": args.base_model,
               "tasks": tasks, "scores": scores, "results": res["results"],
               "n_samples": {t: res.get("n-samples", {}).get(t) for t in tasks},
               "runtime_s": round(dt, 1), "git_commit": run_lib.git_commit(),
               "evaluated_at": run_lib.now_iso()}
    run_lib.write_json(out_path, summary)
    reg = {"run_name": run_name, "kind": "retention", "adapter": args.adapter or "(base)",
           "scores": scores, "git_commit": run_lib.git_commit(), "evaluated_at": run_lib.now_iso()}
    run_lib.append_registry("eval_registry.jsonl", reg)
    print(f"[retention] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
