"""General, RESUMABLE job generator for a (model x domain) sweep campaign.

Emits the full target grid (8 adapter arms x 9 LRs x 3 seeds = 216 cells) for the given
base model + task domain, MINUS any cell that already has results/<run>/summary.json. So
re-running it always produces exactly the work that still needs doing -> the orchestrator
resumes cleanly after any interruption (HW reclaimed, crash, etc.).

  python make_campaign_jobs.py --prefix qwsw --base_model Qwen/Qwen2.5-7B --adapt_task cs --out jobs/auto_qwcs.txt
  python make_campaign_jobs.py --prefix lrswm --base_model meta-llama/Llama-2-7b-hf \
         --data_path repro/LLM-Adapters/ft-training_set/metamathqa_100k.json --adapt_task gsm8k --out jobs/auto_l2m.txt
"""
import os, glob, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/guy/UIOrthoLoRA/.venv/bin/python"

ARMS = {
    "lora_r16":      "--method lora --lora_r 16 --lora_alpha 32",
    "dora_r16":      "--method lora --use_dora 1 --lora_r 16 --lora_alpha 32",
    "corda_r16":     "--method lora --corda 1 --lora_r 16 --lora_alpha 16",
    "clora_k1024":   "--method clora --clora_k 1024 --lora_r 32 --lora_alpha 64",
    "lorawd_wd0p3":  "--method lora --lora_r 32 --lora_alpha 64 --weight_decay 0.3",
    "milora_r32":    "--method lora --milora 1 --lora_r 32 --lora_alpha 32",
    "sclora_r32":    "--method lora --sclora 1 --sclora_beta 0.5 --lora_r 32 --lora_alpha 32",
    "lora_null_r16": "--method lora --lora_null 1 --lora_r 16 --lora_alpha 16",
}
LRS = [("2e5", "2e-05"), ("5e5", "5e-05"), ("1e4", "1e-04"), ("2e4", "0.0002"),
       ("3e4", "0.0003"), ("5e4", "0.0005"), ("1e3", "0.001"), ("2e3", "0.002"), ("5e3", "0.005")]
SEEDS = ["42", "43", "44"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="run-name prefix, e.g. lrsw / qwsw / lrswm / qwswm")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--data_path", default="", help="train data; empty = train_cs default (commonsense_170k)")
    ap.add_argument("--adapt_task", choices=["cs", "gsm8k"], default="cs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    done = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(HERE, "results/*/summary.json"))}
    data_arg = f" --data_path {a.data_path}" if a.data_path else ""

    lines, skipped = [], 0
    for arm, flags in ARMS.items():
        for tok, val in LRS:
            for s in SEEDS:
                run = f"{a.prefix}_{arm}_lr{tok}_s{s}"
                if run in done:
                    skipped += 1
                    continue
                train = (f"{PY} train_cs.py {flags} --learning_rate {val} --seed {s} "
                         f"--base_model {a.base_model}{data_arg} --run_name {run}")
                ev = (f"{PY} eval_one_gpu.py --adapter /scratch/cf_models/{run} --run_name {run} "
                      f"--base_model {a.base_model} --adapt_task {a.adapt_task} "
                      f"--ret_suite broad --ret_limit 0 --ret_max_gen 512")
                lines.append(f"{train} && {ev}")

    with open(a.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"[{a.prefix}] target=216  done={skipped}  remaining={len(lines)} -> {a.out}")


if __name__ == "__main__":
    main()
