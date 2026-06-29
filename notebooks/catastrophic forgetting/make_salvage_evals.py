"""Salvage: find adapters that finished TRAINING but have no results summary (died at eval,
e.g. the contention OOMs), and emit eval-ONLY job lines to recover them without retraining.
Infers base_model / adapt_task / ret_max_gen from the run-name prefix."""
import os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/guy/UIOrthoLoRA/.venv/bin/python"
SCRATCH = "/scratch/cf_models"


def is_math(n):  return n.startswith(("mtxm_", "lrswm_", "qwswm_"))
def is_qwen(n):  return n.startswith(("qwsw_", "qwswm_"))


lines = []
for d in sorted(glob.glob(os.path.join(SCRATCH, "*"))):
    n = os.path.basename(d)
    if not os.path.exists(os.path.join(d, "adapter_model.safetensors")):
        continue  # training didn't finish -> nothing to salvage
    if os.path.exists(os.path.join(HERE, f"results/{n}/summary.json")):
        continue  # already evaluated
    if "_lr2e3_" in n or "_lr5e3_" in n:
        continue  # diverged-to-NaN at extreme LR -> eval crashes (inf/nan); not recoverable
    bm = "Qwen/Qwen2.5-7B" if is_qwen(n) else "meta-llama/Llama-2-7b-hf"
    task = "gsm8k" if is_math(n) else "cs"
    rmg = 256 if is_math(n) else 512
    lines.append(f"{PY} eval_one_gpu.py --adapter {d} --run_name {n} "
                 f"--base_model {bm} --adapt_task {task} --ret_suite broad --ret_limit 0 --ret_max_gen {rmg}")

out = os.path.join(HERE, "jobs/salvage_evals.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + ("\n" if lines else ""))
print(f"salvage eval jobs: {len(lines)} -> {out}")
for l in lines:
    print("  " + l.split("--run_name ")[1].split(" ")[0])
