"""Generate the full curve-tracing matrix (commonsense domain) for jobs/matrix_cs.txt.
8 methods, each sweeping the knob that moves ||dW||_F, x3 seeds, broad retention eval.
Each job = train (full data, 3 epochs) && eval_one_gpu (broad suite, full retention).
Unique run_names (mtx_*) so nothing collides with prior checkpoints."""
import sys
SEEDS = [42, 43, 44]
PY = "/home/guy/UIOrthoLoRA/.venv/bin/python"
EVAL = "--ret_suite broad --ret_limit 0 --ret_max_gen 512"

# domain switch: cs (commonsense, default) or math (MetaMathQA -> GSM8K). Same 34 configs.
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "cs"
if DOMAIN == "math":
    PREFIX = "mtxm"
    # MetaMathQA chain-of-thought is long (p95~513 tok); 256 would truncate the answer.
    DATA = " --data_path repro/LLM-Adapters/ft-training_set/metamathqa_100k.json --cutoff_len 512"
    ADAPT = " --adapt_task gsm8k"
    OUTFILE = "jobs/matrix_math.txt"
else:
    PREFIX, DATA, ADAPT, OUTFILE = "mtx", "", "", "jobs/matrix_cs.txt"


def job(train_args, run):
    tr = f"{PY} train_cs.py {train_args}{DATA} --run_name {run}"
    ev = f"{PY} eval_one_gpu.py --adapter /scratch/cf_models/{run} --run_name {run} {EVAL}{ADAPT}"
    return f"{tr} && {ev}"


jobs = []
for s in SEEDS:
    # 1) LoRA + weight-decay (knob = wd) — the strong baseline
    for wd in [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        t = f"wd{str(wd).replace('.', 'p')}"
        jobs.append(job(f"--method lora --lora_r 32 --lora_alpha 64 --weight_decay {wd} --seed {s}", f"{PREFIX}_lorawd_{t}_s{s}"))
    # 2) LoRA rank curve (wd0, alpha=2r) — r32 already covered by wd0.0 above
    for r in [8, 16, 64, 128]:
        jobs.append(job(f"--method lora --lora_r {r} --lora_alpha {2*r} --seed {s}", f"{PREFIX}_lora_r{r}_s{s}"))
    # 3) CLoRA (knob = k)
    for k in [128, 256, 512, 1024, 2048]:
        jobs.append(job(f"--method clora --clora_k {k} --lora_r 32 --lora_alpha 64 --seed {s}", f"{PREFIX}_clora_k{k}_s{s}"))
    # 4) DoRA (knob = rank, alpha=2r)
    for r in [8, 16, 32, 64]:
        jobs.append(job(f"--method lora --use_dora 1 --lora_r {r} --lora_alpha {2*r} --seed {s}", f"{PREFIX}_dora_r{r}_s{s}"))
    # 5) CorDA-KPA (data-aware, alpha=r, knob = rank)
    for r in [16, 32, 64, 128]:
        jobs.append(job(f"--method lora --corda 1 --lora_r {r} --lora_alpha {r} --seed {s}", f"{PREFIX}_corda_r{r}_s{s}"))
    # 6) MiLoRA (minor-SVD, alpha=r, knob = rank)
    for r in [16, 32, 64, 128]:
        jobs.append(job(f"--method lora --milora 1 --lora_r {r} --lora_alpha {r} --seed {s}", f"{PREFIX}_milora_r{r}_s{s}"))
    # 7) SC-LoRA — handled by the FAITHFUL scl2_/scl2m_ beta-curve (gen_lrsweep), so the
    # matrix only carries it for the legacy commonsense run (deprecated there too). Math excludes it.
    if DOMAIN != "math":
        for b in [0.1, 0.3, 0.5, 0.7]:
            t = f"b{str(b).replace('.', 'p')}"
            jobs.append(job(f"--method lora --sclora 1 --sclora_beta {b} --lora_r 32 --lora_alpha 32 --seed {s}", f"{PREFIX}_sclora_{t}_s{s}"))
        for r in [16, 64]:
            jobs.append(job(f"--method lora --sclora 1 --sclora_beta 0.5 --lora_r {r} --lora_alpha {r} --seed {s}", f"{PREFIX}_sclora_r{r}_s{s}"))

with open(OUTFILE, "w") as f:
    f.write("\n".join(jobs) + "\n")
print(f"{len(jobs)} jobs ({len(jobs)//len(SEEDS)} configs x {len(SEEDS)} seeds)")
