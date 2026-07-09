"""
Fair-LR sweep + faithful SC-LoRA beta-curve, per domain. Answers the thesis question:
can ANY engineered method (CLoRA/DoRA/CorDA/MiLoRA/SC-LoRA), at its OWN best LR and
faithful config, beat the simple baselines (vanilla LoRA, LoRA+wd)?

Two blocks:
  (A) LR sweep: 7 representative configs x 7 LRs (2e-5..1e-3), 1 seed -> best LR per method.
  (B) faithful SC-LoRA beta-curve: repo config (output-side, NQ-open D-), beta in the paper's
      regime {0,0.5,0.8,0.9,0.99} at r=128 (3 seeds) + r=32 (seed42), LR 3e-4.

Usage:  python gen_lrsweep.py [cs|math]
  cs   -> jobs/lr_sweep.txt   (commonsense, adapt=CS-8)
  math -> jobs/lr_sweep_math.txt (MetaMathQA->GSM8K, cutoff 512)
Run names: lrsw_/scl2_ (cs), lrswm_/scl2m_ (math) -> no collisions across phases.
"""
import sys
PY = "/home/guy/UIOrthoLoRA/.venv/bin/python"
EVAL = "--ret_suite broad --ret_limit 0 --ret_max_gen 512"

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "cs"
if DOMAIN == "math":
    DATA = " --data_path repro/LLM-Adapters/ft-training_set/metamathqa_100k.json --cutoff_len 512"
    ADAPT = " --adapt_task gsm8k"
    LRP, SCP, OUT = "lrswm", "scl2m", "jobs/lr_sweep_math.txt"
else:
    DATA, ADAPT, LRP, SCP, OUT = "", "", "lrsw", "scl2", "jobs/lr_sweep.txt"


def jb(targs, run):
    return (f"{PY} train_cs.py {targs}{DATA} --run_name {run} && "
            f"{PY} eval_one_gpu.py --adapter /scratch/cf_models/{run} --run_name {run} {EVAL}{ADAPT}")


def bt(b):
    return "b" + str(b).replace('.', 'p')


jobs = []
# (A) LR sweep — representative config per method, 7 LRs, 1 seed
CONFIGS = [
    ("lora_r16",     "--method lora --lora_r 16 --lora_alpha 32"),
    ("lorawd_wd0p3", "--method lora --lora_r 32 --lora_alpha 64 --weight_decay 0.3"),
    ("clora_k1024",  "--method clora --clora_k 1024 --lora_r 32 --lora_alpha 64"),
    ("dora_r16",     "--method lora --use_dora 1 --lora_r 16 --lora_alpha 32"),
    ("corda_r16",    "--method lora --corda 1 --lora_r 16 --lora_alpha 16"),
    ("milora_r32",   "--method lora --milora 1 --lora_r 32 --lora_alpha 32"),
    ("sclora_r32",   "--method lora --sclora 1 --sclora_beta 0.5 --lora_r 32 --lora_alpha 32"),
]
LRS = [(2e-5, "2e5"), (5e-5, "5e5"), (1e-4, "1e4"), (2e-4, "2e4"), (3e-4, "3e4"), (5e-4, "5e4"), (1e-3, "1e3")]
for tag, targs in CONFIGS:
    for lr, lrtag in LRS:
        jobs.append(jb(f"{targs} --learning_rate {lr} --seed 42", f"{LRP}_{tag}_lr{lrtag}_s42"))

# (B) faithful SC-LoRA beta-curve (repo config), at paper r=128 (3 seeds) + r=32 (seed42)
for s in (42, 43, 44):
    for b in (0.0, 0.5, 0.8, 0.9, 0.99):
        jobs.append(jb(f"--method lora --sclora 1 --sclora_beta {b} --lora_r 128 --lora_alpha 128 --seed {s}",
                       f"{SCP}_{bt(b)}_r128_s{s}"))
for b in (0.0, 0.5, 0.9):
    jobs.append(jb(f"--method lora --sclora 1 --sclora_beta {b} --lora_r 32 --lora_alpha 32 --seed 42",
                   f"{SCP}_{bt(b)}_r32_s42"))

with open(OUT, "w") as f:
    f.write("\n".join(jobs) + "\n")
print(f"[{DOMAIN}] {len(jobs)} jobs -> {OUT}  (49 LR-sweep + {len(jobs)-49} faithful SC-LoRA)")
