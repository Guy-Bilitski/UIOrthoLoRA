"""Generate the LR-sweep EXTENSION job list = the full target grid MINUS whatever is already
queued in jobs/lr_sweep.txt or already has a completed results/<run>/summary.json. So it never
duplicates the running pool and never re-runs a finished cell.

Full target: 8 method-arms x 9 LRs x 3 seeds = 216 cells.
"""
import os, re, glob
from collections import Counter

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
# (run_name token, --learning_rate value) — value strings mirror the running pool's convention
LRS = [("2e5", "2e-05"), ("5e5", "5e-05"), ("1e4", "1e-04"), ("2e4", "0.0002"),
       ("3e4", "0.0003"), ("5e4", "0.0005"), ("1e3", "0.001"), ("2e3", "0.002"), ("5e3", "0.005")]
SEEDS = ["42", "43", "44"]

existing = set()
jf = os.path.join(HERE, "jobs/lr_sweep.txt")
if os.path.exists(jf):
    for line in open(jf):
        existing.update(re.findall(r"run_name (lrsw_[A-Za-z0-9_]+)", line))
for p in glob.glob(os.path.join(HERE, "results/lrsw_*/summary.json")):
    existing.add(os.path.basename(os.path.dirname(p)))

lines = []
for arm, flags in ARMS.items():
    for tok, val in LRS:
        for s in SEEDS:
            run = f"lrsw_{arm}_lr{tok}_s{s}"
            if run in existing:
                continue
            train = f"{PY} train_cs.py {flags} --learning_rate {val} --seed {s} --run_name {run}"
            ev = (f"{PY} eval_one_gpu.py --adapter /scratch/cf_models/{run} --run_name {run} "
                  f"--ret_suite broad --ret_limit 0 --ret_max_gen 512")
            lines.append(f"{train} && {ev}")

out = os.path.join(HERE, "jobs/lr_sweep_ext.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + ("\n" if lines else ""))

print(f"existing/queued run_names: {len(existing)}")
print(f"wrote {len(lines)} extension jobs -> {out}")
c = Counter(l.split("--run_name ")[1].split(" ")[0].rsplit("_lr", 1)[0] for l in lines)
for a in sorted(c):
    print(f"  {a}: {c[a]}")
