"""Generate fleet job lines for the adversarial-review experiment batch (E3, E6; extendable).

Takes each method's existing exact-recipe shard line as the template (same flags the campaign
used), swaps LR / seed / weight_decay / run_name, and emits ready-to-append lines.
Usage: python gen_adversarial_jobs.py > /dev/null   (writes jobs/fleet/_e3.txt, _e6.txt)
"""
import glob, re, sys

LINES = []
for f in ["jobs/fleet/_all.txt"] + sorted(glob.glob("jobs/fleet/d0*.txt")):
    try:
        LINES += [l.strip() for l in open(f) if "--run_name" in l]
    except FileNotFoundError:
        pass

def template(prefix):
    """Prefer a non-ep6, non-permfail template line whose run_name starts with prefix."""
    cands = [l for l in LINES if re.search(rf"--run_name {re.escape(prefix)}_lr\S+_s4[2-9](\s|$)", l)]
    cands = [l for l in cands if "_ep6" not in l] or cands
    if not cands:
        raise SystemExit(f"no template for {prefix}")
    return cands[0]

def transmute(line, new_rn, lr=None, seed=42, wd=None, epochs=None):
    old_rn = re.search(r"--run_name (\S+?)( |$)", line).group(1)
    line = line.replace(old_rn, new_rn)
    if lr is not None:
        line = re.sub(r"--learning_rate \S+", f"--learning_rate {lr}", line)
    line = re.sub(r"--seed \d+", f"--seed {seed}", line)
    if wd is not None:
        line = re.sub(r"--weight_decay \S+", f"--weight_decay {wd}", line)
    if epochs is not None:
        line = re.sub(r"--num_epochs \d+", f"--num_epochs {epochs}", line)
    return line

# ---- E3: Qwen mid-range densification (knee hole), 7 methods x {7e-5, 1.5e-4} x s42 ----
E3_METHODS = {
    "qwsw":  ["qwsw_lora_r16", "qwsw_dora_r16", "qwsw_milora_r32", "qwsw_sclora_r32",
              "qwsw_lorawd_wd0p3", "qwsw_clora_k1024", "qwsw_lora_null_r16"],
    "qwswm": ["qwswm_lora_r32", "qwswm_dora_r16", "qwswm_milora_r32", "qwswm_sclora_r32",
              "qwswm_lorawd_wd0p3", "qwswm_clora_k1024", "qwswm_lora_null_r16"],
}
E3_LRS = [("7e-5", "lr7e5"), ("1.5e-4", "lr15e5")]
e3 = []
for fam, prefixes in E3_METHODS.items():
    for p in prefixes:
        t = template(p)
        for lrval, lrtok in E3_LRS:
            rn = f"{p}_{lrtok}_s42"
            e3.append(transmute(t, rn, lr=lrval, epochs=3))

# ---- E6: wd0.3 on DoRA / MiLoRA (does the simplest control transfer?), 2 LRs x s42 ----
E6 = [("lrsw_dora_r16",   "lrsw_dorawd_wd0p3_r16"),
      ("lrsw_milora_r32", "lrsw_milorawd_wd0p3_r32")]
E6_LRS = [("2e-4", "lr2e4"), ("5e-4", "lr5e4")]
e6 = []
for src, dst in E6:
    t = template(src)
    for lrval, lrtok in E6_LRS:
        rn = f"{dst}_{lrtok}_s42"
        e6.append(transmute(t, rn, lr=lrval, wd=0.3, epochs=3))

# ---- E4: SC-LoRA eval-matched calibration (b4 arm) — fill the missing LR rungs ----
# b4_sclora already has 5e-5/1e-4/3e-4/5e-4 at n>=3; missing: 2e-5, 2e-4, 1e-3.
e4 = []
t_b4 = template("b4_sclora_r32")
for lrval, lrtok in [("2e-5", "lr2e5"), ("2e-4", "lr2e4"), ("1e-3", "lr1e3")]:
    for seed in (42, 43):
        rn = f"b4_sclora_r32_{lrtok}_s{seed}"
        e4.append(transmute(t_b4, rn, lr=lrval, seed=seed, epochs=3))

# ---- E5: replay baseline — LoRA + 5% nq_open replay, 2 LRs x 2 seeds ----
# (replay=0 baselines already exist: the plain lrsw_lora cells.)
e5 = []
t_l = template("lrsw_lora_r16")
for lrval, lrtok in [("3e-4", "lr3e4"), ("5e-4", "lr5e4")]:
    for seed in (42, 43):
        rn = f"lrsw_lorarep05_r16_{lrtok}_s{seed}"
        l = transmute(t_l, rn, lr=lrval, seed=seed, epochs=3)
        l = l.replace("train_cs.py ", "train_cs.py --replay_frac 0.05 ", 1)
        e5.append(l)

# ---- E7: 7B bridging arm — MedMCQA 30k, attention-only targets, LoRA r16, 4 LRs x 2 models ----
# De-confounds the 284B run (model+task+targets changed at once): same task+targets on the 7Bs.
MEDDATA = "repro/LLM-Adapters/ft-training_set/medmcqa_train.json"
e7 = []
for fam, base in (("brl", "meta-llama/Llama-2-7b-hf"), ("brq", "Qwen/Qwen2.5-7B")):
    src = "lrsw_lora_r16" if fam == "brl" else "qwsw_lora_r16"
    t = template(src)
    for lrval, lrtok in [("1e-4", "lr1e4"), ("3e-4", "lr3e4"), ("5e-4", "lr5e4"), ("1e-3", "lr1e3")]:
        rn = f"{fam}_lora_r16_{lrtok}_s42"
        l = transmute(t, rn, lr=lrval, epochs=3)
        l = re.sub(r"--data_path \S+", f"--data_path {MEDDATA}", l)
        l = re.sub(r"--target_modules \S+", "--target_modules q_proj,k_proj,v_proj,o_proj", l)
        l = l.replace("--adapt_task cs ", "--adapt_task medmcqa ")
        e7.append(l)

# ---- E2: full-FT anchor — Llama-2 CS, dense dW, 3 LRs x s42 ----
PY = "/home/guy/UIOrthoLoRA/.venv/bin/python"
e2 = []
for lrval, lrtok in [("1e-5", "lr1e5"), ("3e-5", "lr3e5"), ("1e-4", "lr1e4")]:
    rn = f"fft_full_{lrtok}_s42"
    e2.append(
        f"{PY} train_cs.py --method lora --full_ft 1 --base_model meta-llama/Llama-2-7b-hf "
        f"--data_path repro/LLM-Adapters/ft-training_set/commonsense_170k.json --cutoff_len 256 "
        f"--num_epochs 3 --learning_rate {lrval} --weight_decay 0.0 --batch_size 16 "
        f"--micro_batch_size 16 --warmup_steps 100 "
        f"--target_modules q_proj,k_proj,v_proj,up_proj,down_proj --dropout 0.05 "
        f"--train_on_inputs 1 --max_samples 0 --seed 42 --run_name {rn} && "
        f"{PY} eval_one_gpu.py --adapter none --run_name {rn} "
        f"--base_model /scratch/cf_models/{rn} --adapt_task cs --ret_suite broad "
        f"--ret_limit 0 --ret_max_gen 512")

for name, batch in (("jobs/fleet/_e3.txt", e3), ("jobs/fleet/_e6.txt", e6),
                    ("jobs/fleet/_e4.txt", e4), ("jobs/fleet/_e5.txt", e5),
                    ("jobs/fleet/_e7.txt", e7), ("jobs/fleet/_e2.txt", e2)):
    with open(name, "w") as fh:
        fh.write("\n".join(batch) + "\n")
    print(f"{name}: {len(batch)} jobs")
    for l in batch:
        print("  ", re.search(r"--run_name (\S+?) ", l).group(1),
              re.search(r"--learning_rate (\S+)", l).group(1),
              "wd=" + re.search(r"--weight_decay (\S+)", l).group(1))
