"""Qwen scaling ray for the intruder section (new file, analysis-only, 2026-09-16).

The manuscript's strongest magnitude control is the Llama E1 scaling ray: rescale one
trained update to several F_Delta targets, fit the magnitude-retention curve, and show
that the intruder-deleted arm falls off that curve (R^2=0.99, arm B 2.16 points below).
It exists on Llama only. This builds the Qwen counterpart on the highest-forgetting
Qwen cell, CLoRA k1024 3e-4 seed 43 (arm A forgets 8.1 of the base's 48.0 retention
points, so the curve has real dynamic range).

F_Delta is a token-weighted ||dW x||/||x||, hence exactly linear in a global scale on
dW, so scaling every lora_B by c maps F_Delta -> c * F_Delta. Targets are chosen to
bracket the measured arms of the row so B and D can be priced against the curve at
their own measured F_Delta:

    ray    0.100  0.150  [0.210 = arm C, already evaluated]  0.246  [0.288 = arm A]  0.330
    arms                  C (uniform shrink)                 B      A                D (0.337)

Plus three random-direction controls at arm B's F_Delta (0.246): lora_B replaced by
Gaussian noise with per-matrix ||B A||_F matched to the scaled real update. Same
magnitude profile, no trained direction — the floor for "how much of retention is
explained by magnitude alone".

Writes adapters to /home/kfir/cf_models and an eval queue to jobs/qwen_ray_queue.txt
in the tab-separated <run>\t<cmd> format gpu_pool_dyn.py consumes. Does NOT touch the
running campaign, its queue, or any pipeline script. CPU only.
"""
import json, os, shutil
import torch
from safetensors.torch import load_file, save_file

SRC = "tia1_qwsw_clora_k1024_lr3e4_s43"
CF = "/home/kfir/cf_models"
EVAC = "/home/kfir/tierA_evac"
PY = "/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python"
BASE = "Qwen/Qwen2.5-7B"
TARGETS = [(0.100, "f100"), (0.150, "f150"), (0.246, "f246"), (0.330, "f330")]
CTRL_TARGET = (0.246, "f246")   # arm B's measured F_Delta
N_CTRL = 3

CMD = (
    f"{PY} eval_one_gpu.py --adapter {CF}/{{rn}} --run_name {{rn}} --base_model {BASE} "
    "--adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512"
    f" && {PY} forgetting_ce.py --runs {{rn}} --adapters_root {CF} --base_model {BASE} "
    "--max_length 1024 --max_blocks 0 --batch_size 2"
    f" && bash evacuate_cell.sh {CF}/{{rn}} {EVAC}"
)


def save_variant(src_dir, dst_dir, tensors):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    for f in os.listdir(src_dir):
        if f != "adapter_model.safetensors":
            shutil.copy2(os.path.join(src_dir, f), dst_dir)
    save_file(tensors, os.path.join(dst_dir, "adapter_model.safetensors"))


def main():
    fd0 = json.load(open(f"results/{SRC}__rl50/summary.json"))["headline"]["fdelta"]
    src_dir = f"{CF}/{SRC}"
    T = load_file(f"{src_dir}/adapter_model.safetensors")
    bkeys = [k for k in T if k.endswith("lora_B.weight")]
    assert bkeys, SRC
    print(f"[ray] source {SRC}: F_delta {fd0:.4f}, {len(bkeys)} B matrices")

    jobs, scaled_fro = [], {}
    for target, tag in TARGETS:
        c = target / fd0
        out = {k: (v * c if k.endswith("lora_B.weight") else v).contiguous().clone()
               for k, v in T.items()}
        rn = f"{SRC}__ray{tag}"
        save_variant(src_dir, f"{CF}/{rn}", out)
        jobs.append((rn, CMD.format(rn=rn)))
        print(f"[ray] {rn}: F_delta {fd0:.3f} -> {target} (scale {c:.4f})")
        if (target, tag) == CTRL_TARGET:
            scaled_fro = {
                bk: float(torch.linalg.matrix_norm(
                    out[bk].float() @ T[bk.replace("lora_B", "lora_A")].float()))
                for bk in bkeys
            }

    target, tag = CTRL_TARGET
    for ctrl in range(1, N_CTRL + 1):
        g = torch.Generator().manual_seed(2000 + ctrl)
        out = {}
        for k, v in T.items():
            if k.endswith("lora_B.weight"):
                A = T[k.replace("lora_B", "lora_A")].float()
                B = torch.randn(v.shape, generator=g, dtype=torch.float32)
                cur = float(torch.linalg.matrix_norm(B @ A))
                out[k] = (B * (scaled_fro[k] / max(cur, 1e-12))).to(v.dtype).contiguous()
            else:
                out[k] = v.contiguous().clone()
        rn = f"{SRC}__rayrand{ctrl}{tag}"
        save_variant(src_dir, f"{CF}/{rn}", out)
        jobs.append((rn, CMD.format(rn=rn)))
        print(f"[ray] {rn}: per-matrix ||BA||_F matched to {SRC}__ray{tag}")

    os.makedirs("jobs", exist_ok=True)
    with open("jobs/qwen_ray_queue.txt", "w") as fh:
        for rn, cmd in jobs:
            fh.write(f"{rn}\t{cmd}\n")
        fh.write("STOP\n")
    print(f"\n[ray] jobs/qwen_ray_queue.txt: {len(jobs)} eval jobs + STOP")


if __name__ == "__main__":
    main()
