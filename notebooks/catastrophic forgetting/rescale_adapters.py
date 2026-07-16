"""E1 interventional scale-matching (adversarial-review menu, 2026-07-16).

Takes trained lr5e-4 adapters (one per method) and rescales lora_B so the effective update
BA hits F_Delta targets {0.15, 0.40, 0.80} exactly (F_Delta is linear in a global scale on
dW; sources are stock-base-form PEFT adapters, incl. the converted rank-2r residual-init
ones). DoRA is nonlinear in B (magnitude vector) — we apply the same scale and let the
eval-measured fdelta be the x-axis truth. Plus 3 random-direction controls per target:
lora_B replaced by Gaussian noise, per-matrix ||B A||_F matched to the rescaled real LoRA —
same norm profile, random direction.

Run on d001 (CPU-only). Outputs /scratch/cf_models/e1_* adapter dirs + jobs/fleet/_e1.txt
eval-only job lines. Verify after eval: summary fdelta within ~2% of target (except dora).
"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

SRC = {
    "lora":   ("lrsw_lora_r16_lr5e4_s43",     "s43"),
    "dora":   ("lrsw_dora_r16_lr5e4_s43",     "s43"),
    "milora": ("lrsw_milora_r32_lr5e4_s43",   "s43"),
    "sclora": ("lrsw_sclora_r32_lr5e4_s43",   "s43"),
    "clora":  ("lrsw_clora_k1024_lr5e4_s45",  "s45"),
}
TARGETS = [(0.15, "f015"), (0.40, "f040"), (0.80, "f080")]
CF = "/scratch/cf_models"
EVAL = ("/home/guy/UIOrthoLoRA/.venv/bin/python eval_one_gpu.py --adapter {ad} "
        "--run_name {rn} --base_model meta-llama/Llama-2-7b-hf --adapt_task cs "
        "--ret_suite broad --ret_limit 0 --ret_max_gen 512")

def cur_fdelta(rn):
    return json.load(open(f"results/{rn}/summary.json"))["headline"]["fdelta"]

def save_variant(src_dir, dst_dir, tensors):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    for f in os.listdir(src_dir):
        if f != "adapter_model.safetensors":
            shutil.copy2(os.path.join(src_dir, f), dst_dir)
    save_file(tensors, os.path.join(dst_dir, "adapter_model.safetensors"))

jobs = []
lora_scaled_fro = {}   # target -> {matrix_prefix: ||c*B A||_F} from the real LoRA (control ref)

for method, (rn, seed) in SRC.items():
    src_dir = f"{CF}/{rn}"
    fd0 = cur_fdelta(rn)
    T = load_file(f"{src_dir}/adapter_model.safetensors")
    bkeys = [k for k in T if k.endswith("lora_B.weight")]
    assert bkeys, rn
    for target, tag in TARGETS:
        c = target / fd0
        out = {k: (v * c if k.endswith("lora_B.weight") else v).contiguous().clone()
               for k, v in T.items()}
        new_rn = f"e1_{method}_{tag}_{seed}"
        save_variant(src_dir, f"{CF}/{new_rn}", out)
        jobs.append(EVAL.format(ad=f"{CF}/{new_rn}", rn=new_rn))
        print(f"{new_rn}: fd {fd0:.3f} -> {target} (scale {c:.4f}, {len(bkeys)} B mats)")
        if method == "lora":
            fro = {}
            for bk in bkeys:
                ak = bk.replace("lora_B", "lora_A")
                fro[bk] = float(torch.linalg.matrix_norm(out[bk].float() @ T[ak].float()))
            lora_scaled_fro[tag] = fro

# random-direction controls off the plain-LoRA skeleton
rn_l, seed_l = SRC["lora"]
TL = load_file(f"{CF}/{rn_l}/adapter_model.safetensors")
for ctrl in (1, 2, 3):
    g = torch.Generator().manual_seed(1000 + ctrl)
    for target, tag in TARGETS:
        out = {}
        for k, v in TL.items():
            if k.endswith("lora_B.weight"):
                ak = k.replace("lora_B", "lora_A")
                B = torch.randn(v.shape, generator=g, dtype=torch.float32)
                cur = torch.linalg.matrix_norm(B @ TL[ak].float())
                B = B * (lora_scaled_fro[tag][k] / max(float(cur), 1e-12))
                out[k] = B.to(v.dtype).contiguous()
            else:
                out[k] = v.contiguous().clone()
        new_rn = f"e1_randdir{ctrl}_{tag}_{seed_l}"
        save_variant(f"{CF}/{rn_l}", f"{CF}/{new_rn}", out)
        jobs.append(EVAL.format(ad=f"{CF}/{new_rn}", rn=new_rn))
        print(f"{new_rn}: per-matrix ||BA||_F matched to e1_lora_{tag}")

with open("jobs/fleet/_e1.txt", "w") as fh:
    fh.write("\n".join(jobs) + "\n")
print(f"\njobs/fleet/_e1.txt: {len(jobs)} eval-only jobs")
