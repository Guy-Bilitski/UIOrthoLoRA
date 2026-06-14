"""
Mechanism metric F-delta (CLoRA Table 4): for each updated matrix DW and input x,
    F_delta(DW, x) = ||DW x|| / ||x||
averaged over real-world inputs, over all tokens and all updated matrices. Lower =
less output disruption = less forgetting. Also reports ||DW|| (largest singular
value = capacity proxy), per CLoRA.

Targets to reproduce (rank 32, CS): LoRA F_delta=0.79 ||DW||=22.63;
CLoRA-k2048 F_delta=0.14 ||DW||=5.00.

Works for any registered PEFT tuner whose layers expose get_delta_weight('default')
(LoRA, CLoRA-as-LoRA, UIOrthoLoRA). x is captured as the real input to each updated
matrix via forward pre-hooks while running N real prompts through the model.
"""
import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer

import run_lib

HERE = run_lib.HERE
DATASET_DIR = os.path.join(HERE, "repro/LLM-Adapters/dataset")


def load_inputs(n, tokenizer, max_len=256):
    """N real-world inputs: commonsense eval prompts (boolq + piqa + arc mix)."""
    prompts = []
    for ds in ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
               "ARC-Challenge", "openbookqa"]:
        data = json.load(open(os.path.join(DATASET_DIR, ds, "test.json")))
        for d in data[: (n // 7) + 2]:
            prompts.append(run_lib.eval_prompt(d["instruction"], d.get("input") or None))
    return prompts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n_inputs", type=int, default=100)
    ap.add_argument("--run_name", default="")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter, device_map={"": 0})
    model.eval()

    # collect updated matrices + their DW and top singular value
    layers = {}  # name -> {"dw": tensor, "sv": float}
    hooks = []
    accum = {}  # name -> [sum(||DWx||/||x||), token_count]

    def make_hook(name):
        def pre_hook(module, inputs):
            x = inputs[0]                       # (B, T, in)
            dw = layers[name]["dw"]             # (out, in)
            xf = x.reshape(-1, x.shape[-1]).to(dw.dtype)   # (B*T, in)
            xn = xf.norm(dim=-1)                # (B*T,)
            dwx = torch.matmul(xf, dw.T)        # (B*T, out)
            dwxn = dwx.norm(dim=-1)
            mask = xn > 1e-6
            ratio = (dwxn[mask] / xn[mask])
            s, c = accum.get(name, (0.0, 0))
            accum[name] = (s + ratio.sum().item(), c + int(mask.sum().item()))
        return pre_hook

    for name, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer) and hasattr(mod, "get_delta_weight"):
            try:
                dw = mod.get_delta_weight("default").detach()
            except Exception as e:
                print(f"[fdelta] skip {name}: {e}", flush=True)
                continue
            sv = torch.linalg.svdvals(dw.float())[0].item()
            layers[name] = {"dw": dw, "sv": sv}
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))
    print(f"[fdelta] {len(layers)} updated matrices hooked", flush=True)

    prompts = load_inputs(args.n_inputs, tokenizer)
    with torch.no_grad():
        for i in range(0, len(prompts), 8):
            batch = prompts[i : i + 8]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=256).to("cuda:0")
            model(**enc)
    for h in hooks:
        h.remove()

    # F-delta = mean over matrices of per-matrix mean(||DWx||/||x||); also token-weighted
    per_matrix = {n: (s / c if c else 0.0) for n, (s, c) in accum.items()}
    fdelta = sum(per_matrix.values()) / len(per_matrix)
    tot_s = sum(s for s, c in accum.values())
    tot_c = sum(c for s, c in accum.values())
    fdelta_tokenwt = tot_s / tot_c if tot_c else 0.0
    svs = {n: layers[n]["sv"] for n in layers}
    mean_sv = sum(svs.values()) / len(svs)
    max_sv = max(svs.values())

    run_name = args.run_name or os.path.basename(os.path.normpath(args.adapter))
    summary = {"run_name": run_name, "adapter": args.adapter, "kind": "fdelta",
               "n_inputs": len(prompts), "n_matrices": len(layers),
               "fdelta_matrix_mean": round(fdelta, 4),
               "fdelta_token_weighted": round(fdelta_tokenwt, 4),
               "dw_sv_mean": round(mean_sv, 4), "dw_sv_max": round(max_sv, 4),
               "git_commit": run_lib.git_commit(), "evaluated_at": run_lib.now_iso()}
    run_lib.write_json(os.path.join(HERE, "results", run_name, "fdelta.json"), summary)
    run_lib.append_registry("eval_registry.jsonl", summary)
    print(f"[fdelta] {run_name}: F_delta(matrix-mean)={fdelta:.4f} "
          f"F_delta(token-wt)={fdelta_tokenwt:.4f} ||DW||_mean={mean_sv:.3f} "
          f"||DW||_max={max_sv:.3f}", flush=True)


if __name__ == "__main__":
    main()
