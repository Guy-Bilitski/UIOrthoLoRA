"""DeepSeek-V4-Flash TRAINING-path feasibility gate (run on a drained 8-GPU node).

smoke_load.py proves the FP8-native forward works. THIS proves the path train_deepseek.py/
eval_deepseek.py actually use: base dequantized FP8->bf16, sharded device_map="auto". It checks
the four things the whole run is gated on:
  1. the 568 GB bf16 base loads sharded across the 8 GPUs without OOM (reports per-GPU peak);
  2. the MLA attention target modules exist under the expected names and are now real bf16
     (element_size==2) -> residual-init SVD + LoRA can operate on them;
  3. a residual-init prerequisite works: SVD of a dequantized target weight succeeds (no FP8-code
     garbage) and, where both are available, block-dequant == forward-probe dequant (fp8_dequant);
  4. a forward + short greedy generate on a MedMCQA-style prompt returns text.

Run:
  cd notebooks/catastrophic\ forgetting
  HF_HOME=/scratch/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    /home/guyb/UIOrthoLoRA/.venv/bin/python scripts/deepseek/smoke_train_load.py
"""
import os, sys, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
TARGETS = ("q_a_proj", "q_b_proj", "kv_proj", "o_b_proj")
os.environ.setdefault("HF_HOME", "/scratch/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def peakmem():
    return " ".join(f"g{i}:{torch.cuda.max_memory_allocated(i)/2**30:.0f}G" for i in range(torch.cuda.device_count()))


def main():
    n = torch.cuda.device_count()
    print(f"[smoke2] visible GPUs = {n}", flush=True)
    assert n >= 8, "need all 8 GPUs of a drained node"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    print("[smoke2] loading base dequant->bf16, device_map=auto ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        quantization_config=FineGrainedFP8Config(dequantize=True),
        low_cpu_mem_usage=True, trust_remote_code=True)
    print(f"[smoke2] (1) loaded in {time.time()-t0:.0f}s across "
          f"{len(set(model.hf_device_map.values()))} devices. peak {peakmem()}", flush=True)

    # (2) target modules present + bf16
    found = {t: [] for t in TARGETS}
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in found and hasattr(mod, "weight"):
            found[leaf].append((name, mod))
    for t in TARGETS:
        assert found[t], f"target {t} not found among modules"
        _, m0 = found[t][0]
        es = m0.weight.element_size()
        print(f"[smoke2] (2) {t}: {len(found[t])} modules; example dtype={m0.weight.dtype} "
              f"element_size={es} {'BF16-OK' if es == 2 else '*** STILL PACKED ***'}", flush=True)
        assert es == 2, f"{t} weight not dequantized (element_size={es})"

    # (3) residual-init prerequisite: SVD a target weight; cross-check dequant methods if FP8 remains
    import fp8_dequant as FD
    name, m = found["q_b_proj"][0]
    W = m.weight.data.float()
    sv = torch.linalg.svdvals(W)
    print(f"[smoke2] (3) SVD({name}) top={sv[0]:.3f} min={sv[-1]:.3e} shape={tuple(W.shape)} — SVD OK", flush=True)
    if FD.is_fp8(m):  # only if a target somehow stayed packed
        print(f"[smoke2] (3b) dequant roundtrip maxerr={FD.roundtrip_maxerr(m):.3e}", flush=True)

    # (4) forward + generate
    prompt = ("Below is an instruction that describes a task. Write a response that appropriately "
              "completes the request.\n\n### Instruction:\nA patient presents with polyuria and "
              "polydipsia.\nA. Diabetes\nB. Asthma\nC. Anemia\nD. Fracture\n\nAnswer format: A/B/C/D\n\n### Response:\n")
    ids = tok(prompt, return_tensors="pt").to(model.get_input_embeddings().weight.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
    print("[smoke2] (4) GENERATION:", tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True), flush=True)
    print(f"[smoke2] PASS in {time.time()-t0:.0f}s. peak {peakmem()}", flush=True)


if __name__ == "__main__":
    main()
