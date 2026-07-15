"""DeepSeek-V4-Flash smoke test #1 — the feasibility gate.

Goal: prove we can load the FP8 MoE sharded across a node's 8 GPUs and run a forward pass.
Everything else (LoRA train, magnitude/spread/CE measurement) is gated on this working.

Run ON a drained node (all 8 GPUs free):
  cd notebooks/catastrophic\ forgetting
  HF_HOME=/scratch/hf_cache CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    /home/guy/UIOrthoLoRA/.venv/bin/python scripts/deepseek/smoke_load.py

Expected failure modes to iterate on (this is a first attempt, not validated):
  - transformers version lacks `deepseek_v4` support -> needs trust_remote_code + repo modeling *.py (downloaded).
  - FP8 (fbgemm/fp8 block-quant) kernels missing -> may need a dequant path or a newer transformers/kernels.
  - device_map="auto" MoE sharding imbalance / OOM -> try max_memory per GPU, or accelerate/FSDP.
Report exactly what breaks; we fix on-hardware.
"""
import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
os.environ.setdefault("HF_HOME", "/scratch/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

def gpu_mem():
    return " ".join(f"g{i}:{torch.cuda.memory_allocated(i)/1e9:.0f}G" for i in range(torch.cuda.device_count()))

def main():
    n = torch.cuda.device_count()
    print(f"[smoke] visible GPUs = {n}", flush=True)
    assert n >= 8, "need all 8 GPUs of a drained node for the sharded 284B model"
    t0 = time.time()
    print("[smoke] loading tokenizer", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"[smoke] loading model sharded (device_map=auto, trust_remote_code) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto",
        device_map="auto", low_cpu_mem_usage=True)
    print(f"[smoke] model loaded in {time.time()-t0:.0f}s. mem: {gpu_mem()}", flush=True)
    print(f"[smoke] device_map (sample): {list(getattr(model,'hf_device_map',{}).items())[:6]}", flush=True)
    # forward + short generation
    prompt = "Question: A patient presents with polyuria and polydipsia. Most likely diagnosis?\nAnswer:"
    ids = tok(prompt, return_tensors="pt").to(model.device if hasattr(model,'device') else 'cuda:0')
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=32, do_sample=False)
    print("[smoke] GENERATION OK:", tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True), flush=True)
    print(f"[smoke] PASS in {time.time()-t0:.0f}s total. peak mem: {gpu_mem()}", flush=True)

if __name__ == "__main__":
    main()
