"""Diagnose UIOrthoLoRA save/reload: is the SVD basis persisted? does it reload?"""
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, PeftModel, UIOrthoLoRAConfig
from safetensors import safe_open

BASE = "meta-llama/Llama-2-7b-hf"
TMP = "/scratch/cf_models/_diag_test"

m1 = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
cfg = UIOrthoLoRAConfig(target_modules=["q_proj"], num_svalues_to_adapt=2048,
                        num_svectors_to_adapt=410, use_de=True)
m1 = get_peft_model(m1, cfg)

# grab the first UIOrthoLoRA layer's U3 and rotator
def get_layer(m):
    for mod in m.modules():
        if hasattr(mod, "uiortholora_U3"):
            return mod
L1 = get_layer(m1)
U3_orig = L1.uiortholora_U3["default"].detach().float().cpu().clone()
print("U3_orig shape:", tuple(U3_orig.shape), "norm:", U3_orig.norm().item())
print("U3 requires_grad:", L1.uiortholora_U3["default"].requires_grad)

m1.save_pretrained(TMP)

# inspect saved keys
keys = []
with safe_open(f"{TMP}/adapter_model.safetensors", "pt") as f:
    keys = list(f.keys())
uio_basis_keys = [k for k in keys if any(t in k for t in ["_U1", "_U2", "_U3", "_Vt", "_S1", "_S2", "_S3"])]
print(f"\nTotal saved keys: {len(keys)}")
print(f"SVD-basis keys saved: {len(uio_basis_keys)}")
for k in keys:
    if "q_proj" in k:
        print("  ", k)

del m1; torch.cuda.empty_cache()
m2 = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
m2 = PeftModel.from_pretrained(m2, TMP, device_map={"": 0})
L2 = get_layer(m2)
U3_reload = L2.uiortholora_U3["default"].detach().float().cpu()
err = (U3_orig - U3_reload).abs().max().item()
print(f"\nmax|U3_orig - U3_reload| = {err:.6g}  -> {'MATCH (persisted)' if err < 1e-4 else 'DIVERGED (recomputed!)'}")
import shutil; shutil.rmtree(TMP, ignore_errors=True)
