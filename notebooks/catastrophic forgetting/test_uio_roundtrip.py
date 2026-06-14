"""
Correctness check: UIOrthoLoRA does NOT persist its SVD buffers (peft state-dict
filters to the 'uiortholora_' prefix), so reload recomputes the SVD. The trained
orthogonal rotations are defined relative to the training-time SVD basis, so a
faithful reload REQUIRES the recomputed U/S/Vt to match bit-for-bit. This verifies
that train-time ΔW == reload-time ΔW (and forward outputs match).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel, UIOrthoLoRAConfig

BASE = "meta-llama/Llama-2-7b-hf"
TMP = "/scratch/cf_models/_roundtrip_test"

tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token_id = 0
m1 = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
cfg = UIOrthoLoRAConfig(target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
                        num_svalues_to_adapt=2048, num_svectors_to_adapt=410,
                        uiortholora_dropout=0.0, use_de=True, initial_scaler=0.1, initial_sigma=0.1)
m1 = get_peft_model(m1, cfg)
# perturb trained params so we're not at trivial init (simulate a trained adapter)
with torch.no_grad():
    for n, p in m1.named_parameters():
        if "uiortholora_sigma" in n or "uiortholora_D" in n or "uiortholora_E" in n:
            p.add_(torch.randn_like(p) * 0.05)
m1.eval()

# reference delta weights + a forward output
def first_layer():
    for mod in m1.modules():
        if hasattr(mod, "get_delta_weight") and hasattr(mod, "uiortholora_sigma"):
            return mod
ref_mod = first_layer()
ref_dw = ref_mod.get_delta_weight("default").detach().float().cpu()

ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
with torch.no_grad():
    ref_logits = m1(ids).logits.detach().float().cpu()

m1.save_pretrained(TMP)
del m1
torch.cuda.empty_cache()

# fresh reload
m2 = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
m2 = PeftModel.from_pretrained(m2, TMP, device_map={"": 0})
m2.eval()
for mod in m2.modules():
    if hasattr(mod, "get_delta_weight") and hasattr(mod, "uiortholora_sigma"):
        new_dw = mod.get_delta_weight("default").detach().float().cpu()
        break
with torch.no_grad():
    new_logits = m2(ids).logits.detach().float().cpu()

dw_err = (ref_dw - new_dw).abs().max().item()
dw_rel = dw_err / (ref_dw.abs().max().item() + 1e-9)
logit_err = (ref_logits - new_logits).abs().max().item()
print(f"[roundtrip] max|dW_train - dW_reload| = {dw_err:.6g}  (rel {dw_rel:.6g})")
print(f"[roundtrip] max|logits_train - logits_reload| = {logit_err:.6g}")
print(f"[roundtrip] VERDICT: {'PASS (faithful reload)' if dw_rel < 1e-3 and logit_err < 0.1 else 'FAIL — reload diverges, eval would be invalid'}")

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print(f"[roundtrip] cleaned up {TMP}")
