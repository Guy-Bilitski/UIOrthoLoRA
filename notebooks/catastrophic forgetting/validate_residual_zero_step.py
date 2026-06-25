"""0-step validation gate for residual-init methods (CorDA / MiLoRA / SC-LoRA / LoRA-Null).

For each method: build a LoRA adapter (alpha==r), apply the data-aware init (which overwrites
base.weight = W_res), snapshot the init adapter, save, convert to the W0-relative rank-2r adapter,
then RELOAD a fresh base + the converted adapter and assert the reloaded delta (scaling*B@A) ~ 0
everywhere. A 0-step (untrained) run must reconstruct W0 exactly -> base-model retention. This proves
the save -> convert -> reload plumbing is correct end-to-end (esp. for the new LoRA-Null path).

Exit 0 = PASS (all methods' reloaded |delta| < TOL), else exit 1.
"""
import os, sys, tempfile, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
import residual_save as Rs

BASE = "meta-llama/Llama-2-7b-hf"
TARGETS = "q_proj,k_proj,v_proj,up_proj,down_proj".split(",")
CALIB = ["The capital of France is Paris.", "Water is made of hydrogen and oxygen.",
         "The sun rises in the east.", "Two plus two equals four.",
         "Birds can fly in the sky.", "The ocean is full of water.",
         "Mountains are very tall.", "Books contain many words."]
R = 16
TOL = 1e-2   # bf16 round-trip tolerance on |delta|


def build(model, r):
    cfg = LoraConfig(r=r, lora_alpha=r, target_modules=TARGETS, lora_dropout=0.0,
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def init_method(method, pm, tok, r):
    if method == "corda":
        import corda_init as M
        return M.apply_corda(pm, M.collect_corda_cov(pm, CALIB, tok, calib_size=8), r=r)
    if method == "milora":
        import milora_init as M
        return M.apply_milora(pm, r=r)
    if method == "sclora":
        import sclora_init as M
        return M.apply_sclora(pm, M.collect_sclora_M(pm, CALIB, CALIB, tok, beta=0.5, max_len=64), r=r)
    if method == "lora_null":
        import lora_null_init as M
        return M.apply_lora_null(pm, M.collect_lora_null_cov(pm, CALIB, tok, calib_size=8, max_len=64), r=r)
    raise ValueError(method)


def reload_delta_max(out_dir, r):
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
    m = PeftModel.from_pretrained(base, out_dir)
    mx = 0.0
    for _n, mod in m.named_modules():
        if isinstance(mod, BaseTunerLayer) and "default" in getattr(mod, "lora_A", {}):
            A = mod.lora_A["default"].weight.float()
            B = mod.lora_B["default"].weight.float()
            sc = mod.scaling["default"]
            mx = max(mx, (sc * (B @ A)).abs().max().item())
    del base, m
    torch.cuda.empty_cache()
    return mx


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    results = {}
    for method in ["corda", "milora", "sclora", "lora_null"]:
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
        model.config.use_cache = False
        pm = build(model, R)
        err = init_method(method, pm, tok, R)
        init_ad = Rs.capture_init_adapter(pm)
        with tempfile.TemporaryDirectory() as td:
            pm.save_pretrained(td)
            n, r0 = Rs.convert_saved_to_w0_relative(td, init_ad)
            del model, pm
            torch.cuda.empty_cache()
            dmax = reload_delta_max(td, R)
        results[method] = (err, dmax)
        print(f"[{method:10s}] init loss-preserve err={err:.2e}  "
              f"0-step reloaded |delta|max={dmax:.2e}  {'PASS' if dmax < TOL else 'FAIL'}", flush=True)
    ok = all(d < TOL for _e, d in results.values())
    print("VALIDATION GATE:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
