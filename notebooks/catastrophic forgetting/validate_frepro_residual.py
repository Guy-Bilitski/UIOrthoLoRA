"""0-step residual round-trip validation for the FAITHFUL CLoRA repro setting:
PiSSA + MiLoRA at r=64, alpha=128 (scaling=2). Confirms residual_save.py's scaling-generalized
conversion reconstructs W0 exactly after save->convert->reload (untrained => dW=0), and that the
converted adapter preserves scaling s=alpha/r=2. Exit 0 = PASS.
"""
import os, sys, json, tempfile, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
import residual_save as Rs
import milora_init
import data_aware_init

BASE = "meta-llama/Llama-2-7b-hf"
TARGETS = "q_proj,k_proj,v_proj,up_proj,down_proj".split(",")
R, ALPHA = 64, 128
TOL = 1e-2   # bf16 round-trip tolerance on |delta|


def build(model):
    cfg = LoraConfig(r=R, lora_alpha=ALPHA, target_modules=TARGETS, lora_dropout=0.0,
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def reload_delta_max(out_dir):
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
    ok = True
    for method in ["pissa", "milora"]:
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
        model.config.use_cache = False
        pm = build(model)
        if method == "pissa":
            err = data_aware_init.inject_lora_init(pm, data_aware_init.pissa_BAR(R))
        else:
            err = milora_init.apply_milora(pm, r=R)
        init_ad = Rs.capture_init_adapter(pm)
        with tempfile.TemporaryDirectory() as td:
            pm.save_pretrained(td)
            Rs.convert_saved_to_w0_relative(td, init_ad)
            cfg = json.load(open(os.path.join(td, "adapter_config.json")))
            sc_conv = cfg["lora_alpha"] / cfg["r"]
            del model, pm
            torch.cuda.empty_cache()
            dmax = reload_delta_max(td)
        passed = (dmax < TOL) and (abs(sc_conv - ALPHA / R) < 1e-9)
        ok = ok and passed
        print(f"[{method:7s} r={R} a={ALPHA}] init loss-preserve err={err:.2e}  "
              f"conv r->{cfg['r']} a->{cfg['lora_alpha']} (scaling={sc_conv:.2f})  "
              f"0-step reloaded |delta|max={dmax:.2e}  {'PASS' if passed else 'FAIL'}", flush=True)
    print("FREPRO RESIDUAL GATE:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
