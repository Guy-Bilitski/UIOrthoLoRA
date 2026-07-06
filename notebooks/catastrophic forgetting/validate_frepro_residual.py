"""0-step residual round-trip validation for the FAITHFUL CLoRA repro setting:
PiSSA + MiLoRA + SC-LoRA + LoRA-Null at r=64, alpha=128 (scaling=2). Confirms residual_save.py's
scaling-generalized conversion reconstructs W0 exactly after save->convert->reload (untrained => dW=0),
that the converted adapter preserves scaling s=alpha/r=2, AND that each init is loss-preserving
(init err < TOL). Exit 0 = PASS.

  python validate_frepro_residual.py            # gates pissa/milora/sclora/lora_null at s=2
  python validate_frepro_residual.py --cordapp  # ALSO gate the dynamic-rank CorDA++ path (needs the
                                                #   cordapp wiring; harmless import of cordapp_init.py)
"""
import os, sys, json, argparse, tempfile, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
import residual_save as Rs
import milora_init
import data_aware_init

BASE = "meta-llama/Llama-2-7b-hf"
TARGETS = "q_proj,k_proj,v_proj,up_proj,down_proj".split(",")
R, ALPHA = 64, 128
TOL = 1e-2   # bf16 round-trip tolerance on |delta| AND on init loss-preservation err
# tiny calib set for the data-aware inits (this gate checks the residual PLUMBING, not method
# fidelity, so a short calib that yields a valid loss-preserving init is sufficient).
CALIB = ["The capital of France is Paris.", "Water is made of hydrogen and oxygen.",
         "The sun rises in the east.", "Two plus two equals four.",
         "Birds can fly in the sky.", "The ocean is full of water.",
         "Mountains are very tall.", "Books contain many words."]
METHODS = ["pissa", "milora", "sclora", "lora_null"]


def build(model):
    cfg = LoraConfig(r=R, lora_alpha=ALPHA, target_modules=TARGETS, lora_dropout=0.0,
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def init_method(method, pm, tok):
    """Apply the data-aware init at r=64/alpha=128 (s=2); returns loss-preserving err."""
    if method == "pissa":
        return data_aware_init.inject_lora_init(pm, data_aware_init.pissa_BAR(R))
    if method == "milora":
        return milora_init.apply_milora(pm, r=R)
    if method == "sclora":
        import sclora_init as S
        return S.apply_sclora(pm, S.collect_sclora_M(pm, CALIB, CALIB, tok, beta=0.5, max_len=64), r=R)
    if method == "lora_null":
        import lora_null_init as N
        return N.apply_lora_null(pm, N.collect_lora_null_cov(pm, CALIB, tok,
                                                             calib_size=len(CALIB), max_len=64), r=R)
    raise ValueError(method)


def gate_cordapp(tok):
    """Dynamic-rank CorDA++ 0-step gate: precompute -> build with rank_pattern/alpha_pattern ->
    apply_cordapp -> save -> convert -> finalize_dynamic_rank_config -> reload; reloaded |delta| ~ 0
    and init loss-preserving. Mirrors the train_cs cordapp flow (ordering: finalize AFTER convert)."""
    import cordapp_init as Cpp
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
    base.config.use_cache = False
    res = Cpp.precompute_cordapp(base, CALIB, tok, TARGETS, fixed_rank=R, N=2,
                                 calib_size=len(CALIB), max_len=64, bs=1, scaling=ALPHA / R)
    cfg = LoraConfig(r=R, lora_alpha=ALPHA, target_modules=TARGETS, lora_dropout=0.0,
                     bias="none", task_type="CAUSAL_LM",
                     rank_pattern=res["rank_pattern"], alpha_pattern=res["alpha_pattern"])
    pm = get_peft_model(base, cfg)
    err = Cpp.apply_cordapp(pm, res["chosen_covs"], res["ranks"])
    init_ad = Rs.capture_init_adapter(pm)
    with tempfile.TemporaryDirectory() as td:
        pm.save_pretrained(td)
        Rs.convert_saved_to_w0_relative(td, init_ad)
        Cpp.finalize_dynamic_rank_config(td)   # MANDATORY: double per-layer rank_pattern to 2r^l
        del base, pm
        torch.cuda.empty_cache()
        dmax = reload_delta_max(td)
    passed = (dmax < TOL) and (err < TOL)
    print(f"[cordapp dyn-rank r->2r^l] init err={err:.2e}  0-step reloaded |delta|max={dmax:.2e}  "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return passed


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--cordapp", action="store_true",
                    help="also gate the dynamic-rank CorDA++ residual path (needs cordapp_init.py).")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    ok = True
    for method in METHODS:
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
        model.config.use_cache = False
        pm = build(model)
        err = init_method(method, pm, tok)
        init_ad = Rs.capture_init_adapter(pm)
        with tempfile.TemporaryDirectory() as td:
            pm.save_pretrained(td)
            Rs.convert_saved_to_w0_relative(td, init_ad)
            cfg = json.load(open(os.path.join(td, "adapter_config.json")))
            sc_conv = cfg["lora_alpha"] / cfg["r"]
            del model, pm
            torch.cuda.empty_cache()
            dmax = reload_delta_max(td)
        # init-error gate (MiLoRA expert): the init must be loss-preserving (err < TOL) AND the
        # 0-step reload must reconstruct W0 (dmax < TOL) AND scaling must be preserved at s=2.
        passed = (dmax < TOL) and (err < TOL) and (abs(sc_conv - ALPHA / R) < 1e-9)
        ok = ok and passed
        print(f"[{method:9s} r={R} a={ALPHA}] init loss-preserve err={err:.2e}  "
              f"conv r->{cfg['r']} a->{cfg['lora_alpha']} (scaling={sc_conv:.2f})  "
              f"0-step reloaded |delta|max={dmax:.2e}  {'PASS' if passed else 'FAIL'}", flush=True)
    if args.cordapp:
        ok = gate_cordapp(tok) and ok
    print("FREPRO RESIDUAL GATE:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
