"""
Residual-init adapters (CorDA / MiLoRA / SC-LoRA) overwrite base.weight = W_res = W0 - B_init@A_init
in memory, but PEFT save_pretrained persists ONLY the adapter -> at eval PeftModel reloads the
ORIGINAL W0 and adds the trained adapter on top of the WRONG base (the -B_init@A_init cancellation
is lost). Symptom: healthy training loss but exploded evaluated ||dW||_F and ~0 retention.

Fix = standard PiSSA->LoRA residual conversion. We require scaling==1 (alpha==r) for these methods,
so the W0-relative update is exactly
    dW = B_trained@A_trained - B_init@A_init
which is rank <= 2r. Stack it into a rank-2r LoRA adapter (scaling still 1):
    A'' = [A_trained ; A_init]   (2r, in)
    B'' = [B_trained , -B_init]  (out, 2r)
so that  1 * B''@A'' = B_trained@A_trained - B_init@A_init = dW  relative to the ORIGINAL W0.
The eval/forensics harness (which reloads W0) is then correct and UNCHANGED.

Self-check property: a 0-step run converts to dW=0 (B_init cancels itself) -> eval == base model.
"""
import os
import json
import torch
from safetensors.torch import load_file, save_file
from peft.tuners.tuners_utils import BaseTunerLayer


@torch.no_grad()
def capture_init_adapter(model, adapter="default"):
    """Snapshot the injected init adapter weights (saved-form: lora_A (r,in), lora_B (out,r))
    keyed by module name so keys match the saved safetensors prefix."""
    init = {}
    for name, m in model.named_modules():
        if isinstance(m, BaseTunerLayer) and adapter in getattr(m, "lora_A", {}):
            init[name] = (m.lora_A[adapter].weight.detach().float().cpu().clone(),
                          m.lora_B[adapter].weight.detach().float().cpu().clone())
    return init


def convert_saved_to_w0_relative(out_dir, init):
    """Rewrite the just-saved rank-r adapter into the rank-2r W0-relative adapter. In place."""
    path = os.path.join(out_dir, "adapter_model.safetensors")
    sd = load_file(path)
    out = dict(sd)
    r_old, n = None, 0
    for name, (A_init, B_init) in init.items():
        ka, kb = f"{name}.lora_A.weight", f"{name}.lora_B.weight"
        if ka not in sd or kb not in sd:
            continue
        A_tr, B_tr = sd[ka], sd[kb]                      # (r,in), (out,r)
        r_old = A_tr.shape[0]
        A2 = torch.cat([A_tr.float(), A_init.to(torch.float32)], dim=0)      # (2r,in)
        B2 = torch.cat([B_tr.float(), -B_init.to(torch.float32)], dim=1)     # (out,2r)
        out[ka] = A2.to(A_tr.dtype)
        out[kb] = B2.to(B_tr.dtype)
        n += 1
    if r_old is None:
        raise RuntimeError("convert_saved_to_w0_relative: no matching adapter keys found")
    save_file(out, path)
    cfgp = os.path.join(out_dir, "adapter_config.json")
    cfg = json.load(open(cfgp))
    assert cfg.get("lora_alpha") == cfg.get("r"), \
        f"residual conversion assumes scaling==1 (alpha==r); got alpha={cfg.get('lora_alpha')} r={cfg.get('r')}"
    cfg["r"] = 2 * r_old
    cfg["lora_alpha"] = 2 * r_old                        # keep scaling = alpha/r = 1
    json.dump(cfg, open(cfgp, "w"), indent=2)
    return n, r_old
