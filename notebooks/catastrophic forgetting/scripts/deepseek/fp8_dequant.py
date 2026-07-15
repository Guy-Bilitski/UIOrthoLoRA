"""FP8 block-dequant helper — reconstruct a real bf16/fp32 [out,in] weight from an FP8Linear.

Primary use: the residual-init round-trip VERIFICATION (plan §Verification) — confirm the weight
the residual SVD sees matches what the layer's own forward computes, i.e. that block scales were
applied (a plain `.float()` on FP8 codes silently drops them). The main train/eval path loads the
base already dequantized to bf16, so this is not on the hot path; it's a correctness check + a
fallback for a selective-attention-dequant strategy.

Two independent methods that must agree:
  dequant_blockwise(layer): replicate transformers' FineGrainedFP8 block dequant
     (weight_e4m3 * per-128-block weight_scale_inv, ue8m0 or fp32 scales).
  dequant_via_forward(layer): probe the layer's OWN forward with an identity — W.T = layer(I).
     Correct by construction whenever the forward is correct (validated by the smoke test).
"""
import torch


def is_fp8(layer):
    w = getattr(layer, "weight", None)
    return w is not None and w.element_size() == 1 and hasattr(layer, "weight_scale_inv")


@torch.no_grad()
def dequant_blockwise(layer, out_dtype=torch.float32):
    """weight (e4m3, [out,in]) * weight_scale_inv (per [block_m,block_n] block) -> out_dtype."""
    w = layer.weight
    if w.element_size() > 1:                       # not packed — already real
        return w.data.to(out_dtype)
    scales = layer.weight_scale_inv
    q = w.to(torch.float32)
    out_f, in_f = q.shape
    sr, sc = (scales.shape + (1, 1))[:2] if scales.dim() >= 2 else (1, 1)
    if scales.dim() < 2:
        s = scales.to(torch.float32).reshape(1, 1)
        sr = sc = 1
    else:
        # ue8m0 exponents ship as uint8: scale = 2**(byte-127)
        s = ((scales.to(torch.float32) - 127.0).exp2() if scales.dtype == torch.uint8
             else scales.to(torch.float32))
    bm = out_f // sr
    bn = in_f // sc
    q = q.reshape(sr, bm, sc, bn)
    s = s.reshape(sr, 1, sc, 1)
    return (q * s).reshape(out_f, in_f).to(out_dtype)


@torch.no_grad()
def dequant_via_forward(layer, out_dtype=torch.float32, chunk=2048):
    """W = (layer(I))^T using the layer's own (correct) forward. Bias must be None for the targets."""
    w = layer.weight
    dev = w.device
    in_f = layer.in_features
    out_cols = []
    for s in range(0, in_f, chunk):
        e = min(s + chunk, in_f)
        I = torch.zeros(e - s, in_f, dtype=torch.bfloat16, device=dev)
        for r, c in enumerate(range(s, e)):
            I[r, c] = 1.0
        out_cols.append(layer(I).to(out_dtype))     # (chunk, out) = rows of W.T
    WT = torch.cat(out_cols, dim=0)                  # (in, out)
    return WT.T.contiguous()                         # (out, in)


@torch.no_grad()
def roundtrip_maxerr(layer):
    """Max abs diff between the two dequant methods (should be ~0 up to fp8/bf16 rounding)."""
    a = dequant_blockwise(layer).float()
    b = dequant_via_forward(layer).float()
    return float((a - b).abs().max())
