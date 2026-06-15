> 📦 **HISTORICAL snapshot (Wave-1 era).** For the live results pile use `analyze_magnitude_law.py` /
> `analyze_d1_d2.py` over `results/`, and the tables in 08/06. Current headline numbers live there.

# RESULTS (LLaMA-2-7B, commonsense one-stage) — snapshot

Retention = mean(answer-only-BBH, MMLU-Pro). **Two retention scales — keep them separate:**
- **FULL** (mmlu 12k, 2048-gen): used for base/LoRA/CLoRA gates.
- **FAST** (`ret_limit 64, ret_max_gen 512`): used for all UIOrthoLoRA runs (in-process, for speed).
- **Calibration offset:** LoRA FULL ret 21.66 vs FAST 22.52 ⇒ **FAST ≈ FULL + ~0.9**. CLoRA FAST anchors (`calib_*`) in progress to map their FULL→FAST.

## Baselines & gates (FULL retention) — all PASSED
| run | CS avg | BBH(AO) | MMLU-Pro | retention | F∆ | ‖ΔW‖ | notes |
|---|---|---|---|---|---|---|---|
| **Base** | — | 33.1 | 18.96 | **26.0** | — | — | gate A ✓ (target 34.91/18.56) |
| **LoRA** r32 | 78.1 | 30.7 | 12.6 | **21.7** | 0.738 | 55.4 | gate B ✓; forgets (21.7<26.0) |
| CLoRA k128 | 79.2 | 29.9 | 15.1 | 22.5 | 0.60 | 32.5 | |
| CLoRA k256 | 79.3 | 30.6 | 14.2 | 22.4 | 0.58 | 29.8 | |
| CLoRA k512 | 67.6 | 30.6 | 15.6 | 23.1 | 0.51 | 25.9 | CS dip (format) |
| **CLoRA k1024** | **79.9** | 33.5 | 16.1 | **24.8** | 0.46 | 22.0 | **the bar (high-CS corner)** |
| CLoRA k2048 | 65.4 | 34.1 | 17.3 | **25.7** | 0.34 | 16.4 | over-constrained; **bar (high-retention corner)** |

**CLoRA mechanism reproduces cleanly:** ↑k ⇒ ↑retention, ↓F∆, ↓‖ΔW‖. The bar to beat:
**CS 80 @ ret 24.8 (k1024)** and **ret 25.7 @ CS 65 (k2048)**. Win = land above this frontier, esp.
**ret≈25-26 at CS>65** (dominates k2048).

## UIOrthoLoRA — @ LR 3e-4 (WRONG LR, under-adapted — kept only as cautionary)
k_vec=410 param-matched, use_de=1: CS **47-62** (kval410 53, 1024 48, 2048 51, 3072 62, 4096 55). Train loss ~0.98.
→ under-adaptation, NOT a real result. Caused the LR investigation.

## UIOrthoLoRA — @ LR 1e-2 (correct LR)
- **LR sweep (2000-step proxy, k_val2048/k_vec410):** loss vs LR → 3e-4:1.02, 1e-3:0.83, 3e-3:0.77, **1e-2:0.65** (< LoRA 0.76). Monotone; 1e-2 best. CS proxy undertrained (ignore absolute).
- **UILinLoRA (k_vec=0, magnitude-only)** @1e-2: **CS 70.5, ret 23.6 (FAST), F∆ 0.75, ‖ΔW‖ 9.1**. (vs 48.3 @3e-4 — LR fix = +22 CS.) High F∆ despite tiny ‖ΔW‖ ⇒ **D/E gates leak**.
- **Param-matched frontier (Wave 1, k_vec=410, use_de=1, LR1e-2):** TRAINING ~73%, losses ~0.80 (→~0.72). **Retention pending** — the decisive numbers. (Expectation: adapts well CS~76-79, but big-k_vec+gates ⇒ likely leaky ⇒ retention maybe near LoRA. The win is Wave 2.)

## Wave 2 (queued/auto-launch) — the high-retention corner + leakage
small k_vec × use_de on/off, LR1e-2, full 3ep, **logs leakage thermometers**:
`(kval,kvec,use_de)` = (256,32,0)(512,64,0)(1024,128,0)(2048,410,0)(512,64,1)(1024,128,1)(2048,256,1)(256,32,1).
Hypothesis: use_de=0 + small k_vec ⇒ μ_E≈ν_D≈0 (clean) ⇒ ret near base 26 at CS~72-75 ⇒ dominates CLoRA-k2048.

## Calibration (FAST retention re-eval, for comparability)
| run | CS | ret(FAST) | (FULL ref) |
|---|---|---|---|
| calib_lora_fastret | 78.6 | 22.5 | (21.7) |
| calib_clora_k1024_fastret | running | | (24.8) |
| calib_clora_k2048_fastret | queued | | (25.7) |

## Leakage thermometers — validated, not yet on real runs
Sanity (synthetic + real layer): E=D=I,k_vec=0 ⇒ μ_E=ν_D=Leak11=OffTail≈0 (perfect tail confinement);
use_de=1 ⇒ μ_E/ν_D/drift jump; drift=0 for gapped-no-crossing, 1.0 for tail-crosses-leading. **Wave 2 will be
the first real μ_E/ν_D numbers** (use_de off vs on contrast).
