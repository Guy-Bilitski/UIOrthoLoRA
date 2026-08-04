# Head-to-head vs LoRA+wd, exact p + Holm correction

Same comparisons as adjudication/03_head2head.py (26 method x
family cells vs LoRA+wd at best-adaptation operating points). Paired
per-seed t (df=n-1) where >=2 common seeds; Welch-Satterthwaite
otherwise. Holm within family and across all comparisons per axis.
Deltas in points; CI = 95%. Script: `01_head2head_corrected.py`.

## Llama-2-7B x Commonsense-8 (lrsw sweep, r-matched grid)

| Method (cell) | dRet [CI95] | t | p | p Holm(fam) | p Holm(all) | verdict | dAdapt [CI95] | p Holm(all) | verdict | test |
|---|---|---|---|---|---|---|---|---|---|---|
| SC-LoRA (5e-5) | -1.33 [-5.11, 2.45] | -1.51 | 0.2696 | 0.2696 | 1.0000 | n.s. | -1.19 [-1.81, -0.57] | 0.2325 | n.s. | paired(n=3) |
| LoRA (3e-4) | -2.00 [-3.35, -0.65] | -4.73 | 0.0179 | 0.0451 | 0.2684 | n.s. | -2.58 [-3.11, -2.05] | 0.0132 | WORSE | paired(n=4) |
| LoRA-Null (5e-4) | -4.10 [-6.43, -1.77] | -5.59 | 0.0113 | 0.0451 | 0.2256 | n.s. | -2.88 [-3.15, -2.62] | 0.0012 | WORSE | paired(n=4) |
| CLoRA (5e-4 k1024) | -4.25 [-5.44, -3.07] | -11.40 | 0.0014 | 0.0087 | 0.0348 | WORSE | -3.46 [-3.75, -3.17] | 0.0010 | WORSE | paired(n=4) |
| DoRA (5e-4) | -6.78 [-10.04, -3.51] | -8.92 | 0.0123 | 0.0451 | 0.2256 | n.s. | -5.57 [-9.28, -1.86] | 0.3246 | n.s. | paired(n=3) |
| MiLoRA (5e-4) | -4.42 [-6.09, -2.76] | -8.47 | 0.0035 | 0.0173 | 0.0796 | n.s. | -4.56 [-5.20, -3.92] | 0.0043 | WORSE | paired(n=4) |

## Llama-2-7B x math/GSM8K (frm faithful CLoRA recipe, c256)

| Method (cell) | dRet [CI95] | t | p | p Holm(fam) | p Holm(all) | verdict | dAdapt [CI95] | p Holm(all) | verdict | test |
|---|---|---|---|---|---|---|---|---|---|---|
| LoRA (1e-4) | -2.28 [-5.15, 0.59] | -3.42 | 0.0761 | 0.1522 | 0.6848 | n.s. | -2.81 [-5.25, -0.37] | 0.4619 | n.s. | paired(n=3) |
| MiLoRA (1e-4) | -1.13 [-3.47, 1.20] | -2.09 | 0.1722 | 0.1722 | 1.0000 | n.s. | -3.11 [-6.88, 0.66] | 0.7799 | n.s. | paired(n=3) |
| CLoRA (3e-4 k256) | -4.94 [-7.27, -2.61] | -9.11 | 0.0118 | 0.0592 | 0.2256 | n.s. | -6.14 [-7.15, -5.13] | 0.0306 | WORSE | paired(n=3) |
| SC-LoRA (1e-4) | -5.63 [-9.38, -1.89] | -6.47 | 0.0231 | 0.0922 | 0.3018 | n.s. | -6.32 [-9.53, -3.11] | 0.2323 | n.s. | paired(n=3) |
| DoRA (3e-4) | -5.49 [-9.34, -1.63] | -6.13 | 0.0256 | 0.0922 | 0.3074 | n.s. | -7.61 [-10.78, -4.43] | 0.1859 | n.s. | paired(n=3) |
| LoRA-Null (2e-4) | -4.68 [-6.66, -2.69] | -10.14 | 0.0096 | 0.0575 | 0.2013 | n.s. | -4.93 [-7.97, -1.89] | 0.2980 | n.s. | paired(n=3) |
| PiSSA (3e-4) | -26.34 [--, --] | -- | -- | -- | -- | n.t. | -17.13 [--, --] | -- | n.t. | welch(1v3) |

## Qwen-2.5-7B x Commonsense-8 (qwsw sweep)

| Method (cell) | dRet [CI95] | t | p | p Holm(fam) | p Holm(all) | verdict | dAdapt [CI95] | p Holm(all) | verdict | test |
|---|---|---|---|---|---|---|---|---|---|---|
| SC-LoRA (1e-4) | -12.22 [-50.19, 25.75] | -1.38 | 0.3004 | 0.3004 | 1.0000 | n.s. | -0.29 [-0.91, 0.33] | 1.0000 | n.s. | paired(n=3) |
| LoRA (5e-5) | -2.12 [-3.48, -0.76] | -6.70 | 0.0216 | 0.0862 | 0.3018 | n.s. | -1.00 [-1.45, -0.56] | 0.1984 | n.s. | paired(n=3) |
| LoRA-Null (2e-4) | -1.12 [-2.37, 0.13] | -3.86 | 0.0611 | 0.1221 | 0.6106 | n.s. | -1.20 [-5.61, 3.20] | 1.0000 | n.s. | paired(n=3) |
| CLoRA (1e-4 k1024) | -0.95 [-1.80, -0.11] | -4.86 | 0.0398 | 0.1193 | 0.4374 | n.s. | -0.37 [-1.07, 0.32] | 1.0000 | n.s. | paired(n=3) |
| DoRA (2e-4) | -2.02 [-3.12, -0.92] | -7.91 | 0.0156 | 0.0781 | 0.2593 | n.s. | -0.99 [-3.34, 1.35] | 1.0000 | n.s. | paired(n=3) |
| MiLoRA (2e-4) | -3.41 [-4.77, -2.05] | -10.77 | 0.0085 | 0.0511 | 0.1872 | n.s. | -1.11 [-4.28, 2.06] | 1.0000 | n.s. | paired(n=3) |

## Qwen-2.5-7B x math/GSM8K (qwswm sweep; ep6 variants excluded)

| Method (cell) | dRet [CI95] | t | p | p Holm(fam) | p Holm(all) | verdict | dAdapt [CI95] | p Holm(all) | verdict | test |
|---|---|---|---|---|---|---|---|---|---|---|
| SC-LoRA (5e-5) | +0.17 [-1.14, 1.48] | 0.56 | 0.6318 | 0.8773 | 1.0000 | n.s. | +8.26 [1.41, 15.12] | 0.4582 | n.s. | paired(n=3) |
| LoRA (1e-4) | -1.44 [-5.62, 2.74] | -1.49 | 0.2756 | 0.8773 | 1.0000 | n.s. | -5.48 [-23.16, 12.19] | 1.0000 | n.s. | paired(n=3) |
| LoRA(r32) (5e-4) | -5.75 [-31.99, 20.48] | -2.79 | 0.2193 | 0.8773 | 1.0000 | n.s. | +0.30 [-25.68, 26.29] | 1.0000 | n.s. | paired(n=2) |
| LoRA-Null (1e-3) | -2.77 [-4.26, -1.28] | -8.00 | 0.0153 | 0.0915 | 0.2593 | n.s. | +3.36 [-3.71, 10.43] | 1.0000 | n.s. | paired(n=3) |
| CLoRA (1e-3 k1024) | -7.56 [-8.56, -6.56] | -32.52 | 0.0009 | 0.0066 | 0.0236 | WORSE | +1.49 [-6.65, 9.63] | 1.0000 | n.s. | paired(n=3) |
| DoRA (2e-4) | -1.68 [-5.17, 1.82] | -6.09 | 0.1036 | 0.5180 | 0.8288 | n.s. | -5.79 [-85.27, 73.68] | 1.0000 | n.s. | paired(n=2) |
| MiLoRA (2e-4) | -1.38 [-5.38, 2.62] | -1.48 | 0.2762 | 0.8773 | 1.0000 | n.s. | -3.61 [-5.36, -1.86] | 0.2238 | n.s. | paired(n=3) |

## Summary

- Retention, Holm across all 25 testable comparisons (1 not testable, single-seed group; delta reported only): **0 method(s) significantly better** than LoRA+wd, 2 significantly worse, rest n.s.
  - not testable: PiSSA in llama_math (dRet=-26.34, dAdapt=-17.13, welch(1v3))
  - worse: CLoRA in llama_cs (d=-4.25, p_holm=0.0348)
  - worse: CLoRA in qwen_math (d=-7.56, p_holm=0.0236)
- Adaptation, Holm across all: 0 better, 5 worse, rest n.s.
- SC-LoRA on Qwen-math (the frozen layer's one exception): dAdapt=+8.26 raw p=0.0352, Holm(family)=0.2115, Holm(all)=0.4582; dRet=+0.17 raw p=0.6318.
