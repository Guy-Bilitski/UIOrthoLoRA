# TOST equivalence: method offsets vs LoRA+wd at matched magnitude

Model: ret ~ log10 F_delta + method dummies (reference LoRA+wd),
CR1 cluster-robust SEs at recipe-cell level, deduped n=1034 pool,
retention = retention_mean (ladder convention). Equivalent at margin
m iff the 90% CI of the offset lies inside (-m, +m).
Script: `02_equivalence_tost.py`.

## Llama-2 CS  (G=56 cells)

| method | n | offset (pp) | 90% CI | 95% CI | eq +/-1 | eq +/-2 | eq +/-3 |
|---|---|---|---|---|---|---|---|
| clora | 27 | -2.49 | [-4.19, -0.79] | [-4.53, -0.45] | no | no | no |
| dora | 25 | +0.38 | [-0.71, +1.47] | [-0.93, +1.68] | no | YES | YES |
| lora | 25 | -0.29 | [-1.15, +0.56] | [-1.31, +0.73] | no | YES | YES |
| lora_null | 25 | -1.33 | [-2.27, -0.39] | [-2.45, -0.21] | no | no | YES |
| milora | 25 | -0.66 | [-1.70, +0.39] | [-1.91, +0.59] | no | YES | YES |
| milorawd | 2 | +0.49 | [-0.23, +1.22] | [-0.38, +1.36] | no | YES | YES |
| sclora | 24 | -4.95 | [-7.33, -2.58] | [-7.80, -2.11] | no | no | no |

## Llama-2 math  (G=42 cells)

| method | n | offset (pp) | 90% CI | 95% CI | eq +/-1 | eq +/-2 | eq +/-3 |
|---|---|---|---|---|---|---|---|
| clora | 21 | -1.63 | [-2.04, -1.23] | [-2.12, -1.15] | no | no | YES |
| dora | 21 | +0.53 | [-0.11, +1.16] | [-0.23, +1.29] | no | YES | YES |
| lora | 21 | +0.31 | [-0.19, +0.82] | [-0.29, +0.92] | YES | YES | YES |
| milora | 21 | -0.25 | [-0.64, +0.14] | [-0.72, +0.22] | YES | YES | YES |
| sclora | 15 | -2.92 | [-4.09, -1.75] | [-4.32, -1.52] | no | no | no |

## Qwen-2.5 CS  (G=58 cells)

| method | n | offset (pp) | 90% CI | 95% CI | eq +/-1 | eq +/-2 | eq +/-3 |
|---|---|---|---|---|---|---|---|
| clora | 23 | -3.43 | [-8.76, +1.90] | [-9.82, +2.96] | no | no | no |
| dora | 13 | -3.31 | [-8.65, +2.03] | [-9.71, +3.09] | no | no | no |
| lora | 23 | -4.69 | [-9.92, +0.53] | [-10.95, +1.56] | no | no | no |
| lora_null | 21 | -4.65 | [-9.57, +0.27] | [-10.54, +1.24] | no | no | no |
| milora | 25 | -2.88 | [-8.03, +2.28] | [-9.05, +3.29] | no | no | no |
| sclora | 23 | -3.72 | [-8.36, +0.92] | [-9.28, +1.83] | no | no | no |

## Qwen-2.5 math  (G=62 cells)

| method | n | offset (pp) | 90% CI | 95% CI | eq +/-1 | eq +/-2 | eq +/-3 |
|---|---|---|---|---|---|---|---|
| clora | 20 | -1.09 | [-4.38, +2.19] | [-5.02, +2.84] | no | no | no |
| dora | 11 | -1.82 | [-4.67, +1.03] | [-5.23, +1.59] | no | no | no |
| lora | 46 | -0.91 | [-4.45, +2.62] | [-5.14, +3.32] | no | no | no |
| lora_null | 20 | -0.43 | [-4.38, +3.51] | [-5.16, +4.29] | no | no | no |
| milora | 23 | -0.29 | [-4.25, +3.66] | [-5.03, +4.44] | no | no | no |
| sclora | 20 | -3.49 | [-8.50, +1.53] | [-9.49, +2.52] | no | no | no |

## Llama CS grid  (G=74 cells)

| method | n | offset (pp) | 90% CI | 95% CI | eq +/-1 | eq +/-2 | eq +/-3 |
|---|---|---|---|---|---|---|---|
| clora | 21 | +0.08 | [-0.43, +0.58] | [-0.53, +0.68] | YES | YES | YES |
| dora | 3 | +1.78 | [+1.03, +2.54] | [+0.88, +2.69] | no | no | YES |
| lora | 16 | +1.24 | [+0.69, +1.80] | [+0.58, +1.91] | no | YES | YES |
| lora_null | 26 | -1.92 | [-3.11, -0.73] | [-3.35, -0.50] | no | no | no |
| lorawdr16 | 9 | +0.88 | [+0.49, +1.27] | [+0.41, +1.35] | no | YES | YES |
| milora | 51 | +0.31 | [-0.51, +1.13] | [-0.67, +1.29] | no | YES | YES |
| pissa | 4 | -5.88 | [-6.94, -4.83] | [-7.15, -4.62] | no | no | no |
| sclora | 24 | -3.64 | [-4.87, -2.41] | [-5.11, -2.17] | no | no | no |

## Llama math-395k  (G=51 cells)

| method | n | offset (pp) | 90% CI | 95% CI | eq +/-1 | eq +/-2 | eq +/-3 |
|---|---|---|---|---|---|---|---|
| clora | 10 | +0.24 | [-0.58, +1.06] | [-0.75, +1.23] | no | YES | YES |
| dora | 3 | +4.87 | [+3.35, +6.39] | [+3.05, +6.69] | no | no | no |
| lora | 3 | +0.50 | [-0.43, +1.44] | [-0.62, +1.63] | no | YES | YES |
| lora_null | 4 | -0.39 | [-1.04, +0.26] | [-1.17, +0.39] | no | YES | YES |
| milora | 12 | -1.53 | [-4.27, +1.21] | [-4.81, +1.76] | no | no | no |
| pissa | 1 | -11.17 | [-12.50, -9.84] | [-12.77, -9.58] | no | no | no |
| sclora | 6 | -2.90 | [-4.55, -1.25] | [-4.87, -0.92] | no | no | no |

## pooled(all six, family FE)  (G=343 cells)

| method | n | offset (pp) | 90% CI | 95% CI | eq +/-1 | eq +/-2 | eq +/-3 |
|---|---|---|---|---|---|---|---|
| clora | 122 | -1.02 | [-2.10, +0.07] | [-2.31, +0.27] | no | no | YES |
| dora | 76 | +0.52 | [-1.17, +2.21] | [-1.50, +2.54] | no | no | YES |
| lora | 134 | -0.65 | [-2.11, +0.81] | [-2.39, +1.09] | no | no | YES |
| lora_null | 96 | -1.25 | [-2.55, +0.05] | [-2.80, +0.30] | no | no | YES |
| lorawdr16 | 9 | +0.51 | [+0.01, +1.02] | [-0.09, +1.11] | no | YES | YES |
| milora | 157 | -0.61 | [-1.70, +0.49] | [-1.91, +0.70] | no | YES | YES |
| milorawd | 2 | -0.62 | [-1.37, +0.14] | [-1.52, +0.28] | no | YES | YES |
| pissa | 5 | -7.12 | [-8.96, -5.28] | [-9.31, -4.93] | no | no | no |
| sclora | 112 | -4.08 | [-5.60, -2.56] | [-5.90, -2.27] | no | no | no |

## Summary

- pooled model: 0/9 methods equivalent to LoRA+wd within +/-1 pp at matched magnitude
- pooled model: 3/9 methods equivalent to LoRA+wd within +/-2 pp at matched magnitude
- pooled model: 7/9 methods equivalent to LoRA+wd within +/-3 pp at matched magnitude
- per-family: 14/39 method x family offsets equivalent within +/-2 pp; non-equivalences listed below
  - not eq at +/-2: clora in lrsw (offset -2.49, 90% CI [-4.19, -0.79])
  - not eq at +/-2: lora_null in lrsw (offset -1.33, 90% CI [-2.27, -0.39])
  - not eq at +/-2: sclora in lrsw (offset -4.95, 90% CI [-7.33, -2.58])
  - not eq at +/-2: clora in lrswm (offset -1.63, 90% CI [-2.04, -1.23])
  - not eq at +/-2: sclora in lrswm (offset -2.92, 90% CI [-4.09, -1.75])
  - not eq at +/-2: clora in qwsw (offset -3.43, 90% CI [-8.76, +1.90])
  - not eq at +/-2: dora in qwsw (offset -3.31, 90% CI [-8.65, +2.03])
  - not eq at +/-2: lora in qwsw (offset -4.69, 90% CI [-9.92, +0.53])
  - not eq at +/-2: lora_null in qwsw (offset -4.65, 90% CI [-9.57, +0.27])
  - not eq at +/-2: milora in qwsw (offset -2.88, 90% CI [-8.03, +2.28])
  - not eq at +/-2: sclora in qwsw (offset -3.72, 90% CI [-8.36, +0.92])
  - not eq at +/-2: clora in qwswm (offset -1.09, 90% CI [-4.38, +2.19])
  - not eq at +/-2: dora in qwswm (offset -1.82, 90% CI [-4.67, +1.03])
  - not eq at +/-2: lora in qwswm (offset -0.91, 90% CI [-4.45, +2.62])
  - not eq at +/-2: lora_null in qwswm (offset -0.43, 90% CI [-4.38, +3.51])
  - not eq at +/-2: milora in qwswm (offset -0.29, 90% CI [-4.25, +3.66])
  - not eq at +/-2: sclora in qwswm (offset -3.49, 90% CI [-8.50, +1.53])
  - not eq at +/-2: dora in frc (offset +1.78, 90% CI [+1.03, +2.54])
  - not eq at +/-2: lora_null in frc (offset -1.92, 90% CI [-3.11, -0.73])
  - not eq at +/-2: pissa in frc (offset -5.88, 90% CI [-6.94, -4.83])
  - not eq at +/-2: sclora in frc (offset -3.64, 90% CI [-4.87, -2.41])
  - not eq at +/-2: dora in frm (offset +4.87, 90% CI [+3.35, +6.39])
  - not eq at +/-2: milora in frm (offset -1.53, 90% CI [-4.27, +1.21])
  - not eq at +/-2: pissa in frm (offset -11.17, 90% CI [-12.50, -9.84])
  - not eq at +/-2: sclora in frm (offset -2.90, 90% CI [-4.55, -1.25])
