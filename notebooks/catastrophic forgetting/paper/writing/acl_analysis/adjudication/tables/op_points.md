# Operating-point tables (adjudication, 2026-07-18)

Source: `results/*/summary.json`, quarantine-excluded, `_reeval` duplicate dropped.
Rule: best-mean-adaptation LR per method over landed seeds (n>=2 preferred).
Matched-retention cut: best adaptation among cells with mean retention >= base-1pp
(secondary cut at base-2pp). CorDA/CorDA++ WITHHELD (port bug) — shown, never ranked.
`*` on a cut cell = answer-format-collapse seed(s) inside the cell (adaptation
collapsed, retention intact — 02_operating_points.md section 1 note).
Script: `01_op_points.py`.

## Llama-2-7B x Commonsense-8 (lrsw sweep, r-matched grid)
Base retention ceiling = 26.0 (ret_core); adaptation = CS-8. Cuts: ret >= 25.00 / 24.00.

| Method | best LR | CS-8 | ret | F_D | n | flags | cut(-1pp): LR / adapt / ret | cut(-2pp): LR / adapt / ret |
|---|---|---|---|---|---|---|---|---|
| LoRA+wd | 5e-4 | 81.75 ± 0.17 | 25.86 ± 0.37 | 0.399 | 4 |  | 5e-4 / 81.75 ± 0.17 / 25.86 | 5e-4 / 81.75 ± 0.17 / 25.86 |
| SC-LoRA | 5e-5 | 80.61 ± 0.41 | 24.60 ± 1.85 | 0.376 | 3 |  | 2e-5 / 45.97 ± 30.51* / 26.43 | 5e-5 / 80.61 ± 0.41 / 24.60 |
| LoRA | 3e-4 | 79.17 ± 0.20 | 23.86 ± 0.48 | 0.616 | 4 |  | 1e-4 / 69.54 ± 11.30* / 26.47 | 1e-4 / 69.54 ± 11.30* / 26.47 |
| LoRA-Null | 5e-4 | 78.87 ± 0.17 | 21.76 ± 1.32 | 0.702 | 4 |  | 2e-5 / 70.79 ± 2.10 / 26.11 | 2e-5 / 70.79 ± 2.10 / 26.11 |
| CLoRA | 5e-4 (k1024) | 78.29 ± 0.25 | 21.60 ± 0.39 | 0.645 | 4 |  | 2e-4 / 66.37 ± 13.31* / 25.58 | 3e-4 / 68.06 ± 8.34* / 24.23 |
| MiLoRA | 5e-4 | 77.19 ± 0.42 | 21.43 ± 0.87 | 0.852 | 4 |  | 2e-5 / 67.87 ± 4.34 / 26.17 | 2e-5 / 67.87 ± 4.34 / 26.17 |
| DoRA | 5e-4 | 76.23 ± 1.65 | 19.15 ± 1.39 | 1.226 | 3 |  | 2e-4 / 74.29 ± 8.66* / 25.20 | 2e-4 / 74.29 ± 8.66* / 25.20 |

Note (02_operating_points.md convention): under the s42-best-LR rule, DoRA's retention-relevant point is 2e-4 and MiLoRA's is 3e-4 — the mean-rule 5e-4 picks are the highest LR whose seeds all avoided the answer-format-collapse basin, paying 3–5 pp retention. Alt rows:
- DoRA @ 2e-4: adapt 74.29 ± 8.66 (format-collapse seeds, retention intact), ret 25.20 ± 0.34, n=3
- MiLoRA @ 3e-4: adapt 63.09 ± 21.44 (format-collapse seeds, retention intact), ret 24.37 ± 0.52, n=4

Note (E4, section 18.3): SC-LoRA's Llama retention deficit at its calibrated points is a calibration-set artifact — eval-matched calibration puts it +0.92 pp ABOVE the family curve (n=20). Do not read its below-LoRA+wd retention here as method geometry.

## Llama-2-7B x math/GSM8K (frm faithful CLoRA recipe, c256)
Base retention ceiling = 33.1 (bbh); adaptation = GSM8K. Cuts: ret >= 32.10 / 31.10.

| Method | best LR | GSM8K | ret | F_D | n | flags | cut(-1pp): LR / adapt / ret | cut(-2pp): LR / adapt / ret |
|---|---|---|---|---|---|---|---|---|
| LoRA+wd | 2e-4 | 66.79 ± 0.79 | 33.57 ± 1.04 | 0.280 | 3 |  | 2e-4 / 66.79 ± 0.79 / 33.57 | 2e-4 / 66.79 ± 0.79 / 33.57 |
| LoRA | 1e-4 | 63.99 ± 0.87 | 31.29 ± 0.35 | 0.442 | 3 |  | — / none / — | 1e-4 / 63.99 ± 0.87 / 31.29 |
| MiLoRA | 1e-4 | 63.68 ± 0.80 | 32.44 ± 0.10 | 0.450 | 3 |  | 1e-4 / 63.68 ± 0.80 / 32.44 | 1e-4 / 63.68 ± 0.80 / 32.44 |
| LoRA-Null | 2e-4 | 61.86 ± 0.59 | 28.90 ± 0.63 | 0.886 | 3 |  | 1e-4 / 63.76 ± 0.00 (n=1) / 32.15 | 1e-4 / 63.76 ± 0.00 (n=1) / 32.15 |
| CLoRA | 3e-4 (k256) | 60.65 ± 0.40 | 28.63 ± 0.11 | 1.011 | 3 |  | — / none / — | — / none / — |
| SC-LoRA | 1e-4 | 60.47 ± 0.53 | 27.94 ± 0.52 | 0.856 | 3 |  | — / none / — | — / none / — |
| DoRA | 3e-4 | 59.19 ± 0.50 | 28.09 ± 1.07 | 2.854 | 3 |  | — / none / — | — / none / — |
| PiSSA | 3e-4 | 49.66 ± 0.00 | 7.23 ± 0.00 | 2.206 | 1 |  | — / none / — | — / none / — |
| CorDA++ [WITHHELD] | 1e-4 | 58.76 ± 0.00 | 31.56 ± 0.00 | 0.632 | 1 |  | — / none / — | 1e-4 / 58.76 ± 0.00 (n=1) / 31.56 |

Note (E4, section 18.3): SC-LoRA's Llama retention deficit at its calibrated points is a calibration-set artifact — eval-matched calibration puts it +0.92 pp ABOVE the family curve (n=20). Do not read its below-LoRA+wd retention here as method geometry.

## Qwen-2.5-7B x Commonsense-8 (qwsw sweep)
Base retention ceiling = 44.35 (ret_core); adaptation = CS-8. Cuts: ret >= 43.35 / 42.35.

| Method | best LR | CS-8 | ret | F_D | n | flags | cut(-1pp): LR / adapt / ret | cut(-2pp): LR / adapt / ret |
|---|---|---|---|---|---|---|---|---|
| LoRA+wd | 5e-4 | 87.43 ± 0.23 | 40.07 ± 0.68 | 0.246 | 3 |  | — / none / — | — / none / — |
| SC-LoRA | 1e-4 | 87.15 ± 0.15 | 27.85 ± 15.96 | 0.348 | 3 |  | — / none / — | — / none / — |
| CLoRA | 1e-4 (k1024) | 87.02 ± 0.19 | 39.52 ± 1.15 | 0.128 | 4 |  | — / none / — | — / none / — |
| DoRA | 2e-4 | 86.44 ± 0.76 | 38.05 ± 0.97 | 0.260 | 3 |  | — / none / — | — / none / — |
| LoRA | 5e-5 | 86.43 ± 0.41 | 37.95 ± 0.88 | 0.122 | 3 |  | — / none / — | — / none / — |
| MiLoRA | 2e-4 | 86.39 ± 0.97 | 36.68 ± 0.12 | 0.284 | 4 |  | — / none / — | — / none / — |
| LoRA-Null | 2e-4 | 86.23 ± 1.60 | 38.95 ± 0.68 | 0.204 | 3 |  | — / none / — | — / none / — |

## Qwen-2.5-7B x math/GSM8K (qwswm sweep; ep6 variants excluded)
Base retention ceiling = 47.93 (bbh); adaptation = GSM8K. Cuts: ret >= 46.93 / 45.93.

| Method | best LR | GSM8K | ret | F_D | n | flags | cut(-1pp): LR / adapt / ret | cut(-2pp): LR / adapt / ret |
|---|---|---|---|---|---|---|---|---|
| SC-LoRA | 5e-5 | 77.23 ± 0.79 | 47.71 ± 0.23 | 0.107 | 3 |  | 5e-5 / 77.23 ± 0.79 / 47.71 | 5e-5 / 77.23 ± 0.79 / 47.71 |
| LoRA-Null | 1e-3 | 72.33 ± 1.33 | 44.76 ± 0.64 | 0.385 | 3 |  | 2e-4 / 70.48 ± 4.62 / 47.00 | 2e-4 / 70.48 ± 4.62 / 47.00 |
| CLoRA | 1e-3 (k1024) | 70.46 ± 0.96 | 39.98 ± 0.12 | 0.436 | 3 |  | 5e-5 / 59.16 ± 3.25 / 47.40 | 3e-4 / 69.85 ± 4.46 / 46.02 |
| LoRA(r32) | 5e-4 | 70.44 ± 0.86 | 41.77 ± 2.31 | 0.386 | 2 |  | 2e-4 / 63.28 ± 1.23 / 47.03 | 1e-4 / 66.14 ± 1.33 / 46.19 |
| LoRA+wd | 3e-4 | 68.97 ± 3.33 | 47.54 ± 0.43 | 0.102 | 3 |  | 3e-4 / 68.97 ± 3.33 / 47.54 | 3e-4 / 68.97 ± 3.33 / 47.54 |
| MiLoRA | 2e-4 | 65.35 ± 4.03 | 46.16 ± 1.28 | 0.145 | 3 |  | 1.5e-4 / 57.92 ± 0.00 (n=1) / 46.97 | 2e-4 / 65.35 ± 4.03 / 46.16 |
| DoRA | 2e-4 | 63.91 ± 4.50 | 46.08 ± 0.66 | 0.112 | 2 |  | 2e-5 / 56.14 ± 0.37 / 47.98 | 2e-4 / 63.91 ± 4.50 / 46.08 |
| LoRA | 1e-4 | 63.48 ± 4.31 | 46.09 ± 1.54 | 0.072 | 3 |  | 3e-4 / 61.97 ± 4.80 / 47.13 | 1e-4 / 63.48 ± 4.31 / 46.09 |
