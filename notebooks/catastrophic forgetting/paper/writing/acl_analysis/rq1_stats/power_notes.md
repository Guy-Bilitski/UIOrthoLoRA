# Power notes: minimum detectable retention effects

MDE = smallest |paired mean delta| detectable at alpha=.05 (two-sided),
power=.8, at the observed number of common seeds and the empirical SD
of the paired per-seed deltas vs LoRA+wd (best-adaptation cells).
cell_sd_ret = median within-cell retention SD (cells with n>=2).
Script: `03_power_notes.py`.

## Family summary

| family | median cell SD (ret) | median MDE (pp) | max MDE (pp) |
|---|---|---|---|
| llama_cs | 0.37 | 2.67 | 4.97 |
| llama_math | 0.75 | 3.42 | 5.06 |
| qwen_cs | 0.85 | 1.71 | 49.89 |
| qwen_math | 0.67 | 4.49 | 33.73 |

## Per comparison

| family | method | common seeds | cell SD | SD(paired diff) | MDE (pp) |
|---|---|---|---|---|---|
| llama_cs | SC-LoRA | 3 | 4.16 | 1.52 | 4.97 |
| llama_cs | LoRA | 4 | 0.32 | 0.84 | 1.80 |
| llama_cs | LoRA-Null | 4 | 0.50 | 1.47 | 3.12 |
| llama_cs | CLoRA | 4 | 0.32 | 0.75 | 1.59 |
| llama_cs | DoRA | 3 | 0.28 | 1.32 | 4.29 |
| llama_cs | MiLoRA | 4 | 0.49 | 1.04 | 2.22 |
| llama_math | LoRA | 3 | 1.16 | 1.16 | 3.77 |
| llama_math | MiLoRA | 3 | 0.81 | 0.94 | 3.07 |
| llama_math | CLoRA | 3 | 0.58 | 0.94 | 3.07 |
| llama_math | SC-LoRA | 3 | 1.97 | 1.51 | 4.92 |
| llama_math | DoRA | 3 | 1.07 | 1.55 | 5.06 |
| llama_math | LoRA-Null | 3 | 0.63 | 0.80 | 2.61 |
| llama_math | PiSSA | 1 | -- | -- | -- |
| qwen_cs | SC-LoRA | 3 | 7.08 | 15.29 | 49.89 |
| qwen_cs | LoRA | 3 | 0.85 | 0.55 | 1.79 |
| qwen_cs | LoRA-Null | 3 | 0.81 | 0.50 | 1.64 |
| qwen_cs | CLoRA | 3 | 0.85 | 0.34 | 1.11 |
| qwen_cs | DoRA | 3 | 5.02 | 0.44 | 1.44 |
| qwen_cs | MiLoRA | 3 | 0.41 | 0.55 | 1.79 |
| qwen_math | SC-LoRA | 3 | 0.67 | 0.53 | 1.72 |
| qwen_math | LoRA | 3 | 0.82 | 1.68 | 5.49 |
| qwen_math | LoRA(r32) | 2 | 1.45 | 2.92 | 33.73 |
| qwen_math | LoRA-Null | 3 | 0.82 | 0.60 | 1.96 |
| qwen_math | CLoRA | 3 | 0.96 | 0.40 | 1.31 |
| qwen_math | DoRA | 2 | 0.32 | 0.39 | 4.49 |
| qwen_math | MiLoRA | 3 | 1.03 | 1.61 | 5.26 |
