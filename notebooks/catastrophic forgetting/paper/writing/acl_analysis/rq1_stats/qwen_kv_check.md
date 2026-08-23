# Qwen k/v degeneracy check

Per-matrix store: 369 Qwen runs.

## 1. How much of the update sits in k/v

F^2-weighted share of the update's energy carried by k_proj and v_proj:
mean **0.030**, median **0.029**, 90th percentile **0.043**, max **0.237**.

## 2. Adapter-level coordinates, with and without k/v

| coordinate | mean |delta| | median |delta| | max |delta| |
|---|---|---|---|
| e_top | 0.0119 | 0.0110 | 0.0894 |
| e_bot | 0.0134 | 0.0110 | 0.1274 |
| ein_top | 0.0012 | 0.0006 | 0.0116 |
| ein_bot | 0.0017 | 0.0010 | 0.0163 |
| stable_rank | 0.0541 | 0.0282 | 0.2390 |

## 3. Residual association with retention, given log10 F_delta

Partial correlation of each coordinate with retention after
regressing both on log10 F_delta, per Qwen family.

| family | coordinate | full | k/v excluded |
|---|---|---|---|
| qwsw | e_top | -0.119 (n=151) | -0.184 (n=151) |
| qwsw | stable_rank | -0.004 (n=151) | +0.028 (n=151) |
| qwswm | e_top | -0.145 (n=164) | -0.264 (n=160) |
| qwswm | stable_rank | +0.073 (n=164) | +0.077 (n=160) |
