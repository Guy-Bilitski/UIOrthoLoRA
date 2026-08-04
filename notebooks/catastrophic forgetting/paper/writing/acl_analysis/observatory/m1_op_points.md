# M1 best-adaptation operating point per method x family

> mean+-sd over seeds within the single best-adaptation cell. Op point = recipe cell (config x LR) with highest seed-mean adaptation.

| family | method | lr | cell | n_seeds | adapt_mean | adapt_sd | retention_mean_mean | retention_mean_sd | retention_broad_mean | retention_broad_sd | fdelta_mean | fdelta_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lrsw | lorawd | 0.0005 | lrsw_lorawd_wd0p3_lr5e4 | 4 | 81.75 | 0.168 | 25.858 | 0.372 | 33.293 | 0.55 | 0.399 | 0.012 |
| lrsw | sclora | 5e-05 | lrsw_sclora_r32_lr5e5 | 3 | 80.613 | 0.41 | 24.6 | 1.848 | 33.633 | 1.054 | 0.376 | 0.158 |
| lrsw | milorawd | 0.0005 | lrsw_milorawd_wd0p3_r32_lr5e4 | 1 | 80.22 |  | 26.66 |  | 34.68 |  | 0.296 |  |
| lrsw | lora | 0.0003 | lrsw_lora_r16_lr3e4 | 4 | 79.17 | 0.2 | 23.858 | 0.483 | 31.778 | 1.157 | 0.616 | 0.011 |
| lrsw | lora_null | 0.0005 | lrsw_lora_null_r16_lr5e4 | 4 | 78.865 | 0.166 | 21.76 | 1.321 | 29.518 | 1.699 | 0.702 | 0.005 |
| lrsw | clora | 0.0005 | lrsw_clora_k1024_lr5e4 | 4 | 78.29 | 0.249 | 21.602 | 0.393 | 30.148 | 0.644 | 0.645 | 0.012 |
| lrsw | milora | 0.0005 | lrsw_milora_r32_lr5e4 | 4 | 77.192 | 0.42 | 21.432 | 0.87 | 28.785 | 0.982 | 0.852 | 0.013 |
| lrsw | dora | 0.0005 | lrsw_dora_r16_lr5e4 | 3 | 76.233 | 1.65 | 19.153 | 1.391 | 27.927 | 1.148 | 1.226 | 0.018 |
| lrswm | sclora | 0.0001 | lrswm_sclora_r32_lr1e4 | 2 | 59.095 | 0.375 | 22.67 | 0.198 | 32.185 | 0.078 | 0.391 | 0.0 |
| lrswm | lorawd | 0.0005 | lrswm_lorawd_wd0p3_lr5e4 | 3 | 50.67 | 1.325 | 24.21 | 0.324 | 33.363 | 0.261 | 0.408 | 0.012 |
| lrswm | clora | 0.0003 | lrswm_clora_k1024_lr3e4 | 3 | 48.62 | 1.091 | 23.527 | 0.643 | 31.64 | 0.767 | 0.322 | 0.006 |
| lrswm | lora | 0.0003 | lrswm_lora_r16_lr3e4 | 3 | 47.817 | 1.218 | 23.483 | 0.546 | 31.957 | 0.409 | 0.503 | 0.015 |
| lrswm | milora | 0.0003 | lrswm_milora_r32_lr3e4 | 3 | 47.613 | 0.723 | 23.72 | 0.597 | 32.347 | 0.172 | 0.457 | 0.013 |
| lrswm | dora | 0.0003 | lrswm_dora_r16_lr3e4 | 3 | 46.45 | 1.037 | 23.203 | 0.106 | 30.92 | 0.131 | 0.513 | 0.02 |
| qwsw | lorawd | 7e-05 | qwsw_lorawd_wd0p3_lr7e5 | 1 | 87.77 |  | 39.92 |  | 50.41 |  | 0.116 |  |
| qwsw | sclora | 7e-05 | qwsw_sclora_r32_lr7e5 | 1 | 87.53 |  | 37.79 |  | 45.01 |  | 0.246 |  |
| qwsw | lora | 7e-05 | qwsw_lora_r16_lr7e5 | 1 | 87.47 |  | 36.38 |  | 48.09 |  | 0.135 |  |
| qwsw | milora | 0.00015 | qwsw_milora_r32_lr15e5 | 1 | 87.33 |  | 37.64 |  | 46.51 |  | 0.234 |  |
| qwsw | clora | 0.0001 | qwsw_clora_k1024_lr1e4 | 4 | 87.015 | 0.19 | 39.515 | 1.145 | 49.698 | 0.74 | 0.128 | 0.001 |
| qwsw | dora | 0.0001 | qwsw_dora_r16_lr1e4 | 1 | 86.8 |  | 38.5 |  | 48.43 |  | 0.159 |  |
| qwsw | lora_null | 0.0002 | qwsw_lora_null_r16_lr2e4 | 3 | 86.23 | 1.602 | 38.947 | 0.682 | 47.41 | 0.751 | 0.204 | 0.011 |
| qwswm | sclora | 5e-05 | qwswm_sclora_r32_lr5e5 | 3 | 77.23 | 0.785 | 43.14 | 0.709 | 53.507 | 0.297 | 0.107 | 0.0 |
| qwswm | lora_null | 0.001 | qwswm_lora_null_r16_lr1e3 | 3 | 72.327 | 1.333 | 38.973 | 0.716 | 49.157 | 0.133 | 0.385 | 0.013 |
| qwswm | dora | 0.0003 | qwswm_dora_r16_lr3e4 | 1 | 70.96 |  | 36.39 |  | 50.54 |  | 0.157 |  |
| qwswm | clora | 0.001 | qwswm_clora_k1024_lr1e3 | 3 | 70.457 | 0.96 | 31.733 | 0.439 | 42.85 | 0.748 | 0.436 | 0.006 |
| qwswm | lorawd | 0.0003 | qwswm_lorawd_wd0p3_lr3e4_ep6 | 3 | 68.967 | 2.368 | 44.347 | 0.225 | 53.113 | 0.286 | 0.114 | 0.001 |
| qwswm | lora | 7e-05 | qwswm_lora_r32_lr7e5 | 1 | 67.17 |  | 43.72 |  | 53.62 |  | 0.063 |  |
| qwswm | milora | 0.0002 | qwswm_milora_r32_lr2e4 | 3 | 65.353 | 4.029 | 43.083 | 0.576 | 52.573 | 0.152 | 0.145 | 0.008 |
| frc | lorawd | 0.0005 | frc_lorawd_wd0p3_lr5e4_c256 | 4 | 81.858 | 0.205 | 25.943 | 0.699 | 32.597 | 0.9 | 0.402 | 0.014 |
| frc | sclora | 0.0001 | frc_sclora_lr1e4_c256 | 4 | 80.02 | 0.565 | 20.89 | 0.812 | 28.16 | 0.734 | 0.566 | 0.014 |
| frc | lora_null | 0.0002 | frc_lora_null_lr2e4_c256 | 5 | 79.846 | 0.77 | 23.128 | 0.267 | 31.162 | 0.317 | 0.536 | 0.002 |
| frc | clora | 0.0003 | frc_clora_k512_lr3e4_c256 | 5 | 79.386 | 0.358 | 23.31 | 0.767 | 31.386 | 1.339 | 0.521 | 0.019 |
| frc | lora | 0.0003 | frc_lora_r16_lr3e4_c256 | 3 | 79.367 | 0.662 | 23.56 | 0.711 | 32.14 | 0.195 | 0.603 | 0.001 |
| frc | dora | 0.0003 | frc_dora_r32_lr3e4_c256 | 3 | 78.787 | 0.225 | 21.61 | 1.021 | 29.89 | 0.97 | 0.917 | 0.019 |
| frc | milora | 0.0005 | frc_milora_a1r_lr5e4_c256 | 4 | 77.192 | 0.391 | 21.432 | 0.87 | 28.785 | 0.982 | 0.852 | 0.013 |
| frc | lorawdr16 | 0.0005 | frc_lorawdr16_wd0p3_lr5e4_c256 | 5 | 74.58 | 12.993 | 26.614 | 0.232 | 34.028 | 0.861 | 0.347 | 0.014 |
| frc | pissa | 0.0003 | frc_pissa_r32_lr3e4_c256 | 4 | 69.432 | 0.499 | 11.355 | 1.712 | 22.868 | 0.834 | 1.405 | 0.012 |
| frm | lorawd | 0.0002 | frm_lorawd_wd0p3_lr2e4_c512 | 3 | 68.483 | 0.905 | 26.0 | 0.521 | 35.063 | 0.423 | 0.282 | 0.003 |
| frm | lora_null | 0.0001 | frm_lora_null_lr1e4_c256 | 1 | 63.76 |  | 22.9 |  | 30.17 |  | 0.474 |  |
| frm | milora | 0.0001 | frm_milora_lr1e4_c256 | 3 | 63.683 | 0.798 | 23.973 | 0.058 | 32.28 | 0.48 | 0.45 | 0.002 |
| frm | clora | 0.0003 | frm_clora_k256_lr3e4_c256 | 3 | 60.653 | 0.401 | 18.95 | 0.089 | 27.21 | 0.63 | 1.011 | 0.007 |
| frm | sclora | 0.0001 | frm_sclora_lr1e4_c256 | 3 | 60.473 | 0.532 | 18.01 | 0.37 | 27.737 | 0.785 | 0.856 | 0.004 |
| frm | lora | 0.0003 | frm_lora_lr3e4_c256 | 3 | 59.593 | 1.533 | 18.107 | 0.306 | 27.093 | 0.434 | 1.288 | 0.008 |
| frm | dora | 0.0003 | frm_dora_lr3e4_c256 | 3 | 59.187 | 0.502 | 18.317 | 0.758 | 26.693 | 0.021 | 2.854 | 0.021 |
| frm | pissa | 0.0003 | frm_pissa_lr3e4_c256 | 1 | 49.66 |  | 3.62 |  | 21.7 |  | 2.206 |  |
