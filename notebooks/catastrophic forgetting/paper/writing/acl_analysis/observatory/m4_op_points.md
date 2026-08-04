# M4 CE drift at each method's best-adaptation operating point

> mean+-sd over seeds within the single best-adaptation cell. Qwen cells may have <3 seeds with CE (seed-block deletion) -- treat sd as indicative.

| family | method | lr | cell | n_seeds | adapt_mean | adapt_sd | forgetting_kl_mean | forgetting_kl_sd | forgetting_ce_mean | forgetting_ce_sd |
|---|---|---|---|---|---|---|---|---|---|---|
| lrsw | milorawd | 0.0005 | lrsw_milorawd_wd0p3_r32_lr5e4 | 1 | 80.22 |  | 0.149 |  | 1.959 |  |
| lrsw | lorawd | 0.0005 | lrsw_lorawd_wd0p3_lr5e4 | 4 | 81.75 | 0.168 | 0.189 | 0.003 | 2.041 | 0.003 |
| lrsw | sclora | 5e-05 | lrsw_sclora_r32_lr5e5 | 3 | 80.613 | 0.41 | 0.211 | 0.088 | 2.063 | 0.088 |
| lrsw | lora | 0.0003 | lrsw_lora_r16_lr3e4 | 4 | 79.17 | 0.2 | 0.416 | 0.028 | 2.257 | 0.035 |
| lrsw | lora_null | 0.0005 | lrsw_lora_null_r16_lr5e4 | 4 | 78.865 | 0.166 | 0.557 | 0.03 | 2.399 | 0.032 |
| lrsw | clora | 0.0005 | lrsw_clora_k1024_lr5e4 | 4 | 78.29 | 0.249 | 0.641 | 0.018 | 2.483 | 0.036 |
| lrsw | dora | 0.0005 | lrsw_dora_r16_lr5e4 | 3 | 76.233 | 1.65 | 0.714 | 0.008 | 2.538 | 0.032 |
| lrsw | milora | 0.0005 | lrsw_milora_r32_lr5e4 | 4 | 77.192 | 0.42 | 0.742 | 0.013 | 2.562 | 0.028 |
| lrswm | lorawd | 0.0005 | lrswm_lorawd_wd0p3_lr5e4 | 3 | 50.67 | 1.325 | 0.346 | 0.018 | 2.17 | 0.009 |
| lrswm | sclora | 0.0001 | lrswm_sclora_r32_lr1e4 | 2 | 59.095 | 0.375 | 0.348 | 0.017 | 2.158 | 0.017 |
| lrswm | lora | 0.0003 | lrswm_lora_r16_lr3e4 | 3 | 47.817 | 1.218 | 0.395 | 0.01 | 2.219 | 0.024 |
| lrswm | clora | 0.0003 | lrswm_clora_k1024_lr3e4 | 3 | 48.62 | 1.091 | 0.405 | 0.021 | 2.229 | 0.044 |
| lrswm | dora | 0.0003 | lrswm_dora_r16_lr3e4 | 3 | 46.45 | 1.037 | 0.445 | 0.018 | 2.269 | 0.029 |
| lrswm | milora | 0.0003 | lrswm_milora_r32_lr3e4 | 3 | 47.613 | 0.723 | 0.479 | 0.007 | 2.303 | 0.029 |
| qwsw | lora | 7e-05 | qwsw_lora_r16_lr7e5 | 1 | 87.47 |  | 0.087 |  | 2.022 |  |
| qwsw | lorawd | 7e-05 | qwsw_lorawd_wd0p3_lr7e5 | 1 | 87.77 |  | 0.087 |  | 2.022 |  |
| qwsw | clora | 0.0001 | qwsw_clora_k1024_lr1e4 | 4 | 87.015 | 0.19 | 0.131 | 0.005 | 2.065 | 0.005 |
| qwsw | lora_null | 0.0002 | qwsw_lora_null_r16_lr2e4 | 3 | 86.23 | 1.602 | 0.205 |  | 2.14 |  |
| qwsw | sclora | 7e-05 | qwsw_sclora_r32_lr7e5 | 1 | 87.53 |  | 0.214 |  | 2.149 |  |
| qwsw | milora | 0.00015 | qwsw_milora_r32_lr15e5 | 1 | 87.33 |  | 0.395 |  | 2.33 |  |
| qwsw | dora | 0.0001 | qwsw_dora_r16_lr1e4 | 1 | 86.8 |  |  |  |  |  |
| qwswm | lora | 7e-05 | qwswm_lora_r32_lr7e5 | 1 | 67.17 |  | 0.05 |  | 1.985 |  |
| qwswm | sclora | 5e-05 | qwswm_sclora_r32_lr5e5 | 3 | 77.23 | 0.785 | 0.103 | 0.001 | 2.038 | 0.001 |
| qwswm | lorawd | 0.0003 | qwswm_lorawd_wd0p3_lr3e4_ep6 | 3 | 68.967 | 2.368 | 0.188 | 0.001 | 2.122 | 0.001 |
| qwswm | milora | 0.0002 | qwswm_milora_r32_lr2e4 | 3 | 65.353 | 4.029 | 0.27 | 0.017 | 2.205 | 0.017 |
| qwswm | lora_null | 0.001 | qwswm_lora_null_r16_lr1e3 | 3 | 72.327 | 1.333 | 0.413 |  | 2.348 |  |
| qwswm | clora | 0.001 | qwswm_clora_k1024_lr1e3 | 3 | 70.457 | 0.96 | 0.734 | 0.005 | 2.669 | 0.005 |
| qwswm | dora | 0.0003 | qwswm_dora_r16_lr3e4 | 1 | 70.96 |  |  |  |  |  |
| frc | lorawdr16 | 0.0005 | frc_lorawdr16_wd0p3_lr5e4_c256 | 5 | 74.58 | 12.993 | 0.133 | 0.002 | 1.951 | 0.018 |
| frc | lorawd | 0.0005 | frc_lorawd_wd0p3_lr5e4_c256 | 4 | 81.858 | 0.205 | 0.21 | 0.031 | 2.031 | 0.03 |
| frc | lora | 0.0003 | frc_lora_r16_lr3e4_c256 | 3 | 79.367 | 0.662 | 0.408 | 0.007 | 2.232 | 0.027 |
| frc | sclora | 0.0001 | frc_sclora_lr1e4_c256 | 4 | 80.02 | 0.565 | 0.451 | 0.011 | 2.271 | 0.018 |
| frc | clora | 0.0003 | frc_clora_k512_lr3e4_c256 | 5 | 79.386 | 0.358 | 0.503 | 0.026 | 2.33 | 0.03 |
| frc | lora_null | 0.0002 | frc_lora_null_lr2e4_c256 | 5 | 79.846 | 0.77 | 0.549 | 0.018 | 2.376 | 0.02 |
| frc | dora | 0.0003 | frc_dora_r32_lr3e4_c256 | 3 | 78.787 | 0.225 | 0.646 | 0.012 | 2.47 | 0.029 |
| frc | milora | 0.0005 | frc_milora_a1r_lr5e4_c256 | 4 | 77.192 | 0.391 | 0.742 | 0.013 | 2.562 | 0.028 |
| frc | pissa | 0.0003 | frc_pissa_r32_lr3e4_c256 | 4 | 69.432 | 0.499 | 1.977 | 0.098 | 3.798 | 0.114 |
| frm | lorawd | 0.0002 | frm_lorawd_wd0p3_lr2e4_c512 | 3 | 68.483 | 0.905 | 0.216 | 0.006 | 2.04 | 0.02 |
| frm | milora | 0.0001 | frm_milora_lr1e4_c256 | 3 | 63.683 | 0.798 | 0.672 | 0.027 | 2.524 | 0.027 |
| frm | lora_null | 0.0001 | frm_lora_null_lr1e4_c256 | 1 | 63.76 |  | 0.731 |  | 2.583 |  |
| frm | sclora | 0.0001 | frm_sclora_lr1e4_c256 | 3 | 60.473 | 0.532 | 1.18 | 0.022 | 3.032 | 0.022 |
| frm | clora | 0.0003 | frm_clora_k256_lr3e4_c256 | 3 | 60.653 | 0.401 | 1.396 | 0.007 | 3.22 | 0.022 |
| frm | lora | 0.0003 | frm_lora_lr3e4_c256 | 3 | 59.593 | 1.533 | 1.739 | 0.02 | 3.591 | 0.02 |
| frm | dora | 0.0003 | frm_dora_lr3e4_c256 | 3 | 59.187 | 0.502 | 2.425 | 0.425 | 4.249 | 0.415 |
| frm | pissa | 0.0003 | frm_pissa_lr3e4_c256 | 1 | 49.66 |  | 4.455 |  | 6.307 |  |
