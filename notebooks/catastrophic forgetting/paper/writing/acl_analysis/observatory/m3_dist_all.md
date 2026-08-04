# M3 geometry distributions per method x family (all on-pool runs; CorDA flagged WITHHELD)

> sd is over all on-pool runs of the cell (configs x LRs x seeds); seeds within a cell are correlated (ICC~0.78), so this sd mixes cell-to-cell and seed noise.

| metric | family | method | n | mean | sd | min | max |
|---|---|---|---|---|---|---|---|
| stable_rank_w | lrsw | lora | 25 | 4.418 | 2.9 | 1.26 | 9.982 |
| stable_rank_w | lrsw | lora_null | 25 | 6.671 | 1.881 | 4.348 | 10.484 |
| stable_rank_w | lrsw | lorawd | 27 | 5.432 | 3.642 | 1.017 | 11.741 |
| stable_rank_w | lrsw | milora | 25 | 6.172 | 3.935 | 2.779 | 15.091 |
| stable_rank_w | lrsw | milorawd | 2 | 6.018 | 2.635 | 4.155 | 7.881 |
| stable_rank_w | lrsw | clora | 27 | 7.535 | 5.232 | 1.143 | 16.651 |
| stable_rank_w | lrsw | dora | 25 | 3.842 | 2.432 | 1.262 | 8.002 |
| stable_rank_w | lrsw | sclora | 24 | 12.202 | 3.538 | 5.13 | 17.684 |
| stable_rank_w | lrswm | lora | 21 | 4.376 | 3.03 | 1.315 | 10.511 |
| stable_rank_w | lrswm | lorawd | 21 | 6.066 | 4.222 | 1.546 | 13.971 |
| stable_rank_w | lrswm | milora | 21 | 5.958 | 3.812 | 2.722 | 14.462 |
| stable_rank_w | lrswm | clora | 21 | 7.048 | 4.284 | 2.398 | 15.489 |
| stable_rank_w | lrswm | dora | 21 | 4.161 | 2.642 | 1.292 | 9.29 |
| stable_rank_w | lrswm | sclora | 15 | 12.8 | 3.503 | 7.103 | 18.751 |
| stable_rank_w | qwsw | lora | 23 | 5.25 | 3.228 | 1.564 | 10.634 |
| stable_rank_w | qwsw | lora_null | 21 | 7.291 | 2.144 | 4.707 | 11.081 |
| stable_rank_w | qwsw | lorawd | 23 | 6.543 | 4.197 | 1.1 | 12.877 |
| stable_rank_w | qwsw | milora | 25 | 10.334 | 3.296 | 6.345 | 16.768 |
| stable_rank_w | qwsw | clora | 23 | 8.794 | 5.601 | 2.089 | 17.905 |
| stable_rank_w | qwsw | dora | 13 | 6.375 | 2.957 | 1.604 | 9.487 |
| stable_rank_w | qwsw | sclora | 23 | 13.953 | 3.185 | 6.635 | 17.568 |
| stable_rank_w | qwswm | lora | 46 | 8.045 | 4.499 | 1.698 | 17.635 |
| stable_rank_w | qwswm | lora_null | 20 | 8.827 | 1.972 | 5.744 | 11.637 |
| stable_rank_w | qwswm | lorawd | 24 | 8.533 | 4.976 | 1.059 | 15.862 |
| stable_rank_w | qwswm | milora | 23 | 11.234 | 3.591 | 1.375 | 18.192 |
| stable_rank_w | qwswm | clora | 20 | 11.984 | 5.354 | 4.068 | 19.771 |
| stable_rank_w | qwswm | dora | 11 | 5.709 | 3.317 | 2.307 | 11.087 |
| stable_rank_w | qwswm | sclora | 20 | 15.93 | 2.233 | 12.375 | 19.491 |
| stable_rank_w | frc | lora | 16 | 5.866 | 1.921 | 3.144 | 7.547 |
| stable_rank_w | frc | lora_null | 26 | 12.307 | 3.424 | 5.225 | 16.61 |
| stable_rank_w | frc | lorawd | 121 | 7.651 | 4.26 | 1.485 | 15.515 |
| stable_rank_w | frc | lorawdr16 | 9 | 5.966 | 1.273 | 4.522 | 7.102 |
| stable_rank_w | frc | milora | 51 | 7.956 | 4.088 | 1.644 | 15.204 |
| stable_rank_w | frc | clora | 21 | 7.711 | 1.091 | 5.864 | 8.912 |
| stable_rank_w | frc | dora | 3 | 7.119 | 0.178 | 6.914 | 7.232 |
| stable_rank_w | frc | sclora | 24 | 14.791 | 1.988 | 11.863 | 17.426 |
| stable_rank_w | frc | pissa | 4 | 18.058 | 0.04 | 18.026 | 18.117 |
| stable_rank_w | frm | lora | 3 | 23.244 | 0.102 | 23.159 | 23.357 |
| stable_rank_w | frm | lora_null | 4 | 24.261 | 2.968 | 19.809 | 25.771 |
| stable_rank_w | frm | lorawd | 105 | 16.272 | 6.552 | 1.022 | 29.142 |
| stable_rank_w | frm | milora | 12 | 20.34 | 7.349 | 9.929 | 29.573 |
| stable_rank_w | frm | clora | 10 | 25.549 | 0.258 | 25.238 | 25.959 |
| stable_rank_w | frm | dora | 3 | 19.266 | 0.175 | 19.159 | 19.468 |
| stable_rank_w | frm | sclora | 6 | 25.815 | 2.205 | 23.797 | 27.923 |
| stable_rank_w | frm | pissa | 1 | 32.63 |  | 32.63 | 32.63 |
| stable_rank_w | lrsw | corda [WITHHELD] | 23 | 7.094 | 2.107 | 1.509 | 10.415 |
| stable_rank_w | qwsw | corda [WITHHELD] | 5 | 7.044 | 1.328 | 5.898 | 9.183 |
| stable_rank_w | frc | cordapp [WITHHELD] | 8 | 1.861 | 0.717 | 1.207 | 3.282 |
| stable_rank_w | frm | cordapp [WITHHELD] | 3 | 5.781 | 1.445 | 4.776 | 7.437 |
| eff_rank_w | lrsw | lora | 25 | 9.1 | 4.428 | 2.237 | 15.0 |
| eff_rank_w | lrsw | lora_null | 25 | 15.997 | 0.381 | 15.311 | 16.742 |
| eff_rank_w | lrsw | lorawd | 27 | 14.056 | 8.057 | 1.05 | 25.567 |
| eff_rank_w | lrsw | milora | 25 | 15.52 | 5.751 | 9.704 | 27.139 |
| eff_rank_w | lrsw | milorawd | 2 | 17.236 | 5.289 | 13.496 | 20.976 |
| eff_rank_w | lrsw | clora | 27 | 18.393 | 8.06 | 1.456 | 29.255 |
| eff_rank_w | lrsw | dora | 25 | 7.957 | 4.351 | 2.241 | 13.399 |
| eff_rank_w | lrsw | sclora | 24 | 32.885 | 2.722 | 23.897 | 36.116 |
| eff_rank_w | lrswm | lora | 21 | 8.957 | 4.419 | 2.383 | 15.065 |
| eff_rank_w | lrswm | lorawd | 21 | 15.523 | 8.003 | 3.748 | 27.074 |
| eff_rank_w | lrswm | milora | 21 | 15.443 | 5.467 | 9.539 | 26.363 |
| eff_rank_w | lrswm | clora | 21 | 18.842 | 6.586 | 9.015 | 27.919 |
| eff_rank_w | lrswm | dora | 21 | 8.793 | 4.2 | 2.341 | 14.222 |
| eff_rank_w | lrswm | sclora | 15 | 33.978 | 2.003 | 28.163 | 35.934 |
| eff_rank_w | qwsw | lora | 23 | 10.3 | 4.232 | 3.42 | 15.368 |
| eff_rank_w | qwsw | lora_null | 21 | 15.55 | 1.133 | 13.701 | 17.595 |
| eff_rank_w | qwsw | lorawd | 23 | 16.724 | 8.128 | 1.33 | 27.101 |
| eff_rank_w | qwsw | milora | 25 | 25.851 | 1.878 | 23.319 | 29.827 |
| eff_rank_w | qwsw | clora | 23 | 20.519 | 7.851 | 7.241 | 29.794 |
| eff_rank_w | qwsw | dora | 13 | 11.881 | 3.717 | 3.56 | 14.746 |
| eff_rank_w | qwsw | sclora | 23 | 34.8 | 2.448 | 28.682 | 37.768 |
| eff_rank_w | qwswm | lora | 46 | 17.821 | 7.345 | 3.832 | 30.093 |
| eff_rank_w | qwswm | lora_null | 20 | 17.735 | 1.058 | 16.328 | 19.062 |
| eff_rank_w | qwswm | lorawd | 24 | 20.362 | 7.72 | 1.16 | 28.316 |
| eff_rank_w | qwswm | milora | 23 | 29.491 | 5.971 | 2.62 | 32.442 |
| eff_rank_w | qwswm | clora | 20 | 25.015 | 4.234 | 16.838 | 30.308 |
| eff_rank_w | qwswm | dora | 11 | 11.201 | 3.252 | 6.536 | 15.272 |
| eff_rank_w | qwswm | sclora | 20 | 38.258 | 2.701 | 34.243 | 41.552 |
| eff_rank_w | frc | lora | 16 | 15.783 | 6.896 | 6.156 | 21.57 |
| eff_rank_w | frc | lora_null | 26 | 29.546 | 1.902 | 25.381 | 30.717 |
| eff_rank_w | frc | lorawd | 121 | 18.517 | 7.227 | 3.556 | 28.574 |
| eff_rank_w | frc | lorawdr16 | 9 | 12.254 | 1.209 | 10.91 | 13.331 |
| eff_rank_w | frc | milora | 51 | 18.239 | 5.912 | 4.107 | 27.139 |
| eff_rank_w | frc | clora | 21 | 22.189 | 1.994 | 18.761 | 24.008 |
| eff_rank_w | frc | dora | 3 | 20.803 | 0.122 | 20.662 | 20.879 |
| eff_rank_w | frc | sclora | 24 | 36.273 | 14.235 | 25.195 | 73.289 |
| eff_rank_w | frc | pissa | 4 | 39.724 | 0.036 | 39.689 | 39.774 |
| eff_rank_w | frm | lora | 3 | 52.21 | 0.029 | 52.186 | 52.243 |
| eff_rank_w | frm | lora_null | 4 | 62.456 | 0.935 | 61.967 | 63.858 |
| eff_rank_w | frm | lorawd | 105 | 41.566 | 11.35 | 1.051 | 57.164 |
| eff_rank_w | frm | milora | 12 | 45.742 | 8.808 | 33.025 | 56.205 |
| eff_rank_w | frm | clora | 10 | 54.703 | 0.376 | 54.134 | 55.14 |
| eff_rank_w | frm | dora | 3 | 45.768 | 0.231 | 45.577 | 46.025 |
| eff_rank_w | frm | sclora | 6 | 63.05 | 0.06 | 62.958 | 63.11 |
| eff_rank_w | frm | pissa | 1 | 71.879 |  | 71.879 | 71.879 |
| eff_rank_w | lrsw | corda [WITHHELD] | 23 | 11.692 | 2.558 | 2.907 | 14.789 |
| eff_rank_w | qwsw | corda [WITHHELD] | 5 | 12.547 | 1.432 | 11.384 | 14.9 |
| eff_rank_w | frc | cordapp [WITHHELD] | 8 | 4.869 | 3.193 | 1.713 | 10.177 |
| eff_rank_w | frm | cordapp [WITHHELD] | 3 | 32.031 | 12.294 | 19.193 | 43.696 |
| e_top_w | lrsw | lora | 25 | 0.07 | 0.005 | 0.062 | 0.081 |
| e_top_w | lrsw | lora_null | 25 | 0.159 | 0.054 | 0.079 | 0.249 |
| e_top_w | lrsw | lorawd | 27 | 0.071 | 0.005 | 0.06 | 0.08 |
| e_top_w | lrsw | milora | 25 | 0.066 | 0.007 | 0.052 | 0.08 |
| e_top_w | lrsw | milorawd | 2 | 0.071 | 0.004 | 0.069 | 0.074 |
| e_top_w | lrsw | clora | 27 | 0.061 | 0.004 | 0.058 | 0.073 |
| e_top_w | lrsw | dora | 25 | 0.082 | 0.025 | 0.062 | 0.148 |
| e_top_w | lrsw | sclora | 24 | 0.093 | 0.011 | 0.081 | 0.114 |
| e_top_w | lrswm | lora | 21 | 0.066 | 0.003 | 0.06 | 0.072 |
| e_top_w | lrswm | lorawd | 21 | 0.069 | 0.005 | 0.06 | 0.077 |
| e_top_w | lrswm | milora | 21 | 0.061 | 0.007 | 0.05 | 0.071 |
| e_top_w | lrswm | clora | 21 | 0.06 | 0.002 | 0.057 | 0.062 |
| e_top_w | lrswm | dora | 21 | 0.07 | 0.011 | 0.059 | 0.096 |
| e_top_w | lrswm | sclora | 15 | 0.101 | 0.004 | 0.094 | 0.113 |
| e_top_w | qwsw | lora | 23 | 0.113 | 0.005 | 0.101 | 0.121 |
| e_top_w | qwsw | lora_null | 21 | 0.169 | 0.035 | 0.116 | 0.218 |
| e_top_w | qwsw | lorawd | 23 | 0.116 | 0.012 | 0.103 | 0.166 |
| e_top_w | qwsw | milora | 25 | 0.075 | 0.025 | 0.037 | 0.115 |
| e_top_w | qwsw | clora | 23 | 0.055 | 0.003 | 0.051 | 0.06 |
| e_top_w | qwsw | dora | 13 | 0.115 | 0.007 | 0.101 | 0.128 |
| e_top_w | qwsw | sclora | 23 | 0.127 | 0.033 | 0.102 | 0.185 |
| e_top_w | qwswm | lora | 46 | 0.119 | 0.01 | 0.108 | 0.174 |
| e_top_w | qwswm | lora_null | 20 | 0.17 | 0.037 | 0.114 | 0.214 |
| e_top_w | qwswm | lorawd | 24 | 0.124 | 0.016 | 0.109 | 0.175 |
| e_top_w | qwswm | milora | 23 | 0.067 | 0.027 | 0.036 | 0.144 |
| e_top_w | qwswm | clora | 20 | 0.053 | 0.004 | 0.048 | 0.061 |
| e_top_w | qwswm | dora | 11 | 0.119 | 0.006 | 0.11 | 0.125 |
| e_top_w | qwswm | sclora | 20 | 0.151 | 0.009 | 0.133 | 0.16 |
| e_top_w | frc | lora | 16 | 0.068 | 0.001 | 0.067 | 0.07 |
| e_top_w | frc | lora_null | 26 | 0.102 | 0.017 | 0.085 | 0.142 |
| e_top_w | frc | lorawd | 121 | 0.073 | 0.006 | 0.064 | 0.099 |
| e_top_w | frc | lorawdr16 | 9 | 0.072 | 0.0 | 0.071 | 0.072 |
| e_top_w | frc | milora | 51 | 0.071 | 0.007 | 0.059 | 0.093 |
| e_top_w | frc | clora | 21 | 0.06 | 0.003 | 0.054 | 0.063 |
| e_top_w | frc | dora | 3 | 0.081 | 0.002 | 0.079 | 0.083 |
| e_top_w | frc | sclora | 24 | 0.075 | 0.017 | 0.035 | 0.1 |
| e_top_w | frc | pissa | 4 | 0.221 | 0.0 | 0.221 | 0.221 |
| e_top_w | frm | lora | 3 | 0.072 | 0.0 | 0.072 | 0.072 |
| e_top_w | frm | lora_null | 4 | 0.093 | 0.006 | 0.09 | 0.102 |
| e_top_w | frm | lorawd | 105 | 0.076 | 0.007 | 0.058 | 0.101 |
| e_top_w | frm | milora | 12 | 0.074 | 0.014 | 0.062 | 0.101 |
| e_top_w | frm | clora | 10 | 0.065 | 0.001 | 0.064 | 0.065 |
| e_top_w | frm | dora | 3 | 0.119 | 0.002 | 0.117 | 0.121 |
| e_top_w | frm | sclora | 6 | 0.084 | 0.002 | 0.082 | 0.087 |
| e_top_w | frm | pissa | 1 | 0.188 |  | 0.188 | 0.188 |
| e_top_w | lrsw | corda [WITHHELD] | 23 | 0.089 | 0.027 | 0.049 | 0.144 |
| e_top_w | qwsw | corda [WITHHELD] | 5 | 0.07 | 0.004 | 0.064 | 0.073 |
| e_top_w | frc | cordapp [WITHHELD] | 8 | 0.172 | 0.025 | 0.14 | 0.21 |
| e_top_w | frm | cordapp [WITHHELD] | 3 | 0.159 | 0.038 | 0.118 | 0.194 |
| e_bot_w | lrsw | lora | 25 | 0.048 | 0.002 | 0.046 | 0.051 |
| e_bot_w | lrsw | lora_null | 25 | 0.037 | 0.005 | 0.025 | 0.044 |
| e_bot_w | lrsw | lorawd | 27 | 0.049 | 0.004 | 0.045 | 0.064 |
| e_bot_w | lrsw | milora | 25 | 0.131 | 0.075 | 0.049 | 0.269 |
| e_bot_w | lrsw | milorawd | 2 | 0.082 | 0.024 | 0.065 | 0.099 |
| e_bot_w | lrsw | clora | 27 | 0.049 | 0.003 | 0.046 | 0.059 |
| e_bot_w | lrsw | dora | 25 | 0.048 | 0.003 | 0.045 | 0.054 |
| e_bot_w | lrsw | sclora | 24 | 0.04 | 0.003 | 0.037 | 0.046 |
| e_bot_w | lrswm | lora | 21 | 0.05 | 0.004 | 0.045 | 0.057 |
| e_bot_w | lrswm | lorawd | 21 | 0.051 | 0.003 | 0.046 | 0.056 |
| e_bot_w | lrswm | milora | 21 | 0.15 | 0.079 | 0.052 | 0.265 |
| e_bot_w | lrswm | clora | 21 | 0.051 | 0.004 | 0.046 | 0.057 |
| e_bot_w | lrswm | dora | 21 | 0.05 | 0.003 | 0.046 | 0.056 |
| e_bot_w | lrswm | sclora | 15 | 0.039 | 0.002 | 0.036 | 0.043 |
| e_bot_w | qwsw | lora | 23 | 0.092 | 0.002 | 0.089 | 0.094 |
| e_bot_w | qwsw | lora_null | 21 | 0.077 | 0.009 | 0.059 | 0.092 |
| e_bot_w | qwsw | lorawd | 23 | 0.095 | 0.02 | 0.088 | 0.186 |
| e_bot_w | qwsw | milora | 25 | 0.298 | 0.149 | 0.103 | 0.555 |
| e_bot_w | qwsw | clora | 23 | 0.043 | 0.001 | 0.039 | 0.045 |
| e_bot_w | qwsw | dora | 13 | 0.099 | 0.011 | 0.089 | 0.119 |
| e_bot_w | qwsw | sclora | 23 | 0.105 | 0.023 | 0.068 | 0.134 |
| e_bot_w | qwswm | lora | 46 | 0.102 | 0.012 | 0.088 | 0.175 |
| e_bot_w | qwswm | lora_null | 20 | 0.081 | 0.006 | 0.074 | 0.091 |
| e_bot_w | qwswm | lorawd | 24 | 0.109 | 0.023 | 0.09 | 0.206 |
| e_bot_w | qwswm | milora | 23 | 0.362 | 0.161 | 0.109 | 0.572 |
| e_bot_w | qwswm | clora | 20 | 0.045 | 0.001 | 0.042 | 0.046 |
| e_bot_w | qwswm | dora | 11 | 0.103 | 0.002 | 0.1 | 0.106 |
| e_bot_w | qwswm | sclora | 20 | 0.099 | 0.006 | 0.088 | 0.104 |
| e_bot_w | frc | lora | 16 | 0.046 | 0.0 | 0.046 | 0.047 |
| e_bot_w | frc | lora_null | 26 | 0.043 | 0.002 | 0.038 | 0.048 |
| e_bot_w | frc | lorawd | 121 | 0.047 | 0.002 | 0.045 | 0.052 |
| e_bot_w | frc | lorawdr16 | 9 | 0.047 | 0.001 | 0.047 | 0.048 |
| e_bot_w | frc | milora | 51 | 0.079 | 0.038 | 0.048 | 0.163 |
| e_bot_w | frc | clora | 21 | 0.047 | 0.001 | 0.046 | 0.049 |
| e_bot_w | frc | dora | 3 | 0.046 | 0.0 | 0.046 | 0.046 |
| e_bot_w | frc | sclora | 24 | 0.088 | 0.12 | 0.042 | 0.4 |
| e_bot_w | frc | pissa | 4 | 0.038 | 0.0 | 0.038 | 0.039 |
| e_bot_w | frm | lora | 3 | 0.045 | 0.0 | 0.045 | 0.045 |
| e_bot_w | frm | lora_null | 4 | 0.042 | 0.001 | 0.041 | 0.042 |
| e_bot_w | frm | lorawd | 105 | 0.047 | 0.003 | 0.044 | 0.06 |
| e_bot_w | frm | milora | 12 | 0.061 | 0.017 | 0.046 | 0.088 |
| e_bot_w | frm | clora | 10 | 0.046 | 0.0 | 0.046 | 0.046 |
| e_bot_w | frm | dora | 3 | 0.044 | 0.0 | 0.044 | 0.044 |
| e_bot_w | frm | sclora | 6 | 0.042 | 0.0 | 0.042 | 0.042 |
| e_bot_w | frm | pissa | 1 | 0.039 |  | 0.039 | 0.039 |
| e_bot_w | lrsw | corda [WITHHELD] | 23 | 0.053 | 0.01 | 0.039 | 0.069 |
| e_bot_w | qwsw | corda [WITHHELD] | 5 | 0.071 | 0.009 | 0.055 | 0.075 |
| e_bot_w | frc | cordapp [WITHHELD] | 8 | 0.033 | 0.01 | 0.023 | 0.051 |
| e_bot_w | frm | cordapp [WITHHELD] | 3 | 0.032 | 0.004 | 0.028 | 0.035 |
| amp_top_w | lrsw | lora | 25 | 0.07 | 0.005 | 0.065 | 0.082 |
| amp_top_w | lrsw | lora_null | 25 | 0.117 | 0.031 | 0.07 | 0.175 |
| amp_top_w | lrsw | lorawd | 27 | 0.074 | 0.003 | 0.069 | 0.083 |
| amp_top_w | lrsw | milora | 25 | 0.068 | 0.002 | 0.065 | 0.072 |
| amp_top_w | lrsw | milorawd | 2 | 0.073 | 0.0 | 0.073 | 0.074 |
| amp_top_w | lrsw | clora | 27 | 0.062 | 0.01 | 0.056 | 0.111 |
| amp_top_w | lrsw | dora | 25 | 0.083 | 0.02 | 0.067 | 0.149 |
| amp_top_w | lrsw | sclora | 24 | 0.155 | 0.039 | 0.095 | 0.233 |
| amp_top_w | lrswm | lora | 21 | 0.071 | 0.008 | 0.064 | 0.086 |
| amp_top_w | lrswm | lorawd | 21 | 0.073 | 0.005 | 0.068 | 0.084 |
| amp_top_w | lrswm | milora | 21 | 0.067 | 0.004 | 0.063 | 0.076 |
| amp_top_w | lrswm | clora | 21 | 0.059 | 0.004 | 0.055 | 0.067 |
| amp_top_w | lrswm | dora | 21 | 0.074 | 0.009 | 0.065 | 0.088 |
| amp_top_w | lrswm | sclora | 15 | 0.171 | 0.03 | 0.118 | 0.237 |
| amp_top_w | qwsw | lora | 23 | 0.073 | 0.003 | 0.069 | 0.079 |
| amp_top_w | qwsw | lora_null | 21 | 0.098 | 0.019 | 0.074 | 0.13 |
| amp_top_w | qwsw | lorawd | 23 | 0.075 | 0.005 | 0.072 | 0.099 |
| amp_top_w | qwsw | milora | 25 | 0.056 | 0.012 | 0.033 | 0.075 |
| amp_top_w | qwsw | clora | 23 | 0.047 | 0.003 | 0.043 | 0.052 |
| amp_top_w | qwsw | dora | 13 | 0.075 | 0.006 | 0.069 | 0.087 |
| amp_top_w | qwsw | sclora | 23 | 0.133 | 0.033 | 0.092 | 0.206 |
| amp_top_w | qwswm | lora | 46 | 0.074 | 0.006 | 0.067 | 0.101 |
| amp_top_w | qwswm | lora_null | 20 | 0.099 | 0.018 | 0.073 | 0.126 |
| amp_top_w | qwswm | lorawd | 24 | 0.078 | 0.011 | 0.07 | 0.115 |
| amp_top_w | qwswm | milora | 23 | 0.049 | 0.017 | 0.026 | 0.094 |
| amp_top_w | qwswm | clora | 20 | 0.046 | 0.004 | 0.041 | 0.053 |
| amp_top_w | qwswm | dora | 11 | 0.076 | 0.005 | 0.069 | 0.085 |
| amp_top_w | qwswm | sclora | 20 | 0.155 | 0.022 | 0.111 | 0.172 |
| amp_top_w | frc | lora | 16 | 0.066 | 0.0 | 0.065 | 0.066 |
| amp_top_w | frc | lora_null | 26 | 0.097 | 0.014 | 0.083 | 0.13 |
| amp_top_w | frc | lorawd | 121 | 0.073 | 0.005 | 0.066 | 0.1 |
| amp_top_w | frc | lorawdr16 | 9 | 0.071 | 0.0 | 0.071 | 0.072 |
| amp_top_w | frc | milora | 51 | 0.069 | 0.007 | 0.064 | 0.098 |
| amp_top_w | frc | clora | 21 | 0.058 | 0.003 | 0.053 | 0.061 |
| amp_top_w | frc | dora | 3 | 0.074 | 0.001 | 0.073 | 0.075 |
| amp_top_w | frc | sclora | 24 | 0.122 | 0.031 | 0.055 | 0.159 |
| amp_top_w | frc | pissa | 4 | 0.216 | 0.0 | 0.216 | 0.217 |
| amp_top_w | frm | lora | 3 | 0.068 | 0.0 | 0.068 | 0.068 |
| amp_top_w | frm | lora_null | 4 | 0.087 | 0.004 | 0.085 | 0.094 |
| amp_top_w | frm | lorawd | 105 | 0.075 | 0.007 | 0.065 | 0.112 |
| amp_top_w | frm | milora | 12 | 0.074 | 0.019 | 0.059 | 0.112 |
| amp_top_w | frm | clora | 10 | 0.063 | 0.001 | 0.062 | 0.063 |
| amp_top_w | frm | dora | 3 | 0.107 | 0.002 | 0.105 | 0.108 |
| amp_top_w | frm | sclora | 6 | 0.158 | 0.011 | 0.147 | 0.168 |
| amp_top_w | frm | pissa | 1 | 0.189 |  | 0.189 | 0.189 |
| amp_top_w | lrsw | corda [WITHHELD] | 23 | 0.059 | 0.016 | 0.033 | 0.104 |
| amp_top_w | qwsw | corda [WITHHELD] | 5 | 0.054 | 0.005 | 0.046 | 0.057 |
| amp_top_w | frc | cordapp [WITHHELD] | 8 | 0.175 | 0.088 | 0.107 | 0.322 |
| amp_top_w | frm | cordapp [WITHHELD] | 3 | 0.141 | 0.068 | 0.096 | 0.22 |
