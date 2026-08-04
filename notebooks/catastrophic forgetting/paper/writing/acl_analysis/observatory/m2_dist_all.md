# M2 magnitude-family distributions per method x family (log10 units; all on-pool runs; CorDA flagged WITHHELD)

> sd is over all on-pool runs of the cell (configs x LRs x seeds); seeds within a cell are correlated (ICC~0.78), so this sd mixes cell-to-cell and seed noise.

| metric | family | method | n | mean | sd | min | max |
|---|---|---|---|---|---|---|---|
| log10_fdelta | lrsw | lora | 25 | -0.308 | 0.268 | -0.648 | 0.152 |
| log10_fdelta | lrsw | lora_null | 25 | -0.431 | 0.291 | -0.86 | 0.084 |
| log10_fdelta | lrsw | lorawd | 27 | -0.331 | 0.67 | -0.668 | 1.959 |
| log10_fdelta | lrsw | milora | 25 | -0.365 | 0.313 | -0.77 | 0.185 |
| log10_fdelta | lrsw | milorawd | 2 | -0.567 | 0.054 | -0.605 | -0.528 |
| log10_fdelta | lrsw | clora | 27 | -0.315 | 0.623 | -0.877 | 2.281 |
| log10_fdelta | lrsw | dora | 25 | 0.224 | 1.174 | -0.659 | 3.274 |
| log10_fdelta | lrsw | sclora | 24 | -0.168 | 0.331 | -0.78 | 0.488 |
| log10_fdelta | lrswm | lora | 21 | -0.306 | 0.213 | -0.499 | 0.111 |
| log10_fdelta | lrswm | lorawd | 21 | -0.466 | 0.105 | -0.598 | -0.251 |
| log10_fdelta | lrswm | milora | 21 | -0.367 | 0.247 | -0.589 | 0.127 |
| log10_fdelta | lrswm | clora | 21 | -0.507 | 0.228 | -0.716 | -0.065 |
| log10_fdelta | lrswm | dora | 21 | -0.265 | 0.298 | -0.496 | 0.364 |
| log10_fdelta | lrswm | sclora | 15 | -0.301 | 0.324 | -0.798 | 0.222 |
| log10_fdelta | qwsw | lora | 23 | -0.594 | 0.309 | -0.967 | -0.065 |
| log10_fdelta | qwsw | lora_null | 21 | -0.705 | 0.319 | -1.126 | -0.147 |
| log10_fdelta | qwsw | lorawd | 23 | -0.705 | 0.434 | -1.0 | 1.156 |
| log10_fdelta | qwsw | milora | 25 | -0.552 | 0.304 | -0.992 | -0.006 |
| log10_fdelta | qwsw | clora | 23 | -0.685 | 0.312 | -1.08 | -0.158 |
| log10_fdelta | qwsw | dora | 13 | -0.416 | 0.362 | -0.965 | 0.08 |
| log10_fdelta | qwsw | sclora | 23 | -0.369 | 0.332 | -0.933 | 0.431 |
| log10_fdelta | qwswm | lora | 46 | -0.825 | 0.519 | -1.415 | 1.32 |
| log10_fdelta | qwswm | lora_null | 20 | -1.033 | 0.369 | -1.536 | -0.405 |
| log10_fdelta | qwswm | lorawd | 24 | -0.913 | 0.662 | -1.425 | 1.198 |
| log10_fdelta | qwswm | milora | 23 | -0.758 | 0.621 | -1.33 | 1.66 |
| log10_fdelta | qwswm | clora | 20 | -1.024 | 0.38 | -1.521 | -0.354 |
| log10_fdelta | qwswm | dora | 11 | -0.943 | 0.458 | -1.405 | -0.169 |
| log10_fdelta | qwswm | sclora | 20 | -0.619 | 0.398 | -1.259 | -0.06 |
| log10_fdelta | frc | lora | 16 | -0.188 | 0.068 | -0.295 | -0.118 |
| log10_fdelta | frc | lora_null | 26 | -0.069 | 0.448 | -0.801 | 0.904 |
| log10_fdelta | frc | lorawd | 121 | -0.357 | 0.288 | -0.682 | 0.848 |
| log10_fdelta | frc | lorawdr16 | 10 | -0.474 | 0.021 | -0.515 | -0.44 |
| log10_fdelta | frc | milora | 51 | -0.122 | 0.487 | -0.732 | 2.395 |
| log10_fdelta | frc | clora | 21 | -0.317 | 0.095 | -0.474 | -0.199 |
| log10_fdelta | frc | dora | 3 | -0.038 | 0.009 | -0.047 | -0.029 |
| log10_fdelta | frc | sclora | 24 | 0.018 | 0.442 | -0.71 | 0.972 |
| log10_fdelta | frc | pissa | 4 | 0.148 | 0.004 | 0.142 | 0.15 |
| log10_fdelta | frm | lora | 3 | 0.11 | 0.003 | 0.108 | 0.113 |
| log10_fdelta | frm | lora_null | 4 | -0.12 | 0.136 | -0.324 | -0.049 |
| log10_fdelta | frm | lorawd | 105 | -0.224 | 0.527 | -0.71 | 2.065 |
| log10_fdelta | frm | milora | 12 | 0.173 | 0.495 | -0.348 | 1.075 |
| log10_fdelta | frm | clora | 10 | 0.034 | 0.022 | 0.002 | 0.058 |
| log10_fdelta | frm | dora | 3 | 0.456 | 0.003 | 0.453 | 0.459 |
| log10_fdelta | frm | sclora | 6 | 0.03 | 0.107 | -0.069 | 0.128 |
| log10_fdelta | frm | pissa | 1 | 0.344 |  | 0.344 | 0.344 |
| log10_fdelta | lrsw | corda [WITHHELD] | 23 | 0.059 | 0.85 | -0.73 | 2.712 |
| log10_fdelta | qwsw | corda [WITHHELD] | 5 | -0.899 | 0.172 | -1.04 | -0.617 |
| log10_fdelta | frc | cordapp [WITHHELD] | 8 | 0.304 | 1.063 | -0.76 | 2.043 |
| log10_fdelta | frm | cordapp [WITHHELD] | 3 | 0.116 | 0.437 | -0.199 | 0.614 |
| log10_fro_total | lrsw | lora | 25 | 2.404 | 0.514 | 1.484 | 3.102 |
| log10_fro_total | lrsw | lora_null | 25 | 2.297 | 0.496 | 1.441 | 3.01 |
| log10_fro_total | lrsw | lorawd | 27 | 2.292 | 0.506 | 1.55 | 3.808 |
| log10_fro_total | lrsw | milora | 25 | 2.347 | 0.551 | 1.37 | 3.115 |
| log10_fro_total | lrsw | milorawd | 2 | 2.202 | 0.099 | 2.132 | 2.271 |
| log10_fro_total | lrsw | clora | 27 | 2.461 | 0.67 | 1.46 | 4.683 |
| log10_fro_total | lrsw | dora | 25 | 2.796 | 1.066 | 1.479 | 5.302 |
| log10_fro_total | lrsw | sclora | 24 | 2.474 | 0.474 | 1.62 | 3.199 |
| log10_fro_total | lrswm | lora | 21 | 2.315 | 0.51 | 1.52 | 3.049 |
| log10_fro_total | lrswm | lorawd | 21 | 2.16 | 0.333 | 1.579 | 2.565 |
| log10_fro_total | lrswm | milora | 21 | 2.248 | 0.531 | 1.435 | 3.044 |
| log10_fro_total | lrswm | clora | 21 | 2.225 | 0.452 | 1.505 | 2.87 |
| log10_fro_total | lrswm | dora | 21 | 2.333 | 0.537 | 1.519 | 3.155 |
| log10_fro_total | lrswm | sclora | 15 | 2.329 | 0.49 | 1.638 | 3.089 |
| log10_fro_total | qwsw | lora | 23 | 2.255 | 0.521 | 1.37 | 2.956 |
| log10_fro_total | qwsw | lora_null | 21 | 2.145 | 0.51 | 1.342 | 2.88 |
| log10_fro_total | qwsw | lorawd | 23 | 2.082 | 0.362 | 1.444 | 2.942 |
| log10_fro_total | qwsw | milora | 25 | 2.245 | 0.515 | 1.324 | 3.0 |
| log10_fro_total | qwsw | clora | 23 | 2.261 | 0.485 | 1.407 | 2.9 |
| log10_fro_total | qwsw | dora | 13 | 2.497 | 0.513 | 1.376 | 3.042 |
| log10_fro_total | qwsw | sclora | 23 | 2.377 | 0.468 | 1.59 | 3.178 |
| log10_fro_total | qwswm | lora | 46 | 2.048 | 0.6 | 1.05 | 3.243 |
| log10_fro_total | qwswm | lora_null | 20 | 1.921 | 0.509 | 1.121 | 2.674 |
| log10_fro_total | qwswm | lorawd | 24 | 1.888 | 0.519 | 1.118 | 3.27 |
| log10_fro_total | qwswm | milora | 23 | 1.957 | 0.631 | 1.072 | 3.558 |
| log10_fro_total | qwswm | clora | 20 | 1.958 | 0.507 | 1.122 | 2.748 |
| log10_fro_total | qwswm | dora | 11 | 1.887 | 0.662 | 1.051 | 2.894 |
| log10_fro_total | qwswm | sclora | 20 | 2.149 | 0.539 | 1.325 | 2.928 |
| log10_fro_total | frc | lora | 16 | 2.704 | 0.086 | 2.574 | 2.773 |
| log10_fro_total | frc | lora_null | 26 | 2.777 | 0.467 | 1.69 | 3.519 |
| log10_fro_total | frc | lorawd | 121 | 2.415 | 0.387 | 1.551 | 3.482 |
| log10_fro_total | frc | lorawdr16 | 9 | 2.31 | 0.033 | 2.274 | 2.338 |
| log10_fro_total | frc | milora | 51 | 2.686 | 0.485 | 1.527 | 4.165 |
| log10_fro_total | frc | clora | 21 | 2.598 | 0.086 | 2.467 | 2.695 |
| log10_fro_total | frc | dora | 3 | 2.793 | 0.004 | 2.789 | 2.797 |
| log10_fro_total | frc | sclora | 24 | 2.806 | 0.453 | 1.973 | 3.573 |
| log10_fro_total | frc | pissa | 4 | 2.86 | 0.0 | 2.86 | 2.86 |
| log10_fro_total | frm | lora | 3 | 3.05 | 0.001 | 3.049 | 3.051 |
| log10_fro_total | frm | lora_null | 4 | 2.826 | 0.147 | 2.605 | 2.9 |
| log10_fro_total | frm | lorawd | 105 | 2.58 | 0.406 | 2.032 | 3.862 |
| log10_fro_total | frm | milora | 12 | 3.078 | 0.464 | 2.54 | 3.865 |
| log10_fro_total | frm | clora | 10 | 2.972 | 0.023 | 2.941 | 2.994 |
| log10_fro_total | frm | dora | 3 | 3.216 | 0.01 | 3.206 | 3.226 |
| log10_fro_total | frm | sclora | 6 | 2.867 | 0.146 | 2.734 | 3.0 |
| log10_fro_total | frm | pissa | 1 | 3.151 |  | 3.151 | 3.151 |
| log10_fro_total | lrsw | corda [WITHHELD] | 23 | 3.193 | 0.603 | 2.344 | 4.774 |
| log10_fro_total | qwsw | corda [WITHHELD] | 5 | 2.304 | 0.284 | 2.056 | 2.75 |
| log10_fro_total | frc | cordapp [WITHHELD] | 8 | 4.96 | 1.335 | 3.341 | 6.713 |
| log10_fro_total | frm | cordapp [WITHHELD] | 3 | 5.309 | 0.571 | 4.866 | 5.952 |
| log10_spec_max | lrsw | lora | 25 | 1.519 | 0.506 | 0.687 | 2.422 |
| log10_spec_max | lrsw | lora_null | 25 | 1.252 | 0.55 | 0.395 | 2.236 |
| log10_spec_max | lrsw | lorawd | 27 | 1.457 | 0.643 | 0.742 | 3.63 |
| log10_spec_max | lrsw | milora | 25 | 1.444 | 0.616 | 0.46 | 2.509 |
| log10_spec_max | lrsw | milorawd | 2 | 1.301 | 0.131 | 1.208 | 1.393 |
| log10_spec_max | lrsw | clora | 27 | 1.356 | 0.722 | 0.602 | 4.455 |
| log10_spec_max | lrsw | dora | 25 | 2.081 | 1.249 | 0.667 | 4.997 |
| log10_spec_max | lrsw | sclora | 24 | 1.387 | 0.545 | 0.488 | 2.403 |
| log10_spec_max | lrswm | lora | 21 | 1.427 | 0.419 | 0.843 | 2.252 |
| log10_spec_max | lrswm | lorawd | 21 | 1.241 | 0.233 | 0.883 | 1.654 |
| log10_spec_max | lrswm | milora | 21 | 1.301 | 0.547 | 0.554 | 2.235 |
| log10_spec_max | lrswm | clora | 21 | 1.105 | 0.249 | 0.693 | 1.461 |
| log10_spec_max | lrswm | dora | 21 | 1.53 | 0.563 | 0.846 | 2.641 |
| log10_spec_max | lrswm | sclora | 15 | 1.231 | 0.527 | 0.569 | 2.264 |
| log10_spec_max | qwsw | lora | 23 | 1.333 | 0.418 | 0.637 | 2.094 |
| log10_spec_max | qwsw | lora_null | 21 | 1.096 | 0.426 | 0.442 | 1.93 |
| log10_spec_max | qwsw | lorawd | 23 | 1.107 | 0.38 | 0.709 | 2.712 |
| log10_spec_max | qwsw | milora | 25 | 1.179 | 0.44 | 0.443 | 2.048 |
| log10_spec_max | qwsw | clora | 23 | 1.246 | 0.373 | 0.619 | 1.928 |
| log10_spec_max | qwsw | dora | 13 | 1.546 | 0.455 | 0.67 | 2.166 |
| log10_spec_max | qwsw | sclora | 23 | 1.221 | 0.488 | 0.509 | 2.147 |
| log10_spec_max | qwswm | lora | 46 | 1.073 | 0.533 | 0.346 | 3.007 |
| log10_spec_max | qwswm | lora_null | 20 | 0.885 | 0.397 | 0.261 | 1.607 |
| log10_spec_max | qwswm | lorawd | 24 | 0.974 | 0.673 | 0.356 | 3.154 |
| log10_spec_max | qwswm | milora | 23 | 0.951 | 0.688 | 0.138 | 3.325 |
| log10_spec_max | qwswm | clora | 20 | 0.957 | 0.362 | 0.33 | 1.602 |
| log10_spec_max | qwswm | dora | 11 | 0.989 | 0.58 | 0.33 | 2.047 |
| log10_spec_max | qwswm | sclora | 20 | 0.962 | 0.522 | 0.258 | 1.895 |
| log10_spec_max | frc | lora | 16 | 1.72 | 0.076 | 1.591 | 1.837 |
| log10_spec_max | frc | lora_null | 26 | 1.784 | 0.584 | 0.605 | 2.634 |
| log10_spec_max | frc | lorawd | 121 | 1.528 | 0.423 | 0.697 | 2.828 |
| log10_spec_max | frc | lorawdr16 | 9 | 1.299 | 0.06 | 1.204 | 1.412 |
| log10_spec_max | frc | milora | 51 | 1.826 | 0.609 | 0.612 | 3.961 |
| log10_spec_max | frc | clora | 21 | 1.355 | 0.09 | 1.211 | 1.485 |
| log10_spec_max | frc | dora | 3 | 2.087 | 0.067 | 2.032 | 2.161 |
| log10_spec_max | frc | sclora | 24 | 1.802 | 0.614 | 0.804 | 2.891 |
| log10_spec_max | frc | pissa | 4 | 1.731 | 0.015 | 1.71 | 1.744 |
| log10_spec_max | frm | lora | 3 | 2.188 | 0.022 | 2.164 | 2.206 |
| log10_spec_max | frm | lora_null | 4 | 1.795 | 0.29 | 1.361 | 1.97 |
| log10_spec_max | frm | lorawd | 105 | 1.743 | 0.513 | 1.052 | 3.654 |
| log10_spec_max | frm | milora | 12 | 2.204 | 0.561 | 1.512 | 3.166 |
| log10_spec_max | frm | clora | 10 | 1.456 | 0.024 | 1.416 | 1.489 |
| log10_spec_max | frm | dora | 3 | 2.846 | 0.018 | 2.828 | 2.865 |
| log10_spec_max | frm | sclora | 6 | 1.885 | 0.34 | 1.555 | 2.211 |
| log10_spec_max | frm | pissa | 1 | 2.072 |  | 2.072 | 2.072 |
| log10_spec_max | lrsw | corda [WITHHELD] | 23 | 2.627 | 0.796 | 1.556 | 4.738 |
| log10_spec_max | qwsw | corda [WITHHELD] | 5 | 1.474 | 0.141 | 1.362 | 1.709 |
| log10_spec_max | frc | cordapp [WITHHELD] | 8 | 4.714 | 1.391 | 3.184 | 6.577 |
| log10_spec_max | frm | cordapp [WITHHELD] | 3 | 4.849 | 0.69 | 4.442 | 5.646 |
| log10_dw_sv_max | lrsw | lora | 25 | 1.519 | 0.507 | 0.686 | 2.422 |
| log10_dw_sv_max | lrsw | lora_null | 25 | 1.252 | 0.55 | 0.394 | 2.236 |
| log10_dw_sv_max | lrsw | lorawd | 27 | 1.456 | 0.644 | 0.741 | 3.63 |
| log10_dw_sv_max | lrsw | milora | 25 | 1.444 | 0.616 | 0.46 | 2.509 |
| log10_dw_sv_max | lrsw | milorawd | 2 | 1.301 | 0.131 | 1.208 | 1.393 |
| log10_dw_sv_max | lrsw | clora | 27 | 1.356 | 0.722 | 0.601 | 4.455 |
| log10_dw_sv_max | lrsw | dora | 25 | 2.081 | 1.249 | 0.666 | 4.997 |
| log10_dw_sv_max | lrsw | sclora | 24 | 1.387 | 0.545 | 0.488 | 2.403 |
| log10_dw_sv_max | lrswm | lora | 21 | 1.427 | 0.42 | 0.842 | 2.252 |
| log10_dw_sv_max | lrswm | lorawd | 21 | 1.241 | 0.233 | 0.882 | 1.654 |
| log10_dw_sv_max | lrswm | milora | 21 | 1.301 | 0.547 | 0.553 | 2.235 |
| log10_dw_sv_max | lrswm | clora | 21 | 1.105 | 0.249 | 0.692 | 1.46 |
| log10_dw_sv_max | lrswm | dora | 21 | 1.529 | 0.563 | 0.844 | 2.641 |
| log10_dw_sv_max | lrswm | sclora | 15 | 1.231 | 0.527 | 0.569 | 2.264 |
| log10_dw_sv_max | qwsw | lora | 23 | 1.333 | 0.418 | 0.636 | 2.094 |
| log10_dw_sv_max | qwsw | lora_null | 21 | 1.096 | 0.426 | 0.442 | 1.93 |
| log10_dw_sv_max | qwsw | lorawd | 23 | 1.107 | 0.379 | 0.708 | 2.71 |
| log10_dw_sv_max | qwsw | milora | 25 | 1.178 | 0.44 | 0.443 | 2.048 |
| log10_dw_sv_max | qwsw | clora | 23 | 1.245 | 0.373 | 0.619 | 1.928 |
| log10_dw_sv_max | qwsw | dora | 13 | 1.545 | 0.455 | 0.67 | 2.166 |
| log10_dw_sv_max | qwsw | sclora | 23 | 1.221 | 0.488 | 0.509 | 2.147 |
| log10_dw_sv_max | qwswm | lora | 46 | 1.071 | 0.531 | 0.345 | 3.006 |
| log10_dw_sv_max | qwswm | lora_null | 20 | 0.885 | 0.397 | 0.261 | 1.607 |
| log10_dw_sv_max | qwswm | lorawd | 24 | 0.973 | 0.673 | 0.355 | 3.154 |
| log10_dw_sv_max | qwswm | milora | 23 | 0.951 | 0.688 | 0.138 | 3.325 |
| log10_dw_sv_max | qwswm | clora | 20 | 0.954 | 0.361 | 0.33 | 1.602 |
| log10_dw_sv_max | qwswm | dora | 11 | 0.988 | 0.581 | 0.329 | 2.047 |
| log10_dw_sv_max | qwswm | sclora | 20 | 0.962 | 0.522 | 0.257 | 1.895 |
| log10_dw_sv_max | frc | lora | 16 | 1.72 | 0.076 | 1.59 | 1.837 |
| log10_dw_sv_max | frc | lora_null | 26 | 1.784 | 0.584 | 0.605 | 2.634 |
| log10_dw_sv_max | frc | lorawd | 121 | 1.528 | 0.422 | 0.697 | 2.828 |
| log10_dw_sv_max | frc | lorawdr16 | 10 | 1.295 | 0.058 | 1.204 | 1.412 |
| log10_dw_sv_max | frc | milora | 51 | 1.826 | 0.609 | 0.612 | 3.961 |
| log10_dw_sv_max | frc | clora | 21 | 1.354 | 0.09 | 1.21 | 1.484 |
| log10_dw_sv_max | frc | dora | 3 | 2.087 | 0.067 | 2.032 | 2.161 |
| log10_dw_sv_max | frc | sclora | 24 | 1.802 | 0.614 | 0.804 | 2.891 |
| log10_dw_sv_max | frc | pissa | 4 | 1.731 | 0.015 | 1.709 | 1.743 |
| log10_dw_sv_max | frm | lora | 3 | 2.188 | 0.022 | 2.164 | 2.206 |
| log10_dw_sv_max | frm | lora_null | 4 | 1.795 | 0.29 | 1.361 | 1.97 |
| log10_dw_sv_max | frm | lorawd | 105 | 1.743 | 0.513 | 1.052 | 3.653 |
| log10_dw_sv_max | frm | milora | 12 | 2.204 | 0.561 | 1.512 | 3.166 |
| log10_dw_sv_max | frm | clora | 10 | 1.456 | 0.024 | 1.416 | 1.489 |
| log10_dw_sv_max | frm | dora | 3 | 2.846 | 0.018 | 2.828 | 2.865 |
| log10_dw_sv_max | frm | sclora | 6 | 1.885 | 0.34 | 1.555 | 2.211 |
| log10_dw_sv_max | frm | pissa | 1 | 2.072 |  | 2.072 | 2.072 |
| log10_dw_sv_max | lrsw | corda [WITHHELD] | 23 | 2.627 | 0.796 | 1.555 | 4.738 |
| log10_dw_sv_max | qwsw | corda [WITHHELD] | 5 | 1.474 | 0.141 | 1.362 | 1.709 |
| log10_dw_sv_max | frc | cordapp [WITHHELD] | 8 | 3.209 | 1.391 | 1.678 | 5.072 |
| log10_dw_sv_max | frm | cordapp [WITHHELD] | 3 | 3.043 | 0.69 | 2.636 | 3.839 |
