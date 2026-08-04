# Single-metric league table — predicting retention (core)

Ranked by same-sample dR2 over family FE on the CE∩geometry pool (n=911) so
CE/KL compete on the identical sample. `t (cluster)` = cluster-robust t of the
metric in the pooled family-FE model (cluster = recipe cell, G≈340; per 09 Q1
never quote naive F/t). Per-family columns are plain single-family R2.

| rank | metric | block | dR2 (n=911) | t (cluster, n=911) | dR2 full pool (n) | cell-level dR2 | R2 lrsw | R2 lrswm | R2 qwsw | R2 qwswm | R2 frc | R2 frm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | log10 F_delta | magnitude | +0.420 | -12.0 | +0.395 (1034) | +0.370 | 0.785 | 0.747 | 0.705 | 0.689 | 0.861 | 0.863 |
| 2 | log10 spec_max | magnitude | +0.349 | -15.8 | +0.328 (1034) | +0.320 | 0.669 | 0.571 | 0.626 | 0.605 | 0.691 | 0.752 |
| 3 | log10 ||dW||_F | magnitude | +0.348 | -15.5 | +0.324 (1034) | +0.324 | 0.684 | 0.624 | 0.527 | 0.499 | 0.689 | 0.831 |
| 4 | CE drift (forgetting_ce) | CE drift | +0.340 | -12.1 | +0.340 (911) | +0.291 | 0.739 | 0.853 | 0.407 | 0.627 | 0.737 | 0.813 |
| 5 | KL drift (CE - base H) | CE drift | +0.340 | -12.1 | +0.340 (911) | +0.291 | 0.739 | 0.856 | 0.406 | 0.627 | 0.737 | 0.814 |
| 6 | log10 LR | training knob | +0.207 | -11.5 | +0.198 (1034) | +0.213 | 0.509 | 0.516 | 0.450 | 0.308 | 0.276 | 0.223 |
| 7 | stable rank | geometry | +0.116 | -7.3 | +0.105 (1034) | +0.110 | 0.177 | 0.666 | 0.298 | 0.077 | 0.399 | 0.088 |
| 8 | amp_top | geometry | +0.032 | -4.7 | +0.031 (1034) | +0.024 | 0.030 | 0.038 | 0.027 | 0.029 | 0.237 | 0.142 |
| 9 | effective rank | geometry | +0.025 | -2.9 | +0.022 (1034) | +0.020 | 0.043 | 0.325 | 0.055 | 0.000 | 0.156 | 0.006 |
| 10 | e_top (base top-subspace) | geometry | +0.015 | -2.7 | +0.017 (1034) | +0.010 | 0.000 | 0.223 | 0.019 | 0.022 | 0.180 | 0.067 |
| 11 | e_bot | geometry | +0.008 | +4.7 | +0.007 (1034) | +0.008 | 0.023 | 0.065 | 0.013 | 0.004 | 0.032 | 0.005 |

Notes:
- LR (log10) ranks below every magnitude measure — consistent with §18.5
  (LR is a proxy; F_delta is the variable).
- spec_max sits in the magnitude block (r=+0.93 with log F_delta; 06 §5, 09 Q1c).
- raw CE vs KL nearly identical here because family FE absorbs base-entropy
  offsets; cross-family comparability still requires KL (§18.6).
