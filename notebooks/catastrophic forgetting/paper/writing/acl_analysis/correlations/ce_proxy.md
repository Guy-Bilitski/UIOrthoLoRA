# CE drift as a cheap retention proxy — calibration + honest error

KL drift costs ONE forward pass on ~40 WikiText blocks (wall_s ~= 3.4s in the
store) vs a full BBH+MMLU-Pro retention eval. Question: if you calibrated the
KL->retention mapping once per family, how big is the prediction error?

Caveats up front: (i) CE is DOWNSTREAM — a monitor, not a knob; (ii) it is
quasi-tautologically close to retention (both measure drift of base behavior);
(iii) it misses channel B (format damage, 05 §2) which benchmark evals see;
(iv) Qwen coverage 60-62% (seed-block missingness, ignorable per 09 Q4);
(v) two CE protocols mixed, benign (09 Q2; sensitivity below).

## Per-family calibration error (leave-cells-out CV, pp of retention)

| family | n | Spearman(KL,ret) | RMSE KL | MAE KL | KL form | RMSE logF (form) | RMSE KL+logF | seed SD(ret) | AUC(dmg) KL | AUC(dmg) logF |
|---|---|---|---|---|---|---|---|---|---|---|
| lrsw | 180 | -0.868 | 1.88 | 1.16 | KL knee | 2.97 (logF knee) | 2.79 | 0.91 | 0.996 | 0.992 |
| lrswm | 120 | -0.911 | 1.31 | 0.83 | KL linear | 1.64 (logF knee) | 1.41 | 0.43 | 0.989 | 0.980 |
| qwsw | 93 | -0.787 | 6.00 | 3.70 | KL knee | 7.04 (logF) | 8.41 | 2.04 | 0.995 | 1.000 |
| qwswm | 99 | -0.555 | 5.05 | 2.53 | KL knee | 6.27 (logF) | 6.58 | 1.71 | 0.996 | 1.000 |
| frc | 275 | -0.946 | 1.59 | 1.06 | KL knee | 2.40 (logF) | 2.21 | 0.78 | 0.976 | 0.982 |
| frm | 144 | -0.966 | 1.95 | 1.19 | KL knee | 2.62 (logF) | 2.38 | 0.94 | 0.996 | 0.996 |

seed SD(ret) = mean within-cell seed SD — the noise floor of a single-seed
retention measurement itself. AUC = detecting runs >5pp below the family
healthy ceiling (90th pct) from KL alone.

## KL knee locations (hinge fit, retention ~ log KL)

| family | knee (nats KL) | slope below (pp/decade) | slope above |
|---|---|---|---|
| lrsw | 0.290 | -0.6 | -17.5 |
| lrswm | 0.259 | -0.6 | -10.2 |
| qwsw | 0.286 | +1.0 | -40.0 |
| qwswm | 0.270 | -0.3 | -34.6 |
| frc | 0.399 | -4.0 | -18.8 |
| frm | 1.686 | -8.3 | -23.6 |

Four of six families (both base models, both task types) put the knee at
~0.26-0.29 nats; frc 0.40; frm has no flat region (already-steep below-knee
slope -8.3, knee 1.69 is a slope change, not damage onset).

## Protocol sensitivity (families with both CE protocols)

  lrsw: n_blocks=40.0 n=76 r(KL,ret)=-0.876
  lrsw: n_blocks=330.0 n=104 r(KL,ret)=-0.856
  lrswm: n_blocks=40.0 n=41 r(KL,ret)=-0.953
  lrswm: n_blocks=330.0 n=79 r(KL,ret)=-0.917
  qwsw: n_blocks=291.0 n=86 r(KL,ret)=-0.639
  qwswm: n_blocks=291.0 n=95 r(KL,ret)=-0.791
  frc: n_blocks=40.0 n=90 r(KL,ret)=-0.871
  frc: n_blocks=330.0 n=185 r(KL,ret)=-0.870
  frm: n_blocks=40.0 n=65 r(KL,ret)=-0.918
  frm: n_blocks=330.0 n=79 r(KL,ret)=-0.889

## Within-base-model task transfer (calibrate on sibling task)

| calib on | applied to | predictor | RMSE raw | RMSE intercept-corrected |
|---|---|---|---|---|
| lrsw | lrswm | KL | 1.53 | 1.50 |
| lrsw | lrswm | logF | 1.55 | 1.53 |
| lrswm | lrsw | KL | 4.68 | 4.59 |
| lrswm | lrsw | logF | 3.15 | 3.13 |
| frc | frm | KL | 4.10 | 4.04 |
| frc | frm | logF | 2.81 | 2.79 |
| frm | frc | KL | 3.50 | 3.50 |
| frm | frc | logF | 2.54 | 2.46 |
| qwsw | qwswm | KL | 9.50 | 6.71 |
| qwsw | qwswm | logF | 8.07 | 8.01 |
| qwswm | qwsw | KL | 11.72 | 9.52 |
| qwswm | qwsw | logF | 8.06 | 7.75 |
