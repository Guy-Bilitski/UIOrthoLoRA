# Grouped cross-validation — does anything beat magnitude alone?

Pool n=911 (frozen∩geometry∩CE). OOS R2 = 1 - SSE/SST on pooled held-out
predictions. Leave-cells-out: 10 folds of recipe cells (seeds never split across
train/test). LOFO: family FE removed; 'intercept-oracle' removes the held-out
family's mean offset (scores slope transfer only, vs within-family SST).

## leave-cells-out

| model | OOS R2 | RMSE (pp) |
|---|---|---|
| FE only | +0.351 | 7.58 |
| magnitude | +0.783 | 4.38 |
| magnitude+geometry | +0.807 | 4.13 |
| magnitude+CE | +0.785 | 4.36 |
| magnitude+geometry+CE | +0.815 | 4.04 |
| CE only | +0.700 | 5.15 |

## LOFO raw

| model | OOS R2 | RMSE (pp) |
|---|---|---|
| magnitude | +0.628 | 5.73 |
| magnitude+geometry | +0.659 | 5.49 |
| magnitude+CE | +0.612 | 5.86 |
| magnitude+geometry+CE | +0.639 | 5.65 |
| CE only | +0.279 | 7.99 |

## LOFO intercept-oracle (within-family)

| model | OOS R2 | RMSE (pp) |
|---|---|---|
| magnitude | +0.578 | 4.83 |
| magnitude+geometry | +0.580 | 4.82 |
| magnitude+CE | +0.558 | 4.95 |
| magnitude+geometry+CE | +0.567 | 4.89 |
| CE only | +0.483 | 5.35 |

## LOFO per held-out family (magnitude vs full model, intercept-oracle R2 within family)

| held-out family | magnitude | mag+geo+CE | CE only |
|---|---|---|---|
| lrsw | +0.054 | +0.145 | +0.716 |
| lrswm | +0.526 | +0.372 | +0.798 |
| qwsw | +0.541 | +0.469 | +0.290 |
| qwswm | +0.643 | +0.627 | +0.473 |
| frc | +0.859 | +0.845 | +0.732 |
| frm | +0.771 | +0.812 | +0.189 |
