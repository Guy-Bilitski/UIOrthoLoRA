# Per-benchmark fragility along F_delta (cell level, all 6 families)

norm_slope = pp/decade / base ceiling (fraction of base capability lost per decade of F_delta)

## lrsw (n_cells=56)
```
     bench      r   slope  norm_slope   knee  slope_below  slope_above  r2_hinge  r2_linear  contaminated
  mmlu_pro -0.839  -6.726      -0.357  0.299      -14.860       -2.312     0.887      0.704         False
       bbh -0.933 -11.130      -0.338  0.299      -14.097       -9.521     0.882      0.871         False
      mmlu -0.727  -8.663      -0.212  0.299      -25.943        0.714     0.905      0.529         False
     arc_c -0.840  -6.680      -0.149  0.299      -15.593       -1.843     0.930      0.706          True
truthfulqa  0.787   4.009       0.103 -0.167       -5.242        6.022     0.863      0.619         False
```

## frc (n_cells=75)
```
     bench      r   slope  norm_slope   knee  slope_below  slope_above  r2_hinge  r2_linear  contaminated
  mmlu_pro -0.917 -13.736      -0.730  0.178      -16.533       -8.725     0.868      0.841         False
       bbh -0.939 -17.202      -0.522 -0.182       -7.332      -22.668     0.946      0.881         False
      mmlu -0.832 -18.918      -0.463  0.178      -29.483        0.011     0.860      0.692         False
     arc_c -0.944 -12.293      -0.274 -0.560      -29.781      -11.354     0.911      0.891          True
truthfulqa  0.650   5.037       0.130 -0.026       -2.828       12.735     0.850      0.422         False
```

## qwsw (n_cells=58)
```
     bench      r   slope  norm_slope   knee  slope_below  slope_above  r2_hinge  r2_linear  contaminated
  mmlu_pro -0.847 -27.447      -0.673 -0.632        4.720      -47.102     0.882      0.718         False
       bbh -0.853 -30.810      -0.643 -0.609       -2.009      -50.722     0.849      0.727         False
      mmlu -0.868 -35.043      -0.488 -0.537       -6.541      -63.763     0.891      0.753         False
     arc_c -0.917 -21.202      -0.413 -0.663      -39.857      -11.630     0.930      0.841          True
truthfulqa -0.901 -11.730      -0.208 -0.148      -14.272       10.545     0.881      0.811         False
```

## lrswm (n_cells=42)
```
     bench      r   slope  norm_slope   knee  slope_below  slope_above  r2_hinge  r2_linear  contaminated
  mmlu_pro -0.888 -12.533      -0.666 -0.520        0.814      -14.318     0.825      0.789         False
     arc_c -0.953 -20.906      -0.467 -0.520       -4.462      -23.104     0.932      0.909         False
       bbh -0.814  -9.453      -0.287 -0.479       -1.672      -10.995     0.693      0.663         False
      mmlu -0.616  -7.023      -0.172 -0.450        7.016      -10.703     0.504      0.380         False
truthfulqa  0.403   2.200       0.057 -0.019        0.994        7.190     0.213      0.162         False
```

## frm (n_cells=51)
```
     bench      r   slope  norm_slope   knee  slope_below  slope_above  r2_hinge  r2_linear  contaminated
  mmlu_pro -0.883  -9.184      -0.488  0.429      -14.964       -2.497     0.911      0.780         False
       bbh -0.920 -15.261      -0.463 -0.324       -3.677      -16.642     0.863      0.847         False
     arc_c -0.821  -8.858      -0.198  0.407      -17.684        0.919     0.944      0.673         False
      mmlu -0.709  -6.080      -0.149  0.038      -16.468       -1.028     0.791      0.503         False
truthfulqa  0.759   3.535       0.091  0.429        4.796        2.075     0.607      0.576         False
```

## qwswm (n_cells=62)
```
     bench      r   slope  norm_slope   knee  slope_below  slope_above  r2_hinge  r2_linear  contaminated
  mmlu_pro -0.801 -19.284      -0.473 -0.873        4.393      -32.143     0.818      0.642         False
       bbh -0.821 -15.781      -0.329 -0.605       -4.479      -30.514     0.822      0.673         False
      mmlu -0.825 -19.076      -0.266 -0.797       -0.705      -32.132     0.832      0.680         False
     arc_c -0.764  -9.027      -0.176 -0.797        2.405      -17.153     0.810      0.584         False
truthfulqa -0.872  -4.685      -0.083 -1.199        0.092       -5.247     0.782      0.761         False
```

## Fragility ordering concordance across families
- all benchmarks: Kendall W = 0.906 (chi2=21.7, p=2.3e-04, 6 families x 5 benchmarks)
  mean rank (1=most fragile): arc_c=3.50, bbh=2.17, mmlu=3.33, mmlu_pro=1.00, truthfulqa=5.00
- excl. contaminated ARC-c: Kendall W = 1.000 (chi2=18.0, p=4.4e-04, 6 families x 4 benchmarks)
  mean rank (1=most fragile): bbh=2.00, mmlu=3.00, mmlu_pro=1.00, truthfulqa=4.00

## TruthfulQA sign check (does damage RAISE TruthfulQA?)
- frc: r=+0.650 (p=2.8e-10), slope=+5.04 pp/dec, base=38.85, mean tq top-half-F_delta = 36.3
- frm: r=+0.759 (p=1.1e-10), slope=+3.53 pp/dec, base=38.85, mean tq top-half-F_delta = 46.0
- lrsw: r=+0.787 (p=6.8e-13), slope=+4.01 pp/dec, base=38.85, mean tq top-half-F_delta = 37.0
- lrswm: r=+0.403 (p=8.2e-03), slope=+2.20 pp/dec, base=38.85, mean tq top-half-F_delta = 42.8
- qwsw: r=-0.901 (p=6.3e-22), slope=-11.73 pp/dec, base=56.28, mean tq top-half-F_delta = 41.6
- qwswm: r=-0.872 (p=2.8e-20), slope=-4.68 pp/dec, base=56.28, mean tq top-half-F_delta = 53.7
