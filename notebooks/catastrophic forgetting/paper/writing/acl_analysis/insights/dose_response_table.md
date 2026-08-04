# Dose-response: knob -> F_delta -> retention (frc grid, Llama-2 CS)

## Knob 1: weight decay (lorawd cells, n_cell=33, n_run=121)
- Stage 1 (cell level): partial r(log10 F_delta, wd | log LR) = -0.762 (t=-6.4, p=4.0e-07)
- Raw r(wd, retention) = 0.454; partial r(wd, ret | log F_delta) = -0.246 (t=-1.4, p=0.18); | logF + logLR = -0.744

wd -> mean F_delta (per LR column) [cells]:
```
lr   0.00002  0.00010  0.00020  0.00030  0.00050  0.00070  0.00100
wd                                                                
0.0    0.222    0.299    0.512    0.744    1.153    1.504    5.811
0.1      NaN    0.273    0.420    0.553    0.743      NaN      NaN
0.1      NaN      NaN      NaN      NaN      NaN    0.860    0.991
0.2      NaN    0.252    0.355    0.430    0.527      NaN      NaN
0.2    0.217      NaN      NaN      NaN      NaN    0.590    0.637
0.3    0.214    0.235    0.308    0.350    0.402    0.432    0.451
0.5      NaN    0.209    0.243    0.257    0.275    0.289    0.448
```
wd -> retention (per LR column) [cells]:
```
lr   0.00002  0.00010  0.00020  0.00030  0.00050  0.00070  0.00100
wd                                                                
0.0    26.60    26.50    24.34    22.27    17.67    16.19     4.18
0.1      NaN    27.04    25.53    24.46    22.10      NaN      NaN
0.1      NaN      NaN      NaN      NaN      NaN    21.42    19.39
0.2      NaN    27.36    26.36    25.91    24.52      NaN      NaN
0.2    26.80      NaN      NaN      NaN      NaN    23.21    23.74
0.3    26.85    27.30    26.77    26.35    25.94    25.73    25.50
0.5      NaN    27.88    27.33    25.56    26.32    26.83    21.32
```

## Knob 2: CLoRA null-space dimension k (lr 3e-4 fixed, n_cell=5, n_run=21)
- Stage 1: Spearman rho(log2 k, log F_delta) = -1.000 (p=0.000, n=5)
- Stage 2: Spearman rho(log2 k, retention) = 1.000 (p=0.000)
```
 clora_k    fd    ret  adapt
   128.0 0.615 22.682 76.835
   256.0 0.586 23.307 78.913
   512.0 0.520 23.310 79.386
  1024.0 0.448 24.130 74.875
  2048.0 0.347 25.258 69.384
```
- Run-level partial r(log2 k, ret | log F_delta) = -0.220 (t=-1.0, p=0.35, n=21) [descriptive; seeds not independent]

## Knob 3: LoRA rank (lr 3e-4 fixed, n_cell=3, n_run=12)
```
 rank    fd    ret  adapt
  8.0 0.516 24.695 74.372
 16.0 0.603 23.560 79.367
 32.0 0.747 22.130 77.046
```

## Pooled mediation check (all frc cells, cell level)
- frc family curve (cells, n=75): ret = 18.46 + -15.47*log10 F_delta, r=-0.951
- wd cells: mean on-curve residual = +0.26 pp (SD 1.47, n=33)
- clora-k cells: mean on-curve residual = +0.53 pp (SD 0.61, n=5)
- rank cells: mean on-curve residual = +1.73 pp (SD 0.05, n=3)
