# Free-lunch region + exchange rate

## lrsw: canonical knee (sec 18.2) F_delta=0.955 (log -0.02; own hinge refit +0.30)
- peak adapt overall 81.8 (ret 25.9); peak adapt below knee 81.8 (ret 25.9) -> 100.0% of peak is 'free' (n_below=43 healthy cells)
  per-method (peak, peak-below-knee, n_below/n): clora: 78.3/78.3 (6/8); dora: 76.2/74.3 (5/9); lora: 79.2/79.2 (6/7); lora_null: 78.9/78.9 (6/7); lorawd: 81.8/81.8 (7/7); milora: 77.2/77.2 (6/7); sclora: 80.6/80.6 (5/7)

## frc: canonical knee (sec 18.2) F_delta=0.355 (log -0.45; own hinge refit -0.45)
- peak adapt overall 81.9 (ret 25.9); peak adapt below knee 81.4 (ret 26.8) -> 99.5% of peak is 'free' (n_below=21 healthy cells)
  per-method (peak, peak-below-knee, n_below/n): clora: 79.4/69.4 (1/5); lora: 79.4/nan (0/4); lora_null: 79.8/68.1 (2/7); lorawd: 81.9/81.4 (12/31); milora: 77.2/71.8 (3/13); sclora: 80.0/65.4 (1/7)

## qwsw: canonical knee (sec 18.2) F_delta=0.204 (log -0.69; own hinge refit -0.61)
- peak adapt overall 87.8 (ret 39.9); peak adapt below knee 87.8 (ret 39.9) -> 100.0% of peak is 'free' (n_below=26 healthy cells)
  per-method (peak, peak-below-knee, n_below/n): clora: 87.0/87.0 (3/7); dora: 86.8/86.8 (3/7); lora: 87.5/87.5 (4/9); lora_null: 86.2/86.2 (5/8); lorawd: 87.8/87.8 (6/9); milora: 87.3/86.8 (4/9); sclora: 87.5/87.0 (1/9)

## lrswm: canonical knee (sec 18.2) F_delta=0.331 (log -0.48; own hinge refit -0.48)
- peak adapt overall 59.1 (ret 22.7); peak adapt below knee 58.5 (ret 24.6) -> 99.0% of peak is 'free' (n_below=17 healthy cells)
  per-method (peak, peak-below-knee, n_below/n): clora: 48.6/48.6 (5/7); dora: 46.4/42.6 (2/7); lora: 47.8/42.6 (1/7); lorawd: 50.7/49.6 (4/7); milora: 47.6/41.7 (3/7); sclora: 59.1/58.5 (2/7)

## frm: canonical knee (sec 18.2) F_delta=0.316 (log -0.50; own hinge refit +0.43)
- peak adapt overall 68.5 (ret 26.0); peak adapt below knee 68.5 (ret 26.0) -> 100.0% of peak is 'free' (n_below=8 healthy cells)
  per-method (peak, peak-below-knee, n_below/n): clora: 60.7/nan (0/3); lorawd: 68.5/68.5 (8/34); milora: 63.7/nan (0/5)

## qwswm: canonical knee (sec 18.2) F_delta=0.123 (log -0.91; own hinge refit -0.84)
- peak adapt overall 77.2 (ret 43.1); peak adapt below knee 77.2 (ret 43.1) -> 100.0% of peak is 'free' (n_below=36 healthy cells)
  per-method (peak, peak-below-knee, n_below/n): clora: 70.5/69.9 (5/7); dora: 71.0/67.1 (4/7); lora: 67.2/67.2 (9/17); lora_null: 72.3/71.5 (5/7); lorawd: 69.0/69.0 (7/7); milora: 65.4/60.3 (4/9); sclora: 77.2/77.2 (2/6)

## Exchange rate (marginal retention paid per adaptation point gained, quintile bins of F_delta)
### lrsw
```
  fd    n  adapt   ret  d_adapt  d_ret  ret_paid_per_adapt_point
0.20 11.0  50.78 26.72      NaN    NaN                       NaN
0.25 11.0  59.40 27.00     8.62   0.29                     -0.03
0.40 10.0  66.82 25.39     7.42  -1.61                      0.22
0.68 11.0  76.46 21.81     9.65  -3.59                      0.37
1.52 11.0  61.26 11.72   -15.21 -10.08                     -0.66
```
### frc
```
  fd    n  adapt   ret  d_adapt  d_ret  ret_paid_per_adapt_point
0.25 15.0  63.28 26.88      NaN    NaN                       NaN
0.38 14.0  70.98 25.58     7.70  -1.29                      0.17
0.54 14.0  75.75 23.74     4.77  -1.84                      0.38
0.76 14.0  76.19 20.94     0.43  -2.80                      6.46
1.51 14.0  60.61 12.34   -15.57  -8.61                     -0.55
```
### qwsw
```
  fd    n  adapt   ret  d_adapt  d_ret  ret_paid_per_adapt_point
0.11 12.0  85.58 38.86      NaN    NaN                       NaN
0.14 11.0  86.38 39.10     0.80   0.24                     -0.30
0.22 12.0  83.99 39.01    -2.39  -0.10                     -0.04
0.37 11.0  84.07 32.65     0.08  -6.36                     80.20
0.73 12.0  74.69 12.34    -9.39 -20.31                     -2.16
```
### lrswm
```
  fd   n  adapt   ret  d_adapt  d_ret  ret_paid_per_adapt_point
0.25 9.0  42.89 25.32      NaN    NaN                       NaN
0.32 8.0  41.55 25.21    -1.35  -0.11                     -0.09
0.35 8.0  45.01 24.74     3.47  -0.47                      0.13
0.51 8.0  49.08 22.70     4.06  -2.05                      0.50
0.96 9.0  44.71 18.29    -4.37  -4.40                     -1.01
```
### frm
```
  fd    n  adapt   ret  d_adapt  d_ret  ret_paid_per_adapt_point
0.27 10.0  66.21 25.48      NaN    NaN                       NaN
0.42 10.0  65.22 24.17    -0.98  -1.31                     -1.33
0.56  9.0  64.02 21.74    -1.21  -2.43                     -2.02
0.97 10.0  60.46 19.04    -3.56  -2.70                     -0.76
2.43 10.0  49.95 10.67   -10.51  -8.37                     -0.80
```
### qwswm
```
  fd    n  adapt   ret  d_adapt  d_ret  ret_paid_per_adapt_point
0.04 12.0  56.34 43.45      NaN    NaN                       NaN
0.07 12.0  63.29 43.89     6.95   0.45                     -0.06
0.11 12.0  66.64 44.11     3.35   0.22                     -0.07
0.18 12.0  65.09 42.08    -1.54  -2.04                     -1.32
0.48 12.0  57.09 26.89    -8.00 -15.18                     -1.90
```

## Free-lunch summary
```
  fam  knee_logfd  knee_fd  knee_own_fit  n_below  n_above  peak_adapt  peak_adapt_below_knee  frac_free  ret_at_global_peak  ret_at_free_peak  base_ret  ret_cost_global  ret_cost_free
 lrsw       -0.02    0.955         0.299       43       11      81.750                 81.750      1.000              25.858            25.858     25.89            0.032          0.032
  frc       -0.45    0.355        -0.451       21       50      81.858                 81.413      0.995              25.943            26.833     25.89           -0.053         -0.943
 qwsw       -0.69    0.204        -0.614       26       32      87.770                 87.770      1.000              39.920            39.920     44.35            4.430          4.430
lrswm       -0.48    0.331        -0.479       17       25      59.095                 58.530      0.990              22.670            24.560     25.89            3.220          1.330
  frm       -0.50    0.316         0.429        8       41      68.483                 68.483      1.000              26.000            26.000     25.89           -0.110         -0.110
qwswm       -0.91    0.123        -0.841       36       24      77.230                 77.230      1.000              43.140            43.140     44.35            1.210          1.210
```
