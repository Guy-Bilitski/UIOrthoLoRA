# Per-layer / per-projection update composition (n=1002 runs joined of 1035 pool)

## A. Method fingerprints — mean composition per method (all 6 families pooled)
```
           share_q  share_k  share_v  share_up  share_down  depth_centroid  eff_n_mat    n
method                                                                                    
clora        0.073    0.050    0.057     0.505       0.316           0.552     76.656  120
dora         0.172    0.143    0.066     0.394       0.224           0.509     75.165   76
lora         0.142    0.088    0.057     0.447       0.267           0.539     88.517  133
lora_null    0.150    0.109    0.058     0.429       0.254           0.549     95.827   88
lorawd       0.138    0.139    0.089     0.383       0.251           0.561     99.534  321
lorawdr16    0.116    0.134    0.098     0.368       0.285           0.549    115.328    9
milora       0.149    0.127    0.079     0.366       0.279           0.546     99.256  143
milorawd     0.133    0.158    0.100     0.358       0.251           0.544    118.876    2
pissa        0.185    0.198    0.097     0.305       0.215           0.539    136.409    5
sclora       0.165    0.123    0.055     0.447       0.210           0.523     95.710  105
```

Method-determination of composition (eta^2 of method, one-way, runs):
- share_q: eta^2 = 0.187
- share_k: eta^2 = 0.129
- share_v: eta^2 = 0.110
- share_up: eta^2 = 0.139
- share_down: eta^2 = 0.128
- depth_centroid: eta^2 = 0.083
- eff_n_mat: eta^2 = 0.094

## B. Does composition predict retention beyond log F_delta + family?
- share_q: partial r | (logF, fam) = -0.248 (t=-8.1, p=2.1e-15); | (+method) = -0.285 (t=-9.3, p=7.4e-20)
- share_k: partial r | (logF, fam) = +0.162 (t=5.2, p=2.6e-07); | (+method) = +0.203 (t=6.5, p=1.2e-10)
- share_v: partial r | (logF, fam) = +0.071 (t=2.3, p=2.4e-02); | (+method) = +0.032 (t=1.0, p=3.2e-01)
- share_up: partial r | (logF, fam) = -0.092 (t=-2.9, p=3.5e-03); | (+method) = -0.059 (t=-1.9, p=6.3e-02)
- share_down: partial r | (logF, fam) = +0.159 (t=5.1, p=4.8e-07); | (+method) = +0.131 (t=4.2, p=3.5e-05)
- depth_centroid: partial r | (logF, fam) = +0.047 (t=1.5, p=1.4e-01); | (+method) = +0.026 (t=0.8, p=4.2e-01)
- eff_n_mat: partial r | (logF, fam) = -0.106 (t=-3.4, p=7.9e-04); | (+method) = -0.050 (t=-1.6, p=1.2e-01)
(run level; seeds within cells are correlated — treat |t|<4 as suggestive only)

Cell-level (seed-averaged) partials | (logF, fam):
- share_q: r = -0.279 (t=-5.3, p=1.9e-07, n=343)
- share_k: r = +0.180 (t=3.4, p=8.7e-04, n=343)
- share_v: r = +0.052 (t=1.0, p=3.4e-01, n=343)
- share_up: r = -0.063 (t=-1.2, p=2.5e-01, n=343)
- share_down: r = +0.145 (t=2.7, p=7.6e-03, n=343)
- depth_centroid: r = +0.029 (t=0.5, p=6.0e-01, n=343)
- eff_n_mat: r = -0.162 (t=-3.0, p=2.9e-03, n=343)

## C. Depth profiles (Llama CS arms, mean energy share per layer)
- clora: top layers [np.int64(0), np.int64(26), np.int64(24)] carry 12% of energy; first8 20% / last8 28%
- dora: top layers [np.int64(0), np.int64(30), np.int64(25)] carry 22% of energy; first8 30% / last8 29%
- lora: top layers [np.int64(0), np.int64(28), np.int64(25)] carry 13% of energy; first8 22% / last8 31%
- lora_null: top layers [np.int64(0), np.int64(27), np.int64(28)] carry 12% of energy; first8 21% / last8 30%
- lorawd: top layers [np.int64(0), np.int64(28), np.int64(27)] carry 14% of energy; first8 21% / last8 31%
- lorawdr16: top layers [np.int64(0), np.int64(25), np.int64(26)] carry 13% of energy; first8 20% / last8 31%
- milora: top layers [np.int64(0), np.int64(28), np.int64(25)] carry 15% of energy; first8 22% / last8 31%
- sclora: top layers [np.int64(0), np.int64(1), np.int64(28)] carry 13% of energy; first8 24% / last8 28%
