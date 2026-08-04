# Head-to-head vs LoRA+wd (best-adaptation cells, per family)

Delta = method - LoRA+wd; paired per-seed where >=2 common seeds (ICC-safe),
Welch otherwise; outside-noise = |t| > 2. 'beats LoRA+wd' = wins >=1 axis
outside noise without losing the other outside noise. Script: `03_head2head.py`.

## Llama-2-7B x Commonsense-8 (lrsw sweep, r-matched grid)

| Method (cell) | dAdapt | t | verdict | dRet | t | verdict | test | beats LoRA+wd? |
|---|---|---|---|---|---|---|---|---|
| SC-LoRA (5e-5) vs wd (5e-4) | -1.19 | -8.21 | LOSS | -1.33 | -1.51 | tie | <bound method Series.mode of family              llama_cs
method               SC-LoRA
method_cell             5e-5
ref_cell                5e-4
d_adapt                -1.19
t_adapt                -8.21
adapt_verdict           LOSS
d_ret                  -1.33
t_ret                  -1.51
ret_verdict              tie
mode             paired(n=3)
beats_lorawd              no
Name: 0, dtype: object> | no |
| LoRA (3e-4) vs wd (5e-4) | -2.58 | -15.35 | LOSS | -2.00 | -4.73 | LOSS | <bound method Series.mode of family              llama_cs
method                  LoRA
method_cell             3e-4
ref_cell                5e-4
d_adapt                -2.58
t_adapt               -15.35
adapt_verdict           LOSS
d_ret                   -2.0
t_ret                  -4.73
ret_verdict             LOSS
mode             paired(n=4)
beats_lorawd       dominated
Name: 1, dtype: object> | dominated |
| LoRA-Null (5e-4) vs wd (5e-4) | -2.89 | -35.31 | LOSS | -4.10 | -5.59 | LOSS | <bound method Series.mode of family              llama_cs
method             LoRA-Null
method_cell             5e-4
ref_cell                5e-4
d_adapt                -2.89
t_adapt               -35.31
adapt_verdict           LOSS
d_ret                   -4.1
t_ret                  -5.59
ret_verdict             LOSS
mode             paired(n=4)
beats_lorawd       dominated
Name: 2, dtype: object> | dominated |
| CLoRA (5e-4 k1024) vs wd (5e-4) | -3.46 | -38.25 | LOSS | -4.26 | -11.4 | LOSS | <bound method Series.mode of family              llama_cs
method                 CLoRA
method_cell       5e-4 k1024
ref_cell                5e-4
d_adapt                -3.46
t_adapt               -38.25
adapt_verdict           LOSS
d_ret                  -4.26
t_ret                  -11.4
ret_verdict             LOSS
mode             paired(n=4)
beats_lorawd       dominated
Name: 3, dtype: object> | dominated |
| DoRA (5e-4) vs wd (5e-4) | -5.57 | -6.45 | LOSS | -6.78 | -8.92 | LOSS | <bound method Series.mode of family              llama_cs
method                  DoRA
method_cell             5e-4
ref_cell                5e-4
d_adapt                -5.57
t_adapt                -6.45
adapt_verdict           LOSS
d_ret                  -6.78
t_ret                  -8.92
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd       dominated
Name: 4, dtype: object> | dominated |
| MiLoRA (5e-4) vs wd (5e-4) | -4.56 | -22.67 | LOSS | -4.42 | -8.47 | LOSS | <bound method Series.mode of family              llama_cs
method                MiLoRA
method_cell             5e-4
ref_cell                5e-4
d_adapt                -4.56
t_adapt               -22.67
adapt_verdict           LOSS
d_ret                  -4.42
t_ret                  -8.47
ret_verdict             LOSS
mode             paired(n=4)
beats_lorawd       dominated
Name: 5, dtype: object> | dominated |

## Llama-2-7B x math/GSM8K (frm faithful CLoRA recipe, c256)

| Method (cell) | dAdapt | t | verdict | dRet | t | verdict | test | beats LoRA+wd? |
|---|---|---|---|---|---|---|---|---|
| LoRA (1e-4) vs wd (2e-4) | -2.81 | -4.95 | LOSS | -2.28 | -3.42 | LOSS | <bound method Series.mode of family            llama_math
method                  LoRA
method_cell             1e-4
ref_cell                2e-4
d_adapt                -2.81
t_adapt                -4.95
adapt_verdict           LOSS
d_ret                  -2.28
t_ret                  -3.42
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd       dominated
Name: 6, dtype: object> | dominated |
| MiLoRA (1e-4) vs wd (2e-4) | -3.11 | -3.55 | LOSS | -1.13 | -2.09 | LOSS | <bound method Series.mode of family            llama_math
method                MiLoRA
method_cell             1e-4
ref_cell                2e-4
d_adapt                -3.11
t_adapt                -3.55
adapt_verdict           LOSS
d_ret                  -1.13
t_ret                  -2.09
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd       dominated
Name: 7, dtype: object> | dominated |
| CLoRA (3e-4 k256) vs wd (2e-4) | -6.14 | -26.17 | LOSS | -4.94 | -9.11 | LOSS | <bound method Series.mode of family            llama_math
method                 CLoRA
method_cell        3e-4 k256
ref_cell                2e-4
d_adapt                -6.14
t_adapt               -26.17
adapt_verdict           LOSS
d_ret                  -4.94
t_ret                  -9.11
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd       dominated
Name: 8, dtype: object> | dominated |
| SC-LoRA (1e-4) vs wd (2e-4) | -6.32 | -8.47 | LOSS | -5.63 | -6.47 | LOSS | <bound method Series.mode of family            llama_math
method               SC-LoRA
method_cell             1e-4
ref_cell                2e-4
d_adapt                -6.32
t_adapt                -8.47
adapt_verdict           LOSS
d_ret                  -5.63
t_ret                  -6.47
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd       dominated
Name: 9, dtype: object> | dominated |
| DoRA (3e-4) vs wd (2e-4) | -7.61 | -10.3 | LOSS | -5.49 | -6.13 | LOSS | <bound method Series.mode of family            llama_math
method                  DoRA
method_cell             3e-4
ref_cell                2e-4
d_adapt                -7.61
t_adapt                -10.3
adapt_verdict           LOSS
d_ret                  -5.49
t_ret                  -6.13
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd       dominated
Name: 10, dtype: object> | dominated |
| LoRA-Null (2e-4) vs wd (2e-4) | -4.93 | -6.99 | LOSS | -4.68 | -10.14 | LOSS | <bound method Series.mode of family            llama_math
method             LoRA-Null
method_cell             2e-4
ref_cell                2e-4
d_adapt                -4.93
t_adapt                -6.99
adapt_verdict           LOSS
d_ret                  -4.68
t_ret                 -10.14
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd       dominated
Name: 11, dtype: object> | dominated |
| PiSSA (3e-4) vs wd (2e-4) | -17.13 | -37.52 | LOSS | -26.34 | -43.71 | LOSS | <bound method Series.mode of family           llama_math
method                PiSSA
method_cell            3e-4
ref_cell               2e-4
d_adapt              -17.13
t_adapt              -37.52
adapt_verdict          LOSS
d_ret                -26.34
t_ret                -43.71
ret_verdict            LOSS
mode             welch(1v3)
beats_lorawd      dominated
Name: 12, dtype: object> | dominated |

## Qwen-2.5-7B x Commonsense-8 (qwsw sweep)

| Method (cell) | dAdapt | t | verdict | dRet | t | verdict | test | beats LoRA+wd? |
|---|---|---|---|---|---|---|---|---|
| SC-LoRA (1e-4) vs wd (5e-4) | -0.29 | -1.99 | tie | -12.22 | -1.38 | tie | <bound method Series.mode of family               qwen_cs
method               SC-LoRA
method_cell             1e-4
ref_cell                5e-4
d_adapt                -0.29
t_adapt                -1.99
adapt_verdict            tie
d_ret                 -12.22
t_ret                  -1.38
ret_verdict              tie
mode             paired(n=3)
beats_lorawd              no
Name: 13, dtype: object> | no |
| LoRA (5e-5) vs wd (5e-4) | -1.00 | -9.71 | LOSS | -2.12 | -6.7 | LOSS | <bound method Series.mode of family               qwen_cs
method                  LoRA
method_cell             5e-5
ref_cell                5e-4
d_adapt                 -1.0
t_adapt                -9.71
adapt_verdict           LOSS
d_ret                  -2.12
t_ret                   -6.7
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd       dominated
Name: 14, dtype: object> | dominated |
| LoRA-Null (2e-4) vs wd (5e-4) | -1.20 | -1.18 | tie | -1.12 | -3.86 | LOSS | <bound method Series.mode of family               qwen_cs
method             LoRA-Null
method_cell             2e-4
ref_cell                5e-4
d_adapt                 -1.2
t_adapt                -1.18
adapt_verdict            tie
d_ret                  -1.12
t_ret                  -3.86
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd              no
Name: 15, dtype: object> | no |
| CLoRA (1e-4 k1024) vs wd (5e-4) | -0.37 | -2.31 | LOSS | -0.95 | -4.86 | LOSS | <bound method Series.mode of family               qwen_cs
method                 CLoRA
method_cell       1e-4 k1024
ref_cell                5e-4
d_adapt                -0.37
t_adapt                -2.31
adapt_verdict           LOSS
d_ret                  -0.95
t_ret                  -4.86
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd       dominated
Name: 16, dtype: object> | dominated |
| DoRA (2e-4) vs wd (5e-4) | -0.99 | -1.82 | tie | -2.02 | -7.91 | LOSS | <bound method Series.mode of family               qwen_cs
method                  DoRA
method_cell             2e-4
ref_cell                5e-4
d_adapt                -0.99
t_adapt                -1.82
adapt_verdict            tie
d_ret                  -2.02
t_ret                  -7.91
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd              no
Name: 17, dtype: object> | no |
| MiLoRA (2e-4) vs wd (5e-4) | -1.11 | -1.51 | tie | -3.41 | -10.77 | LOSS | <bound method Series.mode of family               qwen_cs
method                MiLoRA
method_cell             2e-4
ref_cell                5e-4
d_adapt                -1.11
t_adapt                -1.51
adapt_verdict            tie
d_ret                  -3.41
t_ret                 -10.77
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd              no
Name: 18, dtype: object> | no |

## Qwen-2.5-7B x math/GSM8K (qwswm sweep; ep6 variants excluded)

| Method (cell) | dAdapt | t | verdict | dRet | t | verdict | test | beats LoRA+wd? |
|---|---|---|---|---|---|---|---|---|
| SC-LoRA (5e-5) vs wd (3e-4) | +8.26 | 5.18 | WIN | +0.17 | 0.56 | tie | <bound method Series.mode of family             qwen_math
method               SC-LoRA
method_cell             5e-5
ref_cell                3e-4
d_adapt                 8.26
t_adapt                 5.18
adapt_verdict            WIN
d_ret                   0.17
t_ret                   0.56
ret_verdict              tie
mode             paired(n=3)
beats_lorawd             YES
Name: 19, dtype: object> | YES |
| LoRA (1e-4) vs wd (3e-4) | -5.48 | -1.33 | tie | -1.44 | -1.49 | tie | <bound method Series.mode of family             qwen_math
method                  LoRA
method_cell             1e-4
ref_cell                3e-4
d_adapt                -5.48
t_adapt                -1.33
adapt_verdict            tie
d_ret                  -1.44
t_ret                  -1.49
ret_verdict              tie
mode             paired(n=3)
beats_lorawd              no
Name: 20, dtype: object> | no |
| LoRA(r32) (5e-4) vs wd (3e-4) | +0.30 | 0.15 | tie | -5.76 | -2.79 | LOSS | <bound method Series.mode of family             qwen_math
method             LoRA(r32)
method_cell             5e-4
ref_cell                3e-4
d_adapt                  0.3
t_adapt                 0.15
adapt_verdict            tie
d_ret                  -5.76
t_ret                  -2.79
ret_verdict             LOSS
mode             paired(n=2)
beats_lorawd              no
Name: 21, dtype: object> | no |
| LoRA-Null (1e-3) vs wd (3e-4) | +3.36 | 2.05 | WIN | -2.77 | -8.0 | LOSS | <bound method Series.mode of family             qwen_math
method             LoRA-Null
method_cell             1e-3
ref_cell                3e-4
d_adapt                 3.36
t_adapt                 2.05
adapt_verdict            WIN
d_ret                  -2.77
t_ret                   -8.0
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd              no
Name: 22, dtype: object> | no |
| CLoRA (1e-3 k1024) vs wd (3e-4) | +1.49 | 0.79 | tie | -7.56 | -32.52 | LOSS | <bound method Series.mode of family             qwen_math
method                 CLoRA
method_cell       1e-3 k1024
ref_cell                3e-4
d_adapt                 1.49
t_adapt                 0.79
adapt_verdict            tie
d_ret                  -7.56
t_ret                 -32.52
ret_verdict             LOSS
mode             paired(n=3)
beats_lorawd              no
Name: 23, dtype: object> | no |
| DoRA (2e-4) vs wd (3e-4) | -5.80 | -0.93 | tie | -1.68 | -6.09 | LOSS | <bound method Series.mode of family             qwen_math
method                  DoRA
method_cell             2e-4
ref_cell                3e-4
d_adapt                 -5.8
t_adapt                -0.93
adapt_verdict            tie
d_ret                  -1.68
t_ret                  -6.09
ret_verdict             LOSS
mode             paired(n=2)
beats_lorawd              no
Name: 24, dtype: object> | no |
| MiLoRA (2e-4) vs wd (3e-4) | -3.61 | -8.89 | LOSS | -1.38 | -1.48 | tie | <bound method Series.mode of family             qwen_math
method                MiLoRA
method_cell             2e-4
ref_cell                3e-4
d_adapt                -3.61
t_adapt                -8.89
adapt_verdict           LOSS
d_ret                  -1.38
t_ret                  -1.48
ret_verdict              tie
mode             paired(n=3)
beats_lorawd              no
Name: 25, dtype: object> | no |

## Tally (both axes, outside noise)

| method | adapt W/L | ret W/L | beats | dominated | families |
|---|---|---|---|---|---|
| CLoRA | 0/3 | 0/4 | 0 | 3 | 4 |
| DoRA | 0/2 | 0/4 | 0 | 2 | 4 |
| LoRA | 0/3 | 0/3 | 0 | 3 | 4 |
| LoRA(r32) | 0/0 | 0/1 | 0 | 0 | 1 |
| LoRA-Null | 1/2 | 0/4 | 0 | 2 | 4 |
| MiLoRA | 0/3 | 0/3 | 0 | 2 | 4 |
| PiSSA | 0/1 | 0/1 | 0 | 1 | 1 |
| SC-LoRA | 1/2 | 0/1 | 1 | 1 | 4 |