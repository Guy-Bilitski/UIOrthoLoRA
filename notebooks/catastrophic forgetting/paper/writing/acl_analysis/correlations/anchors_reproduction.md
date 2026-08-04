# Reproduction of frozen anchors

```
[PREFLIGHT] frozen-pool reproduction of key_numbers.md §18.1
  lrsw: n=180 r=-0.886  OK
  lrswm: n=120 r=-0.865  OK
  qwsw: n=151 r=-0.840  OK
  qwswm: n=164 r=-0.830  OK
  frc: n=276 r=-0.928  OK
  frm: n=144 r=-0.929  OK
  pooled: n=1035 r=-0.847  OK -- §18.1 reproduced
```

Ladder pool: frozen(deduped) ∩ geometry, n=1034 (target 1034)

| step | R2 (mine) | dR2 (mine) | R2 (frozen) | dR2 (frozen) | match |
|---|---|---|---|---|---|
| M0 family FE | 0.390 | +0.390 | 0.390 | +0.390 | OK |
| M1 + log10 F_delta | 0.785 | +0.395 | 0.785 | +0.395 | OK |
| M2 + geometry (e_top, log spec_max, stable_rank) | 0.802 | +0.017 | 0.802 | +0.017 | OK |
| M3 + method dummies | 0.808 | +0.006 | 0.808 | +0.006 | OK |

Ladder reproduction: ALL STEPS MATCH §19.1 to 3 decimals

Commonality anchor (06 §5, shape-only geometry = e_top + stable_rank):
  unique(magnitude) = +0.296  (frozen +0.296)
  unique(shape-geo) = +0.016  (frozen +0.016)
  shared            = +0.099  (frozen +0.099)
  match: OK

## Join coverage (deduped pool, n=1034)

| family | n | geometry | CE/KL | ret_broad | adaptation |
|---|---|---|---|---|---|
| lrsw | 180 | 180 (100%) | 180 (100%) | 180 | 180 |
| lrswm | 120 | 120 (100%) | 120 (100%) | 120 | 120 |
| qwsw | 151 | 151 (100%) | 93 (62%) | 151 | 151 |
| qwswm | 164 | 164 (100%) | 99 (60%) | 164 | 164 |
| frc | 275 | 275 (100%) | 275 (100%) | 273 | 275 |
| frm | 144 | 144 (100%) | 144 (100%) | 144 | 144 |
| ALL | 1034 | 1034 (100%) | 911 (88%) | 1032 | 1034 |

Duplicate run dropped after preflight: frc_lorawdr16_wd0p3_lr3e4_c256_s42_reeval
