# Ladder-step cluster-bootstrap CIs (exact §19.1 blocks)

n=1034, family FE,
B=2000 cell-level bootstrap, seed 0. Geometry block is the
ladder's own 3-metric block (e_top, log spec_max, stable rank),
unlike 09-Q1's 2-metric shape-unique CI. Script: `04_ladder_ci.py`.

| step | dR2 | 95% CI |
|---|---|---|
| magnitude (log10 F_delta) | +0.395 | [+0.309, +0.483] |
| geometry (e_top, log spec_max, stable rank) | +0.017 | [+0.007, +0.032] |
| method identity | +0.006 | [+0.003, +0.018] |

ordering magnitude > geometry: 2000/2000 replicates
