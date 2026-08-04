# Adaptation-side structure (8 CS datasets, cell level)

## lrsw (n_cells=56)
```
      dataset  peak_acc  peak_logfd  collapse_onset_logfd  r_below_knee  r_above_knee
    hellaswag    84.465      -0.190                -0.034         0.688        -0.849
ARC_Challenge    65.469      -0.190                 0.069         0.536        -0.817
   winogrande    80.926      -0.153                 0.087         0.548        -0.735
     ARC_Easy    79.930      -0.153                 0.087         0.516        -0.860
   openbookqa    78.720      -0.153                 0.087         0.547        -0.873
         piqa    80.846      -0.153                 0.089         0.452        -0.740
        boolq    70.551      -0.190                 0.150         0.547        -0.642
  social_i_qa    78.347      -0.034                 0.182         0.399        -0.848
```

## frc (n_cells=75)
```
      dataset  peak_acc  peak_logfd  collapse_onset_logfd  r_below_knee  r_above_knee
    hellaswag    86.497      -0.211                -0.070         0.418        -0.875
     ARC_Easy    82.251      -0.211                -0.004         0.206        -0.888
ARC_Challenge    67.532      -0.211                -0.004         0.204        -0.904
   openbookqa    79.326      -0.211                 0.054         0.256        -0.836
        boolq    70.879      -0.211                 0.082         0.524        -0.772
         piqa    80.711      -0.095                 0.082         0.292        -0.762
   winogrande    80.713      -0.095                 0.148         0.342        -0.779
  social_i_qa    78.361      -0.095                 0.179         0.147        -0.651
```

## qwsw (n_cells=58)
```
      dataset  peak_acc  peak_logfd  collapse_onset_logfd  r_below_knee  r_above_knee
         piqa    89.754      -0.887                -0.779        -0.242        -0.371
ARC_Challenge    88.306      -0.970                -0.732        -0.336        -0.666
     ARC_Easy    94.934      -0.970                -0.691        -0.314        -0.551
    hellaswag    94.629      -0.867                -0.421         0.418        -0.673
   openbookqa    90.811      -0.878                -0.273         0.280        -0.621
        boolq    73.542      -0.466                -0.163         0.612        -0.620
   winogrande    85.802      -0.732                -0.163         0.634        -0.622
  social_i_qa    79.996      -0.878                -0.112         0.450        -0.556
```

## Collapse-onset ordering concordance: Kendall W = 0.738 (3 fams x 8 ds)
mean rank (1 = collapses first): ARC_Challenge=2.17, ARC_Easy=3.17, boolq=6.33, hellaswag=2.00, openbookqa=4.33, piqa=4.17, social_i_qa=8.00, winogrande=5.83

## Above-knee slopes per dataset (pp/decade, cell level)
- lrsw: openbookqa=-23.0, social_i_qa=-22.7, hellaswag=-22.6, ARC_Easy=-22.4, piqa=-18.9, winogrande=-18.5, ARC_Challenge=-16.2, boolq=-14.3
- frc: hellaswag=-46.9, ARC_Easy=-40.8, openbookqa=-36.3, ARC_Challenge=-32.3, winogrande=-23.8, social_i_qa=-23.7, piqa=-22.8, boolq=-11.0
- qwsw: hellaswag=-40.4, ARC_Challenge=-32.0, ARC_Easy=-25.3, openbookqa=-21.2, winogrande=-17.3, piqa=-15.1, boolq=-12.7, social_i_qa=-12.3
