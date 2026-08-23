# Bound on a between-method geometry feature

Frozen pool, run level, family fixed effects, n = 1034.

| model | R2 |
|---|---|
| family FE | 0.3898 |
| + log10 F_delta | 0.7852 |
| + geometry (e_top, log spec_max, stable rank) | 0.8023 |
| + method dummies | 0.8083 |
| family FE + log10 F_delta + method dummies (no geometry) | 0.8007 |

- ladder method step, after geometry: **+0.0060**
- method identity over magnitude alone: **+0.0155**
- geometry and method together over magnitude: **+0.0231**

The middle line is the bound Section 5.2 needs. A geometry feature that is
constant within a method lies in the span of the method dummies, so what
method identity adds over size alone is what such a feature could at most add.
The ladder's own method step is measured after the geometry block has already
absorbed the between-method variation and is not a bound on it.
