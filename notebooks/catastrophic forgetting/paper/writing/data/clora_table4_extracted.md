# CLoRA paper Table 4 — extracted rows used for the cross-literature overlay

Source: CLoRA paper (repro/CLoRA/), Table 4 — commonsense-trained Llama-2-7B variants;
columns extracted: reported F_Δ (their Eq. 3 metric, same estimator as our
fdelta_token_weighted) and reported BBH. These 10 (log10 F_Δ, BBH) pairs are the
"diamonds" in the artifact §1 cross-literature overlay (JS const XLC) and the basis of
the external-replication fit.

| # | row (their label) | log10 F_Δ | F_Δ | BBH |
|---|-------------------|-----------|------|------|
| 1 | LoRA baseline | −0.1024 | 0.790 | 26.69 |
| 2 | DoRA baseline | −0.0223 | 0.950 | 26.90 |
| 3 | PiSSA baseline | 0.0128 | 1.030 | 26.73 |
| 4 | LoRA-L2 | −0.5376 | 0.290 | 32.93 |
| 5 | rsLoRA | −0.0362 | 0.920 | 25.14 |
| 6 | CLoRA k128 | −0.4437 | 0.360 | 30.82 |
| 7 | CLoRA k256 | −0.4685 | 0.340 | 31.92 |
| 8 | CLoRA k512 | −0.5686 | 0.270 | 34.32 |
| 9 | CLoRA k1024 | −0.6778 | 0.210 | 36.49 |
| 10 | CLoRA k2048 | −0.8539 | 0.140 | 38.67 |

Fit over all 10 rows (verified 2026-07-11): **r = −0.9805, slope = −14.65 pp/decade.**
Subset fits: baselines-only (rows 1–5) slope ≈ −12.7; k-series (rows 6–10) slope ≈ −18.9.
Note the auditor caveat: including their Base (no-FT) row would break the fit — Base is
not a fine-tuned point and is excluded by construction (same rule as our own fits).

Reproduce:
```python
import math
XLC = [[-0.1024,26.69],[-0.0223,26.90],[0.0128,26.73],[-0.5376,32.93],[-0.0362,25.14],
       [-0.4437,30.82],[-0.4685,31.92],[-0.5686,34.32],[-0.6778,36.49],[-0.8539,38.67]]
xs=[p[0] for p in XLC]; ys=[p[1] for p in XLC]; n=len(xs)
mx,my=sum(xs)/n,sum(ys)/n
sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
syy=sum((y-my)**2 for y in ys)
print(sxy/math.sqrt(sxx*syy), sxy/sxx)   # -0.9805 -14.65
```
