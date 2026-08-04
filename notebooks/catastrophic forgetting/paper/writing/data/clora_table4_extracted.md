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

## Primary-source verification, 2026-08-04

Verified directly against the published PDF (arXiv:2410.16801, pdfTeX build dated
2025-03-24, 15 pages; text extracted with PyMuPDF). Findings:

1. **Table 4 (p. 6) has no BBH column.** It has exactly three columns
   (`Method`, `||∆W||`, `F`) and eleven rows. The BBH numbers used in our overlay
   come from their **Table 2 (p. 5)** and its appendix duplicate **Table 8
   (p. 14)**, not from Table 4. Our appendix table therefore joins two of their
   tables and should say so.

2. **Table 4 verbatim** (`Method | ||∆W|| | F`):

   | Method | ‖∆W‖ | F |
   |---|---|---|
   | reference | (blank) | 2.42 |
   | LoRA | 22.63 | 0.79 |
   | MiLoRA | 24.32 | 0.92 |
   | LoRA-r16 | 12.70 | 1.03 |
   | LoRA-r8 | 6.45 | 0.95 |
   | LoRA-L2 | 2.07 | 0.29 |
   | CLoRA-k128 | 10.84 | 0.36 |
   | CLoRA-k256 | 10.25 | 0.34 |
   | CLoRA-k512 | 8.19 | 0.27 |
   | CLoRA-k1024 | 6.64 | 0.21 |
   | CLoRA-k2048 | 5.00 | 0.14 |

3. **The `F = 2.42` value is real but is not a ΔW measurement.** Their own text
   (p. 6): "The 'reference' row is computed using the LoRA trained model, noting
   the output scale of original parameter W instead of ∆W." So 2.42 is
   ‖Wx‖/‖x‖ for the base weight, i.e. the base model's own relative output
   scale, not an update magnitude. It does not lie on the ΔW axis and cannot be
   the base model's F_Δ (which is 0 by construction). The value is therefore
   **removed from the F and log10 F cells of our appendix table** and explained
   in that table's caption; base BBH 34.91 (their Tables 2/8) is retained.

4. **Three row labels in the earlier extraction were wrong.** The ten
   (F, BBH) pairs are all correct and the fit is unaffected, but the names were
   misassigned:
   - `DoRA baseline` (F 0.95, BBH 26.90) is in fact **LoRA-r8**. Their DoRA row
     has BBH 28.24 (Table 2) and appears in no F table.
   - `PiSSA baseline` (F 1.03, BBH 26.73) is in fact **LoRA-r16**. Their PiSSA
     row has BBH 29.54 and appears in no F table.
   - `rsLoRA baseline` (F 0.92, BBH 25.14) is in fact **MiLoRA**. The string
     "rsLoRA" appears zero times in the paper.
   The tex table labels have been corrected accordingly. Numbers untouched;
   r = −0.9805 and slope −14.65 pp/decade are unchanged.

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
