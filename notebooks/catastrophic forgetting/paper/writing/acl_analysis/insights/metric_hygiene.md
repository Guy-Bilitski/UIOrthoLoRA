# Metric hygiene + effect sizes

## 1. TruthfulQA inside retention_broad
- frc: broad-with-TQ slope -11.42 (r=-0.931) vs broad-no-TQ -15.54 (r=-0.952); TQ understates slope by 26%. Top-decile-F_delta TQ = 42.0 (n=8)
- frm: broad-with-TQ slope -7.17 (r=-0.909) vs broad-no-TQ -9.85 (r=-0.914); TQ understates slope by 27%. Top-decile-F_delta TQ = 48.7 (n=6)
- lrsw: broad-with-TQ slope -5.84 (r=-0.828) vs broad-no-TQ -8.30 (r=-0.876); TQ understates slope by 30%. Top-decile-F_delta TQ = 45.3 (n=6)
- lrswm: broad-with-TQ slope -9.54 (r=-0.923) vs broad-no-TQ -12.48 (r=-0.916); TQ understates slope by 24%. Top-decile-F_delta TQ = 43.6 (n=5)
- qwsw: broad-with-TQ slope -25.25 (r=-0.933) vs broad-no-TQ -28.63 (r=-0.925); TQ understates slope by 12%. Top-decile-F_delta TQ = 39.5 (n=6)
- qwswm: broad-with-TQ slope -13.57 (r=-0.838) vs broad-no-TQ -15.79 (r=-0.830); TQ understates slope by 14%. Top-decile-F_delta TQ = 49.8 (n=7)
Base TQ: Llama 38.85, Qwen 56.28. If damage pushes TQ toward an indifference band,
Llama rises toward it, Qwen falls toward it.

Most-damaged runs (retention < 40% of family base): mean TQ
- frc: TQ = 45.0 +- 4.7 (n=15)
- frm: TQ = 49.0 +- 2.0 (n=15)
- lrsw: TQ = 44.0 +- 6.0 (n=15)
- qwsw: TQ = 39.2 +- 3.2 (n=33)
- qwswm: TQ = 48.6 +- 2.7 (n=14)

## 2. share_q effect size (pp of retention)
- OLS ret ~ logF + fam + share_q (n=1002 runs): beta(share_q) = -26.85 pp per unit share
- share_q IQR = 0.063 -> effect across IQR = -1.70 pp (vs magnitude beta -12.10 pp/decade)
- lorawd-only (n=321): beta(share_q) = -20.54, partial r = -0.289

## 3. wd adds nothing beyond the F_delta it produces (cell level)
- ret ~ logF: R2 = 0.902
- ret ~ logF + wd: R2 = 0.908
