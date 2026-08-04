# Brief: Qwen2.5-7B x Commonsense-8 (`qwsw`)

**Pool convention.** Frozen `insights/pool.csv`, `fam=="qwsw"`, finite `ret`+`logfd`, quarantine-included for the relation (2 quarantined runs), quarantine-excluded for operating points. Retention = mean(BBH, MMLU-Pro), base ceiling 44.35; adaptation = CS-8 mean. Seeds 42-45, ~3/cell, within-cell correlated (ICC~0.78) — run-level t/p optimistic.
**Preflight (reproduced exactly).** n=151, r(log10 F_delta, ret) = **-0.840**, Spearman -0.778 (= §18.1). Cell-level: -0.799 with log(mean F_delta) aggregation (58 cells, the §18.1 convention); -0.856 with mean(log F_delta) — quote the former.

## Best-adaptation operating points (cells with >=3 seeds; mean +/- SD)

| Method | LR | Adapt (CS-8) | Ret | F_delta | KL (coverage) |
|---|---|---|---|---|---|
| LoRA+wd (wd0.3) | 5e-4 | 87.43 +/- 0.23 | **40.07 +/- 0.68** | 0.245 | 0.23 (2/3) |
| SC-LoRA | 1e-4 | 87.15 +/- 0.15 | 27.85 +/- 15.96 | 0.348 | n/a |
| CLoRA (k1024) | 1e-4 | 87.02 +/- 0.19 | 39.52 +/- 1.15 | 0.128 | 0.13 (3/4) |
| DoRA | 2e-4 | 86.44 +/- 0.76 | 38.05 +/- 0.97 | 0.260 | 0.28 (2/3) |
| LoRA | 5e-5 | 86.43 +/- 0.41 | 37.95 +/- 0.88 | 0.122 | 0.07 (2/3) |
| MiLoRA | 2e-4 | 86.39 +/- 0.97 | 36.68 +/- 0.12 | 0.284 | 0.48 (3/4) |
| LoRA-Null | 2e-4 | 86.23 +/- 1.60 | 38.95 +/- 0.68 | 0.204 | 0.21 (1/3) |

Frozen headline **verified**: LoRA+wd is top-retaining at the best-adapt point (40.07+/-0.68) and holds the family retention maximum (41.02+/-0.46 at 3e-4; n=1 cells 41.03/39.92 at 15e-5/7e-5 exist but are single-seed — flagged in verification B6). Frontier here is 100% LoRA+wd (verification B6, confirmed).

## Verified bullets (RQ1 magnitude / RQ2 methods)

- **Magnitude relation replicates on Qwen-CS**: r = -0.840 (n=151), robust to quarantine exclusion (-0.837, n=149). Same knee-then-cliff shape as Llama, knee lower: at frozen knee log10 F_delta = -0.69, below-knee r = +0.21 (flat, n=64), above-knee r = -0.85 with slope ~ -41 pp/decade (n=87). Consistent with §18.2 and E3 densification (bottom-half r ~ -0.04: flat, not positive).
- **No absolute safe operating band**: best cell is 3.32 pp below base 44.35; abs-band safe_2pp = 0 for every method at every LR (replicates lr_band.csv exactly). Qwen-CS always pays for adaptation; only a *family-relative* band exists: LoRA+wd within 2 pp of family-top at 7/9 LRs vs CLoRA 3/7, LoRA-Null 3/8, MiLoRA 2/9, SC-LoRA 1/9, LoRA and DoRA 0.
- **Method offsets at matched magnitude are n.s. and bounded** (OLS ret ~ logfd + logfd^2 + method, ref=CLoRA): LoRA+wd +3.7+/-2.0 (largest, positive), all others within +/-1.5, |t| <= 1.9 before any cluster correction — replicates §18.4 "Qwen arms: no significant offsets". Power-limited, not evidence of equivalence.
- **CLoRA's stability distinction verified for qwsw**: 0 training divergences across all 7 LRs including 1e-3 (family total: only 2 quarantined runs — LoRA+wd@1e-3 s44, SC-LoRA@1e-3 s42). But non-divergence is not retention safety: CLoRA's retention still collapses at 1e-3 (cell mean 9.7). Its published stability behavior is faithful; the collapse is a magnitude effect, shared with all methods (every method's minimum retention is at 1e-3).
- **Seed noise is the power bottleneck of this matrix**: median within-cell SD(ret) is 0.4-0.9 pp for MiLoRA/LoRA+wd/LoRA-Null/CLoRA/LoRA but **7.08 for SC-LoRA (max 15.96 — its op point is a seed lottery: 27.85+/-15.96)** and 5.02 for DoRA; LoRA-Null has a 2-seed tail cell at 5e-4 with ret {34.2, 16.7} (SD 12.4; seed_variance.csv prints 9.9 for this max — filter-version difference, editorial-level, same story).
- **Nothing here contradicts magnitude-first**: within-family partial correlations given logfd — e_top -0.12 (p=0.14), stable_rank -0.00, log spec_max -0.12 (p=0.14) — all n.s.; geometry's small pooled contribution (§19.1 +1.7% R^2) is cross-family, not carried by qwsw.
- **CE corroboration (cell-level only)**: coverage 93/151 (62%), seed-blocked (s42: 9/58, s43/44: ~90%, s45: 5/5) — per-seed CE analyses barred. Cell-level r(CE, ret) = -0.67 (51 cells); run-level -0.64 matches frozen -0.631 within rounding.

## Reviewer caveats

1. **CE/KL coverage**: 62% on this family with seed-blocked missingness; all CE/KL claims are cell-level, and KL at op points rests on 1-3 of 3 seeds — state coverage wherever quoted.
2. **Seed noise / power**: with n=3 correlated seeds and within-cell SDs up to ~16 pp (SC-LoRA), "n.s." method contrasts here are underpowered, not null results; SC-LoRA and DoRA cell means on qwsw should not be ranked without their SDs attached.
