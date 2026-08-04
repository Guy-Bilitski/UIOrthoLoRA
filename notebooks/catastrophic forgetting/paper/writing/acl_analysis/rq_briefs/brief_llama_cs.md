# Findings brief — Llama-2-7B x Commonsense-8 (families `lrsw`, `frc`)

**Pool convention:** frozen n=1035 pool (`insights/pool.csv`), quarantined-but-finite runs INCLUDED,
`frc_lorawdr16_wd0p3_lr3e4_c256_s42_reeval` duplicate included (dedupe changes nothing beyond rounding).
Retention = `ret` (mean BBH + MMLU-Pro), base ceiling 26.0; adaptation = CS-8 mean. Cells aggregated to
means +/- SD over seeds (seeds within a cell are correlated, ICC~0.78; paired-per-seed tests only).
**Preflight vs frozen anchors (key_numbers.md §18.1): lrsw n=180, r=-0.886 (anchor -0.886); frc n=276,
r=-0.928 (anchor -0.928) — exact.** Every quoted number below reproduces §18/§19 or the adjudication
tables verified in `verification/verification_report.md`.

## Best-adaptation operating points (cell mean +/- SD over seeds)

| method (family) | adapt | ret | F_delta | n seeds |
|---|---|---|---|---|
| LoRA+wd0.3, lr5e-4 (lrsw) | 81.75 +/- 0.17 | 25.86 +/- 0.37 | 0.399 | 4 |
| LoRA+wd0.3, lr5e-4 (frc)  | 81.86 +/- 0.21 | 25.94 +/- 0.70 | 0.402 | 4 |
| SC-LoRA, lr5e-5 (lrsw) | 80.61 +/- 0.41 | 24.60 +/- 1.85 | 0.376 | 3 |
| LoRA, lr3e-4 (lrsw) | 79.17 +/- 0.20 | 23.86 +/- 0.48 | 0.616 | 4 |
| LoRA-Null, lr5e-4 (lrsw) | 78.86 +/- 0.17 | 21.76 +/- 1.32 | 0.702 | 4 |
| CLoRA k1024, lr5e-4 (lrsw) | 78.29 +/- 0.25 | 21.60 +/- 0.39 | 0.645 | 4 |
| MiLoRA, lr5e-4 (lrsw) | 77.19 +/- 0.42 | 21.43 +/- 0.87 | 0.852 | 4 |
| DoRA, lr5e-4 (lrsw) | 76.23 +/- 1.65 | 19.15 +/- 1.39 | 1.226 | 3 |
| PiSSA, lr3e-4 (frc) | 69.43 +/- 0.50 | 11.36 +/- 1.71 | 1.405 | 4 |
| MiLoRA+wd0.3, lr5e-4 (lrsw, E6 arm) | 80.22 | 26.66 | 0.296 | 1 — excluded from ranking |

## Load-bearing facts (RQ1: are methods different? RQ2: what governs the outcome?)

- **Retention tracks update magnitude in both families.** Run-level r(ret, log10 F_delta) = -0.886
  (lrsw, n=180) and -0.928 (frc, n=276); cell-level -0.915 / -0.951; behavioral corroboration
  r(ret, log KL) = -0.884 / -0.915. Verified against §18.1 anchors (exact) and re-derived here.
- **The relation is flat-then-falling with a knee.** Two-segment beats linear (my cell-level F ~ 35 lrsw
  / 30 frc; §18.2 confirms). frc knee reproduces at log10 F_delta ~ -0.37..-0.43 (frozen -0.45,
  F_delta ~ 0.35), above-knee slope ~ -17..-20 pp/decade. The lrsw knee is fit-convention-sensitive
  (frozen -0.02; my refits land -0.5..-0.02 depending on retention-floor handling) — quote the frozen
  value with the §18.2 wording, not a re-derived one.
- **Method identity is real but second-order.** Cell-level residuals from a per-family quadratic:
  frc SC-LoRA -3.0 +/- 1.6 pp, LoRA-Null -1.5, PiSSA -5.2 (1 cell) below the shared curve; everything
  else within ~+/-2.3 pp. Consistent with frozen §18.4 OLS offsets (sclora -3.7, pissa -5.9,
  lora_null -2.0, ref = CLoRA). SC-LoRA's negative retention offset is an E4 calibration-set effect
  (eval-matched calibration moves it to +0.9 pp above curve, §18.3) — do not attribute it to geometry.
- **At matched capacity, the LoRA+wd operating point pairs the highest adaptation with the highest
  multi-seed retention in both families** (ret 25.86 / 25.94, i.e. within ~0.1 pp of the 26.0 base
  ceiling). Paired per seed vs plain LoRA's best cell (lrsw, 4 seeds): adapt +2.58 +/- 0.34,
  ret +2.00 +/- 0.85 (paired t = 4.7, p = 0.018). Its op point also carries the lowest op-point KL
  (0.189 / 0.210) and the smallest multi-seed op-point F_delta (0.40 vs 0.52-1.4). This matches
  the audited adjudication op-point table exactly. CLoRA's published results are faithful; the
  observation here is only that LoRA+wd reaches the same adaptation at lower update magnitude
  in this harness.
- **The wd effect rides the shared curve — a free-lunch region exists.** All wd cells sit within
  ~+/-1.8 pp of the frc curve (§18.4/insights, verified); peak below-knee adaptation equals
  99.4-100% of the global peak (lrsw 81.8/81.8, frc 81.4/81.9 — my recompute exact vs verification).
- **LR robustness differs by method.** Retention stays within 2 pp of the method's own max at 6/8 swept
  LRs for LoRA+wd (2e-5 to 5e-4) vs 4/7-4/9 for LoRA / CLoRA / MiLoRA / DoRA / LoRA-Null (collapse
  from 3e-4 up) and 2/7 for SC-LoRA (declining already at 1e-4). All 18 quarantined runs in this matrix
  sit at lr >= 1e-3 (lrsw 10/180 — DoRA accounts for 7; frc 8/276; CLoRA's single quarantine is at
  out-of-band lr5e-3, consistent with the verified 0/121 in-band count).
- **Seed stability:** within-cell retention SD (cells with >=3 seeds), median: LoRA+wd 0.34, CLoRA 0.35,
  DoRA 0.35, LoRA 0.45, MiLoRA 0.49, LoRA-Null 0.56 — vs SC-LoRA 1.85 (max 5.96 at its lr5e-4 cell).
- **Geometry reads as a fingerprint, not a driver — nothing at the op points contradicts
  magnitude-first.** Stable-rank fingerprint recurs (PiSSA 18.1, SC-LoRA 12-15 vs LoRA 4.4-5.9,
  DoRA 3.8-7.1); partial r(stable_rank, ret | log F_delta) = -0.595 (lrsw) / -0.333 (frc), both exact
  vs the verified observatory values. Op-point retention ordering follows F_delta/KL, not spec_max
  (DoRA op spec_max ~209 vs SC-LoRA ~8, yet retention follows magnitude).

## Anomalies / reviewer-visible caveats

- **Single-seed points:** MiLoRA+wd (E6) is n=1 per cell and non-dominated on the Pareto frontier
  (80.22/26.66) — report separately, exclude from method ranking (verification CORRECTION 1).
  frc LoRA+wd-r16 (wd0.3, lr5e-4) holds ret 26.61 +/- 0.23 but adapt 74.58 +/- 12.99 (adaptation
  seed lottery); frc LoRA+wd (wd0.5, lr1e-3) has ret SD 8.56 — high-LR wd cells can destabilize,
  so wd is not a free knob at large LR (cf. E6 DoRA+wd degenerate). Low-LR adaptation SDs reach
  ~30 pp (CS format collapse): single-seed adaptation rankings in this matrix are unreliable.
- **Convention sensitivity:** the frozen pool includes 18 quarantined-but-finite runs; excluding them
  strengthens r slightly and (pool-wide) flips the adaptation-retention sign census — any
  adaptation-vs-retention correlation quoted for this matrix must name the pool (verification A2).
