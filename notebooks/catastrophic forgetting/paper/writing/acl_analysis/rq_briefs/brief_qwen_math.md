# RQ brief — Qwen2.5-7B x math/GSM8K (`qwswm`)

**Pool convention:** `insights/pool.csv`, fam=qwswm; operating points use quarantine OUT,
`_ep6_` OUT, cells with >=2 seeds (n=154 runs; 4 quarantined, 6 ep6). Adaptation = GSM8K EM
(`adapt`); retention for method comparisons = **BBH only** (base 47.93; MMLU-Pro parser fails
on MetaMath-tuned outputs); pooled magnitude relation uses `ret`.
**Preflight:** frozen anchor reproduced exactly on the full 164-row qwswm pool:
r(log10 F_delta, ret) = **-0.830, n=164** (r vs bbh -0.828). On the operating convention
(n=154): r = -0.752 (ret) / -0.706 (bbh), rank-r -0.564 — the disclosed clean-subset drift
(key_numbers §18.6). Knee at log10 F_delta ~ -0.91 (§18.2). CE/KL coverage 60% (cell-level
state only). Seeds 42-44, ICC ~0.78.

## Best-adaptation operating points (cell mean +/- SD)

| Method | LR | GSM8K | BBH | F_delta | n |
|---|---|---|---|---|---|
| **SC-LoRA** | 5e-5 | **77.23 +/- 0.79** | **47.71 +/- 0.23** | 0.107 | 3 |
| LoRA-Null | 1e-3 | 72.33 +/- 1.33 | 44.76 +/- 0.64 | 0.385 | 3 |
| CLoRA k1024 | 1e-3 | 70.46 +/- 0.96 | 39.98 +/- 0.12 | 0.436 | 3 |
| LoRA r32 | 5e-4 | 70.44 +/- 0.86 | 41.77 +/- 2.31 | 0.386 | 2 |
| LoRA+wd | 3e-4 | 68.97 +/- 3.33 | 47.54 +/- 0.43 | 0.102 | 3 |
| MiLoRA | 2e-4 | 65.35 +/- 4.03 | 46.16 +/- 1.28 | 0.145 | 3 |
| DoRA | 1e-4 | 63.50 +/- 0.80 | 46.92 +/- 0.04 | 0.071 | 2 |
| LoRA r16 | 3e-4 | 61.97 +/- 4.80 | 47.13 +/- 0.43 | 0.161 | 3 |

(DoRA coverage is sparse here, 11 runs; adjudication's 2e-4 pick is an n=1 cell in this pool —
either way DoRA is bottom-tier on adaptation.)

## Verified findings (RQ1/RQ2)

- **The paper's one candidate counterexample lives here.** SC-LoRA beats LoRA+wd on GSM8K by
  **+8.26 pp [95% CI 1.41, 15.12]** (paired over 3 seeds, t=5.18) at statistically tied BBH
  (+0.17, p=0.63); deterministic Pareto frontier is 100% SC-LoRA, P(frontier)=1.00. This is the
  only matrix of six where LoRA+wd is not retention-top: it trails SC-LoRA's op point by 0.17 pp,
  and the family-best BBH (48.19, above base) is SC-LoRA at 2e-5.
- **But the edge is not significant after multiplicity correction:** raw p=0.035; Holm within
  family p=0.21, Holm across all 25 head-to-heads p=0.46 (`rq1_stats/head2head_corrected.md`).
  Report as *suggestive, not significant*; retention MDE for this pair at 3 seeds is ~1.7 pp,
  but the adaptation CI is wide.
- **The win occurs at small update magnitude, on-curve.** SC-LoRA's winning cell sits below the
  knee at F_delta = 0.107 — second-smallest of its own sweep (2e-5 is 0.055) and essentially
  equal to LoRA+wd's op-point magnitude (0.102). It wins by extracting more adaptation per unit
  magnitude, not by tolerating a large update. Do not say "lowest-magnitude cell" or "smallest
  op-point F_delta of the family" (both false; verification CORRECTION 3).
- **Attribution stays with the method-as-configured, not geometry.** The qwswm run used the
  standard nq_open calibration; the E4 eval-matched control exists only for Llama-CS and showed
  SC-LoRA outcomes move ~4 pp with calibration-corpus choice. No calibration-sensitivity control
  exists on Qwen-math (verification OVERCLAIM 1 — never print "genuine geometry win"
  unqualified). Observationally, its op-point update is spectrally small (spec_max 3.1 vs 8.2
  for LoRA+wd) at matched F_delta.
- **Magnitude relation holds; no method is significantly off-curve.** Below the knee, method
  mean BBH residuals span -1.7 to +1.6 pp (SC-LoRA +1.6, LoRA+wd +0.4, rest slightly negative);
  frozen §18.4 finds no significant Qwen method offsets. Pooled TOST puts SC-LoRA's retention
  offset at matched magnitude *below* LoRA+wd (-4.1 pp, Llama-driven); the qwswm-only offset
  (-3.49, 90% CI [-8.50, +1.53]) is too underpowered to bound at +/-2 pp.
- **LR robustness is where LoRA+wd wins this matrix.** It is the only method with cell-mean BBH
  within 2 pp of base at all 7 LRs (7/7 safe; SC-LoRA and DoRA 3/7, others 4-5). At 1e-3, BBH
  collapses for SC-LoRA (12.7), MiLoRA (15.3), LoRA r16 (23.0), LoRA r32 (18.4); SC-LoRA's peak
  at 5e-5 decays monotonically above 1e-4. CLoRA never collapses (BBH 40.0 at 1e-3, 0/20
  divergences) but is the matrix's only Holm-significant result: **-7.56 pp retention vs LoRA+wd
  at op points (p_holm=0.024, WORSE)** — CLoRA's published numbers are faithful; this is a
  matched-capacity comparison within our harness.
- **Divergences (4/158 attempted, quarantined):** LoRA+wd 2 (both at 1e-3 — its own top LR is
  not divergence-free here), LoRA r32 1 (5e-4), MiLoRA 1 (1e-3); SC-LoRA/CLoRA/LoRA-Null/DoRA 0.
- **Seed noise:** median within-cell BBH SD 0.4-1.5 pp (LoRA+wd lowest, 0.41); GSM8K SD 1.4-4.8
  (plain LoRA r16 worst at 4.8; SC-LoRA 1.7; LoRA+wd 2.1). Family median retention MDE 4.5 pp,
  max 33.7 (LoRA r32) — single-seed rankings unreliable.

## Reviewer caveats

1. **n=3 seed pairs everywhere; seeds within a cell are correlated (ICC ~0.78).** The +8.26 pp
   SC-LoRA delta is the family's headline yet its CI spans 1.4-15.1 pp and it does not survive
   Holm; the honest claim is "SC-LoRA is the only method anywhere that even suggestively beats
   LoRA+wd on adaptation at tied retention, under its nq_open-calibrated configuration."
2. **Retention here is BBH-only** (MMLU-Pro parser failure on MetaMath outputs), and CE/KL
   coverage is 60% cell-level — no per-seed CE claims. The pooled r=-0.830 anchor includes
   quarantined rows; the clean-subset value (-0.70/-0.75) must be quoted alongside it.
