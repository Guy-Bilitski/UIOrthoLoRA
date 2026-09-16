# Intruder numeric audit — 2026-09-16
Read-only audit of the supplied snapshot, with calculations from results JSON. Updated for the concurrent upstream evidence at GitHub origin/ortho_new **0d19c06b** and Overleaf table commit **39cd223**, which add Qwen CLoRA3e-4 arms C and D. Their summary files report evaluation times2026-09-16T05:33:50Z and06:22:07Z, respectively, and evaluation-code commitabd9747af0d3f209f4a17921cd2ee6977141d20a. No manuscript, pipeline, queue, or result file was changed. Printed headline accuracies/retentions are already rounded to two decimals; differences below use those stored headline values. F_delta differences use its four-decimal stored values. Geometry aggregates were independently checked against all per-matrix entries. Missing means absent from this snapshot, not a failed run.

## Immediate corrections
1. The completed main-table B/C retention comparisons favor C in **9 of 10**, not all ten. Qwen MiLoRA C−B = **−0.34 pp**; the other four Qwen comparisons are +0.24 (wd), +1.92 (CLoRA2e-4), +7.52 (Null), and +1.81 (CLoRA3e-4). There is no paired uncertainty estimate here that establishes either a benefit or equivalence for ±0.34 pp. “Never” and “indistinguishable” need qualification. C has lower F_delta than B in **10 of 10 completed main-table pairs**, a different and supported claim. Counts exclude the additional off-frontier Llama MiLoRA row.
2. Llama C−B = 2.72, 6.20, 2.15, 4.46, **24.32** pp across the five frontier rows. “2.15–6.20” is valid only when explicitly excluding the destroyed LoRA-Null row. The off-frontier MiLoRA comparison is another 20.11 pp and is not in the twelve-row printed table.
3. Qwen CLoRA 3e-4 has 47.5714% slot occupancy, below all five Llama frontier rows (56.75–93.625%). The extra cell matches learning rate/design, not actual update magnitude or intruder occupancy. Qwen CLoRA 2e-4 has **532**, not 169, top-ten intruders; the narrative log reused LoRA+wd’s count in several entries.
4. The energy field is the mean across matrices of `energy_share_baseAll_t0.5` at aggregate `topk=64`. It is distinct from top-ten occupancy and from a fraction of ΔW energy removed. The JSON lacks singular-value arrays, so this sub-audit independently verifies the aggregation. The parent additionally inspected origin/ortho_new intruder_pass.py: score_matrix uses energy=S_adapted[:k]**2, mask maxcos<tau, and sum(energy[mask])/sum(energy) with k=64. The precise definition is thus confirmed as top-64 W′ spectral energy, averaged across matrices.
5. A norm ratio b is not itself an energy-deletion fraction: 1−b² gives remaining-norm-squared difference, not the reported W′ intruder share. For b=.720,.722,.706 those scalar differences are 48.16%,47.8716%,50.1564%; they are not .277,.375,.438. Since the arm edits W′, even that scalar difference should not be described as orthogonally removed ΔW energy without construction proof.
6. Table caption incorrectly presents both displayed bases as measured on the reduced protocol. Qwen48.00 is matched; the generator’s historical Llama29.235 is full battery. There are also corrected pool Llama artifacts at25.89. Preserve and distinguish provenance, and avoid a quantitative cross-architecture forgetting comparison until matching is verified.
7. Top-ten occupancy is not a universal monotone function of any one update-size statistic across designs. Qwen Null has more occupancy than CLoRA3 despite lower F_delta and lower max update singular value; MiLoRA has lower occupancy than wd despite higher F_delta. “Broadly associated” is supportable; monotonic/function or causally established dose threshold is not.
8. Do not call the Llama LoRA B=10.81% arm a working model or the intervention “intact at86%”: this is already gross loss of output behavior. Llama CLoRA drops76.19→43.50 (a severe task loss, but not a near-zero collapse). Qwen CLoRA3 loses2.44pp (2.4 to one decimal), not at most2.4 when stated as an exact upper bound.

## Qwen configurations
Source: `qwen_campaign3.sh`, RUNS and ARGS arrays; all successful final runs are reported at seed43 in TIERA_RUN_LOG. The campaign script permits seed44 fallback, but the final log records no such fallback. Shared explicit options: cutoff_len256, base Qwen/Qwen2.5-7B.

| Design | r | alpha | LR | seed | Other explicit settings |
| --- | --- | --- | --- | --- | --- |
| LoRA+wd | 32 | 64 | 1e-4 | 43 | weight_decay .3 |
| MiLoRA | 32 | 32 | 1e-4 | 43 | milora=1 |
| CLoRA | 32 | 64 | 2e-4 | 43 | k1024; lambda1.0 |
| LoRA | 16 | 32 | 5e-5 | 43 | standard |
| LoRA-Null | 16 | 16 | 2e-4 | 43 | lora_null=1 |
| SC-LoRA | 32 | 32 | 2e-5 | 43 | beta.5; calibration256; nq_open |
| CLoRA extra | 32 | 64 | 3e-4 | 43 | k1024; lambda1.0 |

The appendix’s generic nearest-normalized-corner frontier selection is not the rule for every row: the log says r32 designs use locked Pareto points, r16 designs use best-adaptation LR, and the extra CLoRA is selected to match Llama LR.

## Geometry independently recomputed
| Qwen configuration | Intruders /1400 slots | Slot share | Mean energy share | Matrices with ≥1 intruder | max σ1(ΔW) |
| --- | --- | --- | --- | --- | --- |
| LoRA+wd 1e-4 | 169 | 12.071428571% | 0.106804604921 | 75/140 | 11.035325050 |
| MiLoRA 1e-4 | 84 | 6.000000000% | 0.080885265109 | 50/140 | 8.900981903 |
| CLoRA 2e-4 | 532 | 38.000000000% | 0.276542249502 | 75/140 | 19.476591110 |
| LoRA 5e-5 | 42 | 3.000000000% | 0.018203952303 | 34/140 | 10.410126686 |
| LoRA-Null 2e-4 | 861 | 61.500000000% | 0.438422210062 | 135/140 | 16.624164581 |
| SC-LoRA 2e-5 | 3 | 0.214285714% | 0.014692585164 | 3/140 | 3.209360361 |
| CLoRA (Llama-matched LR) 3e-4 | 666 | 47.571428571% | 0.375193714244 | 83/140 | 25.510190964 |

All aggregates use criterion_version2, threshold.5 and full-base-reference key `baseAll`. SC-LoRA’s exact0.2142857% rounds to0.2%, while the generator currently displays0%; this is rounding, not zero intruders. Its reported energy .0146926 rounds to.015 at three decimals and.01 at two decimals.

## Complete Qwen arm values
### LoRA+wd 1e-4
| Arm | Retention | Task CS-8 | F_delta |
| --- | --- | --- | --- |
| A | 48.19 | 87.00 | 0.1342 |
| B | 48.86 | 86.00 | 0.1306 |
| C | 49.10 | 87.38 | 0.1197 |
| D | 47.94 | 85.88 | 0.1463 |
| E | 48.67 | 87.12 | 0.1182 |
| Ep | 50.89 | 82.44 | 0.0621 |
| F | 47.20 | 86.38 | 0.1521 |

### MiLoRA 1e-4
| Arm | Retention | Task CS-8 | F_delta |
| --- | --- | --- | --- |
| A | 43.98 | 86.75 | 0.1827 |
| B | 44.75 | 86.25 | 0.1808 |
| C | 44.41 | 86.94 | 0.1776 |
| D | 44.68 | 86.19 | 0.1860 |
| E | 44.95 | 86.75 | 0.1777 |
| Ep | 49.98 | 86.88 | 0.1093 |
| F | 43.45 | 80.19 | 0.1936 |

### CLoRA 2e-4
| Arm | Retention | Task CS-8 | F_delta |
| --- | --- | --- | --- |
| A | 44.34 | 85.94 | 0.2114 |
| B | 46.36 | 84.69 | 0.1850 |
| C | 48.28 | 87.06 | 0.1513 |
| D | 41.65 | 83.25 | 0.2553 |
| E | missing | missing | missing |
| Ep | missing | missing | missing |
| F | missing | missing | missing |

### LoRA 5e-5
| Arm | Retention | Task CS-8 | F_delta |
| --- | --- | --- | --- |
| A | missing | missing | missing |
| B | missing | missing | missing |
| C | missing | missing | missing |
| D | missing | missing | missing |
| E | missing | missing | missing |
| Ep | missing | missing | missing |
| F | missing | missing | missing |

### LoRA-Null 2e-4
| Arm | Retention | Task CS-8 | F_delta |
| --- | --- | --- | --- |
| A | 44.54 | 86.56 | 0.1966 |
| B | 41.28 | 32.88 | 0.2062 |
| C | 48.80 | 87.19 | 0.1398 |
| D | 29.88 | 3.50 | 0.2922 |
| E | missing | missing | missing |
| Ep | missing | missing | missing |
| F | missing | missing | missing |

### SC-LoRA 2e-5
| Arm | Retention | Task CS-8 | F_delta |
| --- | --- | --- | --- |
| A | missing | missing | missing |
| B | missing | missing | missing |
| C | missing | missing | missing |
| D | missing | missing | missing |
| E | missing | missing | missing |
| Ep | missing | missing | missing |
| F | missing | missing | missing |

### CLoRA (Llama-matched LR) 3e-4
| Arm | Retention | Task CS-8 | F_delta |
| --- | --- | --- | --- |
| A | 39.86 | 85.00 | 0.2881 |
| B | 44.74 | 82.56 | 0.2457 |
| C | 46.55 | 87.38 | 0.2101 |
| D | 36.03 | 78.19 | 0.3365 |
| E | missing | missing | missing |
| Ep | missing | missing | missing |
| F | missing | missing | missing |

## Qwen contrasts
All deltas are first arm minus second; positive retention/task means a higher score. F_delta % is relative to the second arm.

| Config | Contrast | Retention pp | Task pp | F_delta difference | F_delta % |
| --- | --- | --- | --- | --- | --- |
| LoRA+wd 1e-4 | B−A | +0.67 | -1.00 | -0.0036 | -2.683% |
| LoRA+wd 1e-4 | C−B | +0.24 | +1.38 | -0.0109 | -8.346% |
| LoRA+wd 1e-4 | D−A | -0.25 | -1.12 | +0.0121 | +9.016% |
| LoRA+wd 1e-4 | E−B | -0.19 | +1.12 | -0.0124 | -9.495% |
| LoRA+wd 1e-4 | Ep−B | +2.03 | -3.56 | -0.0685 | -52.450% |
| LoRA+wd 1e-4 | F−B | -1.66 | +0.38 | +0.0215 | +16.462% |
| LoRA+wd 1e-4 | F−A | -0.99 | -0.62 | +0.0179 | +13.338% |
| LoRA+wd 1e-4 | C−A | +0.91 | +0.38 | -0.0145 | -10.805% |
| MiLoRA 1e-4 | B−A | +0.77 | -0.50 | -0.0019 | -1.040% |
| MiLoRA 1e-4 | C−B | -0.34 | +0.69 | -0.0032 | -1.770% |
| MiLoRA 1e-4 | D−A | +0.70 | -0.56 | +0.0033 | +1.806% |
| MiLoRA 1e-4 | E−B | +0.20 | +0.50 | -0.0031 | -1.715% |
| MiLoRA 1e-4 | Ep−B | +5.23 | +0.63 | -0.0715 | -39.546% |
| MiLoRA 1e-4 | F−B | -1.30 | -6.06 | +0.0128 | +7.080% |
| MiLoRA 1e-4 | F−A | -0.53 | -6.56 | +0.0109 | +5.966% |
| MiLoRA 1e-4 | C−A | +0.43 | +0.19 | -0.0051 | -2.791% |
| CLoRA 2e-4 | B−A | +2.02 | -1.25 | -0.0264 | -12.488% |
| CLoRA 2e-4 | C−B | +1.92 | +2.37 | -0.0337 | -18.216% |
| CLoRA 2e-4 | D−A | -2.69 | -2.69 | +0.0439 | +20.766% |
| CLoRA 2e-4 | C−A | +3.94 | +1.12 | -0.0601 | -28.430% |
| LoRA-Null 2e-4 | B−A | -3.26 | -53.68 | +0.0096 | +4.883% |
| LoRA-Null 2e-4 | C−B | +7.52 | +54.31 | -0.0664 | -32.202% |
| LoRA-Null 2e-4 | D−A | -14.66 | -83.06 | +0.0956 | +48.627% |
| LoRA-Null 2e-4 | C−A | +4.26 | +0.63 | -0.0568 | -28.891% |
| CLoRA (Llama-matched LR) 3e-4 | B−A | +4.88 | -2.44 | -0.0424 | -14.717% |
| CLoRA (Llama-matched LR) 3e-4 | C−B | +1.81 | +4.82 | -0.0356 | -14.489% |
| CLoRA (Llama-matched LR) 3e-4 | D−A | -3.83 | -6.81 | +0.0484 | +16.800% |
| CLoRA (Llama-matched LR) 3e-4 | C−A | +6.69 | +2.38 | -0.0780 | -27.074% |

Exact D/A F_delta increases: MiLoRA1.806%, wd9.016%, CLoRA2e-4 20.766%, CLoRA3e-4 16.7997%, Null48.627%; at whole-percent precision the last is49%, not48%. D retention is higher than A by.70pp for MiLoRA, lower by.25,2.69,3.83,14.66 for wd/CLoRA2e-4/CLoRA3e-4/Null. The baseline’s broader “D worse” is true for F_delta, not universally retention.

## Base provenance
`results/base_qwen25-7b/summary.json`: adapter=null, evaluated2026-09-16T03:31:54Z, commit8cd4a061; headline BBH44.15, MMLU-Pro51.86, retention48.00, CS-8=21.25, F_delta=0. The component scores are separately rounded, whose naive mean is48.005; use stored retention48.00, not an invented correction to48.01. The campaign evaluates --ret_limit50 --eval_limit200 --ret_max_gen512.

| Qwen source | 48.00 − source retention (pp) |
| --- | --- |
| LoRA+wd 1e-4 | -0.19 |
| MiLoRA 1e-4 | +4.02 |
| CLoRA 2e-4 | +3.66 |
| LoRA-Null 2e-4 | +3.46 |
| CLoRA (Llama-matched LR) 3e-4 | +8.14 |

`results/base_l2-7b/retention_agg.json`: scores BBH39.51, MMLU-Pro18.96; mean of these stored values29.235, evaluated2026-06-09T23:57:32+03:00, commit7561e272, n_shards8, per-subtask n values250 for BBH and hundreds/thousands for MMLU-Pro. This is a historical full-battery artifact. Python binary formatting emits29.23;≈29 or explicitly29.235 avoids implying better source precision. **Do not call it a matched reduced-protocol zero point.**

`results/base_llama2_7b_noft/summary.json` and `base_llama2_noft/summary.json` give BBH32.96, MMLU-Pro18.82, retention25.89. `results/retfix_diag/base_bbh_fullset_current_harness.json` reports32.96 fullset, metric_fix_applied=true, paper_ceiling33.1. `results/base_l2-7b_bbhAO/retention_agg.json` has BBH33.1 alone. Thus the previous manuscript’s≈26 pool baseline has support and must not be indiscriminately replaced by29.235. These snapshots alone do not establish which Llama base is matched to the intervention’s exact prompt/protocol.

Arithmetic historical29.235 minus five Llama sources gives4.365,4.955,4.835,6.115,7.195pp (wd,MiLoRA,CLoRA,LoRA,Null). These are **not validated forgetting estimates** because the protocols differ. Similarly, Qwen arms at48–51 exceed the measured48 base by0–3pp; without uncertainty intervals do not call them statistically “at base”, zero forgetting, or indistinguishable. MiLoRA C is44.41, so the log’s “all C arms at48–51” is also false.

## Norm ratios and training guard
No verify_*.log or saved adapter norms are in this snapshot, so none of these ratios was numerically reconstructed from weights. The quoted sources distinguish inspected narrative support from user-supplied numbers.

| Qwen config | B/C/E norm / A | Ep / A | F / A | Source status |
| --- | --- | --- | --- | --- |
| wd1e-4 | .891 | .573 | 1.100 | all in inspected log |
| MiLoRA1e-4 | .972 | .598 | 1.050 | all in inspected log |
| CLoRA2e-4 | .720 | .723 | 1.067 | B/C in log; E association and Ep/F supplied by user |
| CLoRA3e-4 | .722 | .707 | 1.038 | B in log; C/E association and Ep/F supplied by user |
| LoRA-Null2e-4 | .706 | not supplied | not supplied | B/C in log; E association not numerically verified |
| LoRA5e-5 | .873 (B only) | not supplied | not supplied | B in log |
| SC-LoRA2e-5 | B/C≈1.009; E=1 | .828 | near-source/noise | inspected log reports degeneracy |

A and D are1 by construction; D equality was not independently verified from norms. SC-LoRA: three intruders in three matrices; the log explicitly attributes B/C/F slight growth and E=A to numerical error at tiny deletion strength. Retain A and Ep as meaningful reported arms; B/C/D/E/F should not be read as ordinary deletion controls.

The seven successful Qwen configurations are each reported to skip the **same129/31,956 optimizer steps**, with unchanged weights and Adam state on a non-finite gradient. Recomputed rate=0.40368006%;0.4% is correct. The log’s2026-09-16 01:12 entry says seven of seven identical skip sets and no final-run abort/retry. The actual seven nan_guard.json files are absent, so equality of indices is narrative-verified rather than recomputed here. “Fixed by data order and independent of design” is an observed seven-run pattern, not a proven mechanism. The log also records unseeded LoRA-A initialization before get_peft_model:43 is the configured seed, not a guarantee of fully reproducible initialization.

## Preserved Llama values and ray
| Llama config | Slot share | Mean energy | C−B retention | C−A retention | A→B task |
| --- | --- | --- | --- | --- | --- |
| LoRA+wd 5e-4 | 56.7500% | 0.432573144825 | +2.72 | +0.83 | 80.00→2.81 |
| MiLoRA 3e-4 | 78.4375% | 0.553414643090 | +6.20 | +1.80 | 80.00→1.06 |
| CLoRA 3e-4 | 62.0000% | 0.461447465408 | +2.15 | +3.30 | 76.19→43.50 |
| LoRA 3e-4 | 85.9375% | 0.572550664516 | +4.46 | +5.84 | 79.38→10.81 |
| LoRA-Null 5e-4 | 93.6250% | 0.687036467623 | +24.32 | +5.36 | 78.50→0.50 |
| MiLoRA (high-F) 1e-3 | 99.6875% | 0.923995346203 | +20.11 | +2.51 | 65.69→0.00 |

Uniform shrink C−A gains are.83–5.84pp over the six Llama configurations including off-frontier MiLoRA, exactly as the old scaling paragraph reports. The latter improves task65.69→70.06 and retention17.60→20.11. Preserve these with explicit Llama scope.

Refit using A,C,scale1.05,scale1.12: retention = **20.025640130 -5.107295176 ln F_delta**, R²=**0.989589182**. Thus20.03−5.11lnF_delta,R²=.99 is correct.

| Llama wd arm | F_delta | retention | task | ray residual pp |
| --- | --- | --- | --- | --- |
| A | 0.3951 | 24.87 | 80.00 | +0.101641904 |
| C | 0.3273 | 25.70 | 61.69 | -0.029866249 |
| s1.05 | 0.4149 | 24.48 | 80.75 | -0.038618359 |
| s1.12 | 0.4422 | 24.16 | 79.69 | -0.033157296 |
| B | 0.3675 | 22.98 | 2.81 | -2.158205832 |
| D | 0.4416 | 20.24 | 8.81 | -3.960091846 |
| E | 0.3261 | 26.18 | 55.12 | +0.431374152 |
| Ep | 0.2201 | 27.17 | 31.25 | -0.586416424 |
| G | 0.2119 | 27.96 | 28.38 | +0.009672227 |
| H | 0.0596 | 28.36 | 9.19 | -6.068721749 |

The ray covers42.0–62.875% slots and energy.369540768–.467071083; G is47.25% slots versus source56.75%. H F_delta=.0596 lies far below fit range. Ep is not a uniform scaling-ray point (it is a complementary-component edit), so the phrase “smooth rescaling keeps31 task points (Ep)” mislabels the control; report its actual construction.

## Unsupported or incompletely auditable claims
- “No evidence of a consistent extra retention benefit” is a defensible descriptive synthesis. “Intruder retention is explained by magnitude” or “intruder share alone determines collapse” is stronger than these one-seed comparisons demonstrate, especially with varying designs/ranks and no uncertainty estimates.
- Qwen CLoRA3e-4 A/B/C/D are now complete. C scores46.55 retention/87.38 task/F_delta.2101, beating B by1.81 retention and4.82 task points; C−A is+6.69 retention/+2.38 task. D scores36.03/78.19/.3365; D−A is−3.83 retention/−6.81 task and+16.7997% F_delta. Its earlier4.88pp B−A retention recovery is therefore exceeded by uniform shrink. E/Ep/F remain available only for wd and MiLoRA in Qwen at this snapshot; do not conflate an arm being built/feasible with evaluated. All other missing cells remain missing.
- Whole-table claims that every design occupies57–94%, all runs are Llama-only, or Qwen replication is pending are stale. Five means Llama frontier configurations; main table has twelve configurations (5 Llama+7 Qwen); paper_table also prints off-frontier Llama MiLoRA as a thirteenth.
- Baseline appendix says σ1(ΔW)>σ10(W0) in94–100% of matrices for every configuration. The parent inspected intruder_pass.py and confirmed base_sk=base_S[k−1] with k=64. Llama fractions using stored σ64 are97.5%,100%,93.75%,100%,100%,100%; Qwen0–85%. These are **σ64 exceedances, not σ10**, and need correction and Llama scope.
- The rank-one “at most one new direction in top10” claim is not implied by rank alone: low-rank perturbations can rotate multiple singular vectors. Detector threshold occupancy is not bounded that way without a theorem/additional assumptions.
- Claimed pool-law residual range−.08 to+1.94 and correlation+.78 need the exact frozen-pool law/intercept and configuration source selection; those fit artifacts are not here. Existing Llama scope should be retained until recomputed by coding agent.
- Existing numerical-validation claims (singular-value error<1e-3, synthetic detector tests, one-adapter protocol offsets.29/.20, sevenfold evaluation savings, E infeasibility ratios.83–.89) have no direct verification outputs here. Preserve as historical claims if already established; do not imply independently verified during this audit.
- Guard log claims source Qwen pool agreement within.9 task/.005F, but corresponding full-pool source JSONs are not in this snapshot. One documented log pair wd differs.58 task and.002F; CLoRA2 differs.25/.004; CLoRA3 differs.84/.002. Full blanket verification requires exact pool counterparts.

## Short recalculation list for Guy’s coding agent
1. CLoRA3 C/D are already present in the upstream Overleaf table at39cd223. Regenerate the table again after the remaining queue drains, fixing generator caption/base provenance and distinguishing top64 W′ energy from top10 slot occupancy. Keep pending dots until results exist and exclude SC-LoRA noise controls. Do not hand-edit generated cells.
2. Provide verify_*.log outputs for all Qwen norm ratios and the seven nan_guard.json records to verify identical129-step sets. Energy denominator and base_sk index are now confirmed from parent source inspection. No new experiment needed for these checks.
3. Reconcile historical Llama29.235 with corrected25.89 and identify or measure the Llama base under the exact intervention proxy protocol before reporting matched-base forgetting or cross-architecture point losses.
4. If the prose needs “indistinguishable” or “no measurable forgetting”, compute paired uncertainty from item-level results for MiLoRA C−B=−.34, wd C−B=.24, and Qwen arms relative to base. Otherwise use descriptive point-estimate wording now.
5. Recompute the frozen-pool residual/energy correlation and Qwen source-to-pool agreement from identified source artifacts if retaining exact blanket ranges.
