# NAACL intruder manuscript review — 2026-09-16

The manuscript is reframed around the completed paired comparisons. The underlying table values were correct. The substantive errors were universal claims, conflated magnitude/energy definitions, an unmatched Llama base reference, and overinterpretation of destructive edits. No experiment, queue or results file was changed.

Evidence snapshot: UIOrthoLoRA `ortho_new` initially at `abd9747af0d3f209f4a17921cd2ee6977141d20a`, extended with the two CLoRA 3e-4 C/D summaries in `0d19c06b`. Manuscript baseline: Overleaf project `6a46eb1b48498302a1ab34db`, commit `a0bdf73`. Guy concurrently pushed Overleaf table commit `39cd223`; its two new cells were preserved and independently checked before the final update.

## Sections changed

1. **Abstract:** replaced an absolute no-benefit statement with no consistent retention benefit beyond uniform shrinkage, on both architectures.
2. **Introduction, contribution (iii):** aligned with that claim; described small task costs at low occupancy and possible destructive edits at high occupancy without asserting a unique location of adaptation.
3. **Methods, Metric 4:** corrected energy share to the mean top-64 spectral-energy share of the adapted weight W′. Occupancy and deletion still use the top ten. This is not the fraction of update energy removed.
4. **Methods, intruder intervention Runs/Arms:** stated five Llama plus seven Qwen table configurations, the guarded Qwen recipe, learning-rate versus magnitude matching, and the additional Ep/F controls.
5. **Results, `sec:results:intruder`:** renamed to “Intruder removal and magnitude-matched controls”; reported 9/10 retention comparisons, exact Qwen contrasts, the newly completed CLoRA 3e-4 controls, the distinction between Frobenius norm and F_delta, and the limits of interpreting task collapse.
6. **Conclusion:** removed “never”; matched the abstract and Results.
7. **Limitations, item 1:** retained two-architecture commonsense scope, one seed per intruder configuration, and pending controls; added rank/design/coverage confounding, lack of equivalence tests, and the guard cross-reference.
8. **Practical reading and its risk:** stated that intruder deletion is not established as a repair procedure and may damage task behavior.
9. **Appendix Compute:** scoped the B200 estimate to the frozen pool and identified the separate H200 intruder campaign.
10. **Appendix `app:intruder`:** corrected definitions, selection, protocol, base references, control availability, matching caveats, validity boundary and reconciliation with prior work. Added all seven Qwen configurations with r/alpha/LR/seed, exact counts and shares, construction ratios, SC-LoRA's numerical floor, the common 129/31,956 guard disclosure, and available Ep/F scores. Removed unsupported universal threshold, rank-one occupancy bound, σ10 claim and cross-protocol residual interpretation. Preserved the verified Llama scaling ray with its fitted range and extrapolation caveat.
11. **Table 2, generated caption/block label:** corrected historical Llama base provenance, top-64 energy definition, configuration-selection wording, and SC-LoRA interpretation. Regenerated from the script. All data rows and all 15 remaining `\dots` occurrences are unchanged.
12. **New appendix configuration table:** seven Qwen rows, all blue through `newpart`.

All 11 pre-existing `\cut{...}` blocks were preserved verbatim. Superseded current text is also retained in gray; inserted text uses `\new{...}` or `newpart`. `\ifdiff` remains on in Overleaf.

## Main numbers independently recomputed

| Qwen configuration | C−B retention | D−A retention | D/A increase in F_delta | A−B task loss |
| --- | ---: | ---: | ---: | ---: |
| MiLoRA 1e-4 | −0.34 | +0.70 | 1.806% | 0.50 |
| LoRA+wd 1e-4 | +0.24 | −0.25 | 9.016% | 1.00 |
| CLoRA 2e-4 | +1.92 | −2.69 | 20.766% | 1.25 |
| CLoRA 3e-4 | +1.81 | −3.83 | 16.800% | 2.44 |
| LoRA-Null 2e-4 | +7.52 | −14.66 | 48.627% | 53.68 |

CLoRA 3e-4: A/B retention 39.86/44.74 (+4.88); task 85.00/82.56 (−2.44). C scores 46.55/87.38 with F_delta 0.2101; D scores 36.03/78.19 with F_delta 0.3365. C−B retention is +1.81; D−A retention/task is −3.83/−6.81 and its F_delta rises 16.800%. C−A retention gains 6.69.

Qwen counts out of 1,400 top-ten slots, ordered SC-LoRA, LoRA, MiLoRA, LoRA+wd, CLoRA 2e-4, CLoRA 3e-4, LoRA-Null: **3, 42, 84, 169, 532, 666, 861**. Percentages: **0.214, 3.000, 6.000, 12.071, 38.000, 47.571, 61.500**. Corresponding mean top-64 energy shares: **0.015, 0.018, 0.081, 0.107, 0.277, 0.375, 0.438**.

Qwen base: **48.00**, with BBH **44.15**, MMLU-Pro **51.86**, from the matched-protocol summary. Do not average separately rounded components and “correct” the stored aggregate. Base-minus-source retention: **−0.19, 4.02, 3.66, 3.46, 8.14** for LoRA+wd, MiLoRA, CLoRA 2e-4, LoRA-Null, CLoRA 3e-4. MiLoRA C remains **3.59** below base.

Llama: C−B is **2.15–6.20** only when excluding destroyed LoRA-Null; that row is **24.32**. The off-frontier MiLoRA contrast is **20.11** and is not part of the ten main-table comparisons. Refit of the LoRA+wd scaling ray: **20.025640 − 5.107295 ln F_delta**, **R²=0.989589**; B/D residuals **−2.158206/−3.960092**. Stored σ64-margin exceedance is **93.75–100%**, not a σ10 result.

The historical Llama base record averages **29.235**; later pool summaries report **25.89**. Neither is verified on the intervention's reduced protocol. Both provenance distinctions are retained.

See `INTRUDER_NUMERIC_AUDIT_2026-09-16.md` for every available Qwen arm, all contrasts, exact geometry aggregates, sources, and remaining verification limits.

## Verification limits and follow-up for Guy's coding agent

- Norm ratios were cross-checked against the handoff and available narrative log, not reconstructed from adapter weights. The CLoRA Ep/F ratios **0.723/1.067** and **0.707/1.038** are supplied by Guy's handoff; the raw `logs/verify_*.log` files are absent from this snapshot. Inspect those logs and saved metadata to confirm all ratios before final acceptance.
- The shared **129/31,956** skip count is documented for all seven runs in `TIERA_RUN_LOG.md`; **0.403680%** was recomputed. Compare the seven `nan_guard.json` index sets to independently verify identity. No such records are in this snapshot.
- Reconcile Llama's historical and corrected base artifacts, then locate or obtain its base evaluation on the exact intervention proxy before publishing matched-base forgetting gaps. The present wording does not need that new evaluation.
- The current text uses the energy quantity actually stored: top-64 W′ energy. If a top-ten energy statistic is scientifically desired, compute it separately from saved spectra/weights and name the field explicitly. Do not relabel existing values as top-ten or as removed ΔW energy.
- If future prose needs “indistinguishable” or “no measurable forgetting,” compute paired uncertainty from item-level scores. The present prose reports point estimates and makes neither claim.
- When the existing queue drains, pull the generator correction, regenerate the table, preserve the review markup, and update pending-arm prose from the new summaries. SC-LoRA B/C/D/E/F must remain excluded from scientific contrasts; report A/Ep only. No queue or pipeline change was made or requested by this review.

## Validation

- Original table regenerated byte for byte from the pinned JSON snapshot before edits.
- Every filled arm cell, every other data row, and all existing pending markers preserved after the generator prose corrections.
- All pre-existing cut blocks preserved, diff switch on, accepted-text search completed for stale Llama-only and universal claims.
- Review and clean variants compiled with Tectonic; final logs contain zero TeX errors and zero undefined or multiply defined references/citations. Both table renders were inspected. The table's unchanged tabular body has a minor 3.18 pt width warning; bibliography has the existing volume/number warning for `tang2025loranull`.

## Delivery

Overleaf commits: `5f4f487` (main review), `c322120` (concurrent CLoRA C/D integration), `0a76bd9` (regenerated table provenance). Final manuscript and table were published through MCP.
