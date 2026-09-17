# Qwen intruder finalization — 2026-09-17

Evidence: UIOrthoLoRA `origin/ortho_new` at `ff99de58068e1ff791674798b379c2e0b9eee556`.
Reviewed manuscript baseline: Overleaf project `6a46eb1b48498302a1ab34db`, `f442072`.
All changes are proposed in the existing blue/gray convention; no prior review was cleared.

## 1. Close the result set

The campaign is complete. The table contains five Llama configurations, six informative Qwen configurations, and the Qwen SC-LoRA diagnostic row. The latter's B–E constructions remain excluded from scientific comparisons.

Independently recomputed from the saved headline scores:

| Comparison | Final result |
| --- | --- |
| C > B, retention | 10/11; Qwen MiLoRA is the exception, C−B = −0.34 pp |
| C > B, task accuracy | 11/11 |
| D < A, retention | 10/11; Qwen MiLoRA is the exception, D−A = +0.70 pp |
| Deletion improves retention over A | 7/11; restoring the norm removes the gain in 6/7 |
| E > B, retention, feasible matches only | 6/7; Qwen LoRA+wd is the exception, E−B = −0.19 pp |
| C > B on both BBH and MMLU-Pro | 8/11; LoRA, MiLoRA and LoRA+wd on Qwen have mixed component signs |
| B > C, measured F_delta | 10/11; Qwen LoRA has B/C = 0.1059/0.1066 |
| D > A, measured F_delta | 10/11; Qwen LoRA has D/A = 0.1210/0.1217 |
| Qwen C > A on both measured scores | 6/6; retention gains 0.43–6.69 pp, task gains 0.19–2.38 pp |

These are point-estimate orderings, not significance or equivalence tests. The counts include the previously disclosed destructive Llama deletion outcomes; those scores are not interpreted as graded loss of particular skills.

The closing handoff's “11/11” retention dominance and “12/12” D<A claims are not supported by the saved summaries. Neither was copied into the manuscript. Intruder load is also not monotonic in source F_delta across designs; no such claim was added.

## 2. Repair the table's comparison scope

All 58 previously printed retention/task cells and all geometry values are preserved. The table is regenerated from the corrected generator, not edited by hand.

- SC-LoRA receives a double-dagger. Its B–E scores remain visible as numerical-floor diagnostics, with an explicit exclusion from intervention comparisons.
- Both Qwen CLoRA 2e-4 and LoRA-Null E receive a section-sign flag: neither reaches B's Frobenius norm. The reported E/B source-relative ratios are 0.723/0.720 and 0.788/0.706, respectively. The smaller CLoRA mismatch is still a recorded infeasibility and is excluded consistently; this leaves seven valid E comparisons, not eight or nine.
- Missing Llama E cells remain `--`, for infeasible construction rather than pending evaluation. Pending language is removed from the accepted manuscript and caption.
- Table spacing changes from 3 to 2.9 pt solely to accommodate the added footnote marks without overflow.
- The generator now errors if a required final summary is absent, preventing regeneration of a silently incomplete final table.

The norm ratios and infeasibilities are supported by the campaign's recorded verification results in `TIERA_RUN_LOG.md`, especially the 2026-09-16 08:37, 08:47 and 11:12 entries. Raw adapter tensors and `verify_*.log` files are absent from this snapshot, so this audit does not independently reconstruct those norms from weights.

## 3. Integrate the new scaling control

One new Results paragraph and one Conclusion sentence capture the supported insight: at similar measured magnitudes, task accuracy varies much more than retention. Methods names the completed control; the appendix gives its construction and quantitative limits.

CLoRA 3e-4, Qwen: four new rescalings plus A and C give six measured F_delta values from 0.1022 to 0.3286. Retention falls monotonically from 51.35 to 38.26. The descriptive quadratic fit is

`retention = 54.9191496608 − 22.3510659855 F_delta − 91.8846522408 F_delta²`,

with R² = 0.9869839906, versus 0.9774023295 for a linear fit. The quadratic's retention residuals are B +0.8594 and D −0.9637 pp; the separately fitted quadratic task residuals are B −4.0769 and D −4.8243 pp. D's F_delta = 0.3365 lies beyond the fitted range. These are descriptive fits, without confidence bounds or model-selection inference.

For the scaled trained update, B, and three randomized-factor controls, measured F_delta ranges from 0.2393 to 0.2470. Task accuracy ranges from 5.50 to 86.44 (80.94 pp), while retention ranges from 42.39 to 44.74 (2.35 pp). The randomization replaces output factors with Gaussian draws and **retains trained input factors**; it is not a wholly unstructured random weight update. Its seeds are 2001–2003, all on one trained source.

The three random controls' quadratic retention residuals are −1.9188, −0.6690 and −1.2756 pp. A directional effect on retention remains possible. Qwen B's positive retention residual does not replicate the negative Llama B residual (−2.16).

The SC-LoRA A–E score range is 1.28 retention points and 0.31 task points. This is a descriptive range of near-null constructions, not a calibrated protocol noise floor: the weights and measured magnitudes are not all identical. No “within noise,” sigma, significance or equivalence conclusion is derived from it.

The previously reviewed abstract, contribution list and core conclusion (“no consistent retention benefit”) remain intact. No removed Ep/F sections or table columns are reintroduced. The old 15-rescaling/9-control Llama experiment remains distinct from the new Qwen experiment.

## 4. Verification and review order

Review in this order: Table 2 flags/caption, Section 4.3, the new Qwen appendix paragraph, then the short Methods/Conclusion/Limitations changes.

- The unmodified generator reproduced the received Overleaf table byte for byte before any edits.
- `audit_results.py` recomputes all contrasts, ray fits and diagnostic ranges from the pinned JSON summaries. Its CSV and JSON outputs are retained with the review bundle.
- Review and clean versions compile with Tectonic. Final logs have no TeX errors, overfull boxes, undefined references/citations or duplicate labels. The existing `tang2025loranull` volume/number bibliography warning and Tectonic's repeated bibliography-change warning remain.
- The table and appendix pages were visually inspected in the rendered PDFs.
- The accepted text has no stale pending/in-progress status for this campaign. Archived gray text deliberately retains the earlier wording.

Useful next scientific extension: another training seed for Qwen LoRA-Null A/B/C, followed by paired item-level uncertainty if those outputs can be recovered. Neither is represented as completed or required for this scoped final write-up, and no new experiment was launched.
