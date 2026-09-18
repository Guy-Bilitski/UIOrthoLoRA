# Reported GLUE values and metric reconciliation

`glue_reported.csv` preserves the ten rows from the previous approved manuscript, commit `9c8043c1b531f3afb27fa55ce101c8635736c938`, including all 60 task means, 60 standard deviations, originally reported parameter counts, and original six-task averages. It is a transcription of that table, not a reconstruction from final training runs.

The six imported VeRA/LoRA/RandLoRA rows match [RandLoRA, version 2, Appendix B.1, Table 4](https://arxiv.org/html/2502.00987v2). That source specifies Matthews correlation for CoLA, Pearson correlation for STS-B, and accuracy for the remaining tasks, including MRPC, over five runs.

On 2026-09-14, the author explicitly confirmed that the original Table 2 spectral MRPC values (92.0/90.0 for base and 92.5/93.4 for large) are accuracy. This corrects the prior audit's F1 assignment, inferred from the archived manuscript's metric description and the local training helper `notebooks/glue/training_new/training.py`. Those sources did not establish the metric of the final broad-evaluation table. The correction relies on author confirmation, not newly recovered raw accuracy outputs. The separate mixing-intervention records explicitly report MRPC F1 and are unchanged.

The main table retains every task mean and standard deviation in one MRPC accuracy column. Avg6 is no longer displayed across unmatched protocols; every six-task mean is still recomputed by the audit, and `reported_avg6` retains the originally printed aggregates as historical metadata. The average is not a full GLUE benchmark score. Common task metrics do not establish tuning fairness or uncertainty comparability. No average standard deviation or significance test is inferred from the task standard deviations.

## Protocol grouping in the review-panel revision

The 2026-09-14 revision visually separates imported reference rows (five runs under the RandLoRA protocol) and the original spectral rows (reported six seeds). This preserves the data without suggesting equal search budgets or a uniformly retuned ranking. The original [LoRA protocol](https://arxiv.org/html/2106.09685) reports medians and uses intermediate MNLI adapter initialization for some tasks; [VeRA](https://arxiv.org/html/2310.11454) documents a different protocol. Replacing the present reference rows with those publications would not make the comparison matched. P5 in `../EXPERIMENTS_REQUIRED.md` specifies new common-protocol comparisons with both spectral geometry and task metrics. MNLI/QQP remain explicitly absent from the historical six-task table.

## Parameter-count issue

The old UIOrthoLoRA counts were 0.4M (base) and 0.6M (large); UILinLoRA counts were 0.08M and 0.22M. Under the stated recipe of 256 adapted coefficients, two 64-by-64 stored rotation parameters, ambient scalers, and all four attention projections, the inspected implementation implies the following adapter-only counts:

| Backbone | Modules / width | Unrotated | Rotated |
|---|---|---:|---:|
| RoBERTa-base | 48 / 768 | 86,016 | 479,232 |
| RoBERTa-large | 96 / 1024 | 221,184 | 1,007,616 |

These are analytical counts under the stated configuration, not independently recovered counts of the historical models. In particular, the large-model rotated discrepancy cannot be explained by ordinary rounding. The task head, actual module placement, rotation dimensions, counting convention, and historical implementation must be reconciled before parameter-efficiency claims use the old counts. The original spectral counts remain in this source archive but are not silently substituted for verified inventories. The round-3 main table shows explicitly marked configuration-derived adapter counts (0.086/0.479M base, 0.221/1.008M large), with no task head or basis storage included; imported counts retain their source convention. These derived numbers do not resolve the historical inventory discrepancy.

## Run-level coverage

The original spectral protocol reports six seeds, but a complete matching set of final run manifests or checkpoints was not found. The local `training_new/results/glue/` and `glue_large_search/` CSVs include partial/tuning records, some under different configurations. They are not silently substituted for the reported final table. The author has been asked for the original final summaries/checkpoints. This gate is separate from the fully reaggregated mixing intervention.
