# Archived module-level mixing diagnostics

These files copy existing research records, not new training results. They extend the summary archive in `../legacy_mixing/` with the 48 module records behind each A/B/C run. There are 1,296 rows, 27 runs, and nine task/seed settings. Smoke tests and unmatched C-only reruns are excluded.

Source root in the research checkout: `notebooks/glue/training_new/newer_train/results/` (read only during this work).

| Included file | Source relative path |
|---|---|
| `rte_seed42.csv` | `rte_ablation/rte_ablation_layer_metrics.csv` |
| `rte_seed17.csv` | `rte_ablation_seed17/rte_ablation_layer_metrics.csv` |
| `mrpc_seed42.csv` | `mrpc_ablation/mrpc_ablation_layer_metrics.csv` |
| `mrpc_seed17.csv` | `mrpc_ablation_seed17/mrpc_ablation_layer_metrics.csv` |
| `cola_seed42.csv` | `cola_ablation/cola_ablation_layer_metrics.csv` |
| `cola_seed17.csv` | `cola_ablation_seed17/cola_ablation_layer_metrics.csv` |
| `stsb_seed42.csv` | `sts-b_ablation/sts-b_ablation_layer_metrics.csv` |
| `stsb_seed17.csv` | `sts-b_ablation_seed17/sts-b_ablation_layer_metrics.csv` |
| `sst2_seed42.csv` | `sst-2_ablation/sst2_ablation_layer_metrics.csv` |

Numeric strings are copied without recomputation; line endings may be normalized. `scripts/analyze_archived_layers.py` verifies unique task/condition/seed/module keys, exact module coverage, and all shared summary means before deriving any new fractions.

`Leak11_F`, `Leak12_F`, and `Leak21_F` divide block Frobenius norms by the effective delta norm; `OffTailRatio_F` instead divides the off-tail norm by the norm of the recorded coordinate matrix. Squaring the individual leakage ratios and renormalizing by the individual off-tail squared ratio yields the recorded-frame energy fractions. This avoids silently assuming perfectly orthogonal finite-precision bases. The largest inferred coordinate/ambient squared-norm discrepancy is 0.0002635263. The diagnostic implementation uses a denominator stabilizer of 1e-12. The manuscript gives the exact reconstruction convention.

The resulting plot and table average per-matrix fractions within each run and then across seeds. They are not pooled energy ratios, squared task-mean norm ratios, new independent layer replicates, checkpoint trajectories, or norm-matched controls. Exact generating code/configuration manifests remain a separate provenance requirement.
