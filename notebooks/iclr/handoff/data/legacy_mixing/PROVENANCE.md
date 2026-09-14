# Archived mixing-summary provenance

These are copies of existing CSV summaries, not newly generated training results. The source root was `notebooks/glue/training_new/newer_train/results/` in the research checkout, which was not modified.

| Included file | Original relative path |
|---|---|
| `rte_seed42.csv` | `rte_ablation/rte_ablation_summary.csv` |
| `rte_seed17.csv` | `rte_ablation_seed17/rte_ablation_summary.csv` |
| `mrpc_seed42.csv` | `mrpc_ablation/mrpc_ablation_summary.csv` |
| `mrpc_seed17.csv` | `mrpc_ablation_seed17/mrpc_ablation_summary.csv` |
| `cola_seed42.csv` | `cola_ablation/cola_ablation_summary.csv` |
| `cola_seed17.csv` | `cola_ablation_seed17/cola_ablation_summary.csv` |
| `stsb_seed42.csv` | `sts-b_ablation/sts-b_ablation_summary.csv` |
| `stsb_seed17.csv` | `sts-b_ablation_seed17/sts-b_ablation_summary.csv` |
| `sst2_seed42.csv` | `sst-2_ablation/sst2_ablation_summary.csv` |

Each file contains A/B/C. Smoke and C-only reruns are excluded. The earliest RTE export uses `final_val_accuracy` instead of `final_val_score`; the generator handles that explicit schema difference. It averages already-computed module means across available seeds. There is no interpolation, synthetic replication, or squaring of averaged norm ratios.

The associated 1,296 module records are now included under `../legacy_layers/`. Their analysis reconciles every shared summary field, then derives four-block fractions per matrix before averaging. This extends the old summaries without replacing their normalization or pretending that their already-averaged norm ratios are energy fractions.

The result-file commit previously inspected was `8f1c7186`: repository provenance, not proof of the exact code executed. The current implementation was inspected separately, and the manuscript appendix describes the leading identity core. Exact generating code/configurations remain an author-confirmation item. Line endings may be normalized; numeric fields are preserved.

## Review-panel reanalysis, 2026-09-14

The generator now reports every condition's available seed mean and sample SD, all 27 selected steps/endpoints, and the archived time/memory fields. n=1 receives no estimated SD. Four-block fractions come from the accompanying per-module records, not the square of an averaged norm ratio. No raw CSV was changed.

`total_train_time` is divided by 60 for minutes; `peak_gpu_memory` is divided by 2^30 for GiB. Inspection of the legacy runner places the clock around the training call (including its validation/checkpoint work), and uses CUDA maximum allocated memory after training following a pre-training peak reset. Per-run hardware attribution, setup/post-training diagnostics and reserved/process memory are not established by these fields. The sum is approximately 14.9755 training-call hours, not verified total project GPU-hours. The source inspection fingerprints in `../../GPU_SOURCE_FINGERPRINTS.sha256` identify the current inspected implementation, not a certified historical generating revision.
