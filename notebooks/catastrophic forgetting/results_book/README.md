# Results book

One readable folder with every experiment's results — adaptation, retention, magnitude (F_Δ), geometry, and MiLoRA-style amplification metrics, per seed and per LR. Regenerated automatically as new runs land (`results_book.py`, auto-loop every 30 min).

_Last updated: 2026-07-12 11:23 IDT_

**Retention definitions** — CS runs: mean(BBH, MMLU-Pro), Llama-2-7B base = 26.0. Math runs: BBH-only, Llama-2-7B base = 33.1. Missing values are shown as ·.

| File | What it holds | Data rows |
|---|---|---|
| [01_cs_lr_sweep.md](01_cs_lr_sweep.md) | CS LR sweep on Llama-2-7B (lrsw_): 8 adapters × 7 LRs, per-adapter tables | 70 |
| [02_math_faithful.md](02_math_faithful.md) | Math arm (frm_ faithful repro + lrswm_ math LR sweep): GSM8K/MATH vs BBH retention | 94 |
| [03_cs_faithful.md](03_cs_faithful.md) | Faithful CS repro (frc_) — fills as the spine lands | 30 |
| [04_rank_wd_matrix.md](04_rank_wd_matrix.md) | Rank/weight-decay matrix (mtx_/mtxm_), 3 seeds, per-config mean±SD | 110 |
| [05_qwen.md](05_qwen.md) | Qwen2.5-7B second-model replication: CS + math LR sweeps | 99 |
| [06_calibration_control.md](06_calibration_control.md) | Eval-matched calibration control (b4_) | 4 |
| [07_ce_forgetting.md](07_ce_forgetting.md) | CE-to-base forgetting for every measured run + Spearman(CE, F_Δ) | 76 |
| [08_geometry_fingerprint.md](08_geometry_fingerprint.md) | Per-method geometry fingerprint + per-run appendix | 308 |
| [99_all_runs.md](99_all_runs.md) | Flat master table — every run, all metrics | 559 |

_152 additional exploratory/legacy runs (uio*, clora_*, scl2_*, a5_*, grid_*, …) are classified `misc` and appear only in [99_all_runs.md](99_all_runs.md)._
