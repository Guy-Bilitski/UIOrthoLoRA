# Overheads — the cost of geometry (adjudication)

Wall-clock: live medians from `results/train_registry.jsonl` (lrsw_/qwsw_ runs,
runtime > 1000 s), normalized to LoRA. Init tax + CLoRA per-k memory constant are
[EXTERNAL] measured values (INTERESTING_INSIGHTS.md section 7, fig_efficiency.py) —
cited, not recomputed. init_class: 0 = free, 1 = weight-only SVD/eigh,
2 = needs a calibration forward pass, 3 = heavy multi-pass. deploy_delta_rank_factor:
residual-init methods (MiLoRA/PiSSA/CorDA) modify base weights at init, so the
deployable checkpoint delta is rank-2r (2x adapter bytes) unless the full base is
reshipped. Script: `05_overheads.py`.

| Method | train x (Llama) | train x (Qwen) | params (M) | extra resident GB | init tax | deploy delta |
|---|---|---|---|---|---|---|
| LoRA | 1.0 | 1.0 | 28.0 | 0 | none | rank-r |
| LoRA+wd | 1.0 | 1.0 | 56.1 | 0 | none (free AdamW flag) | rank-r |
| CLoRA | 1.17 | 1.17 | 56.1 | 3.34 | k x d covariance/eigh on base weights (fast); frozen-P build | rank-r |
| MiLoRA | 1.0 | 1.0 | 56.1 | 0 | 160 base-weight SVDs (no forwards) | rank-2r |
| LoRA-Null | 1.0 | 1.0 | 28.0 | 0 | 256 calibration forwards + eigh | rank-r |
| SC-LoRA | 0.99 | 1.0 | 56.1 | 0 | 512 calibration forwards + eigh | rank-r |
| DoRA | 2.15 | 2.15 | 28.9 | 0 | none | rank-r |
| PiSSA | nan | nan | nan | 0 | 160 base-weight SVDs (no forwards) | rank-2r |
| CorDA | 1.0 | 1.0 | 28.0 | 0 | 256 calibration forwards + inv/SVD | rank-2r |
| CorDA++ | nan | nan | nan | 0 | 1280 forwards + 5x inv/SVD (~3.5e16 FLOPs, ~22.5 GB transient) | rank-2r |

CLoRA frozen-P memory vs k: k128=0.42 GB, k256=0.84 GB, k512=1.67 GB, k1024=3.34 GB, k2048=6.69 GB (sweeps use k1024 = 3.34 GB; the frc k2048 boundary point pays 6.68 GB).