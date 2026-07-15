# Data protocol notes (for the methods section / A* submission)

Living record of data-collection protocol decisions and known data-quality caveats. Updated 2026-07-15.

## Per-adapter record (the publication matrix)
qwen/llama × math/cs × learning-rate × seed, with per cell:
- **adaptation** + **retention** (per-task: bbh, mmlu, mmlu_pro, arc_c, truthfulqa; plus retention_broad/bbh) — `results/<run>/summary.json`
- **magnitude** — `dw_sv_max`, `dw_sv_mean`, `fdelta` (token-weighted ‖dW‖F) in `summary.json`
- **geometry** — `results/<run>/geo.json` (per-adapter aggregate: e_top/e_bot, ein_top/bot, amp_top, stable_rank, eff_rank, spec) + per-layer `results/geo_drift/permatrix/<run>.jsonl`
- **CE-shift** — `results/<run>/forgetting.json` (forgetting_ce, forgetting_kl, base_entropy) — MiLoRA/Kalajdzievski CE-to-base on WikiText-103 test
- **provenance** — `git_commit` in summary; full hyperparameters in `results/<run>/config.json` (snapshot of the adapter's run_config.json) and recoverable from the git-committed shard commands

## CE-shift protocol (decision 2026-07-15: "full-test forward + document split")
- Target protocol = **full WikiText-103 test** (`--max_blocks 0`, ~291–330 blocks depending on tokenization), matching MiLoRA Table 8.
- **Split, by necessity:** adapters are ephemeral on `/scratch/cf_models`. Cells scored before the switch whose adapters were already evacuated (~625, mostly the seed-s42 prior phase) retain their **40-block** slice CE and cannot be re-scored without retraining. All cells scored after the switch (remaining ~478 training cells + 124 Qwen s42 recovery + any present adapter) use **full test**.
- **Measured 40-block vs full-test delta ≈ 2%** (e.g. qwsw_corda_r16_lr5e5_s44: 2.0136 vs 2.0602). Rankings preserved; absolute values differ ~2%. Report the protocol per cell (n_blocks is stored in each forgetting.json) and note the delta.

## Quarantine (excluded from analysis, not deleted)
`flag_diverged.py` → `results/quarantine_diverged.txt`. Criteria: NaN/inf magnitude, exploded magnitude (>1000) or fdelta (>50), collapsed retention (<3), or NaN CE. These are REAL training divergences concentrated in high-LR arms (lr1e3 ~25%, 2e3, 7e4), not pipeline errors.

## Sync / integrity
Per-run files (summary/geo/forgetting/config) sync to d001 + GitHub (`ortho_new`) via the 15s collect loop; `derive_supervisor.sh` (d001) keeps per-node `derive_loop.sh` alive. Data hygiene verified 2026-07-15: value ranges valid, geometry invariants hold, no CE↔retention sanity violations.
