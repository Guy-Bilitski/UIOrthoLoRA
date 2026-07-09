# CAMPAIGN STATUS & HANDOFF — as of 2026-06-29

Authoritative current-state doc. **Read this first.** (Older `handoff/00..09` are historical; the
2026-06-10 content previously here was the obsolete UIOrthoLoRA era.) Working dir for everything
below: `notebooks/catastrophic forgetting/`. Branch: `ortho_new`.

---

## 1. The project in one paragraph
We fine-tune **Llama-2-7B** (and will replicate on **Qwen2.5-7B**) with 7 LoRA-family adapters and
measure the **adaptation vs retention** (catastrophic-forgetting) tradeoff. **Central claim = THE LAW:**
*retention is governed by the weight-update magnitude ‖ΔW‖_F, not by the adapter method* — adapters
fall on one retention-vs-‖ΔW‖ curve, so the simplest magnitude control (LoRA + weight decay) matches
elaborate structured inits. We deliberately do **not** headline "LoRA is the best method."

## 2. Central findings (seed 42, Llama-2 CS, n=49 = 7 methods × 7 LRs)
- **Magnitude law:** retention vs ‖ΔW‖_F is tight (pooled r≈−0.87, R²≈0.75; r=−0.93/R²=0.86 on the 5
  well-behaved methods). Use **‖ΔW‖_F (fdelta_token_weighted)** as the magnitude axis — NOT dw_sv_max
  (confounded: CorDA's spiky spectrum inflates σmax).
- **Refinement (honest):** the law is NOT perfectly method-free. ANCOVA: per-method offsets lift R²
  0.75→0.88 (p<0.001) — driven entirely by **CorDA (−3.0pp) and SC-LoRA (−3.3pp), which forget MORE
  than their ‖ΔW‖ budget predicts** (the data-aware inits). The other 5 (LoRA, LoRA+wd, MiLoRA, CLoRA,
  DoRA) straddle the curve → fair among them.
- **Mechanism:** data-aware inits transmit the same LR into a larger ‖ΔW‖ (fig7) → why they fall off.
- **LR is a proxy:** retention~LR R²=0.35 vs retention~‖ΔW‖ R²=0.75. LR matters only via ‖ΔW‖.
- **Budget:** ‖ΔW‖ buys adaptation (+21pp/decade) and costs retention (−16pp/decade); sweet spot
  ‖ΔW‖_F≈0.31–0.62. Per-method operating points in `paper/figs_v2/op_points.png`.
- **Per-benchmark:** MMLU forgets fastest (−23pp/dec); TruthfulQA ~immune.
- **CAVEAT on all the above:** single seed (s42), n=7/method, ranks NOT matched across methods
  (LoRA/DoRA/CorDA r16; MiLoRA/SC-LoRA r32; CLoRA k1024). Fine for the LAW; the CorDA/SC-LoRA off-curve
  verdict needs seed replication + a rank-matched control before it's airtight.

## 3. How to operate (concrete)
- **Check progress:** `ps -eo cmd | grep -E '[g]pu_pool.py'` (running pools);
  `for p in lrsw lrswm qwsw qwswm mtxm; do echo "$p: $(ls results/${p}_*/summary.json 2>/dev/null|wc -l)"; done`
- **Resume the campaign after ANY interruption (HW reclaimed, crash):**
  `cd "notebooks/catastrophic forgetting" && SKIP_VALIDATION=1 nohup bash run_all_experiments.sh > logs/orchestrator.log 2>&1 &`
  (no PID arg). RESUMABLE: `make_campaign_jobs.py` regenerates each phase's REMAINING cells (skips
  completed summaries), so it never re-runs or duplicates done work.
- **Salvage** adapters that trained but died at eval (recover compute, no retrain):
  `python make_salvage_evals.py` → `gpu_pool.py --gpus 8 --tag salvage --jobs jobs/salvage_evals.txt`.
  (`relaunch_clean.sh` chains salvage → orchestrator.)
- **Regenerate ALL figures + tables from live data:** `.venv/bin/python paper_figs_v2.py` (CANONICAL).
- **venv:** `/home/guy/UIOrthoLoRA/.venv/bin/python`. **No wide internet** in this env — give fetch
  tasks to the user (they relay to an internet-capable agent).

## 4. The campaign (single-seed-first 2×2)
Full grid = 2 models × 2 domains × 8 arms × 9 LRs, but **ONE seed (42) first** (~288 runs, ~9 days);
add seeds 43/44 later only where it matters. `run_all_experiments.sh` runs phases in priority order:
**Llama-2 CS → Qwen CS → Llama-2 math → Qwen math**, smoke-gating each new pipeline. Arms
(`<prefix>_<arm>_lr<tok>_s<seed>`): lora_r16, lorawd_wd0p3, dora_r16, clora_k1024, corda_r16(KPA),
milora_r32, sclora_r32(β0.5), lora_null_r16. Prefixes: `lrsw`=L2-CS, `qwsw`=Qwen-CS, `lrswm`=L2-math,
`qwswm`=Qwen-math (`mtx_/mtxm_`=older 3-seed matrices). Model also recoverable from `base_model` in
each `results/<run>/summary.json`. LR grid = 7 LRs: 2e-5,5e-5,1e-4,2e-4,3e-4,5e-4,1e-3 (2e-3/5e-3 dropped — they diverge).

**LIVE STATE 2026-06-29 ~10:49:** L2-CS **49/49 COMPLETE**; L2-math 7 + mtxm 1 (sparse); Qwen 0
(not started). `salvage2` pool running; `relaunch_clean.sh` hands off to `run_all_experiments.sh`
(CS gap = lora_null → Qwen CS → math). Qwen2.5-7B prefetched & module-compatible.

## 5. Figures & tables
- **CANONICAL generator:** `paper_figs_v2.py` → `paper/figs_v2/` and writes `paper/table_main_{cs,math}.tex`.
  Figures: `fig0_hero` (the law — the central figure), `fig1_magnitude_law` (why ‖ΔW‖_F is the fair axis),
  `fig2_fairness_residuals` (5-on-curve / CorDA+SC-LoRA below), `fig3_pareto`, `fig4_lr_sensitivity`,
  `fig5_per_benchmark`, `fig6_supporting_structure`, `fig7_lr_is_the_proxy`, `fig8_magnitude_budget`, `op_points`.
- **DEPRECATED:** old `paper_assets.py` pooled the 3-seed matrix WITH the sweep → inconsistent,
  collapse-inflated main table (the bogus `LoRA+wd 71.6±14.7`). Its table output is redirected to
  `*_LEGACY`; do NOT use it for the main table. (It still has useful LR-fairness/forensics analysis.)
- Figures are >2000px; downscale with PIL into `figs_v2/preview/` to view.

## 6. Method ports & audit (Phase 0 — all faithful)
CorDA(KPA), MiLoRA(minor-init), CLoRA(loss-term), SC-LoRA(output-side balanced cov), LoRA-Null
(null-space, `lora_null_init.py`, unit-tested) all verified faithful vs references. Residual-init
methods (CorDA/MiLoRA/SC-LoRA/LoRA-Null) need the **rank-2r W0-relative conversion at save**
(`residual_save.py`) or eval explodes — wired in `train_cs.py`; 0-step gate = `validate_residual_zero_step.py`
(PASSED). CorDA's large ‖ΔW‖ is GENUINE KPA behavior (C_inv un-whitening), not a bug.

## 7. GOTCHAS (these bit us — do not repeat)
- **ONE GPU scheduler at a time.** Two `gpu_pool.py` each claim all 8 GPUs → 2 runs/GPU → OOM. We lost
  ~45h to two collisions. Never launch a second pool while one runs; route everything through `run_all_experiments.sh`.
- **`pgrep -f "<pat>"` SELF-MATCHES** when the pattern is in your own command line. Use bracket trick
  `grep '[m]txm_'` or kill by explicit PID. Bit us 3×.
- **Divergence at extreme LR:** LR 2e-3/5e-3 → NaN weights → eval crashes (`CUDA assert: probability
  tensor contains inf/nan`) → pool RETRIES forever (one job 13×). Dropped those LRs. Eval is NOT robust
  to NaN adapters — any diverged adapter crashes the whole job.
- **Math eval is heavy:** math-tuned models don't emit EOS → `--ret_max_gen 512` retention eval
  crawls/OOMs (~12h/job). Use **256 for math** (already set for gsm8k in make_campaign_jobs.py).
- **Crashes usually kill EVAL, not TRAINING** → salvageable via `make_salvage_evals.py` (eval-only rerun).

## 8. OPEN ITEMS / next steps
1. **Finish the 2×2** (Qwen CS/math + the remaining math arms) — running; ~5 days clean.
2. **Lock the off-curve claim:** seeds 43/44 on **CorDA & SC-LoRA** + a **rank-matched control**
   (single seed + mismatched ranks is the only real certainty gap). DEFERRED until after the 2×2.
3. Fairness experiment (give every method the wd knob) — DEFERRED; framing around the law makes it optional.
4. Base-ceiling calibration missing for MMLU/ARC/TruthfulQA (only BBH/MMLU-Pro have base evals).
5. Minor port nuances flagged in the init files: SC-LoRA per-sample norm (|max(Y)| vs max|Y|),
   LoRA-Null null_dim default=r — confirm vs raw repos only if challenged.

## 9. Key files
`run_all_experiments.sh` (orchestrator, resumable) · `make_campaign_jobs.py` (per-phase remaining
jobs — **edit `SEEDS`/`LRS`/`ARMS` here**) · `make_salvage_evals.py` · `relaunch_clean.sh` ·
`validate_residual_zero_step.py` · `train_cs.py` (trainer; every method is `--method lora` + a flag,
e.g. `--corda 1` / `--milora 1` / `--sclora 1 --sclora_beta` / `--lora_null 1` / `--weight_decay`) ·
`eval_one_gpu.py` (`--adapt_task cs|gsm8k`, `--base_model`, `--ret_suite broad`, `--ret_max_gen`) ·
`paper_figs_v2.py` (CANONICAL figures/tables) · init modules: `corda_init.py` `milora_init.py`
`sclora_init.py` `lora_null_init.py` `residual_save.py`. Results: `results/campaign_summary.jsonl`
(flat, analysis-ready) + `results/<run>/summary.json` (keys: `headline` flat metrics, `per_dataset`
8 CS tasks, `fdelta`{fdelta_token_weighted, dw_sv_mean, dw_sv_max}). Memory (Claude): see
`port-audit-and-lr-sweep-plan` + `matrix-campaign-results`.
