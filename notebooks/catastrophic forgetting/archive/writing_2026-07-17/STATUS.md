# CAMPAIGN STATUS — as of 2026-07-09

> **LIVE OPS 2026-07-15 (~08:30Z):** 30-node fleet resumed. Fixed offline gsm8k eval failure (pre-cached `openai/gsm8k` fleet-wide) + added `train_cs.py` skip-retrain guard (commit `bbf5eb8f`) so ~270 overnight-banked adapters re-evaluate in minutes. All 30 dispatchers up; 714 planned cells completing + syncing. Full incident writeup: **`handoff/39_FLEET_EVAL_RECOVERY_2026-07-15.md`**.

Live snapshot of the magnitude-law campaign. For onboarding read **`WORKDIR_ALIGNMENT.md`**; for the plan
read **`handoff/20_FAITHFUL_REPRO_SPEC.md`** + **`handoff/25`–`28`**; for numbers read
**`paper/writing/data/key_numbers.md`**. (The prior 2026-06-29 2×2 snapshot is in git history / `handoff/13`.)

---

## 1. The project in one paragraph
We fine-tune **Llama-2-7B** (replicated on **Qwen2.5-7B**) with LoRA-family adapters and measure the
**adaptation vs retention** (catastrophic-forgetting) tradeoff. **THE LAW:** retention is governed by the
weight-update magnitude **F_Δ**, not the adapter method — every adapter falls on one retention-vs-F_Δ curve,
so plain **LoRA + weight decay** matches elaborate structured inits. We deliberately do NOT headline "LoRA is
the best method"; the durable claim is "no geometry beats magnitude control on retention, at zero extra cost."

## 2. Current campaign — faithful CLoRA reproduction, two nodes
Every adapter goes through one pipeline (`train_cs.py` → `eval_one_gpu.py`) on the CLoRA recipe: CS
(Commonsense-170K, r32/α64, prefix `frc_`) and math (MetaMathQA-395K, r64/α128, prefix `frm_`), LR swept as
a controlled variable, F_Δ/‖ΔW‖ measured on every run. Deadline ~Sun; a **two-node 16×B200 fleet** meets
~883 GPU-h of demand (Node A = this host, owns all adapters + analysis; Node B = second host, trains fresh +
syncs summary JSON — see `handoff/28`).

**LIVE on Node A (verify with `ps aux | grep -E 'gpu_pool|auto_dispatch'`):**
- `gpu_pool.py --gpu_ids 0,1,2,3,5 --tag frepro4    --jobs jobs/frepro4_main5.txt`  (CLoRA k-grid + frc_lora_l2 + competitor math)
- `gpu_pool.py --gpu_ids 4       --tag frepro4b4    --jobs jobs/frepro4_b4.txt`      (SC-LoRA/CorDA++ boundary cells)
- `gpu_pool.py --gpu_ids 6,7     --tag frepro4hs    --jobs jobs/frepro4_headline_math2.txt`
- `gpu_pool.py --gpu_ids 7       --tag frepro4inj   --jobs jobs/frepro4_inject.txt`  (2e-5 wd/α=r/SC-LoRA-faithful controls)
- `auto_dispatch.py --jobs jobs/master_dispatch.txt --gpus 0-7 --tag disp`            (self-refilling; absorbs GPUs as pools drain)

## 3. Progress
- **Faithful MATH (`frm_`): ~46/46 done** (+ method-row/β cells in flight); 48 `frm_*` result dirs on disk.
- **Faithful CS (`frc_`): landing now** — the 65-cell reservoir is the paper's spine (0 done at campaign
  start; first cells training/evaluating now).
- `results/campaign_summary.jsonl` ≈ 472 rows (whole project history; dedup rule = latest `evaluated_at`
  per `run_name`, see key_numbers §0). ~482 subdirs under `results/`.
- **Check progress:** `for p in frc frm; do echo "$p: $(ls -d results/${p}_*/ 2>/dev/null | wc -l)"; done`
  and tail `logs/<tag>_*.log`.

## 4. Analysis — DONE & validated (CPU / eval-only, no GPU contention)
- **Magnitude law:** CS pooled r=−0.86 (R²0.74) / on-curve −0.92; within every method −0.86…−0.97;
  **2nd model Qwen CS r=−0.88**; ceiling-robust (Spearman −0.896, saturating fit beats linear+quadratic on
  AIC+LOO-CV, below-ceiling slope −20.8). LR is a weaker proxy (R² 0.32 vs 0.74).
- **CLoRA Table 4 external replication:** r(log F_Δ, BBH) = −0.98, slope −14.7 vs our −14.8.
- **Geometry-drift verdict:** magnitude 1st-order, rank modest 2nd-order; **principal-direction 2nd-order
  axis TESTED & REJECTED** (outlier-driven); geometry = fingerprint/measurement tool (`handoff/27`).
- **CE-to-base:** validates vs MiLoRA Table 8; MiLoRA≈LoRA at matched magnitude (`forgetting_ce.py`).
- **Efficiency/memory:** LoRA+wd on the frontier at zero extra cost; DoRA 2.13×; CLoRA k-memory tax.
- **Metrology:** fdelta = CLoRA's F_Δ (not Frobenius) — corrected in key_numbers.

## 5. Running / queued
CLoRA faithful k-grid (high-k boundary verdict), `frc_lora_l2` (LoRA-L2 side-by-side), SC-LoRA
β0.9/eval-matched/r128 + CorDA++ α=r@2e-5 controls, 2e-5 wd cells (answer CorDA++'s "LoRA forgets at 2e-5"),
3-seed headlines (s43/s44), the Qwen block on Node B (`qwswm_lorawd_wd0p3` math LR-sweep first to fix the
math anti-replication), c2048 MATH-offset anchor, full CE-to-base batch on A.

## 6. Honest boundaries (state in the paper)
High-k CLoRA (k1024/k2048) beats LoRA+wd on CS — faithful verdict running; SC-LoRA −4pp below the law is
**provisional** (recipe-confounded); Qwen-math anti-replicates (high-LR cells unrun); single-seed primary
(3-seed running); CorDA withheld pending fair calibration; ranks not matched (frame the LAW, not a ranking).

## 7. Key files
`train_cs.py` (trainer) · `eval_one_gpu.py` (in-process evaluator) · `gpu_pool.py` / `auto_dispatch.py`
(schedulers) · `make_frepro_jobs.py` + `build_lean.py` (job generation) · init modules `corda_init.py`
`cordapp_init.py` `milora_init.py` `data_aware_init.py` `sclora_init.py` `lora_null_init.py`
`residual_save.py` · `math_eval.py` `bbh_metric_fix.py` · analysis `geo_drift_phase{1,2}.py`
`forgetting_ce.py` `retfix_retention_gate.py` · figures `paper_figs_v2.py` +
`paper/writing/make_figs_split_lora_null.py`. Superseded scripts/jobs are in `archive/` (see
`CLEANUP_MANIFEST.md`). Memory: `~/.claude/projects/-home-guy-UIOrthoLoRA/memory/`.
