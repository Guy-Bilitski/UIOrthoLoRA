> **SUPERSEDED 2026-08-28. Bring-up state from the first session. The campaign plan it describes (18-cell LR sweep) was replaced. Current spec: handoff/EXPERIMENT_FINAL.md**

# H200 session state — 2026-08-26 ~15:00 UTC (for Claude resume)

If resuming with full conversation context (`claude --continue`), this file is
redundant. If context is missing, read this + `handoff/H200_BOOTSTRAP.md` +
`handoff/TIER_A_SPEC_2026-08-23.md` and continue from "NEXT ACTIONS".

## Done (verified)
- venv at repo root `.venv`: torch 2.12.0+cu130, transformers 5.10.2, editable
  fork (peft 0.19.1 from src/peft), lm-eval 0.4.12. requirements-freeze.txt was
  never committed (gitignored) — env reconstructed from README pins.
- HF auth: Guy's token (user GuyBilitski181) at `/home/kfir/.cache/huggingface/token`.
  TRAP: stale rejected token (Orr-z) still sits at `/home/kfir/hf-home/token`
  and HF_HOME points there — every process needs
  `export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token` (+ HF_HUB_OFFLINE=0,
  HF_HUB_DISABLE_XET=1). Llama-2 + Qwen access confirmed with it.
- Data: repro/LLM-Adapters re-cloned from GitHub; metamathqa_100k.json rebuilt.
- Base-SVD stores: Qwen DONE (140 matrices). Llama running (logs/geo1_llama.log).
- `intruder_pass.py --selftest`: PASS 5/5.
- Job files regenerated with real paths: OUT_ROOT=/home/kfir/cf_models,
  EVAC_DEST=/home/kfir/tierA_evac (LOCAL — off-node target still open with Guy).
- SMOKE (cell 2, tia1_frc_lorawd_wd0p3_lr5e4_s43) running via
  `gpu_pool --tag smoke` since ~14:47 UTC: loss 2.73→1.09 @60 steps, 2.75 it/s,
  31,956 steps ≈ 3.2 h train, then eval+CE+evac in the same chain
  (logs/smoke_0.log, logs/smoke_pool.log).
- DETACHED WATCHDOG `auto_launch_tierA1.sh` (logs/auto_launch_tierA1.log):
  on smoke rc=0 + .evacuated marker it regenerates job files and launches the
  full Exp 1 queue (tag=tierA1) BY ITSELF; on failure writes logs/SMOKE_FAILED.flag
  and does nothing. Lock: logs/tierA1_launched.flag. DO NOT also launch tierA1
  manually — check the lock/log first.

## NEXT ACTIONS (in order)
1. When smoke eval lands: run intruder_pass on the smoke adapter
   (`.venv/bin/python intruder_pass.py --adapter /home/kfir/cf_models/<run> --base_model meta-llama/Llama-2-7b-hf`).
2. Queue launches itself (watchdog). After first TWO cells: report per-cell
   timing to Guy (train ~3.2 h ⇒ likely >3.5 h/cell with evals — re-check the
   7-day machine ask, spec fallback says keep all 18).
3. Run intruder_pass after EACH cell's eval (CPU, ~10 min); report R1 the
   moment the first 6 coverage cells have it (cells 2,11,6,15,9,18).
4. Watch cell 12 (Qwen lorawd 1e-3) for divergence → retrain once at 5e-4 as
   `..._lr5e4f_s43`, flag it, never silently substitute.
5. rsync `results/` (and ideally /home/kfir/tierA_evac) off-node regularly —
   dev box or push; that's the data the paper reads.
6. After Exp 1 drains: Exp 2 anchors (jobs/tierA_exp2_anchors.txt, tag=tierA2),
   then ladder/controls per H200_BOOTSTRAP step 6 (needs
   rescale_adapters_qwen.py written from rescale_adapters.py).
7. Standing rules: never modify the tested pipeline; seed 43; evacuation on
   every cell; don't reorder/thin the queue without telling Guy.
