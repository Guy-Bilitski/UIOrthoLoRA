# Qwen intruder campaign — HANDOFF for a fresh agent (written 2026-09-16 ~07:55 UTC)

Read this first, then the 2026-09-14..16 entries of `handoff/TIERA_RUN_LOG.md` (every result and
decision, timestamped). Working dir: `notebooks/catastrophic forgetting/` on branch `ortho_new`.
Box: single H200 (143 GB), venv `/home/kfir/guyb/UIOrthoLoRA/.venv`, models `/home/kfir/cf_models`,
results `results/`, logs `logs/`. Guy Bilitski (PI-side user) is the person you report to.

## 1. What is done
* All 7 Qwen2.5-7B configurations TRAINED under `run_guarded.py` (NaN-step skip guard). Every run
  skipped the same 129/31,956 steps (0.40 %, batch-locked, design- and co-tenancy-independent).
  Seeds 43. Runs: `tia1_qwsw_{lorawd_wd0p3_lr1e4,milora_lr1e4,clora_k1024_lr2e4,lora_r16_lr5e5,
  loranull_r16_lr2e4,sclora_lr2e5,clora_k1024_lr3e4}_s43`. The last is the extra Llama-matched cell.
* Qwen base (no adapter) on the proxy protocol: `results/base_qwen25-7b/summary.json`, retention 48.00.
* Arms A-D (source, intruders deleted, uniform shrink, B rescaled) evaluated for MiLoRA, LoRA+wd,
  CLoRA 2e-4, CLoRA 3e-4, LoRA-Null; E/Ep/F for MiLoRA, LoRA+wd; E, Ep for CLoRA 2e-4.
  Summary per arm: `results/<run><suffix>/summary.json` (headline.cs_avg = task, retention_mean,
  fdelta = F_delta). Suffixes: `__rl50` (A), `__k10allablB/C/D/E/Ep/F1`.
* Geometry per run: `results/intruder/<run>.json` (aggregate.total_intruders_k10_baseAll_t0.5 /
  (n_matrices*10) = slot fraction; mean_energy_share_baseAll_t0.5). Norm ratios: `logs/verify_<run>.log`.
* GitHub: pushed through commit 0d19c06b (ortho_new). Overleaf (project 6a46eb1b48498302a1ab34db,
  branch **main**): table + text revisions pushed (latest 39cd223); revisions are marked \new{}
  (blue) / \cut{} (gray) / newpart per Guy's review convention. Keep that convention.

## 2. What is still running (no training left; evaluation only)
Queue `jobs/qwen_dyn_queue.txt` (tab-separated `<run>\t<cmd>`, `STOP` at the end is already queued).
Order of the ~19 pending arms: CLoRA 2e-4 F1 · LoRA-Null E, Ep, F1 · CLoRA 3e-4 E, Ep, F1 ·
SC-LoRA rl50, Ep · LoRA r16 rl50, B, C, D, E, Ep, F1 · SC-LoRA B, C, D, E, F1 (degenerate: only 3
intruders; these five measure noise — Guy may say to drop them; deleting their lines from the queue
is safe). ~85 min per arm, 2 at a time → informative arms done ~16:30 UTC, all ~20:00 UTC.

Daemons (all `setsid`, survive any session): `bash qwen_watchdog.sh` (restarts pool/campaign,
status every 30 min in `logs/qwen_watchdog.log`, writes `results/FINAL_TABLE_qwen.{md,csv}` via
`paper_table.py` when the pool exits after STOP), `python gpu_pool_dyn.py` (2-wide eval pool; log
`logs/tQd_pool.log`, per-job `logs/tQd_<n>.log`, state `logs/tQd_state.json`, live knobs
`logs/dynpool_control.json` = {"slots_training":1,"slots_idle":2,"min_free_mb":40000}),
`bash qwen_campaign3.sh` (only waiting for the pool to drain).

Health check (use ^-anchored pgrep; unanchored patterns match your own shell):
```
pgrep -fc '^/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python gpu_pool_dyn.py'   # 1
pgrep -fc '^bash qwen_watchdog.sh'; pgrep -fc '^bash qwen_campaign3.sh'      # 1, 1
pgrep -fc '^/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python eval_one_gpu.py'   # 2 while queue non-empty
grep -cE 'rc=[1-9]' logs/tQd_pool.log; grep -liE 'out of memory' logs/tQd_*.log   # 0, none
```
If the pool died: `setsid python gpu_pool_dyn.py --queue jobs/qwen_dyn_queue.txt --tag tQd --slots 2
--slots_training 1 --min_free_mb 40000 --stagger 150 --lock logs/gpu_slot.lock >> logs/tQd_pool.log
2>&1 < /dev/null &` (singleton lock; adopts running chains). A job that failed twice is marked
"FAILED (no more retries)" in the pool log — re-enqueue by appending its exact line again.
NEVER `pkill -f <pattern>` from a shell whose own command line contains the pattern (it killed the
handover shell once). Session-bound watchers (CronCreate every 30 min, Monitor tail -F on the three
logs with `stdbuf -oL`) die with the session: re-create them; background Bash watchers get killed
by the harness on spurious "low memory" claims — use Monitor + cron instead.

## 3. When the queue drains (watchdog logs "queue drained")
1. `python paper_table.py` (full arm table + base footnote) — sanity-read every new cell.
2. `python make_table_intruder_tex.py` → `paper/table_intruder.tex`; replace `\dots` cells only
   by re-running the script (never hand-edit). SC-LoRA degenerate arms: the table prints them if
   evaluated; Guy decides whether to keep them (appendix already explains the degeneracy).
3. Push to Overleaf via the git bridge (token `olp_...` inside `~/.claude.json`; clone
   `https://git:<token>@git.overleaf.com/6a46eb1b48498302a1ab34db`, copy `paper/table_intruder.tex`
   to `tables/`, commit as Guy, `git push origin HEAD:main`). A clone may already exist at
   `<scratchpad>/ovl`; otherwise re-clone. Check the remote `main.tex` equals your local copy
   before overwriting it (diff), then apply text edits with \new/\cut.
4. `git add results/tia1_qwsw_* paper/table_intruder.tex handoff/TIERA_RUN_LOG.md; git commit; git
   pull --rebase origin ortho_new; git push origin ortho_new` (others push iclr commits to the
   same branch; rebase first).
5. Final run-log entry; then the paper edits for the remaining rows (LoRA r16, SC-LoRA) if their
   numbers change any statement in Section "Removing intruder dimensions..." / Appendix.

## 4. The scientific state (for writing)
* Intruder load is a function of update size on both architectures: Qwen slot share 0.2 %
  (SC-LoRA) → 3 (LoRA r16) → 6 (MiLoRA) → 12 (LoRA+wd) → 38 (CLoRA 2e-4) → 48 (CLoRA 3e-4) → 62 %
  (LoRA-Null); Llama 57-94 %. Qwen frontier sources forget 0-4 of 48 retention points; CLoRA 3e-4
  forgets 8.1; Llama sources 4-7 of ~29.
* 10/10 configurations (5 Llama, 5 Qwen with a real deletion): uniform shrink (C) retains >=
  intruder deletion (B) at equal norm; D (delete + restore norm) is worse than A, growing with
  intruder energy share (F +2/+9/+21/+17/+48 % for MiLoRA/LoRA+wd/CLoRA 2e-4/CLoRA 3e-4/LoRA-Null).
* Task collapse under deletion needs the intruders to dominate the top spectrum: -0.5/-1.0/-1.3 pp
  at 6/12/38 % slots, -2.4 at 48 %, collapse at 62 % (LoRA-Null 86.6 → 32.9; D 3.5) and at every
  Llama row (57-94 %). Below-chance = destroyed output, not a removed skill.
* Non-intruder deletion at equal norm (E, Ep) never retains less than B; on Qwen MiLoRA the
  count-matched deletion of aligned directions is the only arm that hurts the task (-6.6 pp).
* Retention proxy (50 items/subtask) reads ~8 pts above the pool's full battery on Qwen; only
  within-table contrasts and the base zero point (48.00) are used. F_delta = token-weighted
  ||dW x||/||x|| over adapted matrices (uio_inprocess.fdelta_inprocess), a perturbation magnitude:
  arms C/D move it by construction; B/E/F are the informative cells.
* Manuscript sections already revised (blue/gray): sec:results:intruder, intro contribution (iii),
  Conclusion 3rd result, Limitations item 1, Appendix app:intruder (guarded training, E/Ep
  availability, validity boundary on Qwen, Qwen base). Guy's PI has a prompt to continue the
  paper work (in the chat log of 2026-09-16 ~05:50); coordinate via Guy.

## 5. Open decisions for Guy
* Drop the five degenerate SC-LoRA arms (saves ~3.5 h)? Default: they run last.
* Off-node evacuation target (adapters are only under /home/kfir/tierA_evac on this disk).
* Whether to keep the seven-row Qwen block or fold CLoRA 3e-4 into an appendix table.

> **SUPERSEDED 2026-09-17.** The campaign described below is complete. See
> `QWEN_CAMPAIGN_COMPLETE_2026-09-17.md` for the closing state, results and open items.
