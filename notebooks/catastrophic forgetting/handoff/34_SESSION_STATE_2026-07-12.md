# SESSION STATE / RESUME DOC — 2026-07-12 ~09:50 (Sunday morning, pre-session-close snapshot)

READ THIS FIRST on resume. Supersedes handoff/29. Campaign: magnitude-law (thesis: PEFT
retention is governed by update magnitude F_Δ; geometry acts through size/allocation;
LoRA+wd is the best efficient operating point). Timeline: PI keeps nodes ~until Monday
morning; artifact is the primary deliverable; paper.tex fixes are the main open task.

## THE ARTIFACT (primary deliverable — treat with care)
- Live URL (NEVER mint a new one): https://claude.ai/code/artifact/5c46636f-036a-4fae-919f-43be8e07639c
- **CRITICAL republish rule for a NEW session:** the Artifact tool mints a NEW URL unless you
  pass `url:` explicitly. From a fresh session: copy the repo source
  `paper/writing/artifact_status_report.html` to your scratchpad, edit there, and publish
  with `url: https://claude.ai/code/artifact/5c46636f-036a-4fae-919f-43be8e07639c`,
  favicon 📉 (keep it). Then `cp` back to the repo path and commit (they must stay identical).
  A stray duplicate URL (0354b1b2...) exists and now shows a "stale — moved" notice; leave it.
- Timestamp bump before publish: `sed -i "s|Last updated · <b>[^<]*</b>|Last updated · <b>$NOW</b>|"`.
- Current state (stamped 2026-07-12 09:35, commit 433d9ceb): 9 sections; §3 has ONE merged
  3-seed error-bar operating-point table (all 7 points, n=3) + Qwen subsection (CS+math
  per-adapter tables, single-seed disclosed) + CLoRA boundary box with the full faithful
  k-grid (F_Δ 0.61→0.34 / retention 22.6→25.0 monotone, k2048 accuracy collapse 68.5);
  §1 has the retention-battery table (5 benchmarks × both arms) + Qwen math law + broad
  composite r=−0.945; §6 CE table covers every method family (n=50, Spearman 0.976);
  whole file went through a verified readability pass (bullets, bolded takeaways).
- PI framing guardrails (HARD): constructive framing, no "geometry doesn't matter", no
  "everyone reports one LR", bold claims need 3 seeds, NO future/pending/roadmap content
  in the artifact.

## KEY RESULTS LEDGER (all verified against summary.json; artifact is the display copy)
- Llama CS law n=49: r=−0.858, slope −14.78; within-method −0.86..−0.97; Spearman −0.90;
  knee ≈0.37; past-knee residual SD ≈3.5 vs 0.6 below.
- Broad retention battery (every eval records BBH/MMLU-Pro/MMLU/ARC-C/TruthfulQA):
  [CORRECTED 2026-07-12, handoff/36 BLOCKER-2: the −0.945/−13.7 figure was the 5-task
  mean INCLUDING TruthfulQA; the true 4-task capability composite is r=−0.937 slope
  −16.96 — use that.] 4-task composite r=−0.937 slope −17.0; per-task CS: MMLU −0.93, ARC-C −0.93, MMLU-Pro
  −0.89, BBH −0.79; TruthfulQA flat −0.10 (control only; INVERTS +0.43 on math arm).
- Qwen CS n=49 r=−0.857 (identical settings); Qwen MATH n=42 non-degenerate r=−0.71
  slope −15.3 (excl. 1 collapsed lorawd lr1e3 cell, F_Δ 15.8/acc 0; SMOKE excluded);
  Qwen ceiling structure: base BBH ≈48, LoRA+wd sweep entirely at ceiling (F_Δ .04–.12,
  BBH 46.8–48.0, GSM8K to 66.6). Slope family: −13.9 (Llama math), −14.78 (Llama CS),
  −15.3 (Qwen math), −14.65 (CLoRA Table 4 external, r=−0.9805 — 10-row extraction +
  verification in paper/writing/data/clora_table4_extracted.md).
- 3-seed CS operating points (ALL 7): LoRA+wd 81.80±0.16 / 25.93±0.42 / F_Δ .38–.41;
  SC-LoRA 80.61±0.41 / 24.60±1.85 / F_Δ bimodal .28–.56 (retention tracks F_Δ);
  LoRA 79.08±0.11 / 23.81±0.58; LoRA-Null 78.93±0.12 / 22.14±1.32 (past-knee
  heteroscedasticity); CLoRA 78.41±0.11 / 21.59±0.48; DoRA 74.29±8.65 / 25.20±0.33;
  MiLoRA 57.69±22.67 / 24.20±0.48. Pattern: accuracy is the seed-fragile axis; retention
  moves only when F_Δ moves.
- Math headline 3-seed: GSM8K 66.79±0.79 / BBH 33.57±1.04 (all seeds > published 64.6).
- CE-to-base: n=50 scored, Spearman(CE,F_Δ)=0.976; per-method rows in artifact §6;
  MiLoRA 2.54 retro-dicted (2.48) by power-law fit; ce chunks 2-8 + chunk_new queued.
- CLoRA faithful CS k-grid (NEW, overnight): k128/256/512/1024/2048 @ lr3e4 monotone
  along law; PiSSA CS collapse anchor: ret 10.59 @ F_Δ 1.41.

## OPEN WORK (priority order)
1. **paper.tex audit fixes INCOMPLETE.** The claims audit (paper/writing/
   claims_coverage_audit_sat.md) found 5 blockers B1–B5 + W1–W5 + N1–N5. An author agent
   started (commit 042c37db = PARTIAL, B1/Qwen only, no fix log) and died at session close.
   Re-run the author task: full brief incl. supervisor adjudications is reproducible from
   the audit file + this doc's Key Results Ledger. B4 adjudicated: artifact number stands
   (see clora_table4_extracted.md). PI said "leave the paper for now" on 2026-07-11 —
   ASK before resuming paper work.
2. **frm SC-LoRA / LoRA-Null math rows** (frm_sclora_lr1e4, frm_lora_null_lr1e4) — in
   flight/queued on A; when landed + CE-scored, add rows to artifact §6 table.
3. **Queues draining:** A master_dispatch ~49 pending (lorawd wd-columns remainder, b4
   retries, ce chunks, frc remainder incl. 2 new frc_lorawdr16_* param-matched controls);
   B frcB.txt ~26 pending (reservoir competitor columns). Watchdogs auto-heal; check
   stale locks each visit (audit: lock + no summary + no live proc ⇒ rm lock).
4. **Monday bonus-hours menu** (handoff/33_FREEZE_PLAN_SAT.md §c): top items = CLoRA
   boundary error bars (frc_clora_k1024/k2048 s43/s44), instrumented peak-memory (needs
   ~5-line patch logging torch.cuda.max_memory_allocated), b4 confound extension, base
   no-FT broad eval (eval_one_gpu has no base-only mode yet — small patch).
5. Registry freeze + final artifact pass when PI calls it.

## INFRA (all hardened this weekend — the fix stack matters)
- Node A (this host): auto_dispatch.py --jobs jobs/master_dispatch.txt --tag disp
  --hf_offline 1 (setsid). Node B (ssh ubuntu@d002, repo at same path): TWO dispatchers —
  jobs/frcB.txt --tag frcB and jobs/frepro4_qwen_B_keep.txt --tag qwenB (qwen done, will
  exit) — launched via `bash /tmp/launch_frcB.sh` / `/tmp/launch_qwenB.sh` (write script,
  run with bash: /tmp is noexec on B).
- Dispatcher hardening (auto_dispatch.py, deployed BOTH nodes): flock single-instance
  guard per jobs file; bounded failure-requeue (rc≠0 + no summary → clear lock, ≤2
  retries); gpus_locked_by_live_runs() gap guard (lock records gpu; live run's gpu is
  busy even if nvidia-smi shows empty — kills the orphan train→eval race);
  --hf_offline 1 (HF_HUB_OFFLINE — killed the BBH-init hang class; everything cached).
- gpu_watchdog.sh (both nodes, 10-min setsid loops): idle-heal + hung-proc heal (util<5%
  ×2 checks → kill) + stale_lock_gc (lock >15min old + no summary + no proc → rm).
- Sync (A): sync_d002.sh loop (30-min, qw* only) AND /home/ubuntu/nodeB_syncback.sh
  (hourly, ALL of B's results/*/summary.json+run_config.json, --skip-old-files);
  results_book_loop (30-min, results_book/ pages + push); evacuate_qwen_adapters.sh.
- B disk: HF cache lives at /data/hf_cache/huggingface (symlink from ~/.cache/huggingface;
  /home is small). Llama-2 weights ARE on B. /scratch/cf_models exists on both.
- Third hang variant to know: proc at 100% util but log frozen >1h (spinning wedge from
  an old GPU-sharing incident) — watchdog can't see it; only manual log-freshness audit
  catches it. Audit: log mtime vs now for every running job.

## OPERATIONAL RULES (hard-won; do not relearn)
- Long GPU jobs: ONLY setsid detached, NEVER Bash run_in_background. Kill and relaunch =
  SEPARATE Bash calls (compound → exit 144). venv: /home/guy/UIOrthoLoRA/.venv/bin/python.
- rsync absent both nodes → tar-over-ssh (or cat file | ssh 'cat > path' for single files;
  scp chokes on the space in the path).
- Session-only crons DIE at session close AND at session continuation. On resume: CronList
  to verify, recreate the 2-hourly check (prompt text in handoff/34_NEXT_AGENT_PROMPT.md).
- Never archive/move a .py that anything imports (norm_trace.py incident). Never suspend
  (SIGSTOP) GPU-sharing procs. git check-ignore new file types before assuming tracked.
- Sync gold data FIRST in every check; commit+push every milestone.

## FILES MAP
- Artifact source of truth: paper/writing/artifact_status_report.html (repo) — publish
  via scratchpad copy + url param as above.
- Audits: paper/writing/claims_coverage_audit_sat.md (claims), handoff/33_FREEZE_PLAN_SAT.md
  (queue plan + bonus menu + CE chunk verification).
- Data provenance: paper/writing/data/clora_table4_extracted.md, key_numbers.md.
- Results book (PI's browsable tables): results_book/ (auto-regenerates).
- Registries: results/campaign_summary.jsonl, train_registry.jsonl; CE:
  results/forgetting*.jsonl (union, dedup by run_name).
- 557 summaries in results/ as of this snapshot; everything through 042c37db pushed.
