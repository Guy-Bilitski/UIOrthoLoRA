# SESSION STATE / RESUME DOC — 2026-07-09 ~16:40 IDT (pre-compaction snapshot)

Read this FIRST on resume. Captures all in-flight work, file locations, and next actions.

## THE ARTIFACT (supervisor deliverable)
- URL (keep same): https://claude.ai/code/artifact/5c46636f-036a-4fae-919f-43be8e07639c
- Source file: /tmp/claude-1000/-home-guy-UIOrthoLoRA/72dbcb26-a4ce-47e1-aaa6-792e52121dea/scratchpad/status_report.html
  (repo backup copy: paper/writing/artifact_status_report.html). Republish via Artifact tool with the
  /tmp path to keep the URL.
- Sections (PI-mandated): 0 Adapters overview, 1 Magnitude, 2 Geometry, 3 CS adapt+ret, 4 Math adapt+ret,
  5 Compute+memory, 6 CE-forgetting, 7 Correctness. Rule: NO future/queued/buggy content (that goes to
  INTERESTING_INSIGHTS.md); constructive framing only.
- DONE this session: full restructure + clarity fixes (adapters section; geometry what-analyzed+column
  defs+CorDA caveat; CS safe-band defined; math grey-bar+retitle+BBH-why; B1 blocker removed = the
  lucky-seed 82.0 in §3; M1 magnitude figure CorDA-excluded + slope −14.78; M2 CE α=2r disclosure; M3/M4
  Qwen −0.88-commonsense / within-method −0.86..−0.97; minors). Timestamp "Last updated" in header.
- DONE 16:55: efficiency §5 rewrite — added init TIMES to table cells (MiLoRA/PiSSA "~minutes",
  SC-LoRA/LoRA-Null "~few min +22GB transient init only", CorDA++ "~1 GPU-h") + caption/paragraph now
  state CLoRA extra memory is ADDITIONAL to the ~55GB plain-LoRA footprint (not total). Republished (same URL).
- IN PROGRESS (spawned 17:10): FINAL "proper review round" the PI demanded — two agents on the FULL artifact:
  adversarial-critic a0668c33096d80385 (verify each original PI comment A–N addressed + framing guardrails +
  "good paper basis?"; writes paper/writing/artifact_review_round_final.md) and data-verifier a7bd29fbc97bb4353
  (recompute every artifact number vs summary.json; writes paper/writing/artifact_number_audit_final.md).
  ON RETURN: apply their fixes, bump timestamp, republish, commit+push.

## RUNNING AGENTS (check tasks/<id>.output on completion notification)
- af7d17f49e76fb8e6 — PROVISION d002 (deliver-from-A + netplan-internet option). On return: verify 8 GPUs
  busy on Qwen shard, sync-back working; then remove Qwen from A's master_dispatch (dedup) + add 7 new cells.
- a8bb67f7db0c02169 — paper.tex integration (fdelta→F_Δ, fold DONE results, citations, claims memo). Owns paper.tex.
- ad80d993beadfc13b — figures (magnitude saturating+ceiling, geometry 4-panel, efficiency, CE, cross-lit). Owns paper/writing/figures/.
- a77a441aeec62ece7 — repo cleanup + INTERESTING_INSIGHTS/README/STATUS + archive stale files.
- a1c8d3... (efficiency §5 artifact clarity) — check tasks/a1c8d30a606cb4ab4.output.
On each return: integrate, commit+push, update artifact if relevant.
PERMANENT agents in .claude/agents/: adapter-paper-expert, section-validator, research-planner,
adversarial-critic, data-verifier.

## NODE A (this host, "d001" delivery source): 8×B200
- auto_dispatch.py (detached, setsid) --jobs jobs/master_dispatch.txt = 109 cells (65 CS reservoir frc_ +
  44 Qwen). Self-refills any GPU freed by a draining pool; skips pool-owned GPUs; per-cell locks in
  results/dispatch_locks/. Log: logs/auto_dispatch.log.
- Live pools (draining): frepro4 (main5, math remainder+competitor math), frepro4b4 (9 eval-matched),
  frepro4inj (7 fleet cells), frepro4hs (headline seeds). DO NOT KILL (mid-flight adapters).
- Persistent monitor: bwpt13aq6 (rc!=0 / ALL DONE / zero-GPU alerts).
- Adapters: /scratch/cf_models (~147GB, 390+). NOT in git. Analyses done reading them → can prune to
  table-subset later (zip saves only ~7%; keep frc_/frm_ table cells).
- CRITICAL GOTCHA: launch long GPU jobs ONLY via setsid (detached), NEVER Bash run_in_background — the
  latter dies at session teardown (cost ~100 GPU-h on Jul 8). See catastrophic-forgetting-workdir memory.

## NODE B (d002 = test-gpu02): 8×B200 IDLE, being provisioned
- Access: `ssh ubuntu@d002` (key-based; d002 in /etc/hosts → 172.20.101.25). Works from main loop.
- Facts: 42GB /home, NO /scratch, bare (python3.12 only). PI added FW access → can netplan-replicate A's
  outbound (github/HF/PyPI). Hosts ONLY Qwen (disk); redirect output to /home/ubuntu/cf_models; stream-
  delete adapters after eval; hourly rsync results/*/summary.json B→A.
- Runs the Qwen shard (qwswm_lorawd_wd0p3 math LR-sweep FIRST = 2nd-model law replication).

## PLAN (handoff/28 = 16-GPU two-node; handoff/26 = single-node): 
- With 2 nodes: ~35% headroom; QWEN SAVED (was sacrificial). CS spine (65 frc_, 0 done) split A+B = the
  bottleneck/paper spine. Only 7 genuinely-NEW cells: frc_lorawd_wd0p3_lr5e4_c256 {s43,s44},
  frc_lorawd_wd0_lr3e4_c256 {s43,s44}, frc_milora_lr3e4_c256 {s43,s44}, frm_lorawd_wd0p3_lr2e4_c2048_s42.
- COORDINATION when B online: remove 57 Qwen lines from A's master_dispatch (avoid dup across non-shared
  FS); add the 7 new cells (dispatcher reads file once at launch → restart dispatcher or add via new pool).

## ANALYSES DONE + VALIDATED (do not recompute):
- Magnitude law: r=−0.86 canonical (n=49 CS), within every adapter −0.86..−0.97, 2nd model Qwen −0.88
  (commonsense), ceiling-robust (Spearman −0.90, saturating>linear/quad, below-ceiling slope −21,
  partial r −0.87, permutation p<5e-5), CLoRA-Table-4 external replication r=−0.98 slope −14.7 vs −14.8.
- fdelta = F_Δ (CLoRA Eq3, ‖ΔWx‖/‖x‖), NOT Frobenius (key_numbers §0 fixed). dw_sv_* = their spectral.
- Geometry (handoff/27, results/geo_drift/): magnitude 1st / rank 2nd (modest) / geometry=FINGERPRINT.
  Principal-direction 2nd-order axis REJECTED (outlier-driven, PiSSA+SC-LoRA). MiLoRA minor-init, SC-LoRA
  principal+erodes, CorDA minor-input+magnitude-blowup. ΔW=(α/r)B@A reconstruction VALIDATED.
- Efficiency (fleet_findings): LoRA+wd zero-cost; CLoRA P-mem 0.42→6.7GB by k; DoRA 2.13×; data-aware
  256-1280 calib forwards + ~22GB transient.
- CE-to-base (forgetting_ce.py): validated vs MiLoRA Table 8 (LoRA 3.57↔3.24, PiSSA 6.31↔6.07); MiLoRA≈
  LoRA at matched magnitude. Full batch over 390 adapters PENDING (run on A, GPU-free window).

## SCRATCHPAD (ephemeral /tmp) — copied to repo:
- fleet_findings.md → paper/writing/fleet_findings.md (all 9 expert digests + efficiency + geometry + CE).
- integration_notes.md → paper/writing/integration_notes.md (reviewer B1/M/m fixes + section clarity content).

## GPU HEALTH (17:09 check): both nodes' 16 GPUs saturated. Node A had 2 B200s (GPU1,2) HUNG 7.5h on
frm_milora_lr2e4/lr5e4_c256_s42 BBH eval — deadlocked at lm-eval BBH init (154 threads sleeping, CPU frozen,
0% util, MATH/CS already computed=64.44/14.6 but no summary.json). Killed both (rc=-9); frepro4 pool
immediately relaunched SC-LoRA cells on the freed slots. WATCH: the milora BBH-init hang may recur; if new
evals stall at "generation_kwargs: {'max_gen_toks': 1024}" with 0% util for >20min, kill+re-queue.
RE-EVAL OWED (non-headline, MiLoRA math competitor): frm_milora_lr2e4_c256_s42, frm_milora_lr5e4_c256_s42.

## GOTCHAS: setsid-not-run_in_background (above); gitignore ignores **/*.md,**/*.txt,.claude/ (negations
added — verify new file types with `git check-ignore`); commit+push every milestone.
