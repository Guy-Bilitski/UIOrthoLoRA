# Prompt for the next agent (paste verbatim)

You are resuming the UIOrthoLoRA magnitude-law campaign as autonomous supervisor. Work dir:
/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting (Node A; Node B = ssh ubuntu@d002,
same path). FIRST ACTION, before anything else: read
handoff/34_SESSION_STATE_2026-07-12.md end-to-end — it is the authoritative state, results
ledger, infra map, and operational rules. Also load your memory directory index.

Then, in order:
1. Recreate the 2-hourly campaign check cron (session-only crons died with the last
   session — verify with CronList, never assume). Cron prompt: sync gold data first
   (bash sync_d002.sh once; git add results/ + commit "results sync <ts>" + push), then
   validate BOTH nodes: 8 GPUs busy each, dispatchers/watchdogs/sync/results-book loops
   alive (relaunch via setsid only, kill and relaunch as separate calls), stale-lock audit
   (lock + no summary.json + no live proc ⇒ rm lock), log-freshness audit for spinning
   wedges (running job whose log mtime is >60 min old at high util ⇒ kill, GC requeues),
   fresh-Traceback grep, B disk. Report one short paragraph.
2. Run that full check once immediately.
3. Continue the standing work from handoff/34 §OPEN WORK: fold newly landed cells into
   the artifact (frm_sclora/frm_lora_null → §6 CE rows; frc lorawd/wd columns and the two
   frc_lorawdr16_* param-matched controls → wherever §3/§2 cite them), keep both queues
   draining, and when the PI opens the Monday bonus window, execute the ranked menu in
   handoff/33_FREEZE_PLAN_SAT.md §c.
4. Artifact rules (CRITICAL): live URL https://claude.ai/code/artifact/5c46636f-036a-4fae-919f-43be8e07639c
   — from a new session you MUST pass that as `url:` when publishing or you will mint a
   duplicate (it happened once; a stray URL already exists as a tombstone). Source of
   truth: paper/writing/artifact_status_report.html — copy to your scratchpad, edit, bump
   the "Last updated" stamp, publish with url:, cp back to the repo, commit+push. Favicon 📉.
   Framing guardrails: constructive, no "geometry doesn't matter", no "everyone reports
   one LR", bold claims need 3 seeds, no future/pending/roadmap content in the artifact.
5. paper.tex: the claims audit (paper/writing/claims_coverage_audit_sat.md) has 5 blockers;
   commit 042c37db is a PARTIAL fix (B1 only). The PI said "leave the paper for now" —
   ASK before resuming paper work; when authorized, re-run the author task with the audit
   + handoff/34's Key Results Ledger as the numbers source (every number must be
   re-verified against results/*/summary.json before it is written).
6. Standing PI orders: all 16 GPUs busy per plan at all times; sync data from both nodes
   to GitHub at every wake-up and milestone (nothing lives only on hosts); keep-everything
   lives on Node A + GitHub, B is workforce; report honestly (failures included, plainly).
