# Handoff 39 — Fleet eval recovery (2026-07-15, ~08:00–08:30Z)

Operational incident + fix. No scientific claims changed; this is infra only. Written after resuming
the 30-node fleet and finding evals silently failing.

## Symptom
"All GPUs running but the evals did not work; data points not arriving." All 30 nodes' dispatchers were
up and GPUs ~8/8 busy, yet almost no *current-campaign* summaries were landing.

## Root causes (three, all confirmed)
1. **gsm8k dataset never pre-cached, dispatch is offline.** Nodes run `auto_dispatch.py … --hf_offline 1`.
   `openai/gsm8k` was absent from `/scratch/hf_cache` on every node, so every `--adapt_task gsm8k` cell
   (~197) trained fully (hours) then died at eval with
   `ConnectionError: Couldn't reach 'openai/gsm8k' (OfflineModeIsEnabled)`, producing **no summary and no
   lock retirement** — pure wasted training. `cs` (local data) and `math_faithful` (local
   `repro/LLM-Adapters/dataset/{gsm8k,MATH}/test.json`) were unaffected; retention "broad" was already cached.
   **Only d001 (head) has internet egress** — compute nodes cannot reach the Hub.
2. **Prior-session shard edits were invalid.** ~48 lines across `jobs/fleet/d0*.txt` had been turned
   *eval-only* (stripping the `train_cs.py &&` prefix, mostly on the failing gsm8k cells). Since all adapters
   were lost at the node handover (handoff/37), every cell must train first; eval-only cells reference
   non-existent adapters. These edits were local/uncommitted on d001 only (guardian hadn't rsynced them out).
3. **No idempotency in `train_cs.py`.** A dispatcher retry of a gsm8k cell reran `train_cs.py && eval` — i.e.
   **retrained the already-banked adapter from scratch** (hours) instead of just re-evaluating.

## Fixes applied
1. **Cached `openai/gsm8k` (config `main`) on d001; rsynced `/scratch/hf_cache/datasets/openai___gsm8k/`
   to all 29 remote nodes; verified offline load on d001/d002/d032.** Recipe for any future offline dataset:
   cache on d001 (network up), then fan out the cache dir by rsync (compute nodes have no egress).
2. **Reverted all `jobs/fleet/d*.txt` from `*.txt.bak`** (correct train+eval form). Verified no diff vs .bak.
3. **Added a skip-guard to `train_cs.py`** (committed `bbf5eb8f` on ortho_new; also live on all 30 nodes):
   right after `out_dir` is computed, if `run_config.json` (written last = completed run) +
   `adapter_model.safetensors` exist, print `SKIP train, proceed to eval` and `return`. Turns a banked-cell
   retry into minutes of eval. Idempotent; eval logic unchanged. Deployed by rsync (all nodes were byte-identical
   `4c3d2beb…` pre-patch; each re-verified with `py_compile` + guard-grep).

## Recovery dynamics (no manual per-cell action needed)
`auto_dispatch.py` loops forever; a cell is pending when it has no `summary.json` and no lock. The guardian
GCs stale failed-cell locks (no summary, no live proc, >8 min) → the running dispatcher re-picks the cell →
guard skips the retrain → eval now succeeds (gsm8k cached) → summary written → `collect_loop` pushes it.
~270 overnight-banked adapters recover this way as GPU slots free. A few pre-patch retrains already in flight
finish and self-correct.

## State at handoff (08:30Z)
- 30/30 dispatchers UP, ~8/8 GPUs busy per node, 0 down. Backlog pushed (`origin/ortho_new`); collect ~20 min.
- Planned campaign cells: 714 (union of shard `--run_name`s). Completing steadily; gsm8k path verified end-to-end.
- guardian + collect + per-node watchdogs all healthy (an earlier "loops wedged 3h" read was a UTC-vs-local
  timezone mistake — they were fine).

## Ops notes / gotchas
- `/home/guy/UIOrthoLoRA` and `/home/guyb/UIOrthoLoRA` are the **same files** (same inode); venv is under
  `/home/guy`. d001 dispatcher runs as **root** (locks root-owned); remote dispatchers run as **ubuntu**.
- Dispatcher reads its jobs file **once at startup**; shard edits only affect future (re)launches.
- Never `git add -A` (results churn + lock dirs); `collect.sh` scopes to `results/` and excludes lock dirs.
- Guardian/collect `ssh` all of `ready_nodes.txt`; a truly dead node would slow (not wedge) a pass.
