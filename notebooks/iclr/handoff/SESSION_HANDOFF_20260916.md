> **Superseded status and queue — 19 September 2026:** Start with
> `EXPERIMENT_PREPARATION_HANDOFF_20260919.md`. All 21 band/head confirmations
> are complete and integrated in the manuscript. The author reconfirmed TWO
> GPUs. Old live-process IDs, four-GPU allocation and missing-band instructions
> below are historical, not a current launch instruction. Preserve all original
> exports, including signed roundoff residuals; do not apply the old cosmetic
> clamping suggestion to raw evidence. Check current artifacts and leases first.

# Session handoff — 2026-09-16 ~11:00 local

> **UPDATE 2026-09-16 ~11:20 local (next session).** Sections 1–4 below are
> superseded by §0. Read §0 first, then §5–§7 (still current).
>
> ## 0b. BLOCK B IS COMPLETE — 2026-09-16 22:25 local (19:25Z)
>
> All 21 registered RTE band entries completed, reload-validated and frozen
> (`block_b_freeze.jsonl`, 21/21 against protocol sha 721935…4fdeb). Coverage
> 3/3 seeds on all six band x flexibility arms plus the three head-only
> references; nothing dropped. Export refreshed (23 band rows incl. 2 pilots).
> Committed ortho_new `dabd48f4`, mirrored to Overleaf `a88118e`.
>
> NEXT, in order:
> 1. Block-B held-aside evaluation is now UNBLOCKED (it required 21/21). It has
>    NOT been run. Same rules as the P1 evaluation: one pass, no direction-based
>    decisions, reported alongside and never merged with inner-selection scores.
> 2. Band results are NOT in the manuscript. `BAND_PRESENTATION_SPEC_20260916.md`
>    (Overleaf root) is Astra's prospective table/figure spec; the user was asked
>    to confirm it is the agreed structure and had not answered as of this note.
> 3. Blocks C (MRPC replication) and D (LoRA-8) remain unstarted; all 4 GPUs are
>    now free. Block C needs its protocol registered + a fresh preflight first.
>
> Section 0 below describes the mid-day state and is kept for history.
>
> ## 0. Live state as of ~11:20 local (08:20Z)
>
> **§2 was wrong: the lane drivers did NOT die with the old session.** They were
> launched detached and survived; three were still chaining entries on GPUs 0/2/3
> and had already queued all three `P1_HEAD_BASE` heads. Before launching
> anything, map surviving drivers to GPUs or you will double-book:
>
>     ps -eo pid,cmd | grep -E "band_lane|gate_lead|chain_lane" | grep -v grep
>
> - Locked held-aside evaluation (§4): **COMPLETE**, all 18/18, every loading
>   check reproduced. Bundle committed (ortho_new `d391c4c3`) and mirrored to
>   Overleaf (`a63b1f2`). Numbers reported to the user for Astra. Do not rerun.
> - `ops/block_b_freeze.py`: **8/21** frozen (the 9 diagonal arms minus TAIL/123,
>   which landed just after the pass). Append-only — rerun as runs land.
> - Block B: all 9 diagonal arms completed+validated; the 3 `P1_HEAD_BASE` heads
>   were running; the 8 ROT64 entries are queued on chained lanes:
>
>       gpu0  LEAD_ROT64/42,  LEAD_ROT64/17     gpu2  LEAD_ROT64/123, MID_ROT64/42
>       gpu3  MID_ROT64/17,   MID_ROT64/123     gpu1  TAIL_ROT64/17,  TAIL_ROT64/123
>
>   (`BAND_TAIL_ROT64/seed_42` was already running on GPU 1.) Expected complete
>   ~23:30 local — GPU 1 is the long pole because it starts its pair ~14:25.
> - New: `ops/chain_lane.sh <gpu> <pred_pid|-> <pred_session|-> <entries…>`
>   queues a lane behind a predecessor that already owns the GPU. It waits on the
>   predecessor's **lane-driver PID**, not on tmux-session absence: `band_lane.sh`
>   sleeps between entries, so "no session on this GPU" is briefly true mid-lane
>   and would double-book. It also refuses a GPU holding a colleague's process.
>   Chain logs: `/tmp/band_chain_gpu<N>.log`.
> - Astra's `BAND_PRESENTATION_SPEC_20260916.md` (Overleaf root) is a prospective
>   table/figure specification for block B. Confirm with the user that it is the
>   agreed structure before integrating band results (§5.4).
> - Do NOT compute an inner-selection-vs-held-aside delta; Astra specified
>   "alongside, never merged", and that difference is not a specified contrast.

For the agent picking up the ICLR campaign after a context clear. Everything
below is verifiable from the repo; when in doubt, trust ledgers and hashes over
this prose. Working checkout: `notebooks/iclr-campaign-20260914` (branch
`ortho_new`). Overleaf clone: `notebooks/overleaf-6aa54397` (project
6aa54397e58b10444b0fa2aa; credential helper already configured via
`~/.git-credentials-overleaf`; ALWAYS `git pull --rebase` before editing —
author/Astra edits win; NEVER force-push). Deadlines: abstract ~2026-09-19,
full paper ~2026-09-26.

The confirmed central result, methodology rules (Astra's, binding), and
history are in auto-memory `iclr-campaign-state.md` and
`notebooks/iclr/handoff/` docs. Astra communicates through the user; his
requests live as files in the Overleaf repo root.

## 1. What is running RIGHT NOW (survives the clear)

Training runs live in tmux on a NAMED SOCKET — plain `tmux ls` does NOT show
them:

    tmux -L iclr_6aa54397_20260914 ls

Four sessions, one per GPU (all four GPUs are ours per user's latest
instruction):

- GPU 0 `band_rte_BAND_LEAD_DIAG_seed_123_gpu0` (run 20260916T070645Z_42d074cd0e07)
- GPU 2 `band_rte_BAND_MID_DIAG_seed_123_gpu2` (run 20260916T070809Z_1ecac81f6784)
- GPU 3 `band_rte_BAND_TAIL_DIAG_seed_123_gpu3` (run 20260916T071122Z_49b9a7623bad)
- GPU 1 `band_TAIL_ROT_42_gpu1` — BAND_TAIL_ROT64/seed_42 (run
  20260916T064705Z_7b77bcef1185), rotated arm, ~4h class (started 06:47Z)

Each session runs the fail-closed controller (`smoke.py --purpose band`) which
trains, validates (checkpoint-reload reproduction) and settles its own GPU
lease, then exits. Session logs: `/tmp/band_<session>.log`. The diagonal
seed-123 runs finish ~1h after their 07:0xZ starts; check the ledger:

    tail -5 campaign_outputs_confirmation_v1/run_ledger.jsonl

A run is DONE only when its ledger status is `completed` (that means
validated). `awaiting_validation` → validation in progress inside the same
session.

## 2. What DIED with the old session (re-establish these)

The lane-chaining monitors and watcher ran inside the old Claude session.
Nothing relaunches the remaining queue until you do. Also a background CPU
process `locked_evaluation.py` may or may not have survived — check
`pgrep -af locked_evaluation` (see §4).

Re-establish a watcher (optional but recommended):
`notebooks/iclr/handoff/ops/watch_campaign2.sh` tails both ledgers + alerts.

## 3. Remaining block-B queue (11 runs) and how to launch

Registered protocol: `campaign_outputs_confirmation_v1/protocols/band_flexibility_rte_20260916.json`
(sha256 721935221ca1b4db38736016fbb256fa1d355d4dcf8a90a693856eaf7614fdeb).
Status at handoff: 6 completed (LEAD/MID/TAIL DIAG × seeds 42,17), 4 running
(above), 11 unlaunched:

    rte/BAND_LEAD_ROT64/seed_42   rte/BAND_LEAD_ROT64/seed_17   rte/BAND_LEAD_ROT64/seed_123
    rte/BAND_MID_ROT64/seed_42    rte/BAND_MID_ROT64/seed_17    rte/BAND_MID_ROT64/seed_123
    rte/BAND_TAIL_ROT64/seed_17   rte/BAND_TAIL_ROT64/seed_123
    rte/P1_HEAD_BASE/seed_42     rte/P1_HEAD_BASE/seed_17     rte/P1_HEAD_BASE/seed_123

Timing: DIAG ≈55 min, ROT64 ≈4h12m, HEAD ≈≤1h. Plan (user-approved, all 4
GPUs): 2 rotated arms per GPU serially + heads slotted after; block B complete
~21:30–23:00 tonight.

Launch via the serial lane driver (waits for each entry to complete+validate
before the next; refuses duplicates unless `retry_of` is passed to the inner
launcher):

    cd notebooks/iclr-campaign-20260914
    nohup bash notebooks/iclr/handoff/ops/band_lane.sh <GPU> <entry_id> [<entry_id> ...] \
      > /tmp/band_lane_gpu<GPU>.log 2>&1 &

e.g. once the current sessions exit:

    gpu0: rte/BAND_LEAD_ROT64/seed_42  rte/BAND_LEAD_ROT64/seed_17
    gpu1: rte/BAND_LEAD_ROT64/seed_123 rte/BAND_MID_ROT64/seed_42
    gpu2: rte/BAND_MID_ROT64/seed_17   rte/BAND_MID_ROT64/seed_123
    gpu3: rte/BAND_TAIL_ROT64/seed_17  rte/BAND_TAIL_ROT64/seed_123
    heads (P1_HEAD_BASE × 3): on whichever lanes finish first.

Note GPUs 0/2/3 free ~1h from their 07:0xZ starts; GPU 1 frees when
TAIL_ROT64/seed_42 ends (~10:50Z, i.e. ~13:50 local). Start each lane the
moment its GPU's current session exits.

Rules that will bite you if ignored:
- The launcher REFUSES to start if the campaign source tree
  (`notebooks/iclr/campaign/*.py`) is dirty or if the preflight token does not
  cover the exact source hash. Current good preflight:
  `notebooks/iclr/handoff/data/campaign_v1/preflight/20260916T022852Z_05451632/report.json`
  (already baked into `ops/launch_band_entry.sh` as PF=). If you must change
  campaign code: commit, run a fresh preflight, then `sed` the PF line and
  VERIFY the sed actually replaced it.
- Never launch on a GPU with a foreign compute process (colleagues Goody /
  danielf share the box). Check `nvidia-smi` first.
- A `failed` ledger status → diagnose from the run dir's `worker_failure.json`
  and `/tmp/band_<session>.log`, then relaunch with 5th arg `retry_of=<old run id>`
  passed through `launch_band_entry.sh` directly.
- NEVER remove seeds/conditions to hit the deadline (Astra, binding).

## 4. Locked held-aside evaluation (Astra's request — in flight at handoff)

Request: `overleaf-6aa54397/LOCKED_EVALUATION_REQUEST_20260916.md`
(sha256 0c736b31…d0535c) + `overleaf-6aa54397/data/locked_evaluation_request_20260916.json`.
Executor: `notebooks/iclr/handoff/ops/locked_evaluation.py` — one-time CPU
evaluation of the 18 frozen P1 confirmation checkpoints on the official locked
splits (RTE 277 / MRPC 408), with strict loading checks (inner accuracy must
reproduce EXACTLY — verified it does; loss diff ~1e-7). Output:
`notebooks/iclr/handoff/data/locked_evaluation_20260916/`
(held_aside_results.csv, per_example/, summary_contrasts.json,
evaluation_manifest.json). It was running as a background process at handoff
(~4/18 done, all checks passing; observed held-aside MRPC MIX ≈0.843–0.863).

On pickup:
1. `pgrep -af locked_evaluation` — if alive, wait (~30 min total).
2. If dead and `evaluation_manifest.json` EXISTS → it finished; proceed.
3. If dead and no manifest → the run was killed mid-flight: DELETE the partial
   `locked_evaluation_20260916/` directory entirely and rerun
   (`CUDA_VISIBLE_DEVICES= .venv/bin/python notebooks/iclr/handoff/ops/locked_evaluation.py`).
   "Preserve first output" applies to the completed export, not an aborted
   partial; the script itself refuses to run if the directory exists.
4. Then run `notebooks/iclr/handoff/ops/block_b_freeze.py` — appends validated
   block-B run/checkpoint IDs to `block_b_freeze.jsonl` in the same output dir
   against the protocol hash. Rerun it as block-B runs land (append-only).
   Block-B held-aside evaluation happens ONLY after 21/21 frozen — do not
   evaluate it piecemeal.
5. Commit the bundle to ortho_new, mirror to the Overleaf clone (copy under
   `data/` there like previous bundles), push both, and report to the user for
   Astra: held-aside scores reported ALONGSIDE, never merged with,
   inner-selection scores or the historical GLUE table; no direction-based
   decisions; everything retained.

## 5. After block B completes (tonight)

1. Rerun `notebooks/iclr/handoff/ops/final_evidence_export.py` — refreshes
   `notebooks/iclr/handoff/data/final_evidence_20260916/` (band_results.csv +
   band_coverage.json are designed to be re-exported as runs land). Known
   cosmetic TODO: clamp tiny negative within_band_offdiagonal_fraction
   (−1e-8, float roundoff on diagonal arms) to 0.
2. Run `ops/block_b_freeze.py` again → should reach 21/21.
3. Commit + mirror to Overleaf + push.
4. Band table/figure PRESENTATION STRUCTURE must be agreed with Astra BEFORE
   integrating rotated results into the paper (his standing rule). Current
   partial numbers (seed 42): LEAD 0.753 / MID 0.731 / TAIL 0.729; TAIL ROT64
   0.743 vs DIAG 0.729. No band-superiority claim anywhere.
5. Abstract: draft with confirmed numbers + Astra's wording rules is at
   `notebooks/iclr/handoff/ops/abstract_draft.md`. Finalize with user+Astra;
   evidence freeze tomorrow evening (day 3).

## 6. Later queue (post-abstract)

- Block C: MRPC band replication (21 runs, ~22 GPU-h) — days 4–5.
- Block D: LoRA-8 (6 runs, ~7 GPU-h).
- Block-B held-aside evaluation once 21/21 frozen (see §4.4).

## 7. Key invariants (do not relax)

- Fail-closed everything: only ledger `completed` counts; extractor scripts
  refuse to run without `INVALIDATED_RUNS.json` present in BOTH output roots.
- Tolerances: reproduction 1e-6, merged-forward 1e-5, p0 (CPU/GPU MLM) 5e-4.
- Storage: write bulk outputs only under `campaign_outputs_confirmation_v1/`
  (user-approved cleanable root). 43G quarantine lives in
  `invalidated_20260914_tokenizer/` — leave it.
- Per-seed matching errors e_s always from confirmation runs, never
  calibration. Numeric task-cost language. No pretrained-preservation claim.
- Overleaf: pull-rebase before any edit; generated table blocks are marked
  GENERATED (regenerate via `notebooks/iclr/handoff/scripts/build_focused_tables.py --check`).
- Git identity/commits: sync results + docs to ortho_new frequently
  (user asked for continuous sync).
