# HANDOFF — canonical state & plans (CF-in-PEFT controlled study)

**Project evolved** from "is UIOrthoLoRA an A*-worthy CLoRA-beater?" (DEAD — it only tied CLoRA) to
**a controlled study of what governs catastrophic forgetting in PEFT** (thesis: retention is governed by
‖ΔW‖_F, not the adapter method). The current work is a **faithful CLoRA-recipe reproduction + LR×wd
sweep** on LLaMA-2-7B (CS + math).

## Read in this order (current → historical)
0. **`../WORKDIR_ALIGNMENT.md`** — ⚡ single onboarding doc: goal/thesis, exact CLoRA settings, LIVE file
   map + pipeline, adapter roster & status, current campaign status, decisions log, gotchas. **START HERE.**
   (`../README.md` = front door; `../STATUS.md` = live snapshot; `../paper/writing/INTERESTING_INSIGHTS.md`
   = consolidated findings, DONE-vs-OPEN.)

### LATEST (2026-07-06 → 07-09) — read after WORKDIR_ALIGNMENT
- **25_SUPERVISION_REPORT_2026-07-09.md** — current supervisor report: thesis, the fair LR sweep, canonical
  numbers (from key_numbers.md), CLoRA Table 2/3/4 published comparison, honest boundaries. [CURRENT]
- **26_RESEARCH_PLAN_2026-07-09.md** — post-fleet prioritized plan: injected GPU cells, CPU/eval-only work,
  paper actions, deferred/dropped items. [CURRENT — the working plan]
- **27_GEOMETRY_DRIFT_2026-07-09.md** — geometry-drift verdict: magnitude 1st-order / rank modest 2nd-order
  / **principal-direction 2nd-order axis TESTED & REJECTED**; geometry = fingerprint/measurement tool. [CURRENT]
- **28_TWO_NODE_PLAN_2026-07-09.md** — 16-GPU two-node fleet plan (saves Qwen + 3-seed + full CE). [CURRENT]
- **21_CONSORTIUM_SYNTHESIS.md** (07-06) — 9-agent verdict: law flat in the competitor blob → BBH-only,
  α=2r ruling, Tier-A cell set. [CURRENT — reference] · **22_RETENTION_FIX.md** — BBH-only retention decision
  + PiSSA real-forgetting gate. · **23_REPO_VERIFICATION.md** — correctness-gate pass. ·
  **24_PI_STATUS_REPORT_2026-07-07.md** — prior PI status snapshot.

1. **20_FAITHFUL_REPRO_SPEC.md** (2026-07-05) — **the live plan.** CLoRA Table 2 (CS r32/α64) + Table 3
   (math r64/α128) faithful reproduction, faithful math eval (`math_eval.py`), residual_save generalized
   to scaling≠1, PiSSA wired, LoRA+wd LR×wd sweep. [CURRENT — primary track]
2. **18_ADAPTER_AUDIT_2026-07-02.md** — per-adapter faithfulness verdicts; the FIX-1 calibration↔eval
   confound that quarantines CorDA/SC-LoRA/LoRA-Null off-curve claims. [CURRENT — reference]
3. **17_CORDAPP_IMPL_PLAN_2026-07-02.md** — CorDA++ (dyn covariance + dyn rank) on PEFT static CorDA.
   NB: the implementation now lives in `../cordapp_init.py` (CPU-validated 14/14, wiring into train_cs
   PENDING, `DEFAULT_N` 8→**5** at next restart). [CURRENT — plan; module built]
4. **19_FAIRNESS_STUDY_PLAN.md** — the 231/532-cell eval-matched-calibration fairness study. VALID but
   **deprioritized** below the faithful repro (20). [CURRENT — deferred]
5. **13_STATE_2026-06-29.md** — the 2×2 (2 models × 2 domains × 8 arms × LRs) campaign state + the
   magnitude-law numbers. Mirrored at top-level `../STATUS.md`. [SUPERSEDED-PARTIAL by 20]
6. **15/16 (2026-07-01)** — eval fixes: gen-cap/max-len (15), BBH metric normalization (16). Rows before
   commits `fe0f9be3`/`2602f57d` are not comparable to post-fix rows. [CURRENT — reference]
7. **14_CORDA_PP_PLAN.md** — earlier CorDA++ math/policy (superseded on the "how" by 17). [REFERENCE]
8. **00–12** — the UIOrthoLoRA-instrument era (k_val/k_vec, use_de, drop_major, leakage thermometers)
   and the original beat-CLoRA plan. UIO is DEAD (survives only as `uio_inprocess.py` helpers imported by
   `eval_one_gpu.py`). [HISTORICAL]
9. **data_snapshots/** — frozen campaign_summary.jsonl, registries, base/LoRA anchor jsons.

Paper writing package: **`../paper/writing/FINALIZATION_PLAN.md`** + **`../paper/writing/data/key_numbers.md`**
(single source of truth for every quoted number).

## One-line status (2026-07-09)
Two-node 16×B200 fleet (`handoff/28`) running the faithful CLoRA reproduction: on Node A, 4 `gpu_pool` pools
(`frepro4*`) + `auto_dispatch` on `jobs/master_dispatch.txt`; Node B trains fresh + syncs. Math `frm_`
~46/46 done; CS `frc_` reservoir (paper spine) landing now. Magnitude law confirmed on 2 models
(CS pooled r=−0.86, on-curve −0.92, Qwen −0.88) + externally in CLoRA Table 4 (r=−0.98); geometry-drift
verdict done (magnitude 1st / rank 2nd / principal-direction axis rejected). CorDA/SC-LoRA off-curve claims
still embargoed pending fair calibration. Full snapshot: `../STATUS.md`. Memory:
`~/.claude/projects/-home-guy-UIOrthoLoRA/memory/`.
