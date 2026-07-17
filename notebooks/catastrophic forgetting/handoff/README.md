# HANDOFF — canonical state & plans (CF-in-PEFT controlled study)

**Project evolved** from "is UIOrthoLoRA an A*-worthy CLoRA-beater?" (DEAD — it only tied CLoRA) to
**a controlled study of what governs catastrophic forgetting in PEFT** (thesis: retention follows the
**magnitude relation** — flat-then-falling in F_Δ with a knee — not the adapter method; title direction
*"Magnitude, Not Geometry"*). The data campaign (faithful CLoRA-recipe reproduction + LR×wd sweeps on
LLaMA-2-7B and Qwen-2.5-7B, CS + math) is **complete and frozen** as of 2026-07-17; current work is
offline analysis + writing.

## Read in this order (current → historical)
0. **`41_EVACUATION_2026-07-17.md`** — ⚡ **THE CURRENT STATE DOC.** Fleet evacuated 2026-07-17; what
   data survived, where it is, how to restore it, what remains (offline analysis + writing only).
   Canonical numbers: `../paper/writing/data/key_numbers.md` **§18 FINAL FREEZE + §19 POST-FREEZE
   ADDENDUM**; story layer: `../paper/writing/analysis_final/` (01–07 + `PAPER_BLUEPRINT.md`).
   **START HERE.**
   (`../WORKDIR_ALIGNMENT.md` = onboarding map, live-fleet sections now INERT; `../README.md` = front
   door; `../STATUS.md` was archived to `../archive/writing_2026-07-17/`.)

### 29–41 (2026-07-09 → 07-17) — the multi-seed / freeze / evacuation era
- **41_EVACUATION_2026-07-17.md** — see item 0. [CURRENT]
- **34–40** (session states, plan validation, completeness audit, node handover, resume prompts,
  fleet-eval recovery) — moved to **`../archive/writing_2026-07-17/handoff/`**. [ARCHIVED]
- **33_FREEZE_PLAN_SAT.md** — the freeze plan; executed as key_numbers §18. ·
  **29–32** — 07-09/10 session state, 3-seed error-bar plan, A* roadmap, plan reassessment.
  [SUPERSEDED by 41 + §18]

### 2026-07-06 → 07-09 — read after the above
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
   NB: the implementation lives in `../cordapp_init.py` (CPU-validated 14/14) and has been **WIRED into
   `train_cs.py` (`--cordapp`) since the 2026-07-06 frepro4 restart** — no longer pending. [REFERENCE — plan; module built + wired]
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

Paper writing package: **`../paper/writing/analysis_final/PAPER_BLUEPRINT.md`** (story layer, docs 01–07) +
**`../paper/writing/data/key_numbers.md` §18–§19** (single source of truth for every quoted number).
(`FINALIZATION_PLAN.md` and the 07-02 writing suite are at `../archive/writing_2026-07-17/`.)

## One-line status (2026-07-17)
Campaign **FROZEN**, fleet **EVACUATED** 2026-07-17 (`41_EVACUATION_2026-07-17.md`) — no live pools;
remaining work is offline analysis + writing. Magnitude relation (flat-then-falling with a knee) final:
pooled **r=−0.847, n=1035**, 6 model×task families, 8 methods, 3–5 seeds (key_numbers §18.1); nested ΔR²
ladder: magnitude +0.395, geometry +0.017, method +0.006 (§19.1); SC-LoRA's deviation resolved as a
calibration artifact by E4 (+0.92pp above the relation, §18.3); DeepSeek-284B geometry fingerprint recurs
(`../paper/writing/analysis_final/07`). Source of record: `results/*/summary.json`
(`campaign_summary.jsonl` / `results_book/` stale). Memory:
`~/.claude/projects/-home-guy-UIOrthoLoRA/memory/`.

*(Prior one-line status, 2026-07-09, for provenance: two-node 16×B200 fleet running the faithful CLoRA
reproduction; math `frm_` ~46/46 done, CS `frc_` reservoir landing; CS pooled r=−0.86 single-seed;
CorDA/SC-LoRA off-curve claims embargoed pending fair calibration — since resolved, see above.)*
