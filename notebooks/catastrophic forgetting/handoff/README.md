# HANDOFF — canonical state & plans for the UIOrthoLoRA forgetting study

Read in order:
1. **00_OPERATING_STATE.md** — env, scripts, critical findings/bugs, what's running, how to resume. READ FIRST.
2. **01_RESULTS.md** — all numbers (gates, CLoRA bar, UIOrthoLoRA/UILinLoRA, calibration, leakage).
3. **02_EXPERIMENT_PLAN.md** — what to run for a publishable result (frontier, seeds, leakage, ablations).
4. **03_LEAKAGE_ANGLE.md** — the leakage-thermometer paper angle ("optimal leakage budget") + the must-fix caveat.
5. **04_LEAKAGE_MAP.md** — B1 deliverable: the realized leakage map (16 runs, both arms). Headline: retention tracks a ΔW *magnitude* budget, not directional leakage. Regenerate via `../make_leakage_map.py --write`.
6. **data_snapshots/** — frozen copies of campaign_summary.jsonl, registries, gate jsons.

Live state also in ../STATUS.md and ../README.md. Persistent cross-session memory:
~/.claude/projects/-home-guy-UIOrthoLoRA/memory/ (uiortholora-phase1-gotchas, reproduction-campaign, ...).

One-line status: gates A/B/C + CLoRA bar done; UIOrthoLoRA under-adapts at LoRA's LR (fixed: use LR=1e-2);
tuned frontier (Wave 1) training, Wave 2 (high-retention + leakage) queued; leakage thermometers implemented & validated.
