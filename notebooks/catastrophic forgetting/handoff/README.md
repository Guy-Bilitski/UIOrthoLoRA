# HANDOFF — canonical state & plans (CF-in-PEFT controlled study)

**Project evolved** from "is UIOrthoLoRA an A*-worthy CLoRA-beater?" (DEAD — it only ties CLoRA) to
**a controlled study of what governs catastrophic forgetting in PEFT**, using corrected UIOrthoLoRA /
UILinLoRA as controllable instruments. See 00 for the current thesis (3 threads).

## Read in this order
1. **00_OPERATING_STATE.md** — READ FIRST. Current thesis (3 threads) + guardrails + env + scripts +
   durable findings/bugs + what's running. [CURRENT]
2. **08_FORWARD_PLAN.md** — the live plan: T1 Frobenius-magnitude law, T2 rank-beyond-Frobenius
   (kingmaker E2b), T3 data-basis leakage frontier. [CURRENT — the plan]
3. **06_INSIGHTS.md** — live findings + the ★ basis-reveal + the honesty/preemption ledger. [CURRENT]
4. **07_RELATED_WORK.md** — lit positioning (OPLoRA / Subspace-Geometry / CorDA / CorDA++ / SC-LoRA);
   the basis axis; [VERIFY] citation handles to confirm before any manuscript. [CURRENT — reference]
5. **05_DISCOVERY_PLAN.md** — the instrument-pivot plan (D1/D2/D3). Mostly folded into 08; experiment
   specifics superseded by 08. [SUPERSEDED-PARTIAL]
6. **01_RESULTS.md** — early numbers (gates, CLoRA bar, calibration). For the LIVE results pile use
   `analyze_magnitude_law.py` / `analyze_d1_d2.py` over `results/`. [HISTORICAL]
7. **02_EXPERIMENT_PLAN.md** — the original beat-CLoRA plan. [SUPERSEDED — goal abandoned]
8. **03_LEAKAGE_ANGLE.md / 04_LEAKAGE_MAP.md** — leakage-thermometer angle + the realized weight-basis
   leakage map. Useful reference; superseded as the headline by the Frobenius law + data-basis reframe. [REFERENCE]
9. **data_snapshots/** — frozen campaign_summary.jsonl, registries, gate jsons.

## One-line status (2026-06-15)
Frobenius-magnitude law holds within-architecture (r −0.96..−0.98 across LoRA/CLoRA/UIO); weight-basis
direction irrelevant (μ_E −0.09). Phase-2 running (grid + λ-sweep + LoRA-wd control + CLoRA fast
re-eval); two pending readouts = scale-unified cross-arch Frobenius curve + E2b matched-CS rank.
Memory: ~/.claude/projects/-home-guy-UIOrthoLoRA/memory/.
