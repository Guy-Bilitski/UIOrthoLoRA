# Writing-readiness plan — 2026-07-16 (post-campaign full-picture)

Status snapshot after the 30-node seed campaign + DeepSeek launch. Companion to
`data/key_numbers.md` §13 (the authoritative recomputed numbers).

## A. What is DONE and writable NOW

| Pillar | State | Evidence |
|---|---|---|
| Magnitude law, 2 models × 2 tasks, n≥3 | **COMPLETE** | r = −0.83…−0.93 per family (§13.1), 287 cells with 3 seeds, seed SD 0.3–2.6 pp (§13.2) |
| Geometry corroboration at full n | **COMPLETE** | spec_mean r=−0.758, n=1222; partial r(spread|F_Δ)=+0.20 → magnitude dominates (§13.3) |
| CE independent corroboration | **COMPLETE** | forgetting_ce vs log F_Δ r=+0.786, n=1094 (§13.3) |
| Pareto / efficiency / method tables | data complete | lorawd wd0p3 + low-LR sclora frontier stable across seeds |
| Base ceilings (Llama) | landed | key_numbers "Base ceilings (C5)" |

**Sections that can be drafted immediately** (once PI unpauses paper.tex): law results,
seed-robustness, geometry/CE corroboration, method comparison at 7B. All numbers cite
key_numbers.md §13; figures should be regenerated from the merged aggregates
(`adapter_metrics_merged.jsonl`, `forgetting_merged.jsonl`) — the fig_* scripts read the
per-node files and need repointing to the merged ones.

## B. What lands within ~24 h (write around it, don't block on it)

1. **Sweep dora tail** — 7 evals, 2–5 h ETAs. Adds the last s43/s44 dora replicates.
   (9 long-ETA dora replicates were deliberately sacrificed for DeepSeek nodes — every
   affected cell keeps its s42 anchor; disclose in the seed-coverage table.)
2. **Gap-fill s45/s46** — ~90 cells evaluating; pushes moderate-LR cells to n=4–5.
   Bonus error-bar tightening only; no claim depends on it.
3. **DeepSeek-V4-Flash 284B×13B-active MoE generalization** — all 7 methods training
   (13 cells up; 8 more auto-launch via `scripts/deepseek/launch8.sh` when staging ends →
   3 seeds × 7 methods). Timeline: train ~9–12 h + eval/CE/geo (first-run timing TBD,
   est. 3–6 h) → **first summaries tonight, full 21-cell set ~tomorrow**.

## C. Gate for "all the data we need" → start writing

- [ ] Sweep: 0 running sweep evals; final collect + derive pass; data_sanity clean.
- [ ] Recompute §13 one final time (script: `analyze_full_2026-07-16.py`) — freeze numbers.
- [ ] DeepSeek: ≥1 sane summary/method (21 preferred) → recompute magnitude+spread→retention
      at 284B; the generalization claim is directional consistency with §13.1.
- [ ] Regenerate figures from merged aggregates; verify against §13.
- [ ] PI unpauses paper.tex (writing outside paper.tex — e.g. results notes — can start now).

## D. Known caveats to disclose in the paper

- 9 dora s43/s44 eval replicates sacrificed (nodes repurposed to DeepSeek); s42 anchors kept.
- 11 extreme-LR cells (lr2e-3/5e-3) PERMANENT-FAIL: diverge to NaN → no valid eval exists;
  they are the (expected) divergent boundary of the LR grid, not missing data.
- CorDA excluded (contaminated init — documented decision).
- Diverged high-LR evals generate to max_new_tokens → their eval cost is 6–18 h/cell;
  scientifically they anchor the high-F_Δ/low-retention end and are retained.
- DeepSeek runs use dequant-bf16 base + MLA-attention-only targets (fair-run protocol,
  `handoff/DEEPSEEK_GEN_EXPERIMENT.md`); clora regularizer device fix + MTP load-retry
  applied 2026-07-16 (commit 09782568 and earlier).

## E. Operational leftovers (before teardown)

- Re-enable sweep orchestration is NOT needed — after the tail drains, run final
  finalize/collect, then teardown per `campaign-final-launch` memory (Phase 4).
- 21 evacuated DeepSeek nodes are listed in `fleet/ready_nodes.txt` comments.
- Artifact status page refresh (fixed URL, favicon 📉) after DeepSeek numbers land.
