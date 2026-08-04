# Missing experiments — launch when compute is available

Audited 2026-07-19 against: handoff/41_EVACUATION_2026-07-17.md (authoritative
survived/lost ledger), key_numbers.md §16/§18.7, STRATEGY_MEMO_2026-07-18.md,
ACL_CAMPAIGN_INSIGHTS_2026-07-19.md, analysis_final/{01,07,08,09}, jobs/.

**CRITICAL COMPUTE-SIZING FACT (handoff/41):** after the evacuation, **no 7B
adapter checkpoints survived** (the seed-42 Qwen adapters were already destroyed
2026-07-03; the rest went with the fleet). Every 7B item below that touches
model outputs is therefore a **retrain-then-eval** job, not an eval pass. The
ONLY surviving checkpoints are the 21 DeepSeek-284B adapters
(`results/ds_adapters_evac/`, SHA256-verified 21/21).

Priority order within each tier. Owner: (assign on relaunch).

**TOOLING NOTE (2026-07-30, resolved):** the lost `scratchpad/make_grand_table.py`
has been rebuilt at `acl_analysis/rq1_stats/07_make_grand_table.py` and verified
to reproduce the frozen `tables/table_grand.tex` exactly (every number, marker,
and bold; default mode writes `tables/table_grand_regen.tex` and diffs against
the frozen file — zero mismatches).
The new RQ1 statistics layer (`acl_analysis/rq1_stats/`: Holm-corrected
head-to-heads, TOST equivalence, MDE power notes + independent verifier) IS
committed and reproducible.

---

## Tier 1 — retire "provisional" markers in the paper

### 1. Complete E4 to 24/24 (SC-LoRA eval-matched ladder, Llama-CS)
The paper's calibration-artifact resolution quotes n=20/24. Four cells lack
benchmark evals (dirs exist with config/forgetting/geo only, no summary.json):
`b4_sclora_r32_lr1e3_s42`, `b4_sclora_r32_lr1e3_s43`,
`b4_sclora_r32_lr3e4_s46`, `b4_sclora_r32_lr5e4_s46`.
Adapters lost -> retrain these 4 cells then run full retention+adaptation
evals. Removes "provisional" from the SC-LoRA resolution (08_thesis_memo:91).
Template: `archive/jobs_superseded/frepro4_b4.txt`.

### 2. B4 companion eval-matched arms: LoRA-Null and CorDA++
The eval-matched calibration control was designed for SC-LoRA AND LoRA-Null AND
CorDA (THESIS_VALIDATION_PLAN B4); only SC-LoRA ran to (near) completion.
Missing full evals (dirs on disk, no summary.json):
`b4_lora_null_r16_lr1e4_s42`, `b4_lora_null_r16_lr3e4_s42`,
`b4_lora_null_r16_lr3e4_s46`, `b4_cordapp_r32_lr1e4_s42`,
`b4_cordapp_r32_lr3e4_s42`. Retrain + eval. Extends the calibration control
beyond one method, closing the "only SC-LoRA was controlled" scope note.

### 3. Qwen rescale ladder (E1 cross-architecture, CS + math)
The interventional claim is single-setting (Llama-CS, n=24). Strategy memo
names "Qwen-CS + math rescale ladder (A2)" the single highest-value missing
run. Requires: retrain a small set of Qwen anchor adapters, then
`rescale_adapters.py` ladder + random-direction controls + evals. No job file
exists yet; write one from the E1 spec.

---

## Tier 2 — coverage the paper currently discloses as gaps

### 4. Qwen CE/KL drift recovery (fills the grand-table `--` cells)
The CE store covers ~60% of Qwen runs; seed-42 CE is missing across nearly all
Qwen cells and two operating cells have no KL at all
(`qwsw_sclora_r32_lr1e4`, `qwswm_dora_r16_lr2e4`).
**Premise correction:** the missing cells' adapters no longer exist, so this is
retrain + `forgetting_ce.py`, NOT an eval-only pass.
Ready-to-launch spec: `jobs/qwen_ce_recovery.txt` (124 train+CE chains).
Disclosure ledger of all missing cells: `jobs/ce_backfill_qwen.txt` (123 runs).
Llama CE backfill lists: `jobs/ce_chunks/chunk2-7.txt` (verify against
`results/forgetting_merged.jsonl` before launch).

### 5. CorDA / CorDA++ faithful nq_open re-run (un-withhold CorDA)
CorDA was mis-calibrated (wikitext-2, F_Delta explodes) and is withheld from
all quantitative results. Retrain with faithful nq_open calibration + 0-step
self-check (Delta W -> 0), then full sweep evals. Lets CorDA re-enter the law
fit, ladder, and grand table (closes "7 of 8 assessed").

### 6. SC-LoRA eval-matched calibration control on Qwen-math
The one head-to-head exception (SC-LoRA Qwen-math +8.3 pp GSM8K at tied BBH)
has no calibration control in its own setting. Retrain SC-LoRA Qwen-math cells
with eval-matched calibration; settles data-aware-init-as-configured vs
subspace geometry. Fresh spec needed.

### 7. MMLU-Pro answer-parser fix + math retention re-score
Math retention is BBH-only because the parser fails on MetaMathQA-tuned output
(chance-floor collapse = artifact). Two parts: (a) parser fix, code-only, no
GPU; (b) re-scoring requires model generations -> retrain the math operating
cells (or fold into any other math retrain above). If sane afterward
(base MMLU-Pro 19.0 Llama / 40.8 Qwen, monotone in F_Delta), math retention can
move to the same core metric as commonsense.

### 8. DoRA F_Delta recompute with magnitude-vector rescaling
DoRA's F_Delta is a lower bound (PEFT get_delta_weight omits the forward-time
magnitude-vector rescaling). Checkpoints were not retained (REBUTTAL_PREP:486)
-> clean re-run of the 7 DoRA cells, then corrected F_Delta. Can only increase
DoRA's already-positive residual; tightens the plot, overturns nothing.

---

## Tier 3 — DeepSeek-284B retention recovery (eval-only; adapters SURVIVED)

### 9. 284B retention + CE evaluations
The only truly lost 284B data is the retention/CE evals (killed at 14-26%
progress). The 21 adapters are archived and SHA256-verified, so this IS an
eval-only job (the one item in this file that is):
- Hardware: one 8xB200-class node (~79 GB/GPU train peak; eval loads the
  158 GB FP8 base with dequant-on-load via `scripts/deepseek/fp8_dequant.py`).
- Re-download `deepseek-ai/DeepSeek-V4-Flash` (~160 GB; d001 cache gone).
- Restore: `cat <run>.tar.part-* | tar xf -` in `results/ds_adapters_evac/`,
  verify against `SHA256SUMS`.
- Run `scripts/deepseek/eval_deepseek.py` (MMLU, GSM8K, HumanEval/MBPP,
  HellaSwag/ARC, TruthfulQA) + `ce_deepseek.py` (WikiText CE drift) per
  `handoff/DEEPSEEK_GEN_EXPERIMENT.md`.
- Also: re-score d016 adapt (`dsv4_lorawd_r16_lr5e4_s42`, its log was lost);
  optionally compute F_Delta at 284B (needs dequantized base matrices).
- Exclude/flag `dsv4_lora_null_r16_lr5e4_s44` (diverged; adapt 25.7 ~ chance).
Would upgrade the paper from "shape fingerprint recurs" to an actual 284B
retention test of the magnitude relation.

---

## Tier 4 — density/completeness (optional, figure-quality)

10. **E7 Qwen bridging 4th cell**: `brq` MedMCQA lr1e3 (quoted r=-0.995 sits on
    n=3). One train+eval cell.
11. **E3 Qwen mid-LR densification, 2nd wave**: ~13 cells (7 methods x
    {7e-5, 1.5e-4}) lost to the fleet kill; figure density only.
12. **Qwen trained-not-evaluated sweep cells**: qwsw 27 + qwswm 22 cells never
    got benchmark evals (adapters lost -> retrain+eval). Ledger: key_numbers
    §18.7.
13. **PiSSA + CorDA++ math seeds**: `frm_pissa_lr3e4_c256_s43/s44`,
    `frm_cordapp_lr3e4_c256_s43/s44` (queued in `jobs/night_final_B.txt` [N1],
    never ran). Puts the single-seed PiSSA math row on 3 seeds.
14. **E5 replay retention (0/4) and E6 MiLoRA+wd/DoRA+wd evals (0/2)**:
    reviewer panel voted E5 "CE-only/partial" acceptable; decide explicitly
    whether to complete or keep scoped out.
15. **Base-ceiling battery remainder**: 4/22 dirs evaluated; the retshard/bbhAO
    ladders never synced. Headline ceilings (Llama 26.0/35.26, Qwen 44.35) are
    landed; the remainder is redundancy only.

---

## Explicitly not planned (decisions, not oversights)

- **Full PiSSA seven-rate sweep**: disclosed design boundary in the paper
  (recipe-rate only); revisit only if a reviewer demands it.
- **Base-model task accuracy (zero-shot CS-8 / GSM8K)**: base rows report
  retention ceilings by design.
- **284B retraining**: not needed; adapters survived (see Tier 3 - retention
  recovery is eval-only).

## Job-file map (what exists vs needs writing)

| Item | Existing spec |
|---|---|
| 4 (Qwen CE) | `jobs/qwen_ce_recovery.txt` + ledger `jobs/ce_backfill_qwen.txt` |
| 1/2 (E4/B4) | template `archive/jobs_superseded/frepro4_b4.txt` |
| 13 (math seeds) | `jobs/night_final_B.txt` [N1] block (dedupe [N2]/[N3], landed) |
| 3, 5, 6, 7, 8, 9 | no job file yet - write fresh specs |

Historical job files (`qwen3seed_B.txt`, `master_dispatch.txt`,
`frc_reservoir_B.txt`, `overflow_evacuated.txt`) largely landed; dedupe against
`results/` before any reuse.
