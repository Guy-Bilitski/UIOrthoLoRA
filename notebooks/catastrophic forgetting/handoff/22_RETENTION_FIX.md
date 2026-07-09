# 22 — Math retention pipeline fix (2026-07-06, zero-GPU pass)

Scope: PI order "fix the math retention pipeline → publication-grade data". Three items:
(P1) MMLU-Pro broken on MATH-trained models; (P2) PiSSA collapse diagnosis; (P3) BBH
convention reconciliation for the paper. Live pool (pid 2932862, tag frepro3) untouched;
no live .py edited — patches below are TEXT, to be applied at the restart window (~T+20h).

---

## P1 — MMLU-Pro on math-trained models: DIAGNOSED (extraction failure, not forgetting)

### Mechanism (confirmed from plumbing + data, no GPU needed)

1. **How retention is computed** (`eval_one_gpu.py` L137–162): `retention_mean =
   mean(bbh_fewshot, mmlu_pro)` via in-memory lm-eval `simple_evaluate`;
   `--ret_suite broad` adds mmlu/arc_c/truthfulqa into `retention_broad`.
2. **The MMLU-Pro task** (installed lm-eval `tasks/mmlu_pro/_default_template_yaml` +
   `utils.py`): **5-shot CoT, generate_until** (stop on "Question:", max_gen 2048), prompt
   ends `"Answer: Let's think step by step."`, and the ONLY extraction filter is the regex
   `answer is \(?([ABCDEFGHIJ])\)?` — it requires the literal text `answer is ` followed
   immediately by an option LETTER (optionally parenthesised). No fallback, no colon
   tolerance, no value-matching.
3. **What MetaMathQA training installs** (`repro/LLM-Adapters/ft-training_set/
   metamathqa_395k.json`, verified on raw rows): every one of 395K responses terminates
   with `"The answer is: <value>"` — **a colon, then a NUMBER/expression, never a
   letter**. After 3 epochs at lr 3e-4 (train_on_inputs=true, train loss 0.23–0.43) the
   model answers MMLU-Pro in exactly that format: `The answer is: 42` (the option's
   VALUE). The stock regex fails twice over: the `:` breaks the match, and the model
   emits the value, not the letter A–J.
4. **Dissociation proof from existing numbers** (no samples were saved —
   `log_samples` was never enabled — but the metric plumbing makes the mode
   unambiguous):
   - Base Llama-2 (mimics the few-shot exemplars, which end "the answer is (X)"):
     MMLU-Pro **18.96**. All 7 faithful math cells: **0.0–14.2**, 6/7 ≤ random-10%.
     Below-random on a 10-option MC task is the signature of systematic
     extraction failure, not graceful forgetting.
   - **Dose-response**: the lightly-tuned `lrswm_*` math cells (100K subset) hold
     MMLU-Pro 12–18 at low LR and slide to 8–10 at lr 1e-3; the faithful `frm_*` cells
     (395K × 3 epochs = strongest format lock-in) are floored ≤ 9.5 with PiSSA (lowest
     train loss 0.227 = hardest lock-in) at exactly **0.0**. Format conversion, not
     knowledge loss, tracks training dose.
   - Likelihood-based tasks need no extraction and behave smoothly across the same
     models (mmlu 23–28, arc_c 24–38) — no 0.0 anywhere.

### Interim fix (DONE, zero-GPU)

`retfix_bbh_only_report.py` re-reports every completed math cell (56 found:
frm_/lrswm_/mtxm_/qwswm_) with `retention_math` = BBH answer-only, from existing
`results/*/summary.json` only. Outputs:
- `results/retention_bbh_only.jsonl` (machine)
- `results/retention_bbh_only.md` (table, with %of-base-33.10 for Llama-2 cells and an
  `mmlu_pro_extract_fail` flag). `campaign_summary.jsonl` untouched.

Faithful-cell ranking under the fixed axis (base ceiling 33.10):
LoRA+wd0/lr1e-4 **30.96** > MiLoRA 30.18 > LoRA 29.14 > CLoRA k256 28.61 > k64 28.24 >
k128 27.55 ≫ PiSSA 7.23. Every qualitative story survives (consistent with synthesis R6).

### Permanent fix — RECOMMENDATION: (ii) drop MMLU-Pro from the math retention axis

- **Why not (i) a tolerant extractor:** a fixed regex chain (`answer is:?\s*\(?([A-J])`,
  `Answer:\s*([A-J])`, …) only recovers cases where the model emits a LETTER. A
  MetaMath-locked model predominantly emits the option VALUE (`The answer is: 42`);
  no extractor can map that back to a letter without option-value matching, which is a
  different (and non-comparable) metric. It also breaks comparability with CLoRA's
  published MMLU-Pro column and would force re-scoring every non-math cell.
- **Paper-facing justification (for §Setup):** "For math-tuned models we report
  retention on BBH (answer-only 3-shot); MMLU-Pro's CoT-extraction protocol is
  confounded with the instruction-tuned answer template installed by MetaMathQA
  (models emit `The answer is: <value>`, unparseable by the benchmark's letter
  regex), producing below-random scores that measure format shift rather than
  knowledge retention (base 18.96 → tuned ≤ 9.5 while likelihood-based MMLU is
  method-invariant). We retain MMLU-Pro as a flagged secondary diagnostic."
- Matches supervisor ruling R6 in handoff/21 (BBH-only primary; MMLU-Pro disclosed
  as broken secondary). Optional Tier-B: the tolerant-extraction experiment is still
  spec'd in the gate script (measures how much IS recoverable) — diagnostics only.

### Patch for `eval_one_gpu.py` (apply at restart; do NOT apply while pool live)

Two hunks; keeps `retention_mean` for backward comparability, adds `retention_bbh`:

```python
# HUNK 1 — after the ret_mean line (currently L160):
     ret_mean = round(((ret.get("bbh") or 0) + (ret.get("mmlu_pro") or 0)) / 2, 2)
+    # Format-robust math retention axis (P1, handoff/22): BBH answer-only alone.
+    # MMLU-Pro's letter-regex cannot parse MetaMath-style answers ("The answer
+    # is: 42"), so retention_mean is broken for math-trained cells. Base ceiling
+    # (Llama-2, full set, answer-only) = 33.10.
+    ret_bbh = ret.get("bbh")

# HUNK 2 — in the headline dict (currently L164-168):
     headline = {"cs_avg": cs_avg, "adapt_task": args.adapt_task, **ret,
                 **{k: cs[k] for k in ("gsm8k", "math") if k in cs},
-                "retention_mean": ret_mean, "retention_broad": ret_broad,
+                "retention_mean": ret_mean, "retention_bbh": ret_bbh,
+                "retention_broad": ret_broad,
                 "fdelta": fd.get("fdelta_token_weighted"),
```

(If fix (i) were ever wanted despite the recommendation: create
`lm_eval/tasks/mmlu_pro/mmlu_pro_tolerant` overriding only `filter_list` with the
regex chain `['answer is:?\\s*\\(?([ABCDEFGHIJ])\\)?', '[Aa]nswer:\\s*\\(?([ABCDEFGHIJ])\\)?']`
+ take_first, task name `mmlu_pro_tolerant`, and add it to `CORE`. NOT recommended —
see above; and per the campaign rule, ship it via an idempotent `ensure_*` self-heal
like `bbh_metric_fix.py`, never by hand-editing site-packages.)

### 1-GPU validation gate (restart window; script ready)

```
CUDA_VISIBLE_DEVICES=0 python retfix_retention_gate.py --mode mmlupro \
    --adapter /scratch/cf_models/frm_lora_lr3e4_c256_s42        # ~20-30 min
```
Runs mmlu_pro with `log_samples=True` (limit 10/subtask), dumps raw generations to
`results/retfix_diag/frm_lora_lr3e4_c256_s42/mmlu_pro_samples.jsonl`, then scores the
stock vs tolerant extractors offline. PASS = tolerant score lands in [10, 19]
(→ failure confirmed AND partially recoverable) OR >50% of generations contain no
option letter at all (→ definitively unrecoverable → the drop decision is final).
Either outcome confirms the diagnosis; the branch only decides whether MMLU-Pro can
be salvaged as a Tier-B secondary.

---

## P2 — PiSSA collapse: verdict from offline artifacts

**Ruled OUT: RESIDUAL-BUG.** `logs/frepro_1.log` L6: `[pissa] major-SVD init applied
(alpha=128,r=64); loss-preserving err=1.94e-03` (correct init); L7417:
`[residual_save] converted 160 layers to W0-relative rank-128 adapter (was rank-64)`
and the saved `adapter_config.json` shows r=128/alpha=256 — the alpha/r=2.0 scaling is
preserved exactly per the residual_save recipe. A broken residual reload zeroes/garbles
ALL generation; GSM8K 49.66 (655/1319 correct, full CoT) rules that out. Training was
healthy: smooth loss to **0.2272 — the LOWEST in the faithful set** (LoRA ~0.28+).

**Evidence table (all offline):**

| signal | PiSSA | peers (6 faithful cells) | reading |
|---|---|---|---|
| BBH answer-only | **7.23** | 27.55–30.96 | far below even MC-subtask guess floors → format suspect |
| MMLU-Pro | 0.0 | 6.9–14.2 | total format lock-in (P1 mechanism, strongest) |
| mmlu (likelihood) | **24.54** | 23.0–27.9 | **identical to peers** — knowledge probe shows NO extra forgetting |
| arc_c (likelihood) | 24.06 | 28.0–37.7 | mildly low |
| truthfulqa_mc2 | **52.65** | 41.0–45.8 | elevated — classic degeneracy/low-confidence artifact |
| GSM8K faithful | 49.66 (pub. 58.23) | 58.5–65.0 | real −8.6 pp vs published: genuine quality loss |
| MATH parse_fail | **1400/5000 (28%)** | 665–804 (~13–16%) | 2× \boxed-format breakdown → generation-format degradation |
| fdelta / train loss | 2.21 (max) / 0.227 (min) | ≤1.28 / ≥0.28 | largest drift + hardest task fit |

**VERDICT: EVAL-ARTIFACT-dominant on the retention axis, with real secondary drift.**
The likelihood-based probe (mmlu = peers) shows PiSSA has NOT catastrophically
forgotten more knowledge than the other adapters; its generative-format lock-in is
simply the most extreme (lowest train loss → most complete conversion to MetaMath
output style → BBH exact-match and MMLU-Pro extraction both starve). On top of that
there is a REAL adaptation-quality deficit (GSM8K −8.6 pp vs published, 2× MATH
parse-fail) consistent with over-drifting the principal directions (‖ΔW‖ largest,
fdelta 2.21) — i.e. our PiSSA repro point is genuinely worse-trained than CLoRA's,
but its BBH 7.23 is NOT a clean forgetting measurement and must not be plotted as one.

**Cannot be 100% settled offline** (no BBH generations were saved). Decisive 1-GPU
diagnostic at restart (~20–30 min):

```
CUDA_VISIBLE_DEVICES=0 python retfix_retention_gate.py --mode pissa \
    --adapter /scratch/cf_models/frm_pissa_lr3e4_c256_s42
```
Dumps BBH generations + targets, reports `target_in_gen − exact_match` (content right
but format wrong ⇒ EVAL-ARTIFACT) vs empty/wrong generations (⇒ REAL-FORGETTING), and
counts MetaMath-style ("answer is:") leakage into BBH answers. Paper action pending the
gate: exclude the PiSSA cell from magnitude-law fits (already the synthesis position —
the 7-point law was leverage-driven by this cell) and report it with the diagnosis note.

---

## P3 — BBH convention reconciliation: they are ALREADY the same axis

**Finding: there is only ONE BBH convention in the adapter data.** The apparent
conflict is a base-ceiling measurement discrepancy, not a task-config difference.

| | old lrsw/n=49 registry rows | faithful frm_* cells |
|---|---|---|
| task | `bbh_fewshot` (answer-only 3-shot) via eval_one_gpu.py | **same task, same script** |
| evidence | summary.json git 21517195 (2026-06-25) | git ec6bfea2 (2026-07-05/06) |
| exact_match metric | raw (pre-fix) | + `bbh_metric_fix` normalization (2026-07-01) |
| metric delta for Llama-2 | — | **verified byte-identical no-op** (handoff/16: n=20 × 5 subtasks, 0.47→0.47) |
| gen plumbing | pre gen_cap | gen_cap=1024 / max_len=4096 (immaterial for short answer-only gens) |

The two "base ceilings" floating around:
- **33.10** = `results/base_l2-7b_bbhAO/retention_agg.json`, answer-only, FULL set
  (250/subtask) → this is the paper's number, and it is the correct full-set ceiling.
- **36.57** = `base_retention_check.py` spot-check at **limit=40/subtask**
  (logs/base_retention_check.log header says so explicitly). lm-eval `limit` takes the
  FIRST 40 docs per subtask — a small, non-random subsample. It is NOT a different
  convention and must not be quoted as a ceiling.
- 39.51 (`results/base_l2-7b`) is `bbh_cot_fewshot` — a different task, used only in
  the very early June-9 base run; no adapter cells were ever scored with it
  (run_retention.py L119–120 already documents answer-only as canonical).

**RECOMMENDATION (cheap option, and it is also the correct one): no re-eval of any
frepro cell.** The frm_* retention numbers are already on the paper's axis. Actions:
1. Paper: keep "BBH answer-only 3-shot, base ceiling 33.10" everywhere; add one
   harness sentence: "identical task config across all campaigns; the exact-match
   normalization added for Qwen (2026-07-01) is a verified no-op for Llama-2."
2. Purge "36.57" from any doc that cites it as a base ceiling (it appears only in the
   spot-check log; grep shows no summary.json carries it).
3. One cheap confirmation run at restart (NOT per-cell): full-set base BBH under the
   current harness —
   `CUDA_VISIBLE_DEVICES=0 python retfix_retention_gate.py --mode base_bbh` (~1–2 GPU-h).
   PASS = |score − 33.10| ≲ 1.0 pp → the no-op claim is confirmed at full scale and the
   two campaigns share one axis by measurement, not just by argument.

**Cost comparison** (why not the re-eval option): re-evaling retention for the frepro
cells under "the paper's config" would be re-running the SAME task — 27 subtasks × 250
docs ≈ 6.5k short answer-only gens ≈ 1–1.5 GPU-h/cell → ~8–10 GPU-h for the 7 done
cells, ~150 GPU-h if applied to the full 103-cell campaign — to reproduce numbers we
already have. The single base confirmation (≤2 GPU-h) buys the same paper claim.
No re-eval job lines are therefore provided; if the gate FAILS (delta > ~1 pp), escalate:
that would mean the metric fix is not a no-op at full scale, and then the 7 frm_* cells
need a BBH-only re-eval (7 × ~1.5 GPU-h ≈ 10 GPU-h; job lines = the gate script in a
loop over adapters, or eval_one_gpu.py with a retention-only flag added at that point).

---

## Artifacts written (this pass — nothing else touched)

1. `retfix_bbh_only_report.py` — zero-GPU re-reporter (ran successfully: 56 math cells).
2. `results/retention_bbh_only.jsonl` + `results/retention_bbh_only.md` — the fixed
   `retention_math` (BBH-only) column for every completed math run.
3. `retfix_retention_gate.py` — the 3-mode 1-GPU gate for the restart window
   (compile-checked; NOT run — zero-GPU pass). No live script imports any retfix_ file
   (verified by grep).
4. This report.

## Restart-window checklist (order matters, ~3 GPU-h total on one GPU)

1. Apply the two-hunk `eval_one_gpu.py` patch above (adds `retention_bbh`).
2. `--mode mmlupro` gate on `frm_lora_lr3e4_c256_s42` (~30 min) → confirms P1 branch.
3. `--mode pissa` gate on `frm_pissa_lr3e4_c256_s42` (~30 min) → settles P2 verdict.
4. `--mode base_bbh` (~1–2 h) → locks P3 single-axis claim at full scale.
5. Re-run `python retfix_bbh_only_report.py` after each new frm_ batch lands to keep
   `retention_bbh_only.*` current; paper tables consume `retention_math` from it (or
   `retention_bbh` from post-patch summaries) — never `retention_mean` for math cells.
