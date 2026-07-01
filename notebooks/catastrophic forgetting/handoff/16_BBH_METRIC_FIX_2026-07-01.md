# BBH + PAD + CAP fixes for Qwen eval (2026-07-01) — read with handoff/15

Bringing Qwen2.5-7B online exposed FOUR eval issues (all Qwen-specific; Llama-2 unaffected,
so the 56 L2-CS + 14 L2-math cells stay valid — verified no-op each time). Bugs 1–2 are in
handoff/15 (committed fe0f9be3). Bugs 3–4 below.

## Bug 3 — pad token = "!" for Qwen (`eval_one_gpu.py`)
We hardcoded `tokenizer.pad_token_id = 0`. Qwen's token 0 decodes to **"!"** (real pad is
`<|endoftext|>`=151643); Llama-2's token 0 is `<unk>` (harmless). Batch-padding finished
sequences with "!" put trailing `!!!!` in decoded responses. FIX: only fall back to 0 when
the tokenizer has no declared pad (`if tokenizer.pad_token_id is None: =0`). No-op for Llama-2.

## Bug 4 — bbh_fewshot exact_match had no normalization (`bbh_metric_fix.py`)
Retention CORE = `["bbh_fewshot", "mmlu_pro"]`. **bbh_fewshot is the NON-CoT variant**: direct
answer, raw `exact_match`, NO filter. Qwen emits a leading space (" -33" for target "-33"), so
even CORRECT answers scored wrong -> **bbh=0.00 for Qwen** (mmlu_pro was fine — it has a regex
that grabs one letter). FIX = patch the installed lm-eval bbh fewshot template metric to
`ignore_case: true` + `regexes_to_ignore: ["^\\s+", "[\\s.]+$"]` (strip surrounding whitespace
+ trailing '.'). **NOT ignore_punctuation** — that deletes minus signs and corrupts numeric
answers. Applied idempotently at eval startup via `bbh_metric_fix.ensure_bbh_fewshot_metric_fix()`
(wired into eval_one_gpu.py) so a lm-eval reinstall can't silently reintroduce bbh=0.
- Measured (raw vs normalized, 5 diverse subtasks, n=20): **Qwen 0.00 -> 0.54**,
  **Llama-2 0.47 -> 0.47 (identical every subtask)** = provable no-op for Llama-2.
- NB: `bbh_cot_fewshot` (a different task, with a regex filter) also works for Qwen (0.875 on
  arithmetic) but we did NOT switch to it — would need a Llama-2 re-run. Non-CoT + normalization
  keeps the existing metric and Llama-2 cells valid.

## Cap decision (`--gen_cap`, default 1024)
The bbh=0 was NOT a cap issue (bbh gen ~343 tok). Cap only bounds mmlu_pro ramblers. Sweep
(cap 512/1024/2048, 2 adapters): mmlu_pro gains 512->1024 on the verbose math model
(43.6->47.0) then diminishing; bbh needs <512. **Chose 1024** = captures Qwen's longer genuine
CoT without the ~2x slowdown (and ~2h/cell for high-LR) of 2048. Consistent across all cells.
`max_len=4096` (handoff/15) still applies (fixes the batch=1 -> 32 issue).

## Net eval config for the 2x2 (both models)
pad = declared-or-0 · gen_cap=1024 · max_len=4096 · retention = bbh_fewshot(normalized)+mmlu_pro
+ mmlu/arc_c/truthfulqa. Relaunched as one 8-GPU pool over jobs/combined.txt (155 jobs; 11 Qwen
reuse-evals + 144 train+eval). Cleaned all measurement artifacts + 2 invalid bbh=0 Qwen cells.

## Gotcha reinforced
`pkill -f '<pat>'` self-matches your own command line (killed my shell via `pkill -f
'qwsw_batchtest'` and via a plain `gpu_pool.py`); always bracket the pattern (`[q]wsw`, `[g]pu_pool`).
