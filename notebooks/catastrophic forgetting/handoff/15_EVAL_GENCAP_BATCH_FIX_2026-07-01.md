# EVAL FIX — Qwen generation cap + batch-size (2026-07-01)

Two eval bugs found while bringing Qwen2.5-7B online for the 2×2. Both are in
`eval_one_gpu.py`. **Neither changes Llama-2 results** (proven below), so the existing
56 L2-CS + 14 L2-math cells stay valid — no re-run.

## Bug 1 — Qwen rambles to 2048 tokens (generation cap didn't bind)
Qwen2.5-7B ships `generation_config.max_new_tokens=2048`. In transformers this OVERRIDES
HFLM's `max_length`, so ~17% of `mmlu_pro`/`bbh` generate_until questions ran to the full
2048 tokens without ever emitting a parseable answer → ~8× slower than Llama-2 (~24h/cell),
and our `--ret_max_gen` flag never bound (it only hits the task GROUP, not the leaf subtasks).
- **FIX:** set `model.generation_config.max_new_tokens = args.gen_cap` (default **512**) on the
  BASE model before the PEFT wrap. New arg `--gen_cap`.
- **Validated:** Qwen base mmlu_pro biology 0.75→0.75, math 0.42→0.44 (the truncated questions
  are ramblers that score 0 either way; genuine CoT is ~140–240 tok, well under 512).
- **A/B no-op proof for Llama-2:** same adapter at cap 512 vs 2048 → retention BYTE-IDENTICAL
  (bbh/mmlu_pro/mmlu/arc/truthfulqa all Δ0; only cs_avg moved 0.75 = bf16/batching noise, which
  is cap-independent). Llama-2 occasionally generates 512–2048 tok but those are non-answers.

## Bug 2 — eval ran at batch size 1 (Qwen's 32k context window)
HFLM's auto batch-sizer reserves memory for a sequence of length `max_length`. Qwen2.5's native
window is 32768 → it concluded only **batch 1** fits (17GB/183GB used, ~50% util) → another ~10×
slowdown. Llama-2 didn't hit this (its window is 4096), which is ALSO why Llama-2 evals were fast.
- **FIX:** pass `max_length=args.max_len` (default **4096**, = Llama-2's window) to both HFLM
  instantiations. New arg `--max_len`. mmlu_pro/bbh 5-shot contexts (~2.5k tok) fit in 4096−512,
  so no truncation; and it MATCHES Llama-2 → better cross-model comparability.
- Net: eval ~8h → ~30–45min/cell. Training is unchanged (3 epochs / full data / eff. batch 16,
  identical to L2 protocol — deliberately NOT sped up, would break comparability).

## Operational
- Both fixes apply automatically to all jobs (defaults gen_cap=512, max_len=4096); `make_*` job
  generators didn't need editing. `--ret_max_gen` is now vestigial (retention uses `gen_cap`).
- Relaunched as ONE 8-GPU pool `camp2` over `jobs/combined.txt` = 11 Qwen reuse-evals (no retrain)
  + 144 train+eval (Qwen-CS 56, Qwen-math 56, L2-math 42), minus the 10 already-trained Qwen cells.
- ETA full 2×2 (seed 42): ~3 days.
- GOTCHA reinforced: `pkill -f '<pat>'` self-matches your own command line — killed my shell once
  here via `pkill -f 'qwsw_batchtest'` while the string was in the same command. Use `[q]wsw...`.
