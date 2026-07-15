# DeepSeek-V4-Flash generalization experiment — PREPARED SPEC (run when nodes free up)

Goal: test whether the **magnitude + spectral-spread → retention** forgetting law (found on Llama/Qwen-7B)
generalizes to a much larger, **MoE** model. Not a full sweep — one config per method, single seed.

## Model
`deepseek-ai/DeepSeek-V4-Flash` — 284B total / 13B active MoE (top-6 of 256 experts + 1 shared),
MIT license, open weights on HuggingFace. Released 2026-04-24.
- Memory: bf16 ≈ 568 GB (does NOT fit a node's ~480 GB /scratch; DOES fit sharded across 8×B200=1.44 TB
  HBM). **FP8 ≈ 284 GB fits /scratch** — VERIFY the HF repo dtype first (decides download+rsync feasibility).
- 13B active ⇒ forward passes far cheaper than dense 284B ⇒ CE/eval tractable for a small adapter set.

## Config: one adapter per method, single seed (LRs = 7B-derived priors; verify on 284B)
lora=3e-4 · milora=5e-4 · sclora=5e-5 · lora_null=5e-4 · clora=3e-4 · dora=2e-4.  EXCLUDE corda
(contaminated, mag 200-586). Core spread-axis set = 6 adapters. lora_null retention-optimal alt = 2e-5.
Rank/alpha: match 7B campaign (r=16/32, alpha=2r). Attention-only target_modules first (dense, clean for
spread analysis); experts optional later. Seed = 42.

## Tasks (research-backed)
DeepSeek-V4-Flash is strong on mainstream code/math/MMLU but weak on specialized domains (SciCode 44.9%,
HLE 32%, terminal 35.6%) — headroom is in narrow/technical domains.
- **ADAPT (narrow domain, headroom, fast to train):** primary = **MedMCQA** (medical MC, subsample ~20-40k;
  big models ~55-65%, clear room, clean single domain, compact). Alt (matches measured weakness) =
  scientific-Python coding (SciCode-style) — sharper but thinner data. [OPEN: confirm with PI/Guy.]
- **RETENTION (5 unrelated domains → real cross-domain forgetting):** MMLU (knowledge), GSM8K (math),
  HumanEval/MBPP (code), HellaSwag/ARC (commonsense), TruthfulQA (truthfulness). None = medicine.
- **CE-drift:** WikiText-103 test, full (`--max_blocks 0`), MiLoRA-comparable — same protocol as 7B.

## Measurements per adapter (same three axes as the 7B campaign)
1. magnitude — fdelta, dw_sv_max/mean (from LoRA factors; size-independent).
2. geometry — **stable_rank, eff_rank, spec, fro from the LoRA factors ONLY** (the predictors that mattered;
   no base-W SVD needed). Skip alignment geometry (e_top/out_top) — needs huge base SVD and didn't matter.
3. CE-drift — forgetting_ce/kl vs base (disable_adapter), full WikiText test.
Then repeat the magnitude+spread→retention correlation/partial-correlation analysis and compare to 7B.

## Pipeline to BUILD (the real engineering — current scripts are single-GPU only)
- Multi-GPU **sharded train+eval stack**: FSDP/DeepSpeed or DeepSeek-native parallelism + PEFT LoRA, model
  sharded across a node's 8 GPUs. train_cs.py / eval_one_gpu.py do NOT shard — need a new path.
- MoE LoRA targeting decision (attention-only first).
- Geometry: reuse geo_drift phase2 spread metrics (factor-only) — works as-is on the saved adapters.
- CE: adapt forgetting_ce.py to load the sharded base model.
- Data: cache MedMCQA + retention/eval sets + WikiText on d001 (egress), rsync to chosen nodes.

## Resourcing / timing / guardrails
- Run only on nodes that have genuinely drained (shard done + derive/finalize complete) — never steal from
  the main sweep. Peel off ~3-5 nodes; each runs a few adapters sharded, sequentially.
- Download+rsync the 284GB(fp8)/568GB(bf16) model is a big one-time op on d001 egress.
- New compute-heavy direction ⇒ get PI nod before committing nodes. Do NOT touch paper.tex / artifact.

## OPEN ITEMS before launch
1. Confirm adapt task (MedMCQA vs SciCode) — needs Guy/PI.  2. Verify HF dtype/size + /scratch fit.
3. Build + smoke-test the sharded train+eval stack on ONE node/ONE adapter.  4. Quick LR-transfer check.
