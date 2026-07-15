# DeepSeek-V4-Flash generalization experiment — PREPARED SPEC (run when nodes free up)

Goal: test whether the **magnitude + spectral-spread → retention** forgetting law (found on Llama/Qwen-7B)
generalizes to a much larger, **MoE** model. Not a full sweep — one config per method, single seed.

## Model (verified from HF config 2026-07-15)
`deepseek-ai/DeepSeek-V4-Flash` — MIT, open weights, **accessible (not gated)**. `model_type=deepseek_v4`,
43 layers, hidden 4096, **256 routed experts + 1 shared, 6/token**, vocab 129280, 284B total / 13B active.
- **FP8 block-quant** (e4m3, dynamic act, weight_block_size 128×128), **~158 GB on disk** → fits a node's
  ~480 GB /scratch with ~320 GB headroom for the ongoing sweep. Downloading to d001 now (egress).
- **FP8 training implication (key):** LoRA training needs the base in a trainable form — either dequant
  FP8→bf16 (~568 GB, shards across 8×B200=1.44 TB HBM) or an FP8-QLoRA path (base frozen FP8, adapter bf16).
  DeepSeek ships FP8→bf16 dequant scripts (V3 lineage). **Residual-init methods (milora/sclora/clora) need
  base-weight SVD → require dequant of the targeted matrices first** (extra work vs plain lora/dora).

## Node → method assignment (d002–d008, one adapter per node, single seed=42; LR = 7B prior)
| node | method    | LR    | notes |
|------|-----------|-------|-------|
| d002 | lora      | 3e-4  | baseline, no base-SVD |
| d003 | milora    | 5e-4  | residual-init → needs base-W dequant+SVD |
| d004 | sclora    | 5e-5  | residual-init → needs base-W dequant+SVD; 7B Pareto star |
| d005 | lora_null | 5e-4  | null-space init → needs base-W SVD |
| d006 | clora     | 3e-4  | residual-init → needs base-W dequant+SVD |
| d007 | dora      | 2e-4  | no base-SVD |
| d008 | lorawd    | 5e-4  | lora + weight decay, no base-SVD |
Run ONLY after a node's sweep shard + derive/finalize complete (never steal from the sweep). Attention-only
target_modules first (dense; clean for spread analysis). If FP8 base-SVD proves too costly, fall back to the
no-SVD methods (lora/dora/lorawd/+lora_null) on more nodes for a first pass.

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
