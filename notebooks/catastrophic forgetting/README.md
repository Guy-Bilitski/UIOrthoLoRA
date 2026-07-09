# Phase-1 Go/No-Go: Is UIOrthoLoRA a real catastrophic-forgetting mitigator?

**Goal.** Decide whether UIOrthoLoRA's adaptation/retention frontier reaches or beats
**CLoRA**'s (and clearly beats plain **LoRA** on retention) on CLoRA's own commonsense
one-stage benchmark (LLaMA-2-7B). See `agent_instructions.nd` for the full spec.

Decision criterion (spec §1): plot **in-domain commonsense avg (x)** vs **out-domain
retention = mean(BBH, MMLU-Pro) (y)**. GO if UIOrthoLoRA's frontier sits on/above
CLoRA's at matched in-domain accuracy AND clearly beats LoRA on retention.

---

## The setup (identical for every method — this is what makes it a fair comparison)

| | |
|---|---|
| Base | `meta-llama/Llama-2-7b-hf` |
| Train | Commonsense-170K (LLM-Adapters), 3 epochs, lr 3e-4, effective batch 16, bf16, seed 42 |
| Target modules | `q,k,v,up,down` (5 modules — **not** gate/o_proj) |
| Rank/α | LoRA r=32, α=64, dropout 0.05 |
| In-domain eval | 8 datasets, generation MC accuracy, beam=4 (LLM-Adapters protocol) |
| Out-domain eval | **answer-only** BBH (`bbh_fewshot`, 3-shot) + MMLU-Pro (5-shot CoT) via lm-eval |
| Mechanism metric | F∆ = mean ‖ΔW·x‖/‖x‖ over 100 real inputs; ‖ΔW‖ = top singular value |

One shared trainer (`train_cs.py`); methods differ only by (a) adapter and (b) an
optional loss term (CLoRA only). UIOrthoLoRA configs are **param-matched** to LoRA's
56.10M trainable (±1%): rotations cost `2·k_vec²`/module and dominate the budget, so
**k_vec≈410** is forced and **k_val** is the real sweep knob.

---

## Scripts

| Script | Role |
|---|---|
| `run_lib.py` | shared prompt templates, logging, registries |
| `train_cs.py` | shared trainer; `--method lora|clora|uiortholora`. CLoRA penalty + GPU-QR P-matrices live here |
| `eval_cs.py` | in-domain commonsense eval (one dataset); `run_eval(model,...)` reusable in-process |
| `eval_retention.py` | retention via lm-eval (one shard of subtasks) |
| `run_cs_eval.py` | orchestrate 8 CS datasets across GPUs (for adapter-reload methods) |
| `run_retention.py` | shard BBH+MMLU-Pro subtasks across GPUs, size-weighted aggregate |
| `fdelta.py` | F∆ / ‖ΔW‖ from a reloaded adapter |
| `uio_inprocess.py` | **UIOrthoLoRA: train+eval in ONE process on ONE GPU (no save/reload)** |
| `eval_one_gpu.py` | eval a LoRA/CLoRA adapter fully on one GPU, in-process (CS+retention+F∆) |
| `gpu_pool.py` | tiny scheduler: run a job list across GPUs (1 job/GPU), per-job logs |
| `make_report.py` | frontier plot + F∆ table + repro-check table |
| `test_uio_roundtrip.py`, `diag_uio_save.py` | UIOrthoLoRA save/reload diagnostics |
| `jobs/*.txt` | the exact command lists fed to `gpu_pool.py` for the campaign |

Run scripts with `/home/guy/UIOrthoLoRA/.venv/bin/python`. Checkpoints go to
`/scratch/cf_models` (526G root vol); small adapters in `models/`. Results in
`results/<run>/` + `results/campaign_summary.jsonl`.

---

## ⚠️ Critical correctness finding: UIOrthoLoRA checkpoints cannot round-trip

UIOrthoLoRA's effect depends on the SVD basis (U,V) of the frozen weight and on
trained orthogonal rotators defined **relative to that basis**. Two independent PEFT
save/reload bugs corrupt a reloaded adapter (verified: ΔW changes ~8%, logits shift 0.63):

1. **U/S/Vᵀ were dropped** from the checkpoint (registered as plain buffers without the
   `uiortholora_` prefix PEFT's saver requires). **Fixed** in `src/peft/.../layer.py`
   by storing them as frozen (`requires_grad=False`) Parameters under the prefix.
2. **Rotators are still dropped on reload** — PEFT's adapter-name remapping mangles the
   orthogonal-parametrization's nested keys (`…left_unitary.default.parametrizations…`
   → inserts `default` twice), so they silently revert to random init. Not cleanly
   fixable in the PEFT layer.

→ **Decision:** evaluate UIOrthoLoRA **in-process** (`uio_inprocess.py`) — train and
eval the same in-memory model, never reload. This is validated and sidesteps both bugs.
The non-round-trippable checkpoint is itself a finding for the verdict.

---

## Other findings already banked

- **UIOrthoLoRA is heavy**: ~4× slower to train (1.1 it/s vs LoRA ~4.5) and ~8× the GPU
  memory (112GB vs ~14), from the full rank-4096 SVD reconstruction in its forward.
  Faithful checkpoints are ~14.5GB (≈ base model). Each run ≈ 8h train + ~1h eval.
- **D/E gates break the "structurally orthogonal" claim** when `use_de=True`:
  `diag(E)·M·diag(D)` leaves the protected subspace. The spec sweep uses use_de=True
  only; a `use_de=False` arm is the key follow-up to isolate the orthogonality claim.
- **BBH config**: CLoRA's base 34.91 reproduces with **answer-only** 3-shot BBH (got
  33.1), NOT chain-of-thought (gives 39.5). Answer-only is also ~15× faster.
- **CLoRA over-constrains at extreme k** (k2048) under the literal repo loss
  (sum-over-modules, λ=1): in-domain collapses on siqa/hellaswag (CS 65 vs paper 83.7),
  while k128 behaves correctly (CS 79.2 > LoRA 78.1). Under investigation — may need a
  reg normalization to faithfully reproduce the paper's strong CLoRA. **If unaddressed,
  an artificially-weak CLoRA would unfairly flatter UIOrthoLoRA.**

---

## Reproduction gates (must pass before trusting any UIOrthoLoRA number)

| Gate | Target | Achieved | Status |
|---|---|---|---|
| A base retention | BBH 34.91 / MMLU 18.56 | 33.1 / 18.96 | ✅ |
| B LoRA CS | ~79.9 | 78.1 | ✅ |
| B LoRA retention | BBH 26.69 / MMLU 14.46 | 30.7 / 12.6 (mean 21.7 < base 26.0 ⇒ forgetting) | ✅ trend |
| C CLoRA-k2048 | CS 83.7 / BBH 38.67 / MMLU 20.59 | CS 65 (over-constraint) — see findings | ⚠️ |

See `STATUS.md` for live campaign state.
