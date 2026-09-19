> **Current decoder plan — reviewed 19 September:** Read
> `DECODER_SUBSPACE_STUDY_20260919.md` and `CODING_AGENT_PROMPT_20260919.md`.
> Test leading/middle/tail support with our strict DIAG/ROT128 adapters; no
> LoRA/PiSSA comparison or new penalty sweep. The plan includes bounded balanced
> tuning, a two-GPU timing budget and a predeclared DIAG-only fallback. Older
> decoder commands, matrices and costs below are historical and superseded.

> **Latest author priority — modern decoder first:** Read
> `DECODER_FIRST_PRIORITY_20260919.md`. Prioritize the prepared decoder pilot,
> tuning/calibration and confirmation runs on the two assigned GPUs. Defer
> CENTER and further encoder controls. The coding agent has now delivered the
> temperature and band held-aside exports; verify/reuse them, do not rerun them.
> Older schedules below are superseded where they place controls first.

# Compact decoder pilot — design for discussion (19 September 2026)

Candidate: **Qwen2.5-1.5B-Instruct on GSM8K**. Neither the model/task nor the
run matrix is frozen; this document gives the implemented, CPU-tested pipeline,
what the CPU feasibility check measured, the proposed protocol, the open
decisions and a two-GPU cost. No GPU run has happened.

## 1. What exists and was tested

Package `notebooks/iclr/decoder_pilot/` (isolated from the RoBERTa runner; reuses
the tested `SpectralLinear`, `LoRALinear`, `CachedRegularizer` and
`block_summary` from `notebooks/iclr/campaign/`):

| Module | Content |
|---|---|
| `adapters.py` | Placement on the square `q_proj`/`o_proj` of every layer (56 modules); arms UNREG/MIX/NORM (spectral, tail 512 of rank 1536, leading core I, scalers), LORA (rank 8, alpha 16, B = 0), PISSA (rank 8, principal-component factors over a frozen residual). One cached full SVD per module is the shared frame for every arm; `validate_arm` checks forward = W_pre + Δ, merge/unmerge, disable/restore for all arms. |
| `data.py` | Pinned GSM8K parquet reading through the campaign's sealed-source verifier; seeded disjoint inner split (10% of official train = selection, never the test split); Qwen chat template with a fixed system prompt; completion-only targets (`-100` on prompt tokens, EOS appended); `#### <number>` extraction with a last-number fallback; normalized exact match. |
| `engine.py` | Fixed-step AdamW with linear decay, float32 masters + bf16 autocast, token-mean completion NLL over the effective batch, penalty added exactly as in the RoBERTa engine, per-step log (task loss, penalty, raw mixing, grad norm, peak memory), adapter-only checkpoints bound to a frozen-source fingerprint, selection by inner token NLL (earliest tie, step 0 excluded) with the fixed endpoint kept separately, `validate_reload` (fresh model + trainable tensors must reproduce the recorded NLL). |
| `evaluate.py` | Greedy decoding (left padding, adapters merged for speed and restored after), one scorer for all arms, held-out completion NLL of the reference solution, per-example export without prompt text. |
| `geometry.py` | Four-block energies of the effective update W_eff − W_pre in the identical pretrained frame for every arm; scaler statistics for spectral arms; PiSSA's frozen-residual shift and learned factor change reported separately. |
| `pilot.py` | `prepare` (pinned download + seal + SVD cache), `feasibility` (CPU), `train` (GPU; refuses without an explicit resource record naming the GPU and `CUDA_VISIBLE_DEVICES` = that GPU's UUID). |
| `tests/test_decoder_pilot.py` | 15 CPU tests on a tiny synthetic Qwen2 config: every arm trains only adapter tensors (lm_head, k/v, MLP, embeddings frozen), forward/delta/merge/disable agreement, PiSSA principal initialization with zero effective delta and frozen residual, rectangular grouped-query k/v handled with full complements, block accounting, masking/NLL/scoring, seeded split, four-step training + checkpoint + exact reload for all five arms. |

Pinned inputs are sealed under
`campaign_outputs_decoder_v1/inputs/` (model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, GSM8K revision
`740312add88f781978c0658806c59bc2815b9866`, SHA256 manifests) with the SVD cache
`svd_references_q_o.pt` (56 modules, 23 s on CPU, sha `140057fb…`). This was a
CPU/network preparation step under the historical public-download policy; no GPU
was touched.

## 2. CPU feasibility (measured, `campaign_outputs_decoder_v1/feasibility_cpu_20260919.json`)

| Item | Value |
|---|---|
| Model | 1.54 B parameters, 28 layers, hidden 1536, 12 heads / 2 KV heads, vocab 151,936; float32 masters 6.17 GB |
| Adapted projections | q_proj, o_proj: 1536×1536 (k_proj/v_proj are 256×1536 and are excluded from adaptation in every arm by the first-scope decision) |
| SVD frame | 56 full SVDs, 23 s CPU; buffers 1.59 GB (u, v, s, w_pre) |
| Splits | train 6,726 / selection 747 (seed 271828) / held-aside test 1,319; gold answer extracted for 100% |
| Sequence length | mean 224 tokens, p95 354, max 512; at max_length 512, 7 train and 3 test examples truncate (0.1%) → **recommend max_length 640** so nothing truncates |
| Prompt / completion | mean 101 / 123 tokens |
| Pretrained completion NLL | 0.598 nats/token on two selection examples (instruct model already follows the format) |

## 3. Proposed protocol (to agree before launch)

- **Shared across arms:** one model revision, splits, prompts, completion-only
  loss, 56 adapted modules, effective batch 16 (4 × 4 accumulation), max_length
  640, fixed **2 epochs = 842 optimizer steps**, linear decay, no warmup, grad-clip
  1.0, bf16 autocast over float32 masters, inner-selection token NLL every 100
  steps and at the prescribed checkpoint fractions, greedy decoding with 320 new
  tokens, one scorer. Held-aside test generation happens once per confirmation
  at the fixed endpoint; tuning/calibration runs decode the selection split only.
- **Arms:** UNREG, MIX (coefficient 1e-3 as in the registered practical
  protocol), NORM (coefficient calibrated to MIX's pooled norm, same rule family
  as the practical study), LoRA rank 8 alpha 16, PiSSA rank 8. Parameter counts
  are reported, not equated (spectral arms: 56 × (1536 + 1536 + 512) = 200,704;
  LoRA/PiSSA rank 8: 56 × 2 × 8 × 1536 = 1,376,256).
- **Declared tuning budget:** learning rate ∈ {3e-4, 1e-3, 3e-3} for the spectral
  family (chosen on UNREG, then shared by UNREG/MIX/NORM as in the practical
  study) and ∈ {1e-4, 3e-4, 1e-3} separately for LoRA and for PiSSA, all at the
  calibration seed 31415, selected by inner-selection token NLL only. Initial
  scaler/coefficient 0.01 as in RTE.
- **Calibration:** MIX at the selected LR gives the target pooled norm; NORM
  grid {1e-2, 1, 100} then at most two rule-derived refinements (same
  midpoint/extension rule as CENTER).
- **Confirmations:** 5 arms × seeds 42, 17, 123 = 15 runs. Every seed is
  reported; null or reversed outcomes are retained.
- **Measured separately:** SVD setup time, reference storage, steady-state
  step time and tokens/s, peak allocated/reserved memory, generation time,
  diagnostics time (all already logged by `pilot.py train`).

## 4. Cost on two RTX 3090s

The first GPU pilot (UNREG, 100 steps + selection decoding) measures the true
step time; the estimate below assumes ≈2.5 s per 16-example step at ≈300 tokens
(1.5 B forward/backward in bf16 with frozen base) and ≈15 min to greedily decode
1,319 test questions at batch 32.

| Stage | Runs | Per run | GPU-hours | Wall on 2 GPUs |
|---|---:|---:|---:|---:|
| GPU pilot (memory/throughput/learning signal, 100 steps) | 1 | 0.3 h | 0.3 | 0.3 h |
| LR tuning (3 spectral + 3 LoRA + 3 PiSSA, decode selection) | 9 | ≈0.9 h | ≈8 | ≈4 h |
| Calibration (MIX + 3–5 NORM) | 4–6 | ≈0.9 h | 4–5.5 | 2–3 h |
| Confirmations (decode test) | 15 | ≈1.0 h | ≈15 | ≈7.5 h |
| Reload validation + geometry (in-run) | — | included | — | — |
| **Total** | **29–31** | | **≈28–30** | **≈14–15 h**, ≈1 day of the two GPUs |

Memory: 6.2 GB masters + 1.6 GB frame + activations for 4 × 640 tokens without
gradient checkpointing ≈ 6–8 GB + optimizer states (< 0.1 GB) → ≈15 GB expected;
`--gradient-checkpointing` is available if the pilot measures more.

## 5. Open decisions for the author

1. **Instruct versus base model.** The instruct checkpoint already emits the
   GSM8K format (pretrained completion NLL 0.60 nats/token). Fine-tuning may
   move exact match only a few points, which weakens the learning signal; the
   base `Qwen2.5-1.5B` has more headroom but no chat template. The pilot's first
   100 steps plus a zero-shot decode of the selection split will show the
   pretrained exact match and the early loss slope; decide after that number,
   not by which arm looks favorable.
2. **Adapted module set.** q/o only (all arms, rectangular k/v excluded) is the
   implemented first scope; adding k/v (rectangular, full complements supported
   and tested) doubles SVD storage and changes every arm equally.
3. **Tail size / cutoff rule** for rank 1536: 512 (one third, mirrors 256/768) is
   the default; 256 is the alternative. Pin before confirmation.
4. **MIX dose** 1e-3 carried over from RoBERTa; a bracketing check {1e-4, 1e-2}
   at the calibration seed costs two runs if wanted.
5. **Fallback:** if the decoder pilot's step time exceeds ≈5 s or memory exceeds
   the card, drop to matched LoRA/PiSSA on the existing RTE/MRPC protocol (12
   confirmations plus tuning; `modeling.LoRALinear` already exists there, PiSSA
   would be ported from `decoder_pilot/adapters.py`). Do not run both expansions.

## 6. Exact commands (after authorization)

```bash
cd /media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
# resource record: assigned_gpu_ids, output_root=.../campaign_outputs_decoder_v1, downloads_permitted=false
RES=notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_DECODER.json
UUID=$(nvidia-smi -i <GPU> --query-gpu=uuid --format=csv,noheader)
# GPU pilot: throughput/memory/learning signal, 100 steps, decode the selection split
CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.pilot train \
  --resources $RES --gpu <GPU> --arm UNREG --stage pilot --seed 31415 --lr 1e-3 --steps 100 \
  --max-length 640 --eval-every-steps 50 --generation-split selection
# tuning example (spectral family LR grid at the calibration seed)
for LR in 3e-4 1e-3 3e-3; do CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.pilot train \
  --resources $RES --gpu <GPU> --arm UNREG --stage tuning --seed 31415 --lr $LR --steps 842 --max-length 640; done
# calibration example (MIX target, then NORM grid)
CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.pilot train --resources $RES --gpu <GPU> \
  --arm MIX --stage calibration --seed 31415 --lr <selected> --coefficient 1e-3 --steps 842 --max-length 640
# confirmation example (decode the held-aside test once at the fixed endpoint)
CUDA_VISIBLE_DEVICES=$UUID PYTHONPATH=src .venv/bin/python -m notebooks.iclr.decoder_pilot.pilot train --resources $RES --gpu <GPU> \
  --arm PISSA --stage confirmation --seed 42 --lr <selected> --rank 8 --steps 842 --max-length 640 --generation-split held_aside_test
```

A registration/admission layer equivalent to `center_plan.py` (sealed protocol
JSON, exact-entry admission, ledger) is not yet written for the decoder; it is
the first coding step after the author fixes the decisions above, and the
`--stage` field is not a substitute for it.
