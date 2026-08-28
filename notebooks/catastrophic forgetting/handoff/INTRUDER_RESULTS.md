# Intruder dimensions — what was run and what came out

Written 2026-08-28. Machine: one H200. All numbers below are measured, not estimated.

---

## 1. The question

Shuttleworth et al. (arXiv 2410.21228) define an **intruder dimension**: a top singular
vector of the fine-tuned matrix `W0 + dW` whose maximum absolute cosine similarity
against *all* singular vectors of the pretrained `W0` is below a threshold. They report
that LoRA creates them, full fine-tuning does not, and that they are linked to forgetting.
Xie (arXiv 2607.23711) adds a per-layer threshold law for when they appear.

We ask two things:
1. Do intruder dimensions **cause** forgetting, or do they just accompany large updates?
2. Do different retention-aware adapter designs create **more or fewer** of them at the
   same update size?

---

## 2. How intruders were measured

Script: `intruder_pass.py` (new file; the trained pipeline is untouched). CPU only.

Per adapted weight matrix:
1. Compute the top-64 singular triplet of `W0 + dW` by warm-started subspace iteration
   using matrix-vector products only (`W0 @ V + scaling * B @ (A @ V)`). The dense sum is
   never formed. Verified against dense SVD to `<1e-3` relative error on singular values.
2. For each of those 64 left singular vectors, take the **maximum |cosine| against the
   full set of pretrained left singular vectors** of `W0` (all 4096, not a top-k subset).
   The full basis is precomputed once per model by `geo_fullref.py`.
3. A direction is an intruder if that maximum is below threshold `tau`.
   Reported at `tau = 0.5, 0.7, 0.9`. Headline uses the paper's canonical setting:
   **`tau = 0.5` within the top `k = 10`**.

Matrices covered: `q_proj, k_proj, v_proj, up_proj, down_proj` in every layer.
- Llama-2-7B: 32 layers x 5 = **160 matrices** (10,240 top-64 slots; 1,600 top-10 slots)
- Qwen2.5-7B: 28 layers x 5 = **140 matrices**

Validation: `intruder_pass.py --selftest` builds synthetic `W0` with planted directions
and checks the detector recovers them exactly — 6/6 cases pass, including a case that
distinguishes a genuinely new direction from one merely aligned with the base model's
*mid*-spectrum (this is why the full basis, not a top-256 subset, is used).

Also recorded per matrix: intruder energy share (sum of squared singular values of
intruder directions / sum over the top-k), `sigma_1(dW)`, and the spike margin
`sigma_1(dW) / sigma_1(W0)`.

---

## 3. Training setup (identical to the frozen pool recipe)

| item | value |
|---|---|
| script | `train_cs.py` (unmodified) |
| training data | `commonsense_170k.json`, **170,420 examples** |
| epochs / batch / cutoff | 3 / 16 / 256 tokens -> **31,956 optimizer steps** |
| precision | bf16 |
| seed | 43 |
| adapter | LoRA r=32, target modules q,k,v,up,down |
| LoRA+wd | alpha=64, weight decay 0.3, lr 5e-4 |
| MiLoRA | alpha=32, minor-subspace init, weight decay 0, lr 1e-3 |
| base models | `meta-llama/Llama-2-7b-hf`, `Qwen/Qwen2.5-7B` |

Training time: ~3.1 h per cell on the H200 (one process on the GPU at a time).

---

## 4. Evaluation setup

Script: `eval_one_gpu.py` (unmodified).

**Task ability ("task" column)** — CS-8 accuracy: mean over `boolq, piqa, social_i_qa,
hellaswag, winogrande, ARC-Easy, ARC-Challenge, openbookqa`, beam search (4 beams),
**200 items per dataset** (`--eval_limit 200`), 1,600 items total.

**Retention** — `retention_mean` = mean of **BBH** and **MMLU-Pro** accuracy via
lm-eval-harness, **50 documents per subtask** (`--ret_limit 50`; BBH has 27 subtasks,
MMLU-Pro 14, so ~2,050 generation requests). MMLU, ARC-Challenge and TruthfulQA-mc2 are
also scored (they are loglikelihood tasks and cost ~3 minutes) but are not in the
headline number, matching the existing campaign convention.

**Update size ("F"):** `fdelta_token_weighted`, the CLoRA F diagnostic, computed by the
eval script itself from the adapter — so every row's size is measured, not assumed.

Why subsets: every comparison here is between *our own* adapters, scored on the
*identical* documents (lm-eval takes the first N deterministically), so these are paired
comparisons. Cost check on the same adapter: full battery (18,543 generation requests,
4 h 16 m) gave retention 25.16 / task 80.20; the subset protocol (2,050 requests, ~28 min)
gave **24.87 / 80.00**. A 0.29 pp difference for a 7x speedup.

---

## 5. The intervention (how "removing intruders" was done)

Script: `intruder_ablate.py`. For each trained adapter, three modified copies were built
on CPU and written as ordinary PEFT adapters (rank r+1, alpha set so PEFT's scaling is
exactly 1), so the frozen eval pipeline scores them unmodified.

- **B — intruders removed.** In every matrix, find the highest-ranking intruder direction
  (`tau=0.5`, full-basis criterion) and subtract `s_j * u_j * v_j^T` from `W0 + dW`.
  Rank-1 per matrix, matching Shuttleworth's own surgical intervention.
- **C — uniform shrink (the control).** Take the *original* update and scale it down
  uniformly until its Frobenius norm equals arm B's. Same size change, no geometry
  targeting. This is the paper's own E1 rescaling intervention used as a control.
- **D — removed then resized.** Arm B scaled back up to the *source's* norm.

Measured amounts:

| source | matrices with an intruder | ||dW||^2 before | after removal | norm ratio |
|---|---|---|---|---|
| Llama LoRA+wd | 158 / 160 | 71,788.9 | 65,292.5 | **0.9537** |
| Llama MiLoRA | 160 / 160 | 1,669,474.6 | 1,459,986.1 | **0.9352** |

So the surgery removes ~9 % of update energy (4.6 % of norm) for LoRA+wd, ~12.5 % energy
(6.5 % of norm) for MiLoRA.

Additionally `scale_adapter.py` built uniform-scaled copies at **1.05x** and **1.12x** of
each source, giving a measured local curve of retention vs update size with direction
held fixed.

---

## 6. Results

### 6a. Intruders are everywhere, including in the best-behaved adapter

Criterion-exact, `tau = 0.5`:

| adapter | F (update size) | retention | intruders in top-10 | intruder energy share |
|---|---|---|---|---|
| Llama LoRA+wd (best operating point) | 0.395 | 24.87 | **908 / 1,600 (57 %)** | 0.43 |
| Llama MiLoRA (high magnitude) | 1.501 | 17.60 | **1,595 / 1,600 (99.7 %)** | 0.92 |

The count **saturates**: by the high-magnitude cell essentially every top-ranked direction
is an intruder, so the canonical *count* cannot discriminate across most of our range.
Intruder **energy share** (0.43 -> 0.92) still has headroom and is the metric to use.

Note the base-model reference matters: using only the base top-256 directions instead of
the full basis gave 3,895 vs 3,894 intruders here (no difference for these adapters), but
the selftest shows a subset reference *can* mislabel mid-spectrum-aligned directions,
which is exactly the confusion that would matter for a minor-subspace method like MiLoRA.

### 6b. Removing intruders does not reduce forgetting (the causal test)

**Llama LoRA+wd** (gentle, best operating point):

| version | F | retention | task |
|---|---|---|---|
| original | 0.395 | 24.87 | 80.00 |
| shrink 0.95x (control) | 0.377 | 24.38 | 79.50 |
| scaled 1.05x | 0.415 | 24.48 | 80.75 |
| scaled 1.12x | 0.442 | 24.16 | 79.69 |
| **B: intruders removed** | 0.402 | **24.45** | **57.62** |
| **D: removed + resized** | 0.421 | **24.26** | **65.75** |

**Llama MiLoRA** (aggressive, high magnitude):

| version | F | retention | task |
|---|---|---|---|
| original | 1.501 | 17.60 | 65.69 |
| shrink 0.94x (control) | 1.412 | 19.09 | 67.00 |
| scaled 1.05x | 1.569 | 16.80 | 65.25 |
| scaled 1.12x | 1.664 | 14.46 | 64.06 |
| **B: intruders removed** | 1.436 | **13.50** | **59.75** |
| **D: removed + resized** | 1.528 | **8.35** | **47.44** |

Retention difference, intruder-removed minus its magnitude-matched control:

| source | B - C | D - source |
|---|---|---|
| LoRA+wd | **+0.07 pp** | **-0.61 pp** |
| MiLoRA | **-5.59 pp** | **-9.25 pp** |

Task-ability cost of removing intruders: **-21.9 / -14.3 pp** (LoRA+wd),
**-7.3 / -18.3 pp** (MiLoRA).

### 6c. Update size alone predicts retention

The uniform-scaling rows, direction held fixed:

- LoRA+wd: F 0.377 -> 0.395 -> 0.415 -> 0.442 gives retention 24.38 -> 24.87 -> 24.48 -> 24.16
- MiLoRA: F 1.412 -> 1.501 -> 1.569 -> 1.664 gives retention 19.09 -> 17.60 -> 16.80 -> 14.46

Monotone decline with size in the MiLoRA case; flat within noise in the low-magnitude
LoRA+wd case (which sits below the forgetting knee, where retention is insensitive).

### 6d. What this means

1. Intruder dimensions are **not** the carriers of catastrophic forgetting. At matched
   update size, deleting them never improved retention; at high magnitude it made
   retention substantially worse.
2. Intruder dimensions carry the **fine-tuning task knowledge**. Deleting them cost 7-22
   points of task accuracy every time, while an equal-size uniform shrink cost ~0.5.
3. Retention is governed by **how large the update is**, consistent with the paper's
   magnitude-first thesis.

### 6e. Caveats to state in the paper

- Arms B and D are *structured* edits: they strip each matrix's largest component
  (`sigma_1(dW)` 219 -> 133 for MiLoRA, against 205 for the size-matched control). Part
  of the harm may come from the surgery being unnatural rather than from the identity of
  the directions. The 1.05x / 1.12x curve bounds this but does not fully separate it.
- We use **full removal** (`lambda = 0`) and accuracy-based retention benchmarks.
  Shuttleworth et al. scaled intruder singular values down *partially* and measured
  pre-training loss. A partial-`lambda` sweep on our metrics is the natural follow-up.
- **One seed per cell** (seed 43). Three separate seed pathologies are documented in this
  codebase (SC-LoRA at seed 42, CLoRA-Llama at seed 43, and a Qwen NaN at seed 43/44), so
  single-seed points should be treated as such.
- Results so far are **Llama only**. Qwen replication is in progress.

---

## 7. What is left to run

Queue `jobs/tierA_final4.txt`, tag `tierAj`, one process on the GPU at a time.
Per training cell: ~3.1 h train + ~40 min eval = **~3.8 h**. Eval-only jobs: ~40 min.

| phase | what | cells | ETA (from 2026-08-28 07:30) |
|---|---|---|---|
| **B** | Llama design quartet at matched size: MiLoRA lr3e-4 (F~0.50), SC-LoRA lr1e-4 (F~0.35), CLoRA lr3e-4 (F~0.44). LoRA+wd (F 0.41) already done. Answers "does any design create fewer intruders at the same update size?" on Llama. | 3 | **~19:00 Aug 28** |
| **C** | Qwen quartet: LoRA+wd lr3e-4, MiLoRA lr1e-4, SC-LoRA lr5e-5, CLoRA lr2e-4 (F 0.18-0.28). Cross-architecture version of the same question. | 4 | **~10:00 Aug 29** |
| **D** | Exp 2 anchors (Qwen rescale ladder: Qwen-CS and Qwen-math, plain LoRA and LoRA+wd). Separate experiment, reviewer-requested. | 4 | **~01:00 Aug 30** |
| **E** | Magnitude-spread and below-knee cells; strengthens the intruder-vs-size trend statistics. Droppable. | 11 | ~Aug 31-Sep 1 |

Intruder scoring of each new adapter runs automatically on CPU (`auto_intruder.sh`),
~10 min per adapter, so geometry is ready as soon as retention is.

Open item: the causal result above is Llama-only. Phase C provides the Qwen replication.
If the machine has to be released early, phases B and C are the ones that matter; phase E
is padding.
