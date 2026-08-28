# Intruder-dimension experiment — final settings

Approved 2026-08-28. This supersedes all earlier plans. Nothing here is provisional.

---

## 1. Question

**Primary.** Do intruder dimensions *cause* catastrophic forgetting, or is forgetting
determined by the *size* of the weight update?

**Secondary.** Do intruder dimensions specifically carry *task adaptation*, or is their
apparent importance just a consequence of their spectral magnitude?

Retention and adaptation are measured separately, and both questions are tested causally.

---

## 2. Configurations — 7

One per adapter design, at its best Pareto operating point on the frozen pool's measured
(adaptation, retention) frontier. **No learning-rate sweep.**

| # | model | method | lr | seed |
|---|---|---|---|---|
| 1 | Llama-2-7B | LoRA+wd (r32, alpha64, wd 0.3) | 5e-4 | 43 |
| 2 | Llama-2-7B | MiLoRA (r32, alpha32) | 3e-4 | 43 |
| 3 | Llama-2-7B | CLoRA (r32, alpha64, k=1024, lambda=1.0) | 3e-4 | 44 |
| 4 | Qwen2.5-7B | LoRA+wd (r32, alpha64, wd 0.3) | 1e-4 | 43 |
| 5 | Qwen2.5-7B | MiLoRA (r32, alpha32) | 1e-4 | 43 |
| 6 | Qwen2.5-7B | SC-LoRA (r32, alpha32, beta 0.5, nq_open calib 256) | 2e-5 | 43 |
| 7 | Qwen2.5-7B | CLoRA (r32, alpha64, k=1024) | 2e-4 | 44 |

**Llama SC-LoRA is excluded** — its only pool run (lr 1e-3) collapsed to 0.64 retention
(F_delta 7.38), so no trustworthy operating point exists.

Kept as an extra, off-Pareto row for the magnitude contrast: Llama MiLoRA @ 1e-3
(F_delta 1.50).

---

## 3. Training — frozen pool recipe, `train_cs.py` unmodified

| | |
|---|---|
| data | `commonsense_170k.json`, 170,420 examples |
| epochs / batch / cutoff | 3 / 16 / 256 tokens = **31,956 steps** |
| precision | bf16 |
| target modules | q_proj, k_proj, v_proj, up_proj, down_proj |
| time | ~3.1 h per configuration |

---

## 4. Baseline evaluation — `eval_one_gpu.py` unmodified

| quantity | how |
|---|---|
| **task accuracy** | CS-8 mean (boolq, piqa, social_i_qa, hellaswag, winogrande, ARC-Easy, ARC-Challenge, openbookqa), 200 items per dataset |
| **retention** | mean of BBH and MMLU-Pro, 50 documents per subtask |
| **update size** | `fdelta_token_weighted` (F_delta), computed by the eval script from the adapter |

Every adapter is scored on the **same documents**, so all comparisons are paired.
Protocol validated against the full battery on one adapter: 24.87 / 80.00 versus
25.16 / 80.20 — a 0.29 pp difference for a 7x speedup.

---

## 5. Intruder measurement — `intruder_pass.py` (CPU)

For every adapted matrix `W' = W0 + dW`:

1. Top-**10** singular directions of `W'`, by warm-started subspace iteration using
   matvecs only (the dense sum is never formed; verified against dense SVD to <1e-3).
2. For each top-10 left singular vector `u'_j`, compute
   `s_j = max_i |<u'_j, u_{0,i}>|` over **all** pretrained left singular vectors.
3. **Intruder iff `s_j < 0.5`.**

Recorded per configuration: number of intruders among the top-10, intruder fraction, and
**intruder spectral-energy share** `sum_{j in I} sigma_j^2 / sum_{j=1..10} sigma_j^2`.
Aggregated over 160 matrices (Llama) / 140 (Qwen).

**What this means.** `W0` is full rank, so its singular vectors span the whole space —
no direction is outside their span. The criterion tests whether a direction aligns with
any *single* pretrained singular vector, i.e. whether it is **delocalised**. Measured:
200 random unit directions give max|cos| ~ 0.059, so 100 % of random directions qualify
as intruders at tau = 0.5. State this explicitly in the paper.

**Primary continuous statistic: energy share**, because the count saturates (99.7 % at
F_delta 1.5).

---

## 6. Intervention arms — 7 per configuration

Built from the trained adapter on CPU, written as stock PEFT adapters so the frozen eval
pipeline scores them unmodified.

| arm | what | matched to |
|---|---|---|
| **A** | source, unmodified (re-scored at the same protocol) | — |
| **B** | delete **all** intruders among the top-10 directions of `W'` | — |
| **C** | shrink the **whole** update uniformly until `\|dW_C\| = \|dW_B\|` | B, magnitude |
| **D** | arm B rescaled back up until `\|dW_D\| = \|dW_A\|` | A, magnitude |
| **E** | remove non-intruder content until `\|dW_E\| = \|dW_B\|` | B, magnitude |
| **Ep** | remove non-intruder content until `\|dW_Ep - dW_A\| = \|dW_B - dW_A\|` | B, energy removed |
| **F** | per matrix, if B deletes X intruders, delete X **randomly chosen non-intruder** singular directions | B, direction count |

Notes that matter:

- **B, C, E share a magnitude; Ep matches removed energy; F matches count.** These are
  different matchings on purpose — the removal is not orthogonal to the rest of the
  update, so magnitude-matching and energy-matching are genuinely different arms.
- **E and Ep cannot delete non-intruder directions outright.** Those directions are
  base-aligned, so removing them from `W'` forces the update to cancel `W0` and `\|dW\|`
  *grows* (measured +13 %). E/Ep therefore act on the update's own decomposition,
  `dW_E = (1-a) dW + a P_I dW`, with `a` solved for the required match.
- **F does delete whole directions**, exactly as B does — that is the point. It removes
  less energy (non-intruders sit deeper: mean rank 38.9) and `\|dW\|` grows to 1.069.
  **This is not corrected**; the actual energy and F_delta are measured and reported.
  Multiple random draws where practical.

Measured for configuration 1 (Llama LoRA+wd), verified by independent recomputation:

| arm | rank | `\|dW\|^2` | ratio to A | energy of the change |
|---|---|---|---|---|
| A | 32 | 71,788.9 | 1.0000 | — |
| B | 42 | 49,411.5 | 0.8296 | 35,914.2 |
| C | 32 | 49,411.5 | 0.8296 | 2,083.7 |
| D | 42 | 71,788.9 | 1.0000 | 40,777.8 |
| E | 32 | 49,411.5 | 0.8296 | 3,862.7 |
| F | 42 | 82,082.5 | 1.0693 | 13,840.7 |

B deletes 908 directions across 135/160 matrices (31.2 % of update energy);
F deletes exactly 908 non-intruder directions.

---

## 7. What each contrast tests

| contrast | matched on | question |
|---|---|---|
| **B vs C** | update magnitude | does targeting intruders matter beyond shrinking the update? |
| **D vs A** | update magnitude | does altered geometry change retention at fixed size? |
| **B vs E** | update magnitude | intruder vs non-intruder removal at equal resulting size |
| **B vs Ep** | energy removed | intruder vs non-intruder removal at equal energy removed |
| **B vs F** | direction count | are intruders special, or would any X directions do? |

**Forgetting** is tested by B vs C and D vs A. **Adaptation** is tested by B vs E/Ep/F.

---

## 8. Cross-configuration analysis

Report per configuration: task, retention, F_delta, intruder fraction, intruder energy.

Because the frozen pool has (F_delta, retention) for ~60 Llama and ~57 Qwen adapters but
**no** intruder measurements (its checkpoints were lost in 2026-07 — the reason this
slice is retrained), the analysis is two-stage:

1. Fit the magnitude law on the **pool**: measured
   Llama `R = 18.49 - 6.96 log F` (R^2 0.914, n=60);
   Qwen `R = 11.68 - 14.08 log F` (R^2 0.732, n=57).
2. For each configuration compute the residual `e_i = R_i - Rhat(F_i)`.
3. Ask whether intruder energy explains those residuals.

With 7 configurations this is **descriptive, not a powered regression**. The causal
interventions carry the argument.

---

## 9. Wording discipline

Claim: *"We find no evidence that intruder dimensions are the source of forgetting; at
matched update magnitude, removing them does not improve retention and substantially
impairs adaptation."*

Not: *"Intruders do not cause forgetting."*

If all seven configurations line up: *"Across adapter designs and models, intruder energy
provides little additional explanatory power for retention beyond update magnitude."*

Central hypothesis: **intruder structure and forgetting are separable** — intruders may
mark where task adaptation is encoded, while the retention cost is governed by the
magnitude of the parameter displacement.

---

## 10. Deliverable

`paper_table.py` prints the final table directly from `results/`, with pending cells as
`--`, so it is correct at any point:

- **Table 1**: per configuration — intruder fraction, intruder energy, and all 7 arms
  (F_delta, task, retention).
- **Table 2**: the five key contrasts as retention/task deltas.
- `--csv` for the raw numbers.

Currently **6 / 56** arm evaluations complete.

---

## 11. Operational notes

- One process on the GPU at a time; every job waits for a free card before starting.
- Qwen training NaNs at batch 70 **nondeterministically** (ruled out: co-tenancy, thread
  env, seed, method, lr, attention backend, padding, data). Mitigation: each training job
  retries up to 6 times, every attempt verified by `adapter_health.py`; a failed attempt
  costs ~2.5 minutes.
- `auto_intruder.sh` scores each finished adapter; `auto_ablate.sh` builds arms B-F and
  queues their evaluations. Both CPU-only.
