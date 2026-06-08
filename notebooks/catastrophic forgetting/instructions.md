# UIOrthoLoRA Ablation Study — Agent Specification

**Audience:** an autonomous coding agent operating in the UIOrthoLoRA repo.
**Purpose:** isolate the contribution of each architectural component of UIOrthoLoRA so the paper can claim, with evidence, *which* parts of the design move the adaptation/retention frontier — above all whether the **learnable orthogonal rotations** (the single clearest novelty vs. the init-only family and vs. CLoRA) actually matter.

**Prerequisite:** this ablation reuses the shared trainer, datasets, harmonized config, and evaluation pipeline defined in `UIOrthoLoRA_reproduction_protocol.md`. Do not build a separate trainer. Every ablation here is a **config diff** against the full UIOrthoLoRA run defined there.

> **Do-not-fabricate rule.** Where this spec references the UIOrthoLoRA implementation (parameter names, tier construction), map them to the *actual* code in the repo. If a referenced component does not exist or is named differently, stop and surface the discrepancy rather than inventing it. Items needing confirmation against the code are marked **[CONFIRM]**.

---

## 1. What UIOrthoLoRA is made of (the knobs we ablate)

Per target linear layer `W₀ ∈ ℝ^{d_out × d_in}`, one-time SVD `W₀ = U Σ Vᵀ`, `r = min(d_out, d_in)`. Partitioned by hyperparameters `k_val ≥ k_vec` into three tiers:

| Tier | Components | Directions | Magnitude | Trainable params |
|---|---|---|---|---|
| **Major** (top `r − k_val`) | `U_major, V_major` | frozen | **stripped to identity (σ=1)** | none (relies on D,E) |
| **Medium** (`k_val − k_vec`) | `U_med, V_med` | frozen | `σ_med` trainable | `σ_med` |
| **Small / null-space** (bottom `k_vec`) | `U_small, V_small` | **rotated** by `R_U, R_V ∈ O(k_vec)` | `σ_small` trainable | `R_U, R_V, σ_small` |
| **Global** | `D ∈ ℝ^{d_in}`, `E ∈ ℝ^{d_out}` | — | unit-invariant scalers | `D, E` |

Orthogonal rotations parameterized via matrix-exponential (or Cayley): `R = exp(θ − θᵀ)` from skew-symmetric `θ ∈ ℝ^{k_vec × k_vec}`. **[CONFIRM]** parameter names: `theta_L/theta_R`, `sigma_med/sigma_small`, `D/E`, `k_val/k_vec`.

**Trainable-param scaling per module** (drives capacity matching in §4):
- `D + E` ≈ `d_in + d_out` (large, fixed regardless of k)
- `σ_med` = `k_val − k_vec`
- `σ_small` = `k_vec`
- `R_U + R_V` ≈ `k_vec·(k_vec − 1)` (≈ `k_vec²`, **dominant when k_vec is large**)

---

## 2. The ablation matrix

All runs are **leave-one-component-out** (or swap) from the full model `A0`, at a **fixed reference partition** `(k_val*, k_vec*)` chosen in §4, on **both settings (commonsense, math)** and the **primary model (LLaMA-2-7B)**. Promote the winning subset to LLaMA-3-8B afterward.

| ID | Name | Change vs A0 | Isolates | Headline? |
|---|---|---|---|---|
| **A0** | Full UIOrthoLoRA | — (reference) | full method | — |
| **A1** | Rotations frozen (`R_U=R_V=I`, not trained) | freeze rotations, keep `σ_small` trainable, same `k_vec` | **value of rotating the null-space basis** (nearly param-matched) | **★ primary** |
| **A2** | No null-space tier (`k_vec=0`) = **UILinLoRA** | remove small tier entirely | whether null-space repurposing matters at all | ★ |
| **A3** | Medium tier frozen (freeze `σ_med`) | medium magnitudes fixed (tier folds into major) | mid-spectrum magnitude tuning | |
| **A4** | No scalers (freeze `D=1, E=1`) | disable unit-invariant scaling | the UI-SVD scaler contribution | |
| **A5** | Major keeps original Σ (no identity stripping) | `σ_major = Σ_major` frozen instead of `1` | the "strip magnitude → force reliance on D,E" design choice | |
| **A6** | Cayley vs matrix-exp | swap rotation parameterization | numerical/parameterization sensitivity | minor |
| **A7** | Partition grid | sweep `(k_val, k_vec)` 3×3 at matched capacity | where the spectral boundaries should sit | (curve) |
| **A8** | `σ_small` init: minor-Σ vs random | change null-space σ initialization | whether MiLoRA-style minor init helps inside your tier | optional |

`A1` is the experiment the reviewers (and you) care about most: it holds the architecture fixed — same tiers, same directions available, same `σ_small` trainable — and freezes only the rotations. The trainable-param delta is just `R_U + R_V` (≈ k_vec²), so at small/medium `k_vec` it is *nearly param-matched*, making it the cleanest possible isolation of rotation trainability. If A0 beats A1, rotations earn their place; if A0 ≈ A1, the rotations are decorative and you should know that before submission.

---

## 3. Detailed per-ablation spec

### A0 — Full (reference)
The full UIOrthoLoRA at `(k_val*, k_vec*)`. Run with ≥3 seeds; it is the baseline every other ablation is differenced against. Record exact trainable-param count `P0`.

### A1 — Rotations frozen ★
- Set `R_U = R_V = I` and **exclude `θ_L, θ_R` from the optimizer** (freeze, do not delete the tier). `σ_small`, `σ_med`, `D`, `E` stay trainable. `k_vec` unchanged.
- **Isolates:** the benefit of *rotating* the null-space basis vs. merely rescaling the fixed minor directions (which is essentially what MiLoRA's minor tier does without orthogonality enforcement).
- **Confound:** trainable params drop by ≈ k_vec². At the reference `k_vec*` this should be a small fraction of `P0` (report it). If it is **not** small (large k_vec*), additionally run a param-matched A1 that widens `σ_med`/`k_val` to recover `≈ P0`, so the comparison is clean.
- **Hypothesis (to test, not assume):** A0 > A1 on adaptation at equal retention — rotations let the null-space directions align to task dynamics rather than being stuck on the pretrained minor axes.

### A2 — UILinLoRA (`k_vec = 0`)
- No small tier. Only `σ_med` + `D, E` trainable. This is your existing GLUE-appendix variant; here run it on the full CS+math+retention suite.
- **Isolates:** whether the null-space tier contributes anything beyond magnitude tuning + scalers.
- **Confound:** large param drop (loses both `σ_small` and rotations). Report params; this is a "what does the whole small tier buy" test, not a clean rotation test (that's A1).

### A3 — Medium tier frozen
- Freeze `σ_med` (exclude from optimizer); directions already frozen. Effectively the medium tier becomes part of the frozen anchor.
- **Isolates:** the safe "magnitude tuning" regime in the mid-spectrum.
- **Hypothesis:** matters more on **math/high-rank** adaptation (where extra magnitude capacity helps) than on retention — connects directly to the CLoRA high-rank tension.

### A4 — No scalers
- Freeze `D = 1`, `E = 1` (exclude from optimizer).
- **Isolates:** the unit-invariant SVD scaling (your Uhlmann-grounded distinguishing feature).
- **Important interaction:** because the major tier's σ is stripped to identity (A0 design), disabling `D, E` makes the **major tier fully inert**. Note this in analysis — A4 tests not just "do scalers help" but "with scalers off, is the major-tier identity-stripping harmful?" (which A5 addresses from the other side).

### A5 — Major keeps original Σ
- Restore `σ_major = Σ_major` (frozen) instead of `1`.
- **Isolates:** whether stripping major-tier magnitude variance (forcing reliance on D,E) was a good call, or whether preserving the pretrained magnitudes is better.
- Run **with** scalers on (so it's a clean A5-vs-A0 swap) and note the A4×A5 interaction in analysis.

### A6 — Cayley vs matrix-exp
- Swap the orthogonal parameterization only. One run per setting.
- **Isolates:** numerical stability / optimization differences. Expected small; report so reviewers don't ask.

### A7 — Partition grid
- 3×3 over `(k_val, k_vec)` spanning small→large null space and thin→thick medium tier. **Hold total trainable params ≈ constant** across grid points by trading `k_val` against `k_vec` (use the §4 calculator). This is the partition-sensitivity result the reviewer asked for ("static thresholds" limitation).
- Produces a small heatmap of frontier quality vs. partition.

### A8 — σ_small init (optional)
- Initialize `σ_small` from the original minor singular values vs. random small init.
- **Isolates:** whether MiLoRA-style minor init helps inside your rotated tier. Cheap; do it if budget allows.

---

## 4. Capacity-matching protocol (do this first)

Component ablations change trainable-param count, which confounds the comparison. Rules:

1. **Pick the reference partition `(k_val*, k_vec*)`** so that full-UIOrthoLoRA's trainable-param count `P0` matches a rank-32 LoRA on the 5 target modules within ±10% (per `UIOrthoLoRA_reproduction_protocol.md` §5.4). Record `P0` and the LoRA reference count.
2. **Always log trainable-param count** for every ablation run. No exceptions.
3. **For headline claims (A1, A2), report params alongside the metric.** The valid argument is directional:
   - A1 has *slightly fewer* params than A0 (only rotations removed). If A1 underperforms → rotations help. If A1 ≈ A0 → rotations don't (and you say so honestly).
   - For a *strict* claim, run the param-matched A1 variant (widen `σ_med`/`k_val` to recover `≈P0`) so any gap can't be attributed to capacity.
4. **A7 must hold params ≈ constant** across the grid by construction — that's the whole point of a partition sweep.

Provide a small helper that, given `(d_in, d_out)` per target module and a target param budget, returns feasible `(k_val, k_vec)` pairs. **[CONFIRM]** the per-module dims for LLaMA-2-7B (q/k/v are `4096×4096`; MLP up/down differ — read from the model config, do not hardcode).

---

## 5. Run configuration

Each run = the shared base config (from reproduction protocol §5) **plus** an `ablation` block. Suggested schema (map to the repo's actual config system):

```yaml
base: # inherited, do not change across ablations
  model: meta-llama/Llama-2-7b-hf
  setting: commonsense        # or: math
  rank_equiv_budget: <P0_target>   # for capacity matching
  target_modules: [q_proj, k_proj, v_proj, up_proj, down_proj]
  optimizer: adamw
  lr: 3.0e-4                  # 1e-4 for llama-3-8b
  scheduler: linear
  batch_size: 16
  warmup_steps: 100
  epochs: 3
  precision: bf16
  seed: 42

ablation:
  id: A1
  k_val: <k_val*>
  k_vec: <k_vec*>
  train_rotations: false      # A1: false ; A0: true
  train_sigma_med: true       # A3: false
  train_sigma_small: true
  train_scalers: true         # A4: false
  major_sigma: identity       # A5: original
  rotation_param: matrix_exp  # A6: cayley
  sigma_small_init: minor     # A8: random
  capacity_matched: false     # set true for the param-matched A1 variant
```

A0–A8 are then just values of the `ablation` block. The agent should generate one config per (ablation × setting × seed) and queue them.

---

## 6. Evaluation

Reuse the reproduction protocol's evaluation exactly (§8 there):
- **In-domain CS:** LLM-Adapters generation eval, 8 datasets, accuracy, last checkpoint.
- **In-domain math:** GSM8K + MATH, EM, PiSSA pipeline, last checkpoint.
- **Out-domain retention:** BBH + MMLU-Pro via lm-eval.
- **Base reference row** (no adapter) in every table.
- **Optional supplementary:** also score with the dual-axis framework (negative-shift + error-set) for continuity with the current paper.

Do **not** introduce a different eval here — comparability across the ablation table and across the main head-to-head table depends on identical evaluation.

---

## 7. Naming, logging, outputs

- **Run name:** `uiortho_{ablation_id}_{setting}_{model_short}_s{seed}` e.g. `uiortho_A1_cs_l2-7b_s42`.
- **Log to W&B** (or repo default): full config, `trainable_param_count`, `P0` reference, train loss, every eval metric, seed, git commit, SVD partition indices used.
- **Persist per run:** adapter weights, the exact `(k_val, k_vec)`, frozen-buffer indices, and which components were trainable (a boolean vector) so the table is auditable.
- **Outputs directory:** `results/ablation/` with one JSON per run + a consolidated `ablation_summary.csv`.

---

## 8. Analysis & figures the agent should produce

1. **Master table:** rows = A0–A8 (+ param-matched A1), columns = trainable params, CS in-domain avg, CS BBH, CS MMLU-Pro, math GSM8K, math MATH, plus retention. Bold A0; mark deltas vs A0.
2. **Frontier overlay (the key figure):** adaptation (x) vs retention (y), one point/curve per ablation, both settings. A0 vs A1 vs A2 is the rotation/null-space story; the gap between A0 and A1 *is* the rotation contribution made visible.
3. **Leave-one-out bar chart:** Δ(retention) and Δ(adaptation) for removing each component (A1–A5), so the relative importance ordering is one glance.
4. **Partition heatmap (A7):** frontier-quality metric over the `(k_val, k_vec)` grid.
5. **High-rank stress callout:** A3 (medium tier) effect on math specifically — ties to whether your design survives the high-rank regime where CLoRA degrades past k=128.

---

## 9. Success criteria & sanity checks

- **A0 reproduces** the full-UIOrthoLoRA numbers from the head-to-head run within seed noise. If not, stop — the ablation harness diverges from the main harness.
- **A2 (UILinLoRA)** should roughly reproduce your existing UILinLoRA GLUE-appendix behavior qualitatively (magnitude-only is weaker on adaptation). If A2 ≥ A0, the null-space tier is doing nothing — a major finding, investigate before trusting.
- **A1 vs A0** is the headline: a positive, consistent A0−A1 gap (esp. on adaptation at fixed retention) is the result that justifies the rotations in the paper. A null result here must be reported honestly and reframes the contribution.
- **Param counts** logged for every run; no headline claim without them.
- **bf16** everywhere; deterministic SVD init under the fixed seed.

---

## 10. Task checklist (for the agent)

1. **[CONFIRM]** map all component names in §1/§5 to the actual repo implementation; surface any mismatch.
2. Implement freeze-flags for rotations, `σ_med`, `σ_small`, `D/E`, and the `major_sigma` (identity|original) and `rotation_param` (matrix_exp|cayley) switches, if not already exposed.
3. Build the capacity calculator (§4); read per-module dims from the model config.
4. Choose `(k_val*, k_vec*)` matching the LoRA-r32 budget; record `P0`.
5. Generate configs for A0–A8 (+ param-matched A1) × {cs, math} on LLaMA-2-7B; seeds: 3 for A0/A1/A2, 1 elsewhere.
6. Run training via the shared trainer; log per §7.
7. Evaluate per §6 (in-domain + out-domain + base row).
8. Produce tables and figures per §8 into `results/ablation/`.
9. Run the §9 sanity checks; flag any failure before reporting.
10. Promote the winning configuration to LLaMA-3-8B for {A0, A1, A2} as cross-model confirmation.

---

## 11. Must-verify list
- **[CONFIRM]** repo parameter/flag names before generating any config.
- **[CONFIRM]** per-target-module dimensions from the model config (do not hardcode 4096).
- **[CONFIRM]** the orthogonal parameterization currently in use (matrix-exp vs Cayley) so A6 is a genuine swap.
- Ensure freezing a component **excludes it from the optimizer** (and from weight decay), not merely zeroes its gradient.
- Re-run, never copy, all numbers — including A0 — so the ablation table is internally self-consistent and consistent with the head-to-head table.