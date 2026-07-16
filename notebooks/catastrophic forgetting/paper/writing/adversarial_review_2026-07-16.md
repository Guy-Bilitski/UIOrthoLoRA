# Adversarial review synthesis — 2026-07-16

Three independent adversarial reviewers (statistics, Reviewer-2/claims, DeepSeek-design) attacked
the paper's claims against the live data (each ran its own computations). Consolidated verdicts,
repairs, and the proposed experiment batch. Full reports preserved in the session transcripts;
attack scripts in the session scratchpad.

## 1. Verdicts on the core claims

| Claim | Verdict | Why |
|---|---|---|
| Monotone magnitude→retention relationship | **SURVIVES, strongly** | replicates per-seed, per-cell, within-cell, at fixed LR, 2 models; per-seed r within ±0.05 of pooled |
| Log-LINEAR form, single slope ("law") | **BREAKS (Qwen), WEAKENS overall** | Qwen bottom-half-range r = +0.38/+0.43 (sign flip!); drop top F_Δ quartile → qwswm r −0.83→−0.28; curvature F=63.8 (lrsw); slopes vary 3× (−9.3…−29.2 pp/dec) → flat-then-falling with a knee, not a line |
| "R² doubles vs LR" (§2) | **STRAWMAN as written** | LR-as-dummies nearly ties F_Δ (lrsw 0.765 vs 0.785). Real rescue: fixed-LR strata (r −0.75…−0.96 across methods at fixed LR) + partial r(F_Δ\|LR)=−0.70…−0.91 vs partial r(LR\|F_Δ)≈0 + frc/frm grids where wd/rank decouple F_Δ from LR (0.86 vs 0.39) |
| "Magnitude, not direction" | **WEAKENS** | partial r(spread\|F_Δ)=+0.198 at cell level (t=4.1); SC-LoRA −3.7±1.5 pp offset at matched F_Δ in 2 families → direction is a real ~3 pp second-order effect. Also F_Δ itself is a *direction-weighted* norm on adapt data (dw_sv R² 0.33-0.36 vs F_Δ 0.74 — the gap IS direction) |
| CE "independent corroboration" | **WEAKENS framing** | r(F_Δ,CE)≈0.82–0.92 within family is partly mechanical; the evidential link is r(CE,ret) −0.63…−0.92 within family; Qwen CE coverage holes (78/136, 91/156) |
| Selection/truncation, pseudo-replication | **SURVIVE** | only 20 non-finite exclusions; per-seed r stable; quote 287 cells as headline n |

**New supportive finding (use it):** within-cell micro-test — seed-level F_Δ fluctuations at a
FIXED recipe predict seed-level retention fluctuations, pooled r=−0.72 (n=943, t=−32). The
closest thing to a causal signal in the dataset.

## 2. Fixes ALREADY APPLIED (2026-07-16, while DeepSeek trains — before evals start)

- `eval_deepseek.py`: now saves **full per-subtask lm_eval rows** (enables retention-without-
  medical-subtasks post-hoc — was irreversible) + **fdelta_adapt** on MedMCQA prompts (F_Δ was
  measured off-distribution). Deployed to all 21 nodes.
- `uio_inprocess.fdelta_inprocess(prompts=…)` backward-compatible parameter.
- **dsv4_base_noft** (284B no-FT ceiling, was never queued) launched on d001.
- **base_qwen25_noft** (Qwen ceiling — blocks slope normalization) launched on d004.

## 3. Zero-GPU analyses to run before writing (owner: analysis pass)

1. Spline/2-segment fits with per-family knee; report Spearman + healthy-only (ret≥15) +
   bottom-75% r alongside pooled. Retitle "law"→"magnitude relation" unless normalized slopes converge.
2. Rewrite key_numbers §2 with the dummy-LR comparison + fixed-LR strata + partials (pre-empt).
3. Report direction as second-order: cell-level partial + per-method offsets ±SE.
4. Adaptation-efficiency ANCOVA: retention cost is universal; methods differ in adaptation
   bought per unit F_Δ (sharper, defensible slogan).
5. Within-cell micro-test → main text.
6. Retention-broad without ARC-c (adapt-suite contamination: ARC-c is trained AND in broad).
7. Parse-failure-rate vs F_Δ; refit law on parse≥90% cells (format-collapse control).
8. Within-family CE corroboration table; backfill missing Qwen CE (~1 GPU-h).
9. F_Δ decomposition: ‖ΔW‖_F × alignment; is alignment method-invariant? (+ relabel axis
   "effective update magnitude on the adaptation distribution").
10. Fix analyze_full geometry keys (stable_rank_w/eff_rank_w — two rows silently empty).

## 4. Experiment menu (fleet-runnable; ranked by value/cost)

| # | Experiment | Fixes | Cost | Where |
|---|---|---|---|---|
| E1 | **Interventional scale-matching**: rescale existing adapters (LoRA/DoRA/MiLoRA/SC-LoRA/CLoRA lr5e-4 s42) to F_Δ∈{0.15,0.40,0.80} exactly + 3 random-direction controls; eval retention | Converts headline from observational→interventional (R2's top; also the matched-magnitude direction test) | ~20 GPU-h, eval-only | 1 sweep node, <3h |
| E2 | **Full-FT anchor**: Llama-2 CS full FT, 3 LRs, F_Δ from dense ΔW | Scope: LoRA-family artifact vs universal law | ~70 GPU-h | 1 node, <6h |
| E3 | **Qwen mid-range densification**: 7 methods × LR {7e-5,1.5e-4} × s42 | The Qwen knee/functional-form hole | ~1 node, <6h | sweep node |
| E4 | **SC-LoRA eval-matched calibration (C2)**: r32, 7 LRs, s42 | The known "kill-shot" gap, still open | ~30 GPU-h | 1 node, <6h |
| E5 | **Replay baseline**: LoRA lr{3e-4,5e-4} × {0,5%} replay × 2 seeds | The practitioner falsification test | ~30 GPU-h | 1 node |
| E6 | **wd on other methods**: DoRA+wd0.3, MiLoRA+wd0.3, 2 LRs | "simplest control wins" completeness | ~20 GPU-h | shared node |
| E7 | **7B bridging arm**: Llama/Qwen-7B, MedMCQA 30k, attention-only targets, LoRA 4 LRs | De-confounds the 284B (model+task+targets changed at once) | ~40 GPU-h single-GPU jobs | idle sweep GPUs |
| E8 | **284B LR ladder**: lora {1e-4,1e-3,3e-3} + dora {6e-4} s42 | Range restriction: 7 fixed-LR methods span only ~0.2 decades → law unmeasurable at 284B without it. **Single highest-value DeepSeek addition** | 4 node-cells (~800 GPU-h) | DS nodes as cells finish |
| E9 | 284B CE full-blocks post-pass (run_node hardcodes 40 blocks vs spec full) + routing-drift covariate | Protocol-fidelity + MoE scoping | hours, post-pass | DS nodes after cells |

## 5. DeepSeek scoping notes (for the paper, regardless)

- n=21 = 7 method-clusters: pre-register seed-averaged method means, cluster bootstrap,
  "consistency with 7B prediction band" framing — NOT an independent r estimate.
- bf16-dequant ≠ deployed FP8: one scoping sentence.
- F_Δ not comparable in absolute scale across architectures (MoE routing): test monotone
  within-model recurrence only.
- MedMCQA↔MMLU-medical overlap: report retention ± medical subtasks (enabled by today's patch).
