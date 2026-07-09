# Supervision progress report — magnitude-law PEFT campaign (2026-07-09, v3)

Presentable version: https://claude.ai/code/artifact/5c46636f-036a-4fae-919f-43be8e07639c
Companion: handoff/24 (prior PI status), handoff/21 (consortium), paper/writing/data/key_numbers.md (AUTHORITATIVE numbers).

v3 note: rebuilt after a 3-expert review pass (adversarial critic + cold-read clarity + data-verifier).
Numbers switched from ad-hoc mtx maxima to the paper's CANONICAL key_numbers.md (single-seed s42,
n=49 LR sweep). Framing changed from "LoRA+wd wins both axes" (a lucky-seed overclaim) to "flat field
governed by ‖ΔW‖; LoRA+wd is Pareto-competitive." Two new PI requirements folded in: published-number
comparison, and the explicit per-adapter LR sweep + graph.

## Thesis
Catastrophic forgetting in PEFT is governed by weight-update magnitude ‖ΔW‖_F, NOT adapter geometry.
Published per-adapter wins are largely a single-learning-rate artifact. Fair test = every adapter through
one pipeline with LR swept as a controlled variable + ‖ΔW‖ measured (the column the papers omit).

## The fair sweep (answers "which LRs?")
Every adapter trained on Commonsense-170K (Llama-2-7B, seed 42) at 7 LRs:
2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3. 8 adapters (LoRA, LoRA+wd, LoRA-Null, MiLoRA, CLoRA, SC-LoRA,
CorDA, DoRA) × 7 LRs + rank/wd grids + Qwen. One cell = one model trained+evaluated.

## Core result (the killer figure)
- Retention vs LEARNING RATE: methods scatter/cross (R²=0.32) — this is how single-LR papers manufacture "wins".
- Retention vs ‖ΔW‖ (same runs): collapse to ONE line (r=−0.86, R²=0.74). Geometry doesn't predict
  position vs the line; ‖ΔW‖ does.

## Magnitude law (canonical, key_numbers.md §1-2)
- CS pooled n=49: r=−0.86, R²=0.74, slope −14.8. On-curve (excl SC-LoRA): r=−0.92.
- Within EVERY adapter: r=−0.86…−0.97. Qwen 2nd model: r=−0.93 (CS). Qwen MATH does NOT yet replicate
  (r≈+0.7, high-LR cells unrun — stated, not buried).
- LR-as-predictor R²=0.32 vs ‖ΔW‖ R²=0.74 (§2) — the fairness result.

## Fairness / ANCOVA (§5): 5 of 6 adapters straddle 0 residual (on the law). Only SC-LoRA deviates:
−4.15pp BELOW the line (p=0.006, provisional pending calibration control). A small, NEGATIVE geometry
effect (SC-LoRA forgets MORE) — disclosed by us.

## Per-adapter best operating point (CS, s42, key_numbers §3; base ret_core ceiling 26.0)
| Adapter | bestLR | CS acc | Ret | ‖ΔW‖ | safe-band /7 |
|---|---|---|---|---|---|
| LoRA+wd(0.3) | 5e-4 | 81.6 | 25.6 | 0.39 | 6/7 |
| SC-LoRA | 5e-5 | 80.1 | 22.5 | 0.56 | 1/7 |
| MiLoRA | 3e-4 | 79.9 | 24.7 | 0.54 | 5/7 |
| LoRA | 3e-4 | 79.1 | 24.4 | 0.62 | 5/7 |
| CLoRA | 5e-4 | 78.4 | 21.9 | 0.64 | 5/7 |
| DoRA | 2e-4 | 78.3 | 24.8 | 0.45 | 4/7 |
Field is flat (78-82, within seed noise). LoRA+wd Pareto-competitive at smallest ‖ΔW‖ + widest safe band.
Durable claim = "no geometry beats it," NOT "it wins."

## PUBLISHED EVIDENCE — CLoRA paper's own Tables 2/3/4 (user-supplied, 2026-07-09)
The CLoRA paper reports these; SC-LoRA & CorDA were NOT in it (our runs only — fair).

**Table 2 (commonsense, Llama-2-7B, published):** CS-avg / BBH / MMLU (base BBH 34.91, MMLU 18.56)
- LoRA 79.9 / 26.69 / 14.46 · DoRA 80.5 / 28.24 / 11.67 · PiSSA 73.8 / 29.54 / 11.33 · MiLoRA 80.0 / 25.14 / 17.74
- **LoRA-L2 (=LoRA+weight decay) 80.8 / 32.93 / 16.59** ← their OWN wd baseline; BEST retention of any non-CLoRA method
- LoRA-r8 78.8/26.90 · LoRA-r16 79.8/26.73
- CLoRA-k128 80.7/30.82 · k256 80.7/31.92 · k512 82.0/34.32 · k1024 82.6/36.49 · k2048 83.7/38.67

**Table 3 (math):** GSM8K/MATH — LoRA 60.58/16.88 · PiSSA 58.23/15.84 · MiLoRA 63.53/17.76 ·
CLoRA-k64 64.29/17.52 · **k128 64.59/18.38 (their best)** · k256 63.45/17.58. NO wd baseline in their math table.

**Table 4 (‖ΔW‖ & forgetting, math):** ‖ΔW‖ / F_Δ(forgetting,lower=better) / F(retention=BBH)
- reference —/2.42/34.91 · LoRA 22.63/0.79/26.69 · MiLoRA 24.32/0.92/25.14 · LoRA-r16 12.70/1.03/26.73 ·
  LoRA-r8 6.45/0.95/26.90 · **LoRA-L2 2.07/0.29/32.93** · CLoRA-k128 10.84/0.36/30.82 · k256 10.25/0.34/31.92 ·
  k512 8.19/0.27/34.32 · k1024 6.64/0.21/36.49 · k2048 5.00/0.14/38.67

**LoRA-L2 DEFINITION (verified 2026-07-09, agent — PDF was firewalled, so from search-surfaced paper
quotes + our repro's hardcoded coeff):** CLoRA's LoRA-L2 = L2 penalty on the LoRA params A,B (coeff
**1e-5**), a DIRECTION-AGNOSTIC MAGNITUDE/NORM penalty on ΔW — SAME FAMILY as our weight decay, NOT a
spectral/largest-SV penalty (paper explicitly contrasts "L2 on the norm" vs CLoRA's "direction of null
space"). NOT identical to our headline LoRA+wd: their λ=1e-5 vs our wd 0.2-0.3 (4-5 orders); theirs likely
loss-term L2 vs our decoupled AdamW. Our repro HAS a matching `lora_l2` arm at wd=1e-5 (make_frepro_jobs).
Report wording: "same KIND of knob, not identical." PI had hypothesized spectral — that is REFUTED.
TODO if 100% certainty needed: get the actual PDF (firewalled) or PI confirms the baseline definition.

**KEY:** CLoRA's OWN paper shows LoRA-L2 (a norm/magnitude penalty) is the strongest forgetting-mitigator except high-k
CLoRA — the magnitude law is in their Tables 2 & 4. Our contribution: turn their point-observation into a
LAW across 8 adapters × 7 LRs × 2 models. Honest boundary (published): CLoRA-k1024/k2048 (83.7/38.67) beat
LoRA-L2 (80.8/32.93) on CS; k2048 BBH 38.67 > base 34.91 = positive transfer. Their retention cols use MMLU
(ours MMLU-Pro) — never merged.

## Math + published comparison (faithful frm_, s42; matched c256)
- LoRA+wd best matched-c256: GSM8K 67.3 (wd0p3_lr2e4_c256), ‖ΔW‖ 0.28. (c512 gives 69.5 but competitors
  not run at c512 — use c256 for fair matched comparison per critic.)
- Published anchors: CLoRA k128 64.59/18.38, MiLoRA 63.53/17.76. LoRA+wd edges both ("edges", cross-harness).
- In-pipeline: CLoRA k128 59.6, MiLoRA 59.0 (~4pp below published for CLoRA — recipe/harness gap; our LoRA
  reproduces pub 60.2 vs 60.6, so pipeline sound). Published & in-pipeline rows kept separate.
- PiSSA collapsed (49.7, BBH 7.2) = real forgetting (target in 37/270 gens); principal-direction init =
  max perturbation = expected high-‖ΔW‖ endpoint.

## Honest boundaries (non-negotiable, all in report)
1. High-rank CLoRA: published k1024/k2048 CS ~83.7 > LoRA+wd 81.6 — directional constraint buys adapt-
   efficiency at high k. Faithful-recipe test running now.
2. SC-LoRA −4pp below the law (real, negative geometry effect).
3. Qwen-math anti-replicates (pending high-LR cells).
4. Single-seed s42 primary; CS eval seed-instable → report LR safe-band not single peaks; 3-seed running.
5. Published-CS anchors compiled for CLoRA only; rest being assembled (ASK PI to supply if available).

## Correctness gates (all passed 2026-07-08)
8 ports faithful; LoRA reproduces pub GSM8K (60.2 vs 60.6); save/reload lossless (Δ6e-9); base BBH 32.96≈
33.10; MMLU-Pro dropped for math (format unparseable); PiSSA real forgetting.

## Review provenance
3 Opus agents: data-verifier (all report numbers reproduce exactly from raw); clarity reviewer (fixed:
"retention" now defined as absolute held-out acc vs base ceiling 26.0, base ref row added, ‖ΔW‖ glossed,
2-panel figure captions conclusion-first, BBH/r/α expanded); critic (fixed: killed lucky-seed 82.0
headline, added LoRA-Null, disclosed SC-LoRA residual, matched-c256 math, flat-field framing).
