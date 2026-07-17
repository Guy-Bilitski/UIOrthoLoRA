# PAPER BLUEPRINT — "Magnitude, Not Geometry" (final, post-fleet-kill)

`[2026-07-17 — synthesis of analysis_final/01–04 + key_numbers §18. This is the story document
the artifact renders and the paper will follow. Every number traces to results/*/summary.json
via the freeze analyzers.]`

---

## 1. Thesis (one paragraph, final wording)

Catastrophic forgetting under parameter-efficient fine-tuning is governed **to first order by the
effective magnitude of the weight update (F_Δ) and only to second order by where the update is
placed**. Across 1,035 usable adapter runs — 6 model×task families, 8 LoRA-family methods, 3–5
seeds — retention of held-out capability tracks log-magnitude at pooled r = −0.847 (per-family
−0.830…−0.929), with a **flat-then-falling shape**: below a per-family knee, retention is flat;
above it, it falls steeply. The relation survives **intervention**: rescaling a trained adapter —
changing nothing but its magnitude — moves it along the curve (15/15), while a random direction at
matched magnitude pays only a bounded −3.05 pp penalty and buys no adaptation. The practical
corollary stands: plain **LoRA + weight decay**, the cheapest magnitude control, sits on the
adaptation–retention frontier of every elaborate initialization we assess, at zero extra cost.

Pre-registered wording rule: headline = "magnitude **relation**" (flat-then-falling with a knee);
the word "law" only with the knee caveat — normalized slopes do not converge across families
(−0.33…−0.70).

## 2. Contributions (two independent layers)

**Layer 1 — measurement contribution (stands even if rankings shift):**
a fair-comparison protocol for forgetting = (i) a method-neutral magnitude axis F_Δ ("effective
update magnitude on the adaptation distribution" — beats ‖ΔW‖_F, R² 0.72 vs 0.56) + (ii) a
per-method LR sweep + (iii) calibration-set matching for data-aware inits. The protocol decomposes
any adapter's apparent win into a magnitude effect, an LR-tuning artifact, and a calibration
artifact.

**Layer 2 — empirical claims:**
- **C1 (mechanism):** forgetting is first-order in F_Δ — observational (r −0.83…−0.93 per family;
  within-cell seed micro-test r = −0.713, t = −31.3) **and interventional** (E1).
- **C2 (diagnosis):** learning rate is only a proxy — partials r(F_Δ|LR) = −0.58…−0.91 vs
  r(LR|F_Δ) = −0.17…+0.29; the relation holds *inside* every fixed LR ≥ 1e-4 stratum.
- **C3 (corollary):** LoRA+wd is the efficient frontier point — CS 81.8 / retention 25.9 (base
  26.0) at the smallest update and the widest safe LR band; math GSM8K 66.8 ± 0.8 above every
  published competitor at BBH ≥ base; replicates on Qwen.
- **C4 (new, unplanned):** **post-hoc adapter rescaling** is a free practical knob — at matched
  magnitude, rescaled adapters retain +1.09 ± 1.80 pp MORE than natively-trained ones while
  keeping most adaptation; downscaling is safe, upscaling is not (−3.9 pp).

## 3. The evidence walk (paper spine, section by section)

### §3 Setup & measurement
- One pipeline, 9 methods, faithful ports (line-audited; 1 found-and-fixed), F_Δ measured on
  every run. Base ceilings: Llama core 25.89≈26.0 / broad 35.26; Qwen 44.35.
- Final dataset: 1,661 result dirs / 1,500 evaluated / 1,429 post-quarantine; 287 cells with ≥3
  seeds. Registry (campaign_summary.jsonl) and results_book are stale — source = summary.json.

### §4 The magnitude relation (C1)
- §18.1 table (six families, pooled −0.847/rank −0.923, n=1035). Strict-quarantine variant −0.864
  (n=1003) — unchanged.
- Shape: 2-segment beats linear in all 6 families (F 1.6–40); knees F_Δ ≈ 0.12–0.95; below-knee
  −4…+2 pp/dec (Qwen literally flat), above-knee −7.5…−40.8.
- **Interventional (E1):** 15/15 rescales on-curve (+1.29 ± 2.07 pp, within-set r −0.732); 9/9
  random-direction controls −3.05 pp penalty, adapt ≈ 0; upscaling asymmetry (−3.86 pp).
- Micro-test: within-cell demeaned r = −0.713 (n=954 obs / 290 cells).
- Qwen math finally quantitative: r = −0.830 (n=164, 3 seeds; clean −0.695 — always quote both).
  E3 densification: below-knee flat (r −0.03/−0.04), anti-replication dead.

### §5 Geometry & direction: second-order, and a fingerprint (refined C1)
- Direction is real but bounded: partial r(spec_max | F_Δ) = +0.117 (t=3.7); method offsets at
  matched F_Δ ±1.2–4.6 pp; E1 random-direction −3.05 pp. Never claim "adds nothing".
- Fingerprint battery unchanged (design signatures recoverable from trained weights on both
  models); DoRA F_Δ is a lower bound (magnitude vector uncounted) — its +2.9…+4.6 positive
  offsets are the same disclosure class as fft's undercount.
- **SC-LoRA resolution (E4, the fairness centerpiece):** eval-matched calibration puts the full
  ladder +0.92 pp ABOVE the curve (n=20, 5 seeds) vs −3.39 below with nq_open (n=24). The study's
  only significant deviation was the calibration-set choice, not method geometry. Data-aware inits
  inherit their calibration distribution — a protocol requirement, not a SC-LoRA deficiency.
- Sharpest slogan (A4): retention-per-unit-magnitude is near-universal (R² 0.69–0.86, offsets
  ≤ few pp); methods differ mainly in **adaptation bought per unit of magnitude** (spread 4.9–16 pp).

### §6 LR is only a proxy (C2) — rewritten battery
- Old "R² 0.74 vs 0.32" exhibit retired (strawman vs LR-dummies). New: partials, fixed-LR strata
  (r ≤ −0.67 at every LR ≥ 1e-4, every family), decoupling grids (frc/frm R² 0.86 vs 0.37–0.39).

### §7 The practical corollary (C3 + C4)
- Llama CS: LoRA+wd 81.75 ± 0.17 / 25.86 ± 0.37 @5e-4 (4 seeds), F_Δ 0.38–0.41, safe band 6/7.
  DoRA/MiLoRA convention note: their mean-rule rows are dragged by format-collapse seeds
  (retention intact) — print retention-relevant points with disclosure.
- Math: LoRA+wd 66.79 ± 0.79 GSM8K / 33.57 ± 1.04 BBH (≥ base 33.1) — beats best published
  (CLoRA 64.6) by +2.2 pp. In-pipeline anchors now 3-seed.
- Qwen CS: 87.43 ± 0.23 / 40.07 ± 0.68 (or lr1e-4 86.85/40.70); retention ≥ 38 at every swept LR.
  Qwen math: SC-LoRA@5e-5 is the standout (77.2 GSM8K at base BBH) — do NOT over-claim LoRA+wd
  on this arm; the corollary is "magnitude control wins", not "LoRA+wd wins everywhere".
- Controls: rank ladder (F_Δ↑/ret↓ monotone, seed-robust); r16 param-matched control (capacity
  confound dead); wd×lr grid monotone (F_Δ 0.75→0.26, ret 21.1→27.5 s42; multi-seed monotone
  through wd0.3); high-k CLoRA = a magnitude dial with an adaptation price (k2048: 69.4 ± 4.3
  CS-8, 4/5 seeds ≤ 71.5) and a memory tax (up to 6.7 GB) — CLoRA works *through* magnitude.
- **E6 split verdict:** wd transfers to MiLoRA (+1.8/+2.4 pp above curve at adapt 80.2); wd
  breaks DoRA-as-implemented (CE 10–21; it decays the magnitude vector). A knob, not a free lunch.
- **C4 rescaling:** +1.09 ± 1.80 pp vs trained twins at matched F_Δ (n=15 pairs); flagship
  26.9 ret / 75.4 adapt vs 24.4 / 79.1.

### §8 Universality & scope
- Full-FT anchor (E2): monotone in F_Δ but −4.1…−8.6 pp below the adapter curve (dense-mass
  undercount disclosed) — universal in form, family-specific in level.
- Bridging arms (E7): MedMCQA + attention-only, brl r −0.878, brq r −0.995 — off-recipe, both 7Bs.
- Adaptation tax: **Qwen-CS-specific** ≈5.3 pp below-knee gap to ceiling; all other arms ±0.8 pp.
- CE corroboration (reframed): lead with r(CE, retention) −0.63…−0.92; r(F_Δ, CE) is partly
  mechanical. Replay (E5): CE-only partial answer (−0.05…−0.09 in 4/4 pairs; benchmarks lost).

### Limitations / honest ledger
- 284B DSV4 generalization: designed-but-lost (0/21 synced). No 284B claims anywhere.
- Lost-with-fleet: E5 benchmark evals (0/4), E6-DoRA benchmark (0/2, degenerate by CE anyway),
  E3 2nd wave (13/26), b4 4 cells, brq lr1e3, qwsw 27 + qwswm 22 trained-not-evaluated,
  base-ceiling ladders (4/22 evaluated).
- CorDA: NOT-ASSESSED (port mis-calibration disclosed; corrected re-run never completed).
  Coverage: 7 of 8 assessed; all 7 consistent with one curve once calibration is fair.
- Qwen within-cell SDs 2.1–2.7 pp (seed-unstable F_Δ cells); qwswm clean-subset rule.
- DoRA F_Δ lower-bound deferred to camera-ready.
- Format-collapse seeds affect *accuracy* means only (retention intact everywhere) — the
  r16 lr3e-4 "13.5 collapse" is seed-42-specific (s43 72.0, s45 80.7), deterministic under re-eval.

## 4. Proposed paper outline (for sign-off)

1. Introduction — thesis + two-layer contributions (C1–C4).
2. Related work — adapter zoo by basis; CLoRA/MiLoRA/Biderman positioning (Biderman's "wd doesn't
   help" reconciled: 3,000–6,000× smaller wd, full-FT, single LR).
3. Setup & measurement — F_Δ axis, ports, ceilings, protocol.
4. The magnitude relation — observational + shape + **interventional** + micro-test.
5. Geometry: a bounded second-order effect and a fingerprint — incl. E4 calibration control.
6. Learning rate is only a proxy — rewritten battery.
7. The practical corollary — LoRA+wd frontier + wd-transfer boundary + rescaling knob (C4).
8. Universality & scope — full-FT, bridging, Qwen, CE, replay-partial.
9. Discussion, Limitations (ledger), Reproducibility appendix.

## 5. Go/no-go decisions for the user

1. **Title wording** — must drop bare "Magnitude Law": e.g. "Magnitude, Not Geometry: Update Size
   is the First-Order Predictor of Forgetting in PEFT" (relation framing inside).
2. **C4 (rescaling) promotion** — headline contribution or discussion paragraph?
3. **Qwen-math SC-LoRA standout** — foreground honestly in §7 (recommended) or keep to appendix?
4. **284B** — mention as designed-but-lost in Limitations (recommended) or omit entirely?
5. **E5 replay** — report CE-only partial answer (recommended) or drop?
6. **DoRA/MiLoRA table treatment** — retention-relevant operating points with collapse disclosure
   (recommended) vs mean-rule rows.
