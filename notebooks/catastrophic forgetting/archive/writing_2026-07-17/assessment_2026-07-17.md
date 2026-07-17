# Experiment-batch assessment vs. paper claims — 2026-07-17 ~02:30 (pre-DeepSeek)

Numbers: `analyze_ebatch_2026-07-17.py` (rerunnable; residuals vs the observational lrsw fit
ret = 19.91 − 9.47·log10 F_Δ, n=201) + key_numbers §17. Status: 55/77 E-cells landed; the
rest + all 21 DeepSeek cells land by ~06:00.

## Claim-by-claim verdicts with tonight's evidence

| Paper claim | Verdict after E-batch | Evidence |
|---|---|---|
| Magnitude→retention relation is REAL and causal-ish | **UPGRADED observational→interventional** | E1: pure rescaling moves adapters along the curve (12 rescales, within-set r=−0.79, mean on-curve residual +1.2±2.0pp). The R2 "kill-shot" demand is answered. |
| Relation holds beyond the original setting | **STRONGLY REINFORCED** | E7: MedMCQA + attention-only targets reproduce it on both 7Bs (Llama r=−0.88, Qwen r=−0.995 so far). E2: monotone within full-FT too. |
| Log-LINEAR "law" | **RETIRED, replaced by flat-then-falling** (already §17.1) | E3 densification CLEANS the Qwen anomaly: bottom-half r goes +0.33→**−0.02 (flat, not positive)** with the 6 new mid-LR cells — "no retention cost below the knee, monotone above" is now the accurate summary. |
| "Magnitude, not direction" | **REFINED with exact numbers** | E1 controls: random direction at matched F_Δ = **−3.0pp vs trained** (9 controls, replicate to ±0.03 within tier). Direction is real, bounded, second-order. |
| SC-LoRA "below the law" (only sig. ANCOVA deviator) | **EXPLAINED — calibration artifact, NOT the method** | E4 full ladder: eval-matched calibration puts SC-LoRA **+0.92pp ABOVE the curve (n=20)** vs −3.39pp below with nq_open (n=24). Fig2's outlier dissolves; huge fairness win for the paper. |
| Relation is LoRA-family-scoped vs universal | **SHARPENED: universal in form, family-specific in slope** | E2: dense full-FT is monotone but sits −4…−8.6pp below the LoRA curve at matched F_Δ (its ΔW spreads across many directions: dw_sv_max 4.2 vs 30-40). Adaptation-per-F_Δ is far higher for dense (80.5 adapt at F_Δ 0.08). |
| LoRA+wd(0.3) is the practical winner | **GENERALIZES beyond LoRA** | E6: MiLoRA+wd0.3 lands +1.8…+2.4pp above curve with adapt 80.2 at lr5e4 — wd transfers. (DoRA+wd pending.) |
| Base ceilings for normalization | **COMPLETE (7B), capped-basis (284B in-flight)** | Llama 26.0; **Qwen 44.35 (landed tonight)** — note Qwen adapters plateau ~37-39 even below the knee, i.e. a ~6pp adaptation tax before any magnitude effect; worth one paragraph. 284B ceiling on the same capped basis as its 21 cells, lands ~06:00. |
| 284B generalization (consistency framing) | **PENDING tonight** | All 21 trains done; capped evals running; geometry already computed (CPU, done). |

## New results the paper did NOT plan for (consider adding)

1. **E1 upscaling asymmetry**: CLoRA upscaled past its trained magnitude (0.65→0.78) falls
   −3.9pp BELOW the curve while all downscales sit ON/above it — shrinking generalizes,
   stretching extrapolates badly. One paragraph + it reinforces the causal reading.
2. **Rescale > retrain**: rescaled adapters slightly BEAT natively-trained ones at the same
   F_Δ (+1.2pp mean) while keeping most adaptation (e.g. lora f040: 26.9 ret / 75.4 adapt —
   vs the trained lr3e-4 cell 24.4/79.1). "Post-hoc adapter shrinking" is a free practical
   knob — arguably a headline practitioner takeaway.
3. **Qwen adaptation tax** (above): flat region sits ~6pp under the ceiling on Qwen, ≈0 on
   Llama — model-dependent intercept, disclose alongside the knee.

## Anomalies / footnotes

- `b4_sclora_r32_lr2e5_s42` adapt=13.2 vs s43=77.3 (same recipe): undertrained-format seed
  fluke at the lowest LR; retention unaffected (27.5/27.3). Excluded from adapt comparisons.
- E1 F_Δ measurement offset: measured fd = nominal ×1.06-1.09 (sclora ×1.3-1.6) — env/
  measurement-context offset; analysis uses measured fd (axis convention unaffected).
- fft F_Δ under-counts dense updates (o_proj/embeddings/norms moved but unmeasured) —
  part of E2's below-curve gap may be unmeasured mass; disclosed.

## Open items to finish tonight (all in flight; deadline T+8h ≈ 09:50)

| item | ETA | blocking? |
|---|---|---|
| E1 last 3 (dora rescales, slow evals on d027) | ~02:30-03:30 | fig only |
| E3 second wave (14 qwswm + 8 remaining qwsw incl. dora/lorawd/clora/lora_null) | ~05:00-06:00 | knee fig |
| E4 lr1e3 ×2, E5 replay ×4, E6 dorawd ×2 (d022) | ~03:00-04:30 | E5 = last open reviewer demand |
| E7 brq lr1e3 | ~03:00 | no |
| 21 DeepSeek cells (capped evals + CE + geo; geometry already done) | ~04:00-06:00 | the generalization section |
| 284B ceiling (capped, same basis) | ~06:00 | no (nice-to-have) |
| Final: rerun analyze_full + analyze_adversarial + analyze_ebatch, freeze key_numbers §18, artifact refresh, final collect+push | ~06:00-08:00 | — |
