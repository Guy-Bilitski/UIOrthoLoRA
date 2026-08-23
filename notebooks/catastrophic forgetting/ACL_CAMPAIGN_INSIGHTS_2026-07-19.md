# ACL Repositioning Campaign — Insights & Thoughts for Guy (2026-07-19)

Five analysis agents + an independent verifier + an author/reviewer/supervisor writing loop ran over the frozen dataset (n=1035 pool). Every number below survived an adversarial recompute from raw `summary.json`/geometry/CE stores (34 confirmed / 8 corrected / 2 overclaims caught before they reached the paper). Detailed reports: `paper/writing/acl_analysis/{strategy,observatory,correlations,adjudication,insights,verification}/`.

## The big picture (my honest take)

Your instinct was right and the panel's thesis survives it: **the consolidation framing and "Magnitude First, Geometry Second" are the same paper described from two ends.** The four metrics work as *one apparatus applied identically to every adapter* — that's the ACL-native contribution — while magnitude stays the answer. Giving the four metrics equal *rhetorical* weight would have turned the paper into a leaderboard without a thesis; the strategy memo's rule "equal method, unequal message" is the right compromise, and the rewritten paper follows it.

## What the deep-dive actually found (ranked by how much I'd care)

1. **Your question 3 has a clean answer.** Best single retention predictor: **log F_Δ (ΔR² +0.420 over family effects) > spec_max +0.349 ≈ ‖ΔW‖_F +0.348 > CE/KL drift +0.340 > LR +0.207 (5th of 10) > stable rank +0.116 > everything else ≤ +0.032.** CE adds only +0.005 unique beyond magnitude; magnitude keeps +0.085 beyond CE. Cross-validated, magnitude alone transfers best to unseen families. So: magnitude for control, CE for monitoring, geometry for fingerprinting — three roles, not three rivals.

2. **CE drift is a practical product, not just a metric.** KL-to-base (one forward pass, no benchmark evals, no weight access) predicts held-out retention within 1.3–2.0 pp on Llama families and detects >5 pp damage at AUC ≥ 0.976 in all six. The knee sits at **≈0.26–0.30 nats in 4/6 families across both base models** — a "keep drift under ~0.3 nats" practitioner rule. This is the most paper-worthy *new* thing the campaign produced; it's now the Behavioral Drift section's corollary.

3. **The "trade-off" is mostly not a trade-off.** Run-level adaptation↔retention correlation is *positive* over most of the pool (above-knee runs lose both). 99–100% of peak adaptation is reachable *below* the retention knee in every family — but plain LoRA at standard 3e-4 has **zero** below-knee cells, while LoRA+wd has 12/26. One sentence version: *the free lunch exists, but you need a magnitude knob to sit down at the table.* Three regimes along the curve: free → priced (0.1–0.5 ret-pp per adapt-pp) → dominated (negative-sum, all six families).

4. **Method superiority verdict:** LoRA+wd loses none of 26 paired head-to-heads on retention and is the sole method on the observed Pareto frontier in 3 of 4 families. The one honest exception — SC-LoRA on Qwen-math, +8.3 pp GSM8K at tied base-level BBH — is attributable to its *data-aware initialization as configured*, not subspace geometry per se (no eval-matched control exists there), comes with ±16 pp adaptation seed-lottery and the narrowest safe-LR band, and its winning cell is itself low-magnitude. CLoRA's real distinction: **zero in-band divergences (0/121)** — geometry buys optimization stability, not retention.

5. **Three knobs, one curve.** Weight decay, CLoRA's k, and rank all move F_Δ monotonically and land on the same retention curve (residual effects beyond F_Δ all n.s.). At fixed LR, k=2048 also costs adaptation (76.8→69.4) — the k-knob is a magnitude limiter with an adaptation tax (phrased CLoRA-faithfully in the paper: our harness's dose-response, not a critique of their results).

6. **A universal fragility ordering** (new): normalized by base ceiling, retention benchmarks die in the order **MMLU-Pro > BBH > MMLU > TruthfulQA** in all six families (Kendall W = 1.000). And the training task itself starts to forget: hellaswag/ARC gains collapse first, social_i_qa last — part of `cs_avg` behaves like a hidden retention benchmark.

7. **A metrology landmine we defused:** TruthfulQA *rises* with forgetting on Llama (r up to +0.79) and falls on Qwen — inside `retention_broad` it attenuates the measured broad slope 24–30% on Llama. Disclosed in the paper; worth remembering whenever anyone quotes broad retention.

8. **Hygiene catches:** `spec_max` (geometry pipeline) ≡ `dw_sv_max` (summary pipeline), r > 0.9999 — same measurement, two names; the paper now cites one. Doc-05's "KL beats F_Δ in 5/6 families" flips to 1/6 under the frozen-pool convention — pool-convention footnotes now mandatory. The geometry second-order axis (stable rank) is Llama-specific (partial −0.3…−0.67) and ≈0 on Qwen.

## What changed in the paper (Overleaf CF_PEFT)

- **Title:** *Magnitude First, Geometry Second: A Fair, Multi-Metric Audit of Catastrophic Forgetting Across PEFT Adapters* (panel thesis + your consolidation framing).
- **Landed the frozen numbers at last:** n=49/r=−0.86 → n=1035/r=−0.847, multi-seed, ΔR² ladder, E1 intervention, E4 resolution. (The manuscript had been one generation behind the analysis layer.)
- **New sections:** §3 "The Four Instruments" (your 4-metric protocol as a unit); §7 "Behavioral Drift" (CE as behavior-space view + the ~0.3-nat monitor).
- **New MAIN exhibits:** ΔR² ladder table (the one-table title claim: magnitude +0.395, geometry +0.017, method +0.006), cost-of-geometry table (DoRA 2.15× train; CLoRA +3.34 GB at k1024, analytical; init taxes; rank-2r deploy delta), multi-seed Pareto figure. New appendix: league table, CE-proxy, dose-response, LR-band figures.
- **Guardrails held everywhere:** "first-order predictor" never "mechanism"; "matches or edges" never "beats"; CLoRA published numbers faithful (the embargoed cross-harness +2.2 pp claim was caught and deleted); 23× = incremental ΔR², 18× = unique variance, never conflated.
- Review verdict after revision: *accept-shaped*; all three MAJOR findings and all minors closed. Body fits the 8-page ACL limit; compiles clean.
- Pre-freeze manuscript backed up at `paper/writing/paper_prefreeze_backup_2026-07-18.tex`.

## What I'd do next (priority order)

1. **The Qwen rescale ladder (A2)** — the one GPU experiment that most raises the ceiling: it takes E1's causal claim cross-architecture and closes the "n=1 interventional arm" objection.
2. **Complete E4 (20/24 → 24/24)** and, if possible, an eval-matched SC-LoRA Qwen-math control — it would settle whether the one head-to-head exception is calibration or something real.
3. **Consider a short "drift monitor" artifact release** (the ~0.3-nat rule + per-family calibration script) — cheap, practitioner-facing, very ACL.
4. The per-benchmark fragility ordering could seed a follow-up on *what* forgets first and why (format-following vs knowledge) — the two-channel story wants a mechanistic sequel.

## Caveats I'd keep in mind

- The CE-monitor is a **ruler on Llama, only a tripwire on Qwen** (RMSE ≈4–6 pp, tail-driven), and calibration does not transfer raw across tasks — per-family calibration is a design requirement, stated in the paper.
- Several colorful findings stayed out of the main text deliberately (q_proj energy share, adaptation-side ordering, per-layer fingerprints) — split-dependent or not independently verified; they live in `acl_analysis/insights/findings.md` with honesty tags.
- Verification stats caveat stands: seeds within a cell are correlated (ICC ≈ 0.78); everything quoted is cluster-robust.
