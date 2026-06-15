> ⛔ **SUPERSEDED (2026-06-15).** Original "beat CLoRA" plan; that goal is abandoned (corrected
> UIOrthoLoRA only ties CLoRA). Current plan: **08_FORWARD_PLAN.md**. Kept for provenance.

# EXPERIMENT PLAN — what to run for a publishable result

Three threads: **(A) the UIOrthoLoRA/UILinLoRA frontier** (the go/no-go), **(B) the leakage
thermometer story** (see `03_LEAKAGE_ANGLE.md` — the most novel/exciting), **(C) robustness/ablations**.
All UIOrthoLoRA runs: `uio_inprocess.py`, **LR 1e-2**, full 3 epochs, `--ret_limit 64 --ret_max_gen 512`,
leakage logged by default. One GPU each via `gpu_pool.py`.

## A. The frontier (in flight)
- **A1 [RUNNING] Wave 1** — param-matched, k_vec=410, use_de=1, k_val∈{410,1024,2048,3072,4096} + LR{5e-3,2e-2} brackets + UILinLoRA. → high-CS region + confirms LR=1e-2.
- **A2 [QUEUED] Wave 2** — high-retention corner: small k_vec × use_de on/off (`jobs/uio_wave2_retention.txt`). → the win candidate.
- **A3 seeds** — once best 2-3 (k_val,k_vec,use_de) points known, run **3 seeds** each (42/43/44) for error bars. REQUIRED for publication credibility.
- **A4 full retention on the winners** — re-eval the best 2-3 UIOrthoLoRA points with FULL retention (no ret_limit/cap) so the headline numbers aren't fast-only. (Merge→shard, or in-process full overnight.)
- **A5 [conditional] corrected-major-term layer** — IF clean (use_de=0) configs retain poorly despite ~0 thermometers (finding #4 in OPERATING_STATE), build a layer variant with the leading block as true identity (drop `E·U₁·I·V₁ᵀ·D`) and re-run the frontier. Likely UIOrthoLoRA's *true* shot.

**Win conditions (any one ⇒ publishable):** (i) a UIOrthoLoRA point with ret>24.8 at CS≥80 (beats CLoRA-k1024); (ii) ret≈25-26 at CS>65 (dominates CLoRA-k2048 in the high-retention corner); (iii) clearly better CS–retention *frontier area* than CLoRA across the sweep.

## B. Leakage thermometer story — THE novel angle (user-flagged: "how much leakage between low tail & major")
See `03_LEAKAGE_ANGLE.md` for the full framing. Core experiments:
- **B1 leakage map (free):** Wave 1 + Wave 2 already log μ_E, ν_D, Leak11, OffTailF, Drift. Plot **leakage (μ_E,ν_D) vs retention vs CS** → show the *leakage budget* tradeoff (too little→under-adapt, too much→forget).
- **B2 leakage penalty on Llama [HOLD per user until frontier in]:** add `R_mix = λ_E‖M_E‖_F² + λ_D‖M_D‖_F²` to uio_inprocess training (variants λ∈{0, 1e-3, ...}); test whether controlling leakage improves Llama retention at fixed CS. (`M_E,M_D` from `leakage.py`, computed in autograd — NOT no_grad.)
- **B3 the paper's exact diagnostic (RoBERTa-base GLUE):** reproduce variants A(0,0)/B(1e-3,0)/C(1e-3,1e-3) on RTE,MRPC,CoLA,STS-B,SST-2, k_val=256, k_vec=0, q/k/v/o (48 modules), 2 seeds. **Separate harness** (RoBERTa+GLUE) — not built yet. Expected: A→B drops μ_E but RAISES ν_D (escape route); A→C drops both. This is the "must watch both sides" result.

## C. Robustness / ablations (Phase-2, after GO)
- C1 **LR×retention** full curve (does lower LR within the adapting range trade CS for retention?).
- C2 **init_scaler/init_sigma** sweep (affects effective adapter magnitude / gradients).
- C3 **use_de on/off** clean ablation at matched (k_val,k_vec) — isolates the orthogonality-breaking effect (partly in Wave 2).
- C4 math task (GSM8K/MATH, rank64) — CLoRA degrades past k128 there; UIOrthoLoRA's medium tier may exploit it.
- C5 LLaMA-3-8B secondary model.
- C6 init-only family (MiLoRA, LoRA-Null, SC-LoRA) — recipes in memory `method-port-recipes`.

## Priority order for an A* story
1. Finish A1+A2 → see if any win condition hits. (~hours)
2. B1 leakage map (free from A1/A2 logs) — likely the strongest novelty regardless of frontier outcome.
3. A3 seeds + A4 full-retention on winners.
4. If frontier disappoints: A5 (corrected major term) — most likely to unlock UIOrthoLoRA's real retention.
5. B2/B3 leakage-penalty experiments (the "leakage budget" paper section).
