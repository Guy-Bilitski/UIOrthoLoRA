# Artifact integration pass — fixes to apply (all sections at once)

## NEW SECTION: "The adapters" (top, after glossary) — content ready (agent ab3e5b3c)
Lead: all nine add a trainable low-rank update ΔW=A·B to frozen weights; differ only in how it's
initialized/constrained. Then 9 blurbs (≤2 sent each): LoRA, LoRA+wd (ours), DoRA, PiSSA(principal),
MiLoRA(minor), CLoRA(orthog penalty/k), SC-LoRA(calib subspace/β), LoRA-Null(null-space of activations),
CorDA++(context SVD + dynamic rank). Only LoRA+wd = "(ours)". [full text in transcript]

## REVIEWER af09a9 fixes:
- **B1 (BLOCKER):** DELETE §3 boundary sentence "in our controlled pipeline at matched update size,
  LoRA+wd0.5 (82.0/27.3) beats CLoRA-k2048 (79.9/25.1)" — resurrects KILLED lucky-seed 82.0; s44
  collapses (82→50.9; CLoRA 79.9→23.0). Replace with published-only high-k boundary: CLoRA k1024/k2048
  (82.6/83.7, BBH 36.5/38.7) exceed LoRA+wd BUT via smaller update (their Table 4 → on the law); null-
  space buys adapt-per-update at high rank; law governs both. (Also fixes M5 CLoRA over-weight.)
- **M1:** magnitude DW panel plots CorDA + draws n=55 slope −15.43 but labels r=−0.86 (n=49). FIX: drop
  corda from S{} in DW panel (keep in LR panel ok); set DWFIT.slope=−14.78 so line+label+"7 assessed" agree.
- **M2:** CE table (§6) rows are faithful MATH α=2r runs (F_Δ 1.28) — contradicts §3 LoRA F_Δ 0.62 (α=r CS).
  FIX: caption "CE on faithful math-recipe adapters (MetaMathQA, r64/α128); F_Δ on α=2r scale ~2× the CS
  §3 runs"; relabel rows "LoRA (math, 3e-4)" etc. Comparison valid (matched), just disclose.
- **M3:** Qwen r = −0.88 (not −0.86); scope = COMMONSENSE. FIX: "second architecture (Qwen2.5-7B
  commonsense, r = −0.88)". (tile + line 118)
- **M4:** within-method range "−0.75..−0.94 / 7 assessed / CorDA" is wrong scoping. FIX §1+tile: "within
  every one of the 7 assessed adapters (r −0.86 to −0.97)". Reserve −0.75..−0.94 (incl CorDA) for §2 geo.
- **M5:** CLoRA over-weighted → B1 trim + §4 bar reframe title "GSM8K — our LoRA+wd vs each method's best
  published number", caption "published by each method's paper (LoRA/MiLoRA/PiSSA/CLoRA); grey=in-pipeline".
- **M6:** CorDA law-residual −3.0 is uncalibrated (key_numbers §8 withhold). FIX: blank/caveat that cell
  "CorDA retention uncalibrated (fairness fix pending); shown for geometry signature only".
- minors: m1 grey bar CSS (add .bar-fill.repro grey); m2 "six"→"all seven" 78-82; m3 soften §2 "magnitude
  not geometry" → "adds essentially nothing (ΔR²≈0.0002) among on-curve methods"; m4 geo residual caption
  "vs geometry-battery law"; m6 "20,000-shuffle"→"permutation (p<5e-5)"; m7 dek "…commonsense, with a
  faithful math reproduction"; m8 add small GSM8K/BBH-retention 2-col table to §4.

## PENDING (integrate together): geometry columns (ad4ee5), CS safe-band (acd612), efficiency times+mem (a1c8d3)

## GEOMETRY clarity (agent ad4ee5, all numbers CONFIRMED) — ready to paste
- (A) "what we analyzed" paragraph: reconstruct ΔW=(α/r)B·A from saved adapter (validated), SVD the
  BASE weight, measure per-layer how much update energy lands in base top(principal)/bottom(minor)
  singular dirs, output & input sides, over 320 adapters, no retraining. Two questions: does the
  advertised geometry survive into trained weights, and does it predict forgetting beyond size. Answer:
  size 1st, rank modest 2nd, subspace adds nothing measurable among on-curve → fingerprint, not axis.
- (B) column defs: e_top/e_bot = share of update energy in base top-256/bottom-256 OUTPUT singular dirs;
  ein_top/ein_bot = same INPUT side; stable rank = effective # dirs (higher=spread); law residual =
  retention above(+)/below(−) magnitude-law prediction (pp). Baseline: top/bottom-256 = 6.25% of 4096,
  so neutral update ≈ e_top≈e_bot≈0.06; LoRA/LoRA+wd/CLoRA/DoRA sit at neutral band.
- (C) reading guide: MiLoRA only method e_bot>e_top (minor-init, +1.6); SC-LoRA input-principal spike
  ein_top 0.410, −5.7 below law, erodes 0.70→0.21 w/ LR; CorDA input-MINOR spike ein_bot 0.494, −3.0,
  driven by magnitude blowup F_Δ≈27.5.
- PROSE FIX: intro "two principal-direction methods pay extra cost" is WRONG — CorDA is minor-input not
  principal. Change to "the two methods with the strongest input-side alignment spikes"; add CorDA's
  ein_bot line to closing para. (reconciles with M6: CorDA residual uncalibrated → caveat.)
