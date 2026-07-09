# Fleet findings digest (accumulating; full outputs in tasks/<id>.output)

## MiLoRA expert (DONE)
- Port FAITHFUL (minor-SVD init exact; all Table-10 hyperparams match our defaults).
- **Their protocol, verbatim: "the same hyper-parameter configurations as Hu et al. (2023) without tuning
  for all methods"** — ALL methods shared LR 3e-4 (math), 2e-4 (vision). So for THIS paper we can say,
  verified: single shared untuned LR. Their §5.5 even argues higher LR > PiSSA's 2e-5 — supports us.
- **Their Table 8 (forgetting loss, CE-to-base on WikiText-103, from Kalajdzievski 2024 scaling-laws-of-
  forgetting): PiSSA 6.07 > LoRA 3.24 > MiLoRA 2.54** — tracks their Table 7 ΔW-amplification (LoRA 68.2 /
  PiSSA 55.8 / MiLoRA 44.9; amplification 37.5 / 3.2 / 24.2). THIRD-PARTY MAGNITUDE→FORGETTING EVIDENCE.
- **CONFOUND IN THEIR EVIDENCE (our opening):** their LoRA ran α=2r, their MiLoRA α=r → MiLoRA's smaller
  ΔW is partly 2× scaling, not the minor subspace. Their design can't separate subspace from magnitude.
  → KILLER EXPERIMENT: matched-α, matched-LR LoRA-vs-MiLoRA logging ‖ΔW‖ + CE-to-base. If the advantage
  shrinks to ~0, direct strike for magnitude thesis.
- **Their Table 6 (init ablation, fixed budget): Principal 60.7 < Random 63.2 < Minor 64.0** — genuine
  second-axis (direction matters at fixed magnitude); acknowledge in paper.
- Gap attribution: GSM8K reproduced (our 62.85@1e-4/α2r vs pub 63.53; the 58.98@3e-4 trails purely via
  α=2r overshoot — fdelta 1.26 vs 0.45); MATH gap 14.24 vs 17.76 = cutoff 256 vs ~2048 → c2048 anchor is
  the right lever. milora_a1r@3e-4 expected ≈63.5.
- ADD METRIC: CE/KL-to-base on WikiText-103 (their Table-8 metric) — lets us plot our fdelta against
  their own forgetting axis. Cheap eval, no retraining.
- CITE: Kalajdzievski 2024 (arXiv 2401.05605) — magnitude-forgetting scaling-law lineage.
- Watch: our own α split (CS arms α=r, math arms α=2r) — keep explicit in cross-task comparisons.
- Note: frm_milora_lr7e4 trained-not-evaled; no milora_a1r/c2048 results yet (queued).

## CLoRA expert (DONE)
- Port faithful incl. the ½ factor (code has it, paper Eq 1 doesn't — footnote if quoting Eq 1).
- **LoRA-L2 verbatim confirmed**: "L2 regularization for trainable parameters... 1e-5" (1e-4 "too large");
  mechanism (loss-term vs optimizer wd) UNSPECIFIED, code unreleased → keep "same family, not identical
  (mechanism unspecified)".
- **Single shared LR verified verbatim**: "same ... configurations as [MiLoRA] ... for all methods";
  Table 6: one LR per base model (3e-4 L2-7B). No baseline LR-tuned.
- **CRITICAL METROLOGY: our `fdelta` = CLoRA's F_Δ (Eq 3, ‖ΔWx‖/‖x‖ on 100 inputs), NOT Frobenius.**
  key_numbers §0 + analyze_matrix.py mislabeled → FIXED in key_numbers (2026-07-09). PAPER ACTION:
  relabel axis "F_Δ (effective update magnitude)". dw_sv_* = their spectral ‖ΔW‖ column. Our axis is
  DIRECTLY comparable to their Table 4.
- Their Table 4 = external magnitude-law corroboration (k↑ ⇒ F_Δ↓ ⇒ F↑; LoRA-L2 F_Δ 0.29 → BBH 32.93).
  Overlay their (F_Δ,F) on our figure (shape, not intercept — their levels ~2× lower).
- HONEST TENSION: their CLoRA-k2048 EXCEEDS base out-domain (29.63 vs 26.74) in their lm-eval config;
  ours sits at/under base ceiling. Never claim "CLoRA doesn't help retention" — harness/config gap;
  recommended experiment: re-score under both harness configs to attribute.
- Their best out-domain variant is CLoRA-MINOR-k2048 (BBH 40.96), not random — if citing "best CLoRA".

## LoRA-Null expert (DONE)
- Port EXACT (loader byte-match; eigh-ascending ≡ svd bottom-r; covariance equiv). Remaining divergences
  uniform-by-design: α=2r, rank, 5 vs 7 target modules (they adapt o_proj+gate_proj too — citable diff).
- **Their protocol verbatim: single LR 2e-5, wd=0 for EVERY method.** No per-method tuning.
- Their mechanism = DIRECTIONAL theory (BA ⊥ knowledge); no ‖ΔW‖ or magnitude-forgetting analysis at all.
  BRIDGE: null-space init mechanically minimizes the knowledge-projected component of the update →
  their mechanism is a special case of the magnitude law restricted to the knowledge subspace.
- Borrowed evidence: their Table 4b CorDA rank collapse (r256: 89%→73% retention) = more capacity ⇒ more
  forgetting. Threat to note: their Fig 1 (small relative-A change, big retention gaps) argues direction.
- No Qwen/model-dependence in their paper → our Qwen finding is novel.

## CorDA++ expert (DONE)
- Port FAITHFUL (KPM bottom-r correct mode; CO-SVD, damped inverse, π(C), Eq 7-10, N=5×256, windowed-NQ
  calib all exact). 14 cells SAFE TO DISPATCH. Rank-alloc sign ambiguity = low-stakes (<0.5pt; Table VII
  shows ++ gain is ~all covariance selection).
- **Their protocol verbatim: single LR 2e-5, wd=0, α=r, r=128, all linear layers, for ALL methods.**
- FLAGS: our cells are campaign-harmonized (α=2r, r=64/32, LR sweep) → NOT paper repro except the 2e-5
  cell. LABEL non-2e-5 cordapp cells "campaign extrapolation". RECOMMENDED: add α=r cordapp anchor at
  2e-5 (else all CorDA++ numbers confounded by 2× scaling).
- THREAT to answer: at 2e-5, their LoRA catastrophically forgets (NQ 14.99→1.91, worse than full-FT)
  while CorDA++ holds 12.02 AND adapts slightly better. Our LoRA+wd@2e-5 cells must show magnitude
  control closes this. Their paper has NO LR sweep, NO wd, NO trained-‖ΔW‖ measurement = our gap to fill.
- Their Eq 5-6 = prior art acknowledging norm-of-moved-directions governs loss (truncation bound only).

## SC-LoRA expert (DONE)
- Port faithful (init bit-match; reference SVD dead-code correctly omitted). Deviations are SETTINGS:
  **our −4.15pp residual rests on β=0.5 + calib(nq_open)≠eval + r32 — ALL off the paper's recommended
  recipe (β=0.8/0.9, calib=eval-matched, r128). REVIEWER TRAP.**
- Their protocol: baselines swept {2e-5,5e-5,1e-4} best-picked in §4.2/4.3 (SC-LoRA fixed 2e-5); §4.1 all 2e-5.
  (NOTE: unlike the other papers, SC-LoRA DID tune baseline LR in 2 of 3 experiments.)
- Their Table 1: at fixed rank/LR, retention swings 18.88→22.73 purely with β → direction genuinely
  matters for SC-LoRA (it IS the directional control). Their Limitations verbatim: init constraint
  erodes with steps/complex tasks → endorses erosion in our 3-epoch regime.
- Our scl2 r128 β-sweep: raising β shrinks F_Δ 2.01→1.11 AND lifts both axes → β's benefit largely
  magnitude-mediated at hot LR; still below curve at β=0.99.
- CONTROLS NEEDED (priority order): (1) β=0.8/0.9 LR sweep (does −4pp survive recommended β?);
  (2) finish B4 eval-matched; (3) one CROSSED cell β=0.9 × eval_matched (fully paper-faithful);
  (4) r128 law check at 2-3 LRs; (5) init-erosion probe (B drift off Q_r vs step) → converts anomaly
  into measurement-tool demo.

## §I-II validator (DONE)
- All 55 figure points exact; both fits confirmed; within-method r all −0.86…−0.97; SC-LoRA sole sig
  deviator (−4.65 linear; DoRA +1.9 ns). FIXED in report: "8 adapters" tile → 7 assessed; PiSSA/DoRA
  motivation wording; panel-B line/label mismatch (drawn line = n=55 fit r=−0.87); "7 non-CorDA".
- FREE STRENGTHENERS (computed): on-curve-six method dummies add only ΔR²=0.03 (partial R² 0.20) —
  sharpest "geometry adds almost nothing given ‖ΔW‖" stat; alt-axis row (F_Δ 0.74 vs sv_mean 0.36 /
  sv_max 0.33 / LR 0.32); LOMO out-of-sample RMSE 1.7-3.6pp (SC-LoRA 9.0); shared-slope F(5,30)=0.28
  p=0.92; adaptation inverted-U peak ≈ ‖ΔW‖ 1.0 (quad coeff −47). Add per-method r table + Pareto panel.

## §IV validator (DONE)
- Table numerically EXACT incl. LoRA-Null row + all safe bands. FIXED in report: SC-LoRA band placement
  (0.56 is INSIDE [0.31,0.62] — its issue is geometry not magnitude); "ordered by ‖ΔW‖" → Spearman −0.82
  trend + SC-LoRA exception; "beyond noise" → single-seed honesty + robustness metrics (mean-ret 26.3
  highest, safe band 6/7) + at-least-as-good-on-both-axes; CorDA "on the law" → not-yet-assessed;
  k1024=82.6 fix + cross-eval caveat on boundary.
- FREE STRENGTHENERS: usable band (cs≥75 AND ret≥24): LoRA+wd 3/7, MiLoRA/LoRA 2/7, CLoRA/LoRA-Null 0/7
  (flattering AND honest); mean-ret column; Pareto plot; efficiency taxonomy AUDITED (no precompute:
  LoRA/LoRA+wd/DoRA; weight-SVD: MiLoRA/PiSSA; SVD+calibration fwd passes: LoRA-Null/SC-LoRA/CorDA;
  CLoRA: per-step regularizer overhead growing with k; cells ~2.8-5.5 GPU-h, CorDA longest).
- GPU asks: 2 seeds × top 3-4 cells ≈ 27-36 GPU-h (confirmatory); k1024 s43/s44 for 3-seed parity.

## §III validator / STATISTICS PACKAGE (DONE) — the PI ceiling question, answered
- All CS/ANCOVA/wd numbers verified exact. ONE real discrepancy FIXED in report: Qwen row was silently
  BROAD metric (−0.937/−26.1); core = −0.857/−32.0. Now quoted core-first (−0.86 ≈ Llama's −0.858 — a
  cleaner 2nd-model story), broad noted. "Four independent settings" rephrased (rows 1-3 same dataset).
- CEILING PACKAGE (all computed, now in report): Spearman ρ=−0.896 (p=3.5e-18) > |Pearson|; quadratic
  term real (b2=−9.0, p=0.004, concave); MODEL COMPARISON: saturating/hockey-stick BEATS linear AND
  quadratic on AIC + LOO-CV (quadratic is worst out-of-sample — unphysical upturn); knee at ‖ΔW‖≈0.36
  ≈ LoRA+wd op point 0.39; asymptote 26.8 ≈ ceiling. BELOW-CEILING slope −20.8 (ret<25) / Tobit −22.3
  vs pooled −14.8 → true effect ~40-50% STEEPER; partial r controlling method −0.868; permutation
  p<5e-5; per-method slope CIs all overlap ≈−10 except SC-LoRA −26 [−33,−19]; LOMO r stable.
- Inverted-U: significant on both tasks; math peak ‖ΔW‖≈0.70 (baseline 1.29 clearly past) → both-axes
  wd claim SCOPED TO MATH in report (CS peak CI too wide).
- FIGURE ACTIONS (paper): replace straight line with saturating curve + ceiling line + knee annotation;
  add below-ceiling-slope inset; label Qwen metric; GSM8K inverted-U sketch with peak + baseline marks;
  reconcile drawn slope (−15.43) to canonical (−14.78).

## §V-VI validator (DONE)
- ALL 26 published numbers transcribed EXACT vs the PDF; all 6 gates verify vs logs; LoRA-L2
  characterization verbatim-confirmed (soften "explicitly not spectral" → "a norm penalty, not
  direction-aware"); our 67.3 pick verified honest (matched c256, not the c512 69.5).
- **HEADLINE EXTERNAL REPLICATION (computed): CLoRA's own Table 4 → r(log10 F_Δ, BBH) = −0.98,
  slope −14.7 pp/decade vs our −14.8. Two independent datasets, one line.** Added to report §V.
- FIXED in report: their "MMLU" column IS MMLU-Pro (base 18.56 < random 25 ≈ ours 18.96) — metrics
  commensurable, old caveat was wrong against us; "~4 pts" → ~5 pts (5.0); §VI "paper 33.10" → our
  registered ceiling (CLoRA's base BBH = 34.91, ~2pp harness offset → compare Δ-from-base);
  in-pipeline-competitors-single-LR note + MiLoRA pipeline anchor (62.85↔63.53).
- STRENGTHENERS for paper: Table-4 scatter figure w/ fitted r; per-adapter published-vs-ours Δ-from-base
  columns-pair; extract BBH/forgetting tables from the other 4 PDFs → "the law is in the 2025
  literature" (P3, medium cost).

## EFFICIENCY + MEMORY analyst (DONE) — PI's compute/memory requirement
- LoRA+wd = frontier at ZERO cost: no init, wall-clock identical to LoRA (17,126 vs 17,138s; wd is a
  free AdamW flag). Trainable r64 = 112,197,632 (1.638%, log-verified).
- DoRA: 2.13x wall-clock (r16 2.22x) for no retention/adapt benefit.
- CLoRA: pays at TRAIN time — frozen P block (P_u out×k + P_v in×k per module, bf16) resident memory
  = k×1,753,088 floats: **0.42GB(k128) / 0.84 / 1.67(k512 default) / 3.34 / 6.7GB(k2048)**. At default
  k512 that's 8× the trainable LoRA weights; +9.5-14% train wall-clock. Memory-dominated, k-scaling.
- Data-aware INIT tax (one-time, then trains as vanilla LoRA): MiLoRA/PiSSA = 160 base-W SVDs, no
  forwards; SC-LoRA = 512 calib forwards (D+ 256 + D- 256) + eigh; LoRA-Null/CorDA = 256 forwards +
  eigh/inv+SVD; CorDA++ = 1280 forwards (N=5×256) + 5× per-layer inv+SVD + rank alloc (~5× CorDA,
  ~3.5e16 init FLOPs). ~22.5GB transient covariance accumulator during init.
- Peak GPU mem not instrumented (no torch memory prints); analytical resident comparison is the
  substitute. ~55GB per 7B+LoRA r64 process (incidental OOM traces).
- PAPER: efficiency table + CLoRA-memory-vs-k subtable; framing "LoRA+wd on the frontier at zero init,
  zero k-memory-tax, LoRA-identical train cost; fancy methods pay strictly more for same trainable budget."
- Evidence: full file:line map in tasks/ae2310183ffb6dd65.output.

## GEOMETRY DRIFT phase2 (DONE, 320 adapters) — HEADLINE two-factor result (PRELIMINARY, needs stress-test)
Master table: results/geo_drift/adapter_metrics.jsonl (per-adapter) + permatrix/ (per-layer).
- Magnitude law holds: r(retention, log F_Δ) = −0.81 (n=303 joined).
- **SECOND-ORDER GEOMETRY EFFECT beyond magnitude**: partial r(retention, e_top | log F_Δ) = −0.51;
  partial r(retention, amp_top | log F_Δ) = −0.58. At MATCHED magnitude, updates concentrated in W's
  TOP singular subspace (principal directions) forget MORE.
- Per-method energy signature (e_top vs e_bot, top-256/bot-256 of 4096): PiSSA e_top 0.188 (highest,
  principal-init) → forgets most; MiLoRA UNIQUE with e_bot 0.120 > e_top 0.065 (minor-init) → retains;
  SC-LoRA family (sclora+scl2 b0p*) e_top 0.08-0.115 + amp up to 0.19 → EXPLAINS its −4pp below-law
  deviation (concentrates in principal dirs despite modest magnitude); LoRA/LoRA+wd/CLoRA mild top-lean.
- This is the EXPLANATORY payoff: magnitude = 1st-order dominant lever; principal-direction
  concentration = 2nd-order effect the geometric methods implicitly exploit. Constructive framing:
  reconciles thesis with SC-LoRA/high-k-CLoRA boundaries by giving the mechanism. NOT "geometry
  doesn't matter" — "magnitude dominates; the residual geometry effect is measurable and explains the
  method differences."
- CAVEATS to stress-test (interpreter mandate): method labels imperfect (run.split[1] → sclora split
  across 'sclora'/'b0p*'; clora across 'clora'/'k1024'/'k2048'); PiSSA n=1; is 2nd-order effect robust
  to dropping PiSSA+SC-LoRA? within-method vs between-method? per-method partials? significance?

## CE-TO-BASE forgetting metric (DONE, validated on GPU) — 3rd-party-comparable forgetting axis
- Metric = soft CE of ft next-token dist vs BASE dist on WikiText-103 test (MiLoRA §5.4 / Kalajdzievski
  2024). Script forgetting_ce.py; disable_adapter==fresh base (0.0 diff, verified). WikiText-103 cached.
- VALIDATED vs MiLoRA Table 8: our LoRA CE 3.57 (their 3.24), PiSSA 6.31 (their 6.07), PiSSA>LoRA ✓.
  Monotone with magnitude: Spearman ρ(CE, fdelta)=0.943 (p=0.005). 40-block slice converged (=full ±1%).
- **KEY THESIS RESULT: at matched LR 3e-4 + matched rank, MiLoRA CE 3.66 ≈ LoRA 3.57 — MiLoRA does NOT
  forget less once magnitude is held fixed.** Their published advantage came from a lower-magnitude r=8
  operating point. Independent 3rd-party metric corroborates the magnitude-budget law.
- CROSS-CHECK vs geometry 2nd-order effect: MiLoRA(minor)≈LoRA(mild-top) at matched magnitude on CE
  suggests the geometry 2nd-order term is SMALL for on-curve methods (may be PiSSA/SC-LoRA-driven) —
  interpreter reconciling.
- BATCH plan: ~4.7h/GPU full (330 blocks) or ~2h at 128-block slice; shard across idle GPUs. DEFER full
  batch until CS-grid (paper spine) milestone frees GPUs; slice numbers already sufficient for the point.

## GEOMETRY interpreter (DONE) — VERDICT: preliminary 2nd-order claim OVERTURNED (rigor worked)
- Method labels were WRONG in adapter_metrics (run.split fallback) AND in every summary/registry
  ("LORA" for all). Re-labeled from registry args → results/geo_drift/master_labeled.jsonl (303 w/ outcomes).
- **The "principal-direction concentration forgets more" 2nd-order effect is NOT robust — it was carried
  entirely by SC-LoRA + PiSSA (principal-init outliers).** Partial r(ret, amp_top|logfΔ): pooled −0.58 →
  drop PiSSA+SC-LoRA +0.25 (FLIPS) → on-curve −0.14 ns. ANCOVA: amp_top beyond logfΔ+method ΔR²=0.0002
  (p=0.53, nothing). CE cross-check agrees (MiLoRA≈LoRA at matched mag). DO NOT claim it in paper.
- **Robust findings:** (1) MAGNITUDE = 1st-order dominant lever, within EVERY method r −0.75..−0.94, all
  p<0.02, survives rank control. (2) RANK = genuine 2nd-order lever: partial r(ret, log r|logfΔ)=−0.56
  (higher rank → more forgetting at matched magnitude); honest caveat = stable_rank 0.84 collinear w/ r,
  residual marginal once r+method controlled → "magnitude first, rank second."
- **GEOMETRY = MEASUREMENT TOOL (the real geometry contribution, fits guardrails):** SVD-alignment
  metrics fingerprint each method's init from the TRAINED adapter and it PERSISTS 3 epochs: MiLoRA only
  method e_bot>e_top (minor-init, both tasks); SC-LoRA input-principal spike ein_top 0.41 (q/k early
  layers 0.61-0.75); CorDA input-minor ein_bot 0.49 (MLP, + magnitude blowup fΔ27.5, 92% energy in
  down_proj); PiSSA principal (n=1); LoRA/LoRA+wd/CLoRA/DoRA near random baseline. SC-LoRA constraint
  ERODES with LR: ein_top 0.70→0.21 (r=−0.96) — quantitatively confirms SC-LoRA paper's erosion claim.
- Explains the 2 outliers (PiSSA worst forgetter; SC-LoRA −5.7pp below law) WITHOUT a universal
  geometric axis. Metric caveats: amp_top≈ein_top (r0.98)≈e_top(0.78) = one axis; e_top/ein not
  cross-comparable across target types w/o normalizing by k/d; PiSSA n=1, LoRA-Null n=7.
- Figure spec (4 panels): A magnitude law; B stress-test bars (amp collapses/flips, rank stays); C
  method-fingerprint heatmap; D per-layer + SC-LoRA erosion. Data: results/geo_drift/master_labeled.jsonl.
