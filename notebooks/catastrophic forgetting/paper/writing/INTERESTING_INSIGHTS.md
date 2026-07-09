# Interesting empirical insights — corrected & consolidated (2026-07-09)

Living store of concrete, defensible findings from the faithful-reproduction / magnitude-law campaign.
It is also the **holding pen for clarifications and future/uncertain items deliberately kept OUT of the
supervisor artifact** — so nothing is lost. Each item is tagged:

- **[DONE]** validated, reproduces from raw data, safe to state in the paper.
- **[OPEN]** in-progress, pending a run/control, or explicitly uncertain — do NOT state as settled.
- **[RESOLVED-NEG]** tested and *rejected*; recorded so no one re-chases it.

Authoritative numbers = `data/key_numbers.md`. Framing guardrails (PI): constructive; **magnitude is the
1st-order lever, rank a modest 2nd-order lever, geometry is a fingerprint** (the principal-direction
"2nd-order axis" is rejected — see §3); never "geometry doesn't matter"; no unverified claims about other
papers.

---

## 1. THE MAGNITUDE LAW — retention is governed by ‖ΔW‖, not by the adapter method  **[DONE]**

**Finding.** Retention (mean of answer-only BBH 3-shot + MMLU-Pro 5-shot CoT, base ceiling 26.0) collapses
onto ONE curve in the effective update magnitude F_Δ, across every adapter:

- **Llama-2 CS, pooled n=49 (7 adapters × 7 LRs):** r = **−0.86**, R² = 0.74, slope −14.8 pp/decade.
- **On the 6 well-behaved adapters (excl. SC-LoRA, n=42):** r = **−0.92**, R² = 0.84, slope −10.0.
- **Within EVERY adapter individually:** r ∈ [−0.86, −0.97] (LoRA −0.95, LoRA+wd −0.89, DoRA −0.97,
  MiLoRA −0.94, CLoRA −0.90, SC-LoRA −0.97, LoRA-Null −0.86). The points interleave across methods —
  each method just slides along the same line.
- **2nd model (Qwen-2.5-7B CS, LoRA sweep):** r = **−0.88 (core)** / −0.92 (broad), n=7 — the law
  replicates on a second architecture.
- **Llama-2 math, pooled n=14:** r = −0.97 (sparse; LoRA/LoRA+wd/DoRA only).

**Ceiling-robust statistics** (the "is it just a line?" question, answered): Spearman ρ = **−0.896**
(p=3.5e-18) exceeds |Pearson|; the quadratic term is real (b₂=−9.0, p=0.004, concave); on AIC **and**
LOO-CV a **saturating/hockey-stick fit beats both linear and quadratic** (the quadratic is worst
out-of-sample — its upturn is unphysical). Knee at F_Δ ≈ 0.36 (≈ LoRA+wd's operating point 0.39),
asymptote 26.8 ≈ the base ceiling. The **below-ceiling slope is −20.8** (ret<25) / Tobit −22.3 vs the
pooled −14.8 → the true effect is ~40–50% *steeper* once the ceiling is accounted for. Partial-r
controlling for method −0.868; permutation p<5e-5.

**Why it matters.** This is the paper's spine: the simplest magnitude control (plain **LoRA + weight
decay**) matches elaborate structured/data-aware inits at equal magnitude. Reported single-LR "wins" for
fancy adapters are an artifact of landing at a different F_Δ (see §2).

---

## 2. LEARNING RATE IS A WEAKER PROXY THAN ‖ΔW‖ — the fairness result  **[DONE]**

**Finding.** On the same n=49 CS runs, log-LR predicts retention with R² = **0.32** (r=−0.57); the F_Δ it
produces predicts with R² = **0.74** (r=−0.86). R² *more than doubles* by switching the x-axis from the
knob to the update it produces. Among *available* magnitude axes F_Δ is uniquely predictive: F_Δ 0.74 vs
σ_max (dw_sv_max) 0.33 vs dw_sv_mean 0.36 vs log-LR 0.32.

**Why it matters.** This is *how* single-LR papers manufacture method "wins": at each method's favored LR it
sits at a different magnitude. Sweeping LR as a controlled variable and measuring ΔW (the column the papers
omit) dissolves the wins. Use **F_Δ**, never σ_max, as the magnitude axis — σ_max is confounded (MiLoRA/LoRA
show huge σ_max ~155 at moderate F_Δ; CLoRA shows small σ_max ~28 at similar F_Δ).

---

## 3. GEOMETRY-DRIFT VERDICT — magnitude 1st, rank 2nd, principal-direction axis REJECTED  **[DONE / RESOLVED-NEG]**

Computed over the 320 saved adapters (validated ΔW=(α/r)B@A reconstruction). Master labeled table:
`results/geo_drift/master_labeled.jsonl` (labels from `train_registry.jsonl` args — the run.split/summary
`method` field is a broken "LORA" fallback and must NOT be used). Pipelines: `geo_drift_phase1.py`,
`geo_drift_phase2.py`.

- **[DONE] 1st-order = MAGNITUDE.** retention vs log F_Δ within every method r∈[−0.75 (CorDA), −0.94
  (DoRA)], all p<0.02, survives controlling for rank. Rock-solid.
- **[DONE] 2nd-order = RANK (honest, modest).** partial r(retention, log r | log F_Δ) = −0.56 (p=5e-19):
  at matched magnitude, higher-rank / more-spread updates forget a bit more. **Caveat:** stable_rank is
  0.84-collinear with log r and the residual is marginal once BOTH r and method are controlled → state as
  "magnitude first, rank second," not a mystery axis.
- **[RESOLVED-NEG] "principal-direction concentration is a 2nd-order forgetting axis" — TESTED and REJECTED.**
  The preliminary partial r (amp_top −0.58, e_top −0.51) was carried **entirely by two principal-init
  outliers (SC-LoRA, PiSSA)**. Drop them → amp_top **flips to +0.25**; among on-curve methods it is −0.08
  to −0.14 (n.s.); ANCOVA amp_top beyond log F_Δ + method adds ΔR²=**0.0002 (p=0.53)**. The independent
  CE-to-base metric agrees (§4). **Do NOT put this in the paper as a competing geometric axis.** Recorded
  here as a closed negative so it is not re-chased.
- **[DONE] Geometry's REAL contribution = a MEASUREMENT / FINGERPRINT tool** (the constructive framing).
  The SVD-alignment metrics recover each method's *init design* from the *trained* adapter, and the
  signature persists through 3 epochs:
  - **MiLoRA** — the only method with e_bot > e_top (minor-singular init), both CS and math.
  - **SC-LoRA** — input-side principal spike (ein_top 0.41; q/k early layers 0.61–0.75) that **erodes with
    LR** (ein_top 0.70→0.21, r=−0.96) — quantitatively confirms the SC-LoRA paper's own "constraint erodes
    with steps" limitation.
  - **CorDA** — input-side minor spike (ein_bot 0.49, MLP) + magnitude blow-up (F_Δ 27.5, 92% of energy in
    down_proj).
  - **PiSSA** (n=1) — principal (e_top 0.188), the worst forgetter.
  - **LoRA / LoRA+wd / CLoRA / DoRA** — near the random-alignment baseline.
  This explains the two law-outliers (PiSSA worst; SC-LoRA below the CS law) mechanistically **without**
  positing a universal geometric axis.

---

## 4. CE-TO-BASE FORGETTING METRIC — independent, third-party-comparable corroboration  **[DONE]**

**Finding.** Metric = soft cross-entropy of the fine-tuned next-token distribution vs the BASE distribution
on WikiText-103 (MiLoRA §5.4 / Kalajdzievski 2024 scaling-laws-of-forgetting). Script `forgetting_ce.py`
(`disable_adapter` == fresh base, 0.0 diff verified).

- **Validates vs MiLoRA Table 8:** our LoRA CE 3.57 (their 3.24), PiSSA 6.31 (their 6.07), PiSSA>LoRA ✓.
  Monotone with magnitude: Spearman ρ(CE, F_Δ) = **0.943** (p=0.005).
- **KEY:** at matched LR 3e-4 + matched rank, **MiLoRA CE 3.66 ≈ LoRA 3.57** — MiLoRA does NOT forget less
  once magnitude is held fixed; its published advantage came from a lower-magnitude r=8 operating point.
- Cross-check of §3: MiLoRA(minor) ≈ LoRA(mild-top) on CE at matched magnitude → confirms the geometry
  2nd-order term is negligible for on-curve methods (the earlier signal was PiSSA/SC-LoRA-driven).

**Why it matters.** A completely independent forgetting axis, comparable to the numbers other papers report,
lands on the same conclusion as F_Δ.

---

## 5. METROLOGY CORRECTION — our magnitude axis IS CLoRA's F_Δ (not the Frobenius norm)  **[DONE]**

**Finding (CLoRA-paper-expert audit vs the PDF, 2026-07-09).** The `fdelta` field is **NOT** ‖ΔW‖_F. It is
CLoRA's **F_Δ (their Eq 3)**: mean over tokens of ‖ΔW·x‖/‖x‖ on 100 real eval inputs, averaged over
updated matrices — a data-dependent effective-output-change measure. `fdelta.py` /
`uio_inprocess.fdelta_inprocess` implement exactly this; the old "token-weighted Frobenius" label
(key_numbers §0, `analyze_matrix.py:9`) was wrong and a CLoRA-reading reviewer would catch it. `dw_sv_*`
correspond to their spectral ‖ΔW‖ column.

**Upside:** our axis is therefore **directly comparable to CLoRA Table 4's F_Δ column**. PAPER ACTION
(pending): relabel every "‖ΔW‖_F / Frobenius" mention of this axis to "F_Δ (effective update magnitude,
CLoRA Eq 3)" — done in key_numbers.md; still pending in `paper.tex`, `analyze_matrix.py`, and figure labels.

---

## 6. PUBLISHED EVIDENCE — CLoRA's own Table 4 IS the magnitude law  **[DONE]**

**Finding.** CLoRA Table 4 reports F_Δ vs BBH: LoRA 0.79→26.7, LoRA-L2 0.29→32.9, k128 0.36→30.8,
k512 0.27→34.3, k1024 0.21→36.5, k2048 0.14→38.7. Fitting it: **r(log₁₀ F_Δ, BBH) = −0.98, slope −14.7
pp/decade — vs our −14.8.** Two independent datasets, one line.

- Their **LoRA-L2** (their own weight-decay-family baseline) is the strongest forgetting-mitigator except
  high-k CLoRA — and it is a **single untuned point** (L2 on the LoRA params, coeff 1e-5; "1e-4 too large";
  loss-term-vs-optimizer mechanism unspecified, code unreleased). It is the **same *kind* of magnitude/norm
  knob** as our weight decay, **not** a spectral/largest-SV penalty (the PI's spectral hypothesis is
  REFUTED), and **not identical** to our LoRA+wd (λ=1e-5 vs our wd 0.2–0.3). Our LoRA+wd LR×wd sweep is
  exactly the fair baseline they omitted.
- **Cross-literature corroboration also in:** MiLoRA Table 7/8 (ΔW-amplification LoRA 68.2 / PiSSA 55.8 /
  MiLoRA 44.9; CE-to-base PiSSA 6.07 > LoRA 3.24 > MiLoRA 2.54), LoRA-Null Table 4b (CorDA rank collapse
  r256 89%→73% retention = more capacity ⇒ more forgetting), CorDA++ Eqs 5-6 (norm-of-moved-directions
  bounds loss). The law is already visible across the 2025 literature; our contribution turns those
  point-observations into a LAW across 8 adapters × 7 LRs × 2 models.

---

## 7. EFFICIENCY + MEMORY — LoRA+wd is on the frontier at ZERO extra cost  **[DONE]**

- **LoRA+wd:** no init, wall-clock **identical to LoRA** (17,126 vs 17,138 s); weight decay is a free
  AdamW flag. Trainable params (r64) = 112,197,632 (1.638%, log-verified).
- **DoRA:** 2.13× wall-clock (r16 2.22×) for no retention/adapt benefit (see §8). A 2× training tax that
  buys nothing on our axes.
- **CLoRA:** pays at TRAIN time via the frozen P block (resident memory k×1,753,088 floats bf16):
  **0.42 / 0.84 / 1.67 (k512 default) / 3.34 / 6.7 GB** for k128…k2048 — at default k512 that is 8× the
  trainable LoRA weights; +9.5–14% train wall-clock, k-scaling.
- **Data-aware init tax** (one-time, then trains as vanilla LoRA): MiLoRA/PiSSA = 160 base-W SVDs (no
  forwards); SC-LoRA = 512 calib forwards + eigh; LoRA-Null/CorDA = 256 forwards + eigh/inv+SVD; CorDA++ =
  1280 forwards (N=5×256) + 5× per-layer inv+SVD + rank alloc (~5× CorDA, ~3.5e16 init FLOPs; ~22.5 GB
  transient covariance accumulator).
- Peak GPU mem not instrumented (no torch memory prints); the analytical resident comparison is the
  substitute. Evidence map: `tasks/ae2310183ffb6dd65.output`.

**Framing:** "LoRA+wd sits on the frontier at zero init, zero k-memory-tax, LoRA-identical train cost;
fancier methods pay strictly more for the same trainable budget."

---

## 8. METHOD-SPECIFIC FINDINGS

- **[DONE] DoRA = 2× train tax, no payoff.** ~2.0 vs ~4.3 steps/s at identical config; sits on the same
  retention-vs-F_Δ curve as LoRA and its F_Δ blows up at high LR (Llama-math lr1e-3: DoRA F_Δ 2.19 vs LoRA
  1.28 → retention 17.9 vs 19.5, adapt collapses GSM8K 31.9).
- **[DONE] PiSSA's catastrophic forgetting is REAL.** Largest F_Δ (2.21, ~1.7× LoRA), worst retention
  (BBH 7.23), worst adapt (GSM8K 49.7). Gate (2026-07-08, 270 BBH generations inspected): correct target
  in only 37/270 (13.7%); MetaMath contamination negligible (3/270); ~22% empty generations mean the
  number mildly *underestimates* retained ability (likelihood-MMLU parity 24.5 shows recognition survives
  while generation collapses). Principal-direction init = maximal perturbation = the expected high-F_Δ
  endpoint of the law.
- **[OPEN] SC-LoRA sits ~4pp BELOW the law — but the deviation is recipe-confounded.** Spline residual
  −4.15pp (p=0.006), the only significant deviator (per-method slope −26 [−33,−19] vs shared ≈−10). BUT
  our runs use β=0.5 + calib(nq_open)≠eval + r32 — **all off the paper's recommended recipe (β=0.8/0.9,
  eval-matched calib, r128)**, a reviewer trap. The `frc_sclora_b0p9_em_r128 @5e-5` control (fully
  paper-faithful) is queued to test whether −4pp survives their own recipe. A small **negative** geometry
  effect (SC-LoRA forgets MORE), disclosed by us — not a win for geometry.
- **[DONE] CLoRA's over-LoRA advantage does not reproduce in a controlled harness.** In our identical
  pipeline CLoRA retention ≈ LoRA at matched F_Δ, and CLoRA GSM8K (58.5–60.8) ≈ LoRA (60.2); the paper's
  ~+4pp edge shrinks to noise once LoRA is evaluated faithfully on a shared ruler. We anchor on
  *reproducing LoRA* and beating CLoRA's *published* number, so the claim does not rest on our
  (possibly-conservative) CLoRA reproduction. In-pipeline CLoRA k128 59.6 / MiLoRA 59.0 run ~4pp below
  their published values (recipe/harness gap) — kept as a separate row from published anchors.
- **[DONE] Plain LoRA, tuned, matches/beats CLoRA's best *published* number.** LoRA (wd=0) at a
  well-chosen LR reaches GSM8K ~65 ≳ CLoRA-k128's published 64.59 at smaller F_Δ — before adding weight
  decay. With weight decay, LoRA+wd best matched-c256 GSM8K = **67.3** (wd0.3/lr2e4, F_Δ 0.28), edging
  published CLoRA-k128 64.59 and MiLoRA 63.53 (cross-harness "edge," stated carefully). c512 gives 69.5
  but competitors were not run at c512 → c256 is the fair matched comparison.

---

## 9. MEASUREMENT HYGIENE — how you *evaluate* a MetaMath model changes GSM8K by ~20 points  **[DONE]**

The same adapter scores **46.55%** under lm-eval's default GSM8K (5-shot, "Question:/Answer:" template,
strict `#### x`) but **60–66%** under the faithful protocol (0-shot Alpaca template the model trained on,
last-number extraction) — a **~+19.5 pp** swing from the harness alone. Reporting the train/eval-template
match (and both numbers) is itself a measurement-hygiene contribution; it is also why our LoRA reproduces
CLoRA's published 60.58 (we get 60.2). Related: base BBH reproduces CLoRA's 34.91 only with **answer-only**
3-shot `bbh_fewshot` (33.1), NOT CoT (39.5); MMLU-Pro was dropped for math (format unparseable).

---

## 10. PER-BENCHMARK DEGRADATION  **[DONE]**

Slope of each retention benchmark per decade of F_Δ (Llama-2 CS, n=49): MMLU **−23.4** (r−0.93) dies
fastest; MMLU-Pro −15.2; ARC-c −14.9; BBH −14.3; **TruthfulQA −0.5 (r−0.10) essentially immune/flat.**
Knowledge-recall benchmarks forget; a truthfulness/style benchmark does not.

---

## 11. HONEST BOUNDARIES (non-negotiable; all belong in the paper)  **[OPEN where noted]**

1. **[OPEN] High-rank CLoRA beats LoRA+wd on CS.** Published CLoRA-k1024/k2048 CS ~83.7 (BBH 38.67 > base
   34.91 = positive *transfer*, not just retention) exceed LoRA+wd 81.6 on both axes; forcing LoRA+wd to
   that retention collapses its adaptation. So CLoRA's directional (null-space) constraint buys real
   adapt-efficiency at *high k* that pure magnitude control does not. The defensible claim is NARROWER than
   "geometry is useless": the magnitude LAW governs retention universally, and LoRA+wd matches fancy
   adapters on math and mid-regularization CS — but at high-k CS geometry adds value. **Faithful k-grid
   verdict is running (frepro4_main5).**
2. **[OPEN] SC-LoRA −4pp is provisional** — recipe-confounded, single seed (§8).
3. **[OPEN] Qwen-math anti-replicates** (core r=+0.67, ns) — the high-LR cells are unrun; Node-B runs the
   `qwswm_lorawd_wd0p3` math LR-sweep (incl. 5e-4/1e-3) FIRST to convert this into a positive 2nd-model +
   2nd-task replication. Present as in-progress, not buried.
4. **[OPEN] Single seed (s42) primary; CS eval is seed-unstable** → report the LR **safe-band**, not
   single peaks; 3-seed (43/44) headlines running.
5. **[OPEN] CorDA is withheld from every law/figure/table** pending the nq_open re-run + fair-calibration
   pass (wikitext-calib bug fixed; the calib↔eval mismatch is unresolved). Do NOT cite any CorDA number
   (incl. old "77.9/19.9" or "−3.0pp off-curve").
6. **[DONE] Ranks are NOT matched** across methods (LoRA/DoRA/LoRA-Null r16; MiLoRA/SC-LoRA r32; CLoRA
   k1024) — frame the LAW, not a method ranking. A matched-α LoRA-vs-MiLoRA decomposition (their published
   design confounds α=2r-LoRA vs α=r-MiLoRA) is analysis-only once the frc cells land.
7. **[DONE] MATH scorer ~3pp low / cutoff 256-vs-512 sensitivity** — disclosed; c2048 anchor queued.
8. **[DONE] LoRA-Null data-labeling bug** — the generator sets `method=run_name.split("_")[1]`, so
   `lrsw_lora_null_*` (a distinct adapter) is silently classified `lora`. The pooled law (49 pts) and the
   best-adapt LoRA point are unaffected; fix the legend / robustness-count / "6 methods" wording before
   camera-ready (analysis-level fix already in `analysis_a1_a4.py` / `make_figs_split_lora_null.py`).

---

## 12. OPEN / FUTURE / UNCERTAIN — items kept OUT of the supervisor artifact (do not lose)

- **[OPEN] Faithful CS reservoir (65 `frc_` cells) — the paper's spine — is landing now** (0 done at
  campaign start; being trained across the 2-node fleet). Every within-reproduction CS number is
  provisional until these land.
- **[OPEN] CorDA++ is now wired into `train_cs.py`** (29 refs; DEFAULT_N set for the campaign) — a change
  since the last manifest; the α=r @2e-5 paper-faithful anchor (`frc_cordapp_a1r`) is queued to remove the
  α=2r confound.
- **[OPEN] CorDA++'s "LoRA catastrophically forgets at 2e-5" threat** — answered by the injected
  `frc_lorawd wd0/0.2/0.3 @2e-5` cells (does magnitude control close their 2e-5 gap?).
- **[OPEN] Param-matched LoRA+wd control** — only LoRA has the wd knob; a param-matched control is still
  needed to be airtight.
- **[OPEN] Full CE-to-base batch over ~390 adapters** — slice numbers already suffice for the point;
  full batch deferred to idle GPUs (A-node only, reads local /scratch).
- **[OPEN] SC-LoRA init-erosion probe** (B drift off Q_r vs step) needs unsaved per-step checkpoints →
  post-submission; would convert the anomaly into a measurement-tool demo.
- **[DEFERRED] CLoRA harness-attribution rescore** — their k2048 exceeds base out-domain (transfer) in
  their lm-eval config; ours sits at/under the ceiling. A footnote suffices if time runs out; a both-config
  rescore would attribute the gap cleanly.
- **[DEFERRED] DoRA/PiSSA full LR sweeps** — single 3e-4 points suffice for the Table-2 mirror.
- **[DEAD] "principal-direction concentration" as a competing geometric axis** — see §3, do not revive.
- **[DEAD] UIOrthoLoRA as an A*-worthy CLoRA-beater** — it only tied CLoRA; survives only as
  `uio_inprocess.py` helpers imported by `eval_one_gpu.py`.

_Watch-items: MATH scorer/cutoff offset; the CS LoRA+wd wd-sweep verdict; the faithful high-k CLoRA
verdict; the SC-LoRA recommended-recipe control; the Qwen-math high-LR cells; 3-seed robustness._
