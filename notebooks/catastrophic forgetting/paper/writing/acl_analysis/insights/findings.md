# INSIGHT PROSPECTING — findings (2026-07-18)

Prospector pass over the frozen n=1035 pool (+ per-matrix SVD dumps, CE merge). All
numbers reproduce from scripts in this directory; the pool loader (`00_build_pool.py`)
is convention-identical to `analysis_final/ladder_2026-07-17.py` and hard-asserts §18.1
(n=1035, pooled r=-0.847, per-family n/r) before writing anything. Inference at cell
(seed-averaged) level wherever a claim is made (within-cell ICC≈0.78); run-level numbers
are labeled descriptive. Framing guardrail respected throughout: magnitude first,
geometry second; CLoRA's published numbers are faithful.

---

## RANKED INSIGHTS

### 1. A UNIVERSAL FRAGILITY ORDERING of retention benchmarks — Kendall W = 1.000
**Claim.** Normalize each benchmark's degradation slope by its base ceiling (fraction of
base capability lost per decade of F_Δ). The fragility ordering is then IDENTICAL in all
six families (2 models × 2 tasks, 6 recipes): **MMLU-Pro > BBH > MMLU > TruthfulQA**
(most→least fragile). Kendall W = 1.000 (χ²=18.0, p=4.4e-4, 6 families × 4 benchmarks;
contaminated ARC-c excluded; with ARC-c in, W=0.906, p=2.3e-4).
**Evidence.** `02_benchmark_fragility.py` → `benchmark_fragility.{csv,md}`,
`fig_benchmark_fragility.*`. Normalized slopes (per decade of F_Δ): MMLU-Pro −0.36…−0.73
of base; BBH −0.29…−0.64; MMLU −0.15…−0.49; TQ +0.13…−0.21. Cell level, 42–75 cells/family.
**Why it matters.** (a) Refines frozen §7 (raw pp/decade, Llama-CS n=49, "MMLU dies
fastest"): in fraction-of-base units MMLU-Pro is most fragile everywhere — and MMLU-Pro
is exactly the generative/format-following channel (Channel B of 05's mediation), so the
ordering is mechanistically coherent: format-following dies first, likelihood-style
knowledge recall second, calibration-style benchmarks last. (b) A universal ordering is a
falsifiable prediction for other stacks. Per-benchmark knees also differ within a family
(e.g. frc: BBH knee −0.18 vs MMLU-Pro +0.18 in log10 F_Δ).
**Caveat.** Part of the ordering could reflect distance-to-chance-floor (MMLU-Pro 10-way,
MMLU/ARC 4-way); we did not re-normalize by (base − chance) because BBH-AO and our TQ
metric lack clean chance levels. Say "fraction of base" and disclose.
**Confidence:** HIGH (ordering), MED (mechanism reading). **New vs known:** ordering,
normalization, and per-benchmark knees NEW; raw Llama-CS slopes known (§7).

### 2. TruthfulQA RISES with forgetting on Llama-2 — and it sits inside `retention_broad`
**Claim.** At full n, TruthfulQA is not "flat/immune" (frozen §7 wording, n=49): it rises
significantly with F_Δ in all four Llama families (cell-level r = +0.40…+0.79, slopes
+2.2…+5.0 pp/decade, p ≤ 8e-3) and falls on both Qwen families (r = −0.87/−0.90). Because
`retention_broad` = mean(BBH, MMLU-Pro, MMLU, ARC-c, **TQ**), including TQ attenuates the
measured broad forgetting slope by **24–30% on Llama** (12–14% on Qwen).
**Evidence.** `02_benchmark_fragility.py` (sign-check section), `06_metric_hygiene_effects.py`
→ `metric_hygiene.md`. Most-damaged runs (ret < 40% of base): TQ 44–49 on Llama (base
38.85, rising) vs 39–49 on Qwen (base 56.28, falling).
**Why it matters.** Metrology point: a benchmark that anti-correlates with capability sits
inside the broad retention aggregate, systematically flattening it. Quote broad-without-TQ
alongside, or add one disclosure sentence. The convergence of damaged models toward a
common ~40–50 band from both sides suggests TQ under damage measures regression toward
indifference (TQ's known inverse-scaling behavior) — that reading is SPECULATIVE; the
sign flip and the 24–30% attenuation are solid.
**Confidence:** HIGH (sign flip + attenuation), SPECULATIVE (indifference mechanism).
**New vs known:** NEW (supersedes §7's "immune/flat" at ~20× the n; model-dependent sign
flip never noted).

### 3. THREE KNOBS, ONE CURVE — the dose-response exhibit ("every knob is a magnitude knob")
**Claim.** Three mechanistically unrelated knobs — AdamW weight decay (wd 0→0.5), CLoRA's
null-space dimension (k 128→2048), LoRA rank (8→32), all inside the frc grid — each move
F_Δ monotonically, and their retention effects vanish once F_Δ is known:
- wd: partial r(logF_Δ, wd | logLR) = −0.762 (t=−6.4, cells n=33); retention residual
  beyond F_Δ: partial r = −0.25 (p=0.18, ns); ΔR² from adding wd to ret~logF_Δ = **0.006**
  (0.902 → 0.908).
- CLoRA k (lr3e-4, 5 cells × 3–5 seeds): Spearman ρ(log k, F_Δ) = −1.00, ρ(log k, ret) =
  +1.00 (F_Δ 0.615→0.347, ret 22.7→25.3 for k128→2048); residual beyond F_Δ ns (−0.22).
- rank (lr3e-4): F_Δ 0.52→0.75, ret 24.7→22.1 for r8→32; on-curve residual +1.7 pp.
All three knob-sweeps land on the single frc family curve (cells r=−0.951): mean on-curve
residuals +0.26 / +0.53 / +1.73 pp.
**Evidence.** `01_dose_response.py` → `dose_response_table.{csv,md}`, `fig_dose_response.*`
(panel A: wd→dose per LR; panel B: k and rank→dose; panel C: all doses on one curve).
**Bonus (new; frame carefully).** At fixed lr3e-4 in OUR harness, raising k to 2048
*costs adaptation* (cs_avg 76.8→69.4, peak at k256–512) while buying retention exactly
along the curve — at matched LR the k-knob behaves as a magnitude limiter with an
adaptation tax. This does NOT contradict CLoRA's published high-k wins (different
operating points; their numbers are faithful) — it localizes the k-knob's
mechanism-of-action on retention to dose reduction.
**Confidence:** HIGH. **New vs known:** pieces known (§6 wd effect, §17.4, CLoRA Table 4
externally); the unified three-knob mediation exhibit and the fixed-LR k adaptation tax
are NEW.

### 4. THE FREE-LUNCH REGION IS THE WHOLE LUNCH — 99–100% of peak adaptation lies below the knee
**Claim.** Using the frozen per-family knees (§18.2), the maximum adaptation achieved by
any healthy cell (adapt≥25) *below* the retention knee is 99.0–100% of the family-wide
peak in ALL SIX families: peak-below-knee 81.8/81.4/87.8/58.5/68.5/77.2 vs global
81.8/81.9/87.8/59.1/68.5/77.2 (lrsw/frc/qwsw/lrswm/frm/qwswm). No achievable adaptation
requires paying measurable retention.
**Evidence.** `03_freelunch_exchange.py` → `freelunch_table.csv`, `freelunch_exchange.md`,
`fig_freelunch.*`.
**Second half (practitioner point).** Reaching the free region is not automatic: in the
frc grid at the standard lr3e-4, plain LoRA has **zero** cells below the knee (lora 0/4,
sclora 1/7, clora 1/5), while LoRA+wd has 12/31 there and attains 81.4 adapt — the free
lunch exists, but without a magnitude knob (wd, k, or LR discipline) a standard recipe
never sits in it.
**Caveat.** Knee sensitivity: our own hinge refit agrees with §18.2 in 4/6 families but
drifts right for lrsw/frm (tail-anchored SSE); the conclusion holds under both knee
choices, but quote the §18.2 knees.
**Confidence:** HIGH (robust to knee choice). **New vs known:** knee known (§18.2);
saturation known qualitatively for lrsw (05 §5); the 6-family "peak-adapt-below-knee ≈
100%" quantification and the reachability corollary are NEW.

### 5. THE EXCHANGE-RATE TABLE — and a strictly dominated region beyond ~2× the knee
**Claim.** Binning healthy cells by F_Δ (quintiles): the marginal price of adaptation is
~0 below the knee, 0.1–0.5 retention-pp per adaptation-pp in a narrow mid band, and in
the top bin of every family the trade turns **negative-sum** — adaptation itself falls
while retention keeps collapsing (Δadapt −1.5…−15.6 pp, Δret −4.4…−20.3 pp between the
last two bins; 6/6 families). There is no rational operating point beyond roughly 2× the
knee: you pay retention and lose adaptation.
**Evidence.** `03_freelunch_exchange.py` → `exchange_rate.csv`, `freelunch_exchange.md`.
**Why it matters.** One table converts the curve into practitioner language and sharpens
05's saturation story into a three-regime description: free / priced / dominated.
**Confidence:** MED-HIGH (bin-dependent magnitudes; top-bin negative-sum direction is
unambiguous 6/6). **New vs known:** lrsw saturation table existed (05 §5); the price
framing, 6-family replication, and dominated-region universality are NEW.

### 6. WHERE THE UPDATE LANDS: q_proj energy share is a real but HETEROGENEOUS second-order axis
**Claim.** From the unmined per-matrix dumps (160 matrices/adapter): the fraction of
update energy (Σ fro²) in **q_proj** carries a retention signal beyond magnitude — pooled
partial r(share_q, ret | logF_Δ + family) = −0.28 at cell level (t=−5.3, n=343 cells,
p=1.9e-7), surviving method dummies (−0.285, run level) and present within lorawd alone
(partial −0.29). Effect size: −1.7 pp across the share_q IQR (vs −12 pp/decade for
magnitude) — bounded second-order, exactly as the thesis requires. BUT family-split shows
it lives in the CS LR-sweeps (lrsw −0.25, qwsw −0.45) and is absent in grid/math arms
(frc +0.01, lrswm −0.04, frm +0.09, qwswm −0.09); seed-stable 3/4 (s45, small frc-heavy
slice, null).
**Evidence.** `04_permatrix_layers.py`, `07_shareq_stability.py` →
`permatrix_features.csv`, `permatrix_layers.md`, `shareq_stability.txt`, `fig_permatrix.*`.
**Read.** "Updates that load on query projections drift retention more per unit dose" is
plausible (q rotates attention), but given the heterogeneity this is an appendix
observation with the split disclosed — fingerprint-grade, not a headline axis.
**Confidence:** MED (pooled effect robust; cross-arm generality NOT established).
**New vs known:** NEW (per-matrix granularity unmined; adapter-level e_top/stable_rank
axes known, §18.4/§19.2).

### 7. METHOD FINGERPRINTS AT THE PROJECTION LEVEL — CLoRA is an MLP-mover
**Claim.** Update-energy composition is method-determined (η² = 0.08–0.19 per feature)
and interpretable: **CLoRA puts 82% of its energy in MLP** (up 50.5% + down 31.6%; only
18% in q/k/v — the covariance null-space constraint suppresses attention drift and
reroutes the update to MLP), PiSSA is the most attention-heavy and most spread (38% q+k,
eff_n_mat 136), DoRA is q/k-heavy (31%), and every method's depth profile is U-shaped
with a layer-0 spike (last-8 layers carry 28–31% of energy). Depth centroid carries NO
retention signal beyond F_Δ (partial r +0.03, ns) — a useful "we checked layers" null
that supports magnitude-first.
**Evidence.** `04_permatrix_layers.py` (fingerprint table, η², depth profiles),
`fig_permatrix.*`.
**Confidence:** HIGH (descriptive). **New vs known:** NEW at this granularity
(fig_geometry_4panel panel D only showed SC-LoRA's ein_top q/k profile).

### 8. THE TRAINING TASK ITSELF STARTS TO FORGET — adaptation-side collapse ordering
**Claim.** Within the 8-dataset CS adaptation suite, over-dosing degrades datasets in a
consistent order: **hellaswag and ARC collapse first** (onset at/before the retention
knee), **social_i_qa collapses last** (rank 8/8 in all three CS families), boolq
second-last; Kendall W = 0.738 across lrsw/frc/qwsw. Above the knee hellaswag falls at
−23…−47 pp/decade vs social_i_qa/boolq −11…−24. The datasets dying first are those
closest to pretraining knowledge (likelihood/commonsense completion); the survivor is the
most "new-skill-like" (social norms).
**Evidence.** `05_adaptation_side.py` → `adaptation_side.{csv,md}`, `fig_adaptation_side.*`.
**Why it matters.** (a) `cs_avg` is a mixture: part of the adaptation suite behaves like
a retention benchmark, so the adaptation-side knee partially IS forgetting leaking into
the training task. (b) Practitioner: over-training first destroys the components of your
task that overlap general capability. (c) Explains differential "wins" of high-F_Δ runs
(they hold social_i_qa while losing hellaswag).
**Confidence:** MED-HIGH (W=0.738; social_i_qa-last is 3/3 exact).
**New vs known:** NEW (the per_dataset field was never analyzed).

---

## DEAD ENDS TRIED (recorded so nobody re-chases)

- **Depth centroid / effective-matrix-count as retention axes beyond F_Δ:** depth_centroid
  partial r = +0.03 (ns, cells); eff_n_mat −0.16 weak and non-robust under method dummies
  (t=−1.6 run level). Layer placement is a fingerprint, not an axis. (`04_permatrix_layers.py`)
- **share_v / share_up as axes:** ns at cell level (+0.05 / −0.06).
- **Own hinge-refit knees as free-lunch boundaries:** SSE-optimal knees on cells are
  tail-anchored for lrsw (+0.30 vs frozen −0.02) and frm (+0.43 vs −0.50) — quantile-grid
  hinge fitting on skewed cell distributions is unstable; use §18.2 knees. (`03_...py`)
- **wd partial conditioned on (F_Δ AND LR) simultaneously:** unstable (−0.74) because
  within fixed LR, wd and F_Δ are nearly collinear — over-controlled, uninterpretable.
  The honest mediation statement is ΔR²=0.006 / single-conditioned ns partial.
- **TQ "regression to indifference" as a quantitative law:** damaged-run TQ does not
  converge to one value (39–49 across families); sign flip solid, mechanism qualitative
  only. Keep as labeled speculation.

## CANDIDATE PAPER EXHIBITS

- **MAIN:** `fig_dose_response` (three knobs → one curve; insight 3) — strongest new
  visual argument for "control the dose, not the knob"; complements E1.
- **MAIN (small table) or APPENDIX:** universal fragility ordering, 6×4 normalized-slope
  table + W=1.000 (insight 1).
- **APPENDIX:** free-lunch table (insight 4) + exchange-rate table (insight 5), with
  `fig_freelunch` as the 6-panel visual.
- **APPENDIX:** permatrix fingerprint table + depth profiles (insight 7); share_q as a
  disclosed heterogeneous second-order observation (insight 6).
- **DISCUSSION-only:** TQ sign flip + broad-attenuation disclosure, quote broad-no-TQ
  slopes (insight 2); "even the training task forgets" paragraph (insight 8).

## FILES

- `00_build_pool.py` → `pool.csv` (n=1035, §18.1 preflight-asserted), `pool_all.csv`
- `01_dose_response.py` → `dose_response_table.csv/.md`, `fig_dose_response.png/.pdf`
- `02_benchmark_fragility.py` → `benchmark_fragility.csv/.md`, `fig_benchmark_fragility.png/.pdf`
- `03_freelunch_exchange.py` → `freelunch_table.csv`, `exchange_rate.csv`,
  `freelunch_exchange.md`, `fig_freelunch.png/.pdf`
- `04_permatrix_layers.py` → `permatrix_features.csv`, `permatrix_layers.md`, `fig_permatrix.png/.pdf`
- `05_adaptation_side.py` → `adaptation_side.csv/.md`, `fig_adaptation_side.png/.pdf`
- `06_metric_hygiene_effects.py` → `metric_hygiene.md`
- `07_shareq_stability.py` → `shareq_stability.txt`
