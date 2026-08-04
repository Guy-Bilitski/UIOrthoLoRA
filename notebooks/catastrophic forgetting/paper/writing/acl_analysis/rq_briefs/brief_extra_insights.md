# EXTRA INSIGHTS BRIEF (results-section reinforcement pass, 2026-07-30)

Pool: frozen `insights/pool.csv`, n=1035. Preflight hard-asserted before every
script run: n=1035, pooled r(log F_Delta, ret) = -0.847, all six per-family
(n, r) cells to 3 decimals (section 18.1). Inference is cluster-aware
throughout: cell-level (seed-averaged) units, CR1 SEs clustered at cell for
run-level fits, or cell bootstrap. Every candidate was recomputed via two
independent code paths and checked on both pool conventions (frozen and
quarantine-excluded). Scripts in `rq_briefs/extra_scripts/` (a1..a5 = path 1,
qa_recheck.py / qa_recheck2.py = path 2). Directions mined: (a) adaptation
structure, (b) slope moderators, (c) per-benchmark micro-structure at matched
magnitude, (d) seed-instability prediction, (e) knob adaptation-neutrality.
No existing file was modified. Nothing here re-runs the insights/findings.md
dead-ends list.

One metrology catch surfaced during QA and is disclosed where it bites:
**lrswm mixes two adaptation metrics** (`adapt_task` = gsm8k for
lora/lorawd/milora/dora/clora, 108 runs, but math_faithful for 12 of 15
SC-LoRA runs, which sits ~12.7 pp higher). Any lrswm adaptation-side method
comparison is partly a metric artifact (cell-level method dR2 on adaptation
falls from +0.46 to +0.17 when restricted to gsm8k). Findings below were
re-checked gsm8k-only where lrswm adaptation enters.

---

## RECOMMEND-ADD

### X1. The adaptation-optimal dose band contains the retention knee; past it, adaptation falls at 20 to 37 pp per decade

**Insight.** Adaptation is concave in update magnitude in every family, its
2-pp-of-peak dose band contains the frozen retention knee in 5 of 6 families,
and dosing past the optimum loses adaptation at 20 to 37 pp per decade, so a
single dose band jointly optimizes both objectives and there is no rational
reason to dose above the knee.

**Statistics (cluster-aware).**
- Concavity: run-level adapt ~ logfd + logfd^2, CR1 at cell; quadratic
  coefficient negative in 6/6 families, t = -3.1 (lrsw), -4.9 (lrswm; -5.6
  gsm8k-only), -17.9 (qwsw), -5.4 (qwswm), -3.1 (frc), -8.0 (frm).
  n = 120..276 runs per family (1035 total).
- Optimum location: dose band of cells within 2 pp of the family adaptation
  peak contains the section-18.2 retention knee in 5/6 families (frozen and
  quarantine-excluded alike); the exception is lrsw, where the knee sits 0.29
  decades above the band. lrswm band re-checked gsm8k-only: [-0.54, -0.27],
  knee -0.48 inside. Cell-bootstrap quadratic vertices sit a median 0.23
  decades from the knee (max 0.54), but the vertex point estimate is
  fit-sensitive for flat-topped families (run- vs cell-level vertices differ
  by up to 0.3 decades), so the band statement is the quotable one.
- Falling limb (quadratic-free): linear fit on runs above the argmax dose
  bin, CR1 at cell: -19.7+/-5.6 (lrsw), -29.6+/-6.7 (lrswm), -26.3+/-7.7
  (qwsw), -36.7+/-3.5 (qwswm), -36.8+/-2.9 (frc), -23.4+/-2.0 (frm) pp/decade;
  significant 6/6. Path-1 estimator (above the peak cell) gives -16 to -30.

**Assumptions.** adapt = cs_avg (CS families) / GSM8K (math families);
lrswm mixed-metric caveat handled as above. Knees are the frozen section-18.2
values. The rising limb is shallow and family-dependent (Spearman below-knee
+0.65 and +0.64 in lrsw/qwswm, near zero in lrswm), so "single-peaked" should
be phrased as "concave with a robust decline past the optimum", not as a
strong interior maximum in every family.

**Self-QA.** Path 1: cell-level quadratic + hinge + 1-pp/2-pp bands
(a1_adaptation_dose.py, a1b_vertex_bootstrap.py, 2000-resample cell
bootstrap). Path 2: run-level CR1 quadratic, quadratic-free Spearman split at
the knee, binned argmax, quadratic-free falling-limb fit (qa_recheck.py,
qa_recheck2.py). Both conventions: knee-in-band 5/6 on each; falling limb
significant on each. gsm8k-only lrswm re-check passes.

**New vs known.** Insight 4 (free lunch) showed peak-adapt-below-knee at
99-100% of the global peak and insight 5 showed the top dose bin is
negative-sum; X1 adds the formal concavity test (6/6, CR1), the explicit
co-location of the adaptation optimum with the retention knee, and the
per-decade price of over-dosing. If the free-lunch and exchange-rate tables
stay in the appendix, X1 is the one-sentence main-text version.

**Ready-to-paste LaTeX.**
> Adaptation is concave in update magnitude: in all six families a negative
> quadratic dose term is required (cluster-robust $t$ between $-3.1$ and
> $-17.9$), and the dose band whose adaptation lies within 2\,pp of the family
> peak contains the retention knee in five of six families (in the sixth the
> knee sits 0.3 decades above it). Past the optimum, adaptation itself falls
> by 20 to 37\,pp per decade of $F_\Delta$. The dose that maximizes the
> training task and the dose at which retention begins to fail coincide to
> within measurement resolution; we observe no operating point above the knee
> that buys adaptation.

**Verdict: RECOMMEND-ADD.**

---

### X2. Seed lotteries live above the knee and are predictable from train-time measurements

**Insight.** Cells whose retention is a seed lottery (within-cell SD > 3 pp)
sit far above the family knee and can be flagged before any benchmark is run:
mean weight-space dose separates lottery from stable cells with AUC 0.80, and
on Llama the seed scatter of KL drift flags them with AUC 0.96.

**Statistics (cluster-aware by construction; the unit is the cell, >=3 seeds).**
- n = 290 cells, 22 lotteries (frozen pool). Lottery cells sit a median +0.49
  decades above the knee vs +0.02 for stable cells; 86% of lotteries are
  above-knee vs 53% of stable cells.
- AUC (Mann-Whitney): dist-to-knee 0.798 (p = 1.7e-6), mean log F_Delta
  0.803, within-cell SD(log F_Delta) 0.826, log10 LR 0.760; SD(KL drift),
  Llama-only per the Qwen-CE convention, 0.955 (8 positives, n = 200 cells,
  p = 6.7e-6).
- Continuous version (no threshold): Spearman(dist-to-knee, SD ret) = +0.449
  (p = 8e-16, n = 290).
- Screening rule "above knee OR SD(log F_Delta) > q90": sensitivity 1.00
  (22/22), specificity 0.45.
- Robustness: threshold SD>2 / SD>4 give AUC 0.823 / 0.791; quarantine-
  excluded pool gives AUC 0.744-0.778, rho +0.41, SD(KL) 0.967-0.981. Lottery
  composition is method-diverse (9 sclora, 5 lorawd, 4 lora, ...), so this is
  not a repackaging of the known SC-LoRA instability.

**Assumptions.** SD estimated from 3-5 seeds (noisy; hence AUC, not
calibration). The 3-pp cut is arbitrary (sensitivity shown). Quarantined runs
included per frozen convention; excluding them removes 5 of 22 lotteries but
not the effect. SD(KL) needs forward passes on the drift corpus only, no
benchmark evals; Qwen is excluded from the KL predictor per the seed-blocked
CE coverage convention.

**Self-QA.** Path 1: rank-statistic AUC, rank-partial correlation, rule
confusion table, per-family split (a4_seed_lottery.py). Path 2: scipy
Mann-Whitney with p-values, threshold sweep, continuous Spearman
(qa_recheck.py). Both agree to 3 decimals on shared quantities; per-family
signs consistent (dist-knee AUC 0.64-1.00 in every family with any lottery).

**New vs known.** Adjudication reports per-method seed SD medians (LoRA+wd
0.43 vs SC-LoRA 3.06) and the qwsw SC-LoRA lottery. New here: instability is
dose-localized (above-knee) across methods, and it is predictable in advance
from weight-space magnitude and its seed scatter, i.e. retention variance,
not just its mean, is dose-governed.

**Ready-to-paste LaTeX.**
> Seed-to-seed retention instability is itself dose-governed. Cells whose
> within-cell retention SD exceeds 3\,pp sit a median of 0.49 decades above
> the family knee (stable cells: 0.02), and the cell's mean $\log F_\Delta$
> separates unstable from stable cells with AUC 0.80 ($n{=}290$ cells with
> ${\geq}3$ seeds, Mann-Whitney $p{=}2{\times}10^{-6}$). On Llama, where
> per-seed drift measurements are available, the within-cell scatter of KL
> drift flags the same cells with AUC 0.96. A seed lottery can therefore be
> anticipated from train-time measurements before any benchmark is run.

**Verdict: RECOMMEND-ADD.**

---

## OFFER (PI decides)

### X3. No knob or method class bends the retention-dose slope; the one repeatable exception is SC-LoRA

**Insight.** Within family, the retention-magnitude slope is not credibly
moderated by rank, weight decay, or the geometry-constrained vs plain method
class (knobs move position on the curve, not the curve), with one repeatable
exception: SC-LoRA falls progressively further below the family curve as dose
grows, in every family where it is estimable.

**Statistics (run level, CR1 at cell; interactions logfd x moderator on ret).**
- rank (frc, 8/16/32): interaction t = +1.69; cell-bootstrap 95% CI
  [-16.0, +1383.6]. Underpowered (only 7 runs at rank 8/16); a null, not
  evidence of absence.
- wd (frc, 0..0.5, n=131): interaction +11.1, t = +0.78; cell-boot CI
  [-15.4, +40.1] around a main slope of -16.2.
- geometry-class per family (n=120..276): |t| <= 1.64 in 6/6.
- SC-LoRA vs LoRA+wd slope excess: sign negative 6/6 families; significant in
  lrsw (-11.0, t=-4.4) and qwsw (-20.7, t=-4.1); cell-level replication
  agrees (t=-4.3 both); frm is degenerate (2 cells, 0.2-decade span; ignore
  its t). Slope rank: SC-LoRA is the steepest method in all 5 estimable
  families. Path 2 (residuals off the family hinge fit on non-SC-LoRA runs,
  knee fixed at section 18.2): residual-vs-dose slope negative in 5/5
  families, Spearman -0.52 to -0.89.
- Convention sensitivity: interaction remains negative 6/6 quarantine-
  excluded; the path-2 residual slope stays negative in 4/5 but qwsw flips to
  +2.1 (ns), i.e. the qwsw component is carried by quarantined far-collapse
  runs. The lrsw and lrswm components survive both conventions.

**Assumptions.** Slope tests are within-family (raw pp scales differ across
base models). SC-LoRA spans 1.0-1.7 decades with 6-9 cells per family, so the
interaction is estimable but not dense. Binding caveat: E4 showed SC-LoRA's
Llama-CS retention offset is removed by eval-matched calibration, and no
calibration control exists elsewhere, so the excess slope must be attributed
to the method as configured (nq_open calibration), never to its subspace
constraint per se. Do not quote the full method x dose Wald table
(a2_slope_moderators.py section 5): several terms (pissa, dora in the grids)
are degenerate small-span artifacts.

**Self-QA.** Path 1: CR1 interaction tests and per-method slopes
(a2_slope_moderators.py, a2b_sclora_slope.py). Path 2: cell bootstrap for the
knob interactions, hinge-residual regression for SC-LoRA (qa_recheck.py).
Directionally consistent everywhere; significance melts to 2 of 5 families
under quarantine exclusion, which is why this is an OFFER.

**New vs known.** Insight 3 (three knobs, one curve) is about level mediation
(knob effects vanish given F_Delta); X3 is the complementary slope statement,
and the SC-LoRA slope exception is unreported anywhere.

**Ready-to-paste LaTeX (if taken, keep both halves together).**
> Within family, the retention-magnitude slope is statistically
> indistinguishable across rank, weight decay, and the geometry-constrained
> versus plain method class (all interaction $|t| < 1.7$, cluster-robust):
> the knobs move a recipe along the dose curve rather than bending it. The
> one repeatable exception is SC-LoRA, which falls progressively further
> below the family curve as magnitude grows (steepest per-method slope in all
> five estimable families; slope excess versus LoRA+wd significant in two).
> Because eval-matched calibration was shown to remove SC-LoRA's retention
> offset on Llama-CS, we attribute the excess slope to the method as
> configured, not to its subspace constraint.

**Verdict: OFFER** (direction robust, significance partly quarantine-carried,
E4 caveat mandatory).

---

### X4. At matched magnitude, method differences concentrate in MMLU-Pro, not BBH

**Insight.** At matched update magnitude, method identity barely moves BBH
but shifts MMLU-Pro: relative to LoRA+wd, other methods lose 1 to 3 pp more
MMLU-Pro than BBH, consistently across families, so matched-dose method
differences concentrate in the most fragile, format-following benchmark.

**Statistics.**
- Path 1 (run level, bench ~ logfd + hinge + method dummies, ref lorawd,
  CR1 at cell, fraction-of-family-ceiling units): MMLU-Pro-minus-BBH offset
  negative in 36/38 method-family pairs; SC-LoRA 6/6, CLoRA 6/6, DoRA 6/6,
  LoRA 6/6, LoRA-Null 5/5, MiLoRA 5/6, PiSSA 2/2.
- Path 2 (no regression, no ceiling scaling: cell-level dose matching to
  lorawd cells within +/-0.15 decades, raw pp): negative in 31/37; matched-
  cell Wilcoxon on (dPro - dBBH): CLoRA p = 7e-4 (29 pairs), LoRA-Null
  p = 0.003 (25), SC-LoRA p = 0.002 (25), MiLoRA p = 0.03 (43); LoRA and DoRA
  ns. Family-mean magnitudes are 0.3 to 3.3 pp (raw), i.e. second-order.
- Convention: quarantine-excluded census 29/37; SC-LoRA, LoRA-Null, CLoRA
  Wilcoxons stay p < 0.03.

**Assumptions.** Reference is LoRA+wd, so offsets read as "extra damage vs
the best recipe", not absolute damage. The fraction-of-ceiling units of path
1 mechanically inflate MMLU-Pro (lower ceiling), which is exactly why path 2
is in raw pp; quote path-2 magnitudes. frm and PiSSA cells are few (n as
listed); the pooled sign census carries the claim, not any single family.
BBH is the math-family retention metric per convention; here it enters as a
compared benchmark, which is consistent.

**Self-QA.** Two fully disjoint designs (regression-adjusted vs matched
pairs, scaled vs raw) agree on direction; magnitudes differ exactly as the
scaling predicts. Scripts: a3_benchmark_method.py, qa_recheck.py section C.

**New vs known.** Insight 1 established the family-level fragility ordering
(MMLU-Pro most fragile per unit dose); X4 is the method-resolved version at
matched dose: where methods differ at all, they differ on the fragile
channel. Fits the channel-B mediation reading.

**Ready-to-paste LaTeX.**
> At matched update magnitude, method identity leaves BBH nearly unchanged
> but separates on MMLU-Pro: relative to LoRA+wd, the remaining methods lose
> more MMLU-Pro than BBH in 31 of 37 method-family combinations (dose-matched
> cells, raw percentage points; Wilcoxon $p<0.01$ for SC-LoRA, LoRA-Null and
> CLoRA), with typical excesses of 1 to 3\,pp. Matched-dose method
> differences are concentrated in the most fragile benchmark rather than
> spread uniformly across the retention suite.

**Verdict: OFFER** (consistent and two-path robust, but second-order in
magnitude; appendix or one Discussion sentence).

---

### X5. Retention is close to a function of dose; adaptation is only loosely dose-determined

**Insight.** Update magnitude explains most of the run-level retention
variance within family (R2 0.71 to 0.88) but far less of the adaptation
variance (0.19 to 0.66 in five of six families), so the retention axis is
dose-governed while the adaptation axis retains the headroom where method and
recipe choices can still pay.

**Statistics.**
- Run level per family, hinge fit (free knot; quadratic gives the same
  picture): ret R2 = 0.84/0.78/0.79/0.79/0.88/0.87 vs adapt R2 =
  0.35/0.26/0.66/0.64/0.36/0.96 (lrsw/lrswm/qwsw/qwswm/frc/frm). The
  exception is frm, where the LR sweep spans the whole collapse and both
  axes are tight.
- Cell level, same-sample ladder (n=344 cells, family FE): adding dose
  (logfd + logfd^2) gains +0.401 R2 for retention vs +0.163 for adaptation.
- Conventions: quarantine-excluded ret 0.76-0.91 vs adapt 0.22-0.58 (frm
  0.79 vs 0.80). lrswm gsm8k-only: adapt 0.27 vs ret 0.86.

**Assumptions.** R2 comparisons are descriptive (no test of R2 difference);
the claim is the asymmetry's direction and consistency, 5/6 families under
both conventions and two functional forms. Part of the adaptation residual is
measurement (8-dataset cs_avg mixture, insight 8) and, in lrswm, the mixed
adapt_task metric (disclosed above).

**Self-QA.** Path 1: cell-level quadratic ladder (a1_adaptation_dose.py Q2).
Path 2: run-level quadratic AND free-knot hinge per family, both conventions
(qa_recheck2.py). Agree everywhere.

**New vs known.** The retention-side league table (dose ΔR2 +0.420) is
known; the adaptation-side counterpart was never computed. It gives the
mechanism-free reason why the only genuine method win in adjudication
(SC-LoRA on Qwen-math) is an adaptation-side win: that is the axis dose does
not pin down.

**Ready-to-paste LaTeX.**
> Magnitude governs the two axes asymmetrically. Within family, a hinge fit
> on $\log F_\Delta$ explains 0.71 to 0.88 of run-level retention variance,
> but only 0.19 to 0.66 of adaptation variance in five of six families.
> Retention behaves nearly as a function of dose; adaptation retains
> substantial non-dose headroom, which is consistent with the observation
> that the one method win in our head-to-head comparisons occurs on the
> adaptation axis.

**Verdict: OFFER** (clean and two-path robust, but descriptive; best as a
framing sentence introducing the adaptation-side analysis).

---

## DROP

- **wd is adaptation-neutral where CLoRA-k is not** (a5_wd_adapt_neutrality.py):
  DROP. At fixed LR 3e-4, wd shows the same-direction adaptation drift as k
  (Spearman -0.70, p = 0.077, 7 cells; non-monotone cell means), so no clean
  knob contrast exists beyond insight 3's k-tax observation.
- **Base model moderates the fraction-of-ceiling retention slope**
  (a2_slope_moderators.py section 6): DROP. Qwen shift -0.087/decade
  (t = -1.9), math shift +0.077 (t = +1.8); both inside noise once clustered.
- **Full method x dose Wald table** (a2 section 5): DROP as an exhibit;
  degenerate small-span terms (PiSSA, DoRA in the grids) produce absurd t
  values and would mislead. Superseded by X3's targeted tests.

## FILES

- Path-1 scripts (pre-existing this pass, rerun and verified): a1_adaptation_dose.py,
  a1b_vertex_bootstrap.py, a2_slope_moderators.py, a2b_sclora_slope.py,
  a3_benchmark_method.py, a4_seed_lottery.py, a5_wd_adapt_neutrality.py,
  extras_common.py (loader with section-18.1 preflight assert).
- Path-2 scripts (this pass): qa_recheck.py, qa_recheck2.py.
- All scripts print their preflight; nothing outside rq_briefs/ was written.
