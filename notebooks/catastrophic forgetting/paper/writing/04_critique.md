# 04 — Adversarial Pre-Draft Critique

*Written 2026-07-02 in the voice of a hostile ICLR/NeurIPS reviewer, BEFORE the paper is drafted.
Grounded in a direct re-read of `01_project_narrative.md`, `02_figures_tables_explained.md`,
`03_gaps_and_roadmap.md`, `CONCLUSIONS_AND_IDEAS.md`, `data/key_numbers.md`, the tables in
`tables/`, and a fresh recomputation from `data/campaign_summary.jsonl`
(`/home/guy/UIOrthoLoRA/.venv/bin/python`). Every finding below is either verified against the
registry or flagged as a claim I could not verify. Tags: [BLOCKER] = paper is not submittable /
not honest until fixed; [IMPORTANT] = a competent reviewer will attack it and likely win;
[NICE] = strengthens but not fatal.*

The good news first, so the rest is calibrated: the *core empirical object* — retention falls
monotonically with `‖ΔW‖_F`, tightly, and the fit is method-agnostic for most adapters — is real
and I reproduced it from the raw registry (pooled r = -0.858, n=49; on-curve r = -0.915, n=42).
The problem is not the phenomenon. The problem is that **three of the paper's four headline claims
are currently either unsupported by the data present, contradicted by the data present, or
computed on a silently different adapter set than the prose advertises.** Below, in priority order.

---

## [BLOCKER] B1 — "The law across 8 adapters" is actually across 6, and the 8th (CorDA) was DROPPED because it did not fit.

This is the single most damaging honesty gap, and a sharp reviewer will find it in ten minutes.

- The thesis, abstract outline, and every summary say the law holds **"across eight adapters."**
- The hero figure (`fig0_hero`), the ANCOVA (`fig2`), the pooled r=-0.86, and the main table are
  **all computed with CorDA entirely excluded** (`02_figures...` line 16; `key_numbers.md` §5, §8;
  `CONCLUSIONS` §4 CorDA bullet). Verified: dropping CorDA is what produces n=49.
- LoRA-Null is **not a separate series either** — it is silently pooled into "LoRA" by a generator
  bug (`parts[1]=="lora"`), so the plots show **6 visible series, not 8** (`CONCLUSIONS` §6.10).
- So the honest count is: the law is demonstrated on **6 adapters** (LoRA, LoRA+wd, DoRA, MiLoRA,
  SC-LoRA, CLoRA), of which one (SC-LoRA) is a significant deviator and one (LoRA-Null) is
  mislabeled. **CorDA — a flagship data-aware method and arguably the most important adapter to
  test the "geometry is inert" claim against — is absent from every headline exhibit.**

Why this is a BLOCKER, not a caveat: the paper's whole selling point is *method-independence*. An
excluded method is not a footnote; it is the exact case that could falsify the thesis. Worse, when
I include the current CorDA points, the CorDA rows I found in the registry sit systematically
**below** the curve (e.g. `lrsw_corda_r16_lr1e4` fdelta 0.42 → ret 19.9, vs LoRA/MiLoRA ~24–25 at
comparable fdelta), i.e. CorDA looks like a *second* off-curve deviator alongside SC-LoRA — which
directly undercuts "geometry adds nothing." The exclusion is defensible ONLY as a temporary,
loudly-flagged state ("CorDA re-running on nq_open"); it is NOT defensible in a submitted paper.

Required fix (either):
1. Finish the nq_open CorDA re-run + the eval-matched calibration (C2) and put CorDA **back into
   fig0/fig2/the table** as a full 8th series, OR
2. If it stays off-curve after fair calibration, **report it as a second off-curve method and
   retreat the "geometry is inert" claim** to "geometry is inert for calibration-free inits;
   data-aware inits carry a residual penalty we trace to calibration↔eval mismatch." Do not ship
   "8 adapters, one law" while 2 of 8 are off the curve and 1 is excluded.

Either way: **the word "eight" cannot appear next to "one curve" until CorDA is either on it or
openly declared off it.** Right now the prose says eight and the figures show six.

---

## [BLOCKER] B2 — The registry still contains the CorDA residual-save explosion and duplicate rows; the "faithful port / gated by 0-step check" claim is not clean.

Fresh scan of `lrsw_corda_*`: there are **14 rows for 7 LR cells — every CorDA LR is duplicated**,
and one duplicate is the catastrophic `lrsw_corda_r16_lr1e3` with **fdelta = 515.77, ret = 0.0,
cs = 0.0** — the exact residual-save-explosion signature the narrative (§d bug 9) claims was fixed
and gated. The data file the paper is drawn from therefore still contains poisoned CorDA rows and
un-deduplicated cells.

- This is a **data-hygiene BLOCKER**: any pooled/ANCOVA statistic is only trustworthy if the input
  is deduplicated and the exploded rows are purged. The narrative asserts a 0-step ΔW→0 self-check
  passed, but the artifact of the failure mode is sitting in the shipped `campaign_summary.jsonl`.
- A reviewer who is given (or reconstructs) the data will see fdelta=515 next to fdelta=0.42 for
  the "same" run and lose all trust in the pipeline.

Required: purge exploded/duplicate rows, document the dedup rule, and state which CorDA calibration
(wikitext vs nq_open) each surviving row came from. Until then, no CorDA number is quotable and the
"port is faithful" methods-section shield is undermined by its own data file.

---

## [BLOCKER] B3 — Claim 3 ("their wins are an LR artifact") has NO figure and is currently an assertion.

The roadmap itself flags this (C1, "BUILD THIS EXPLICITLY"), and it is correct to. The
"fixed-single-LR-point vs full-LR-sweep-collapse" gotcha exhibit **does not exist** — `fig4` shows
LR optima differ and `fig7` shows the R² contrast, but neither shows the actual rhetorical move:
"here is method X's paper-reported single-LR win; here is that same win dissolving onto the shared
`‖ΔW‖` curve when you sweep." Claim 3 is 1 of the 4 spine claims and the entire "wake-up call"
framing (Claim 4) rests on it.

Additionally — and this is the deeper problem — **the paper never actually reproduces any published
adapter's reported win at its published LR.** The "artifact" narrative requires showing that at the
LR the original papers used, the fancy adapter *does* look better, and that this advantage is what
evaporates. Right now the paper shows all methods roughly tie at their own best LR (see B5); it
never establishes the "win" that is supposedly an artifact. **You cannot debunk a win you never
reproduced.** Without a "here is the win → here is it dissolving" pair, Claim 3 is unfalsifiable
hand-waving and a reviewer will say so.

Required: (a) build the gotcha figure; (b) for at least CorDA and SC-LoRA, reproduce (or cite +
reproduce) the single-LR configuration under which the method beats LoRA, then show the sweep
collapses it. If you cannot reproduce any fancy-adapter win at any LR, Claim 3 must be **deleted or
demoted to a hypothesis**, and the paper becomes "magnitude governs forgetting" (Claims 1–2) only.

---

## [BLOCKER] B4 — The calibration↔eval mismatch makes every "data-aware inits forget more" statement (SC-LoRA -4.15pp, CorDA off-curve) currently uninterpretable.

The docs already know this (C2, §6.3) and to their credit call it the biggest kill-shot. I am
elevating it to BLOCKER-for-a-specific-claim: the **one statistically significant result in the
entire ANCOVA** (SC-LoRA -4.15pp, p=0.006) is confounded by the fact that CorDA/SC-LoRA/LoRA-Null
calibrate on nq_open (factoid QA) while retention is measured on academic/reasoning benchmarks.
The single "interesting" deviation the paper has is exactly the one most likely to be a
calibration-set artifact.

Consequence for framing: the paper cannot simultaneously claim (i) "geometry is inert, all on one
curve" AND (ii) "SC-LoRA is a real off-curve deviator" — those are in tension — while ALSO
admitting (iii) the deviation may be a calibration artifact. Pick a lane after the C2 experiment.
Until C2 runs, **all "forget more than their budget" language must be struck**, including the fig2
ringing of SC-LoRA as a finding. Present it as "one method's residual is nonzero; we cannot yet say
whether that is geometry or calibration" — nothing stronger.

---

## [BLOCKER] B5 — "LoRA+wd wins/surpasses the Pareto frontier" is within single-seed noise, and the paper's own priors say single-seed rankings are unreliable.

Verified best-adapt operating points (CS, s42): LoRA+wd **81.6/25.6** vs SC-LoRA 80.1/22.5,
MiLoRA 79.9/24.7, LoRA 79.1/24.4, DoRA 78.3/24.8. So the "win" is:
- **+1.5pp adaptation** over the runner-up (SC-LoRA), and
- **+0.8pp retention** over the runner-up (DoRA).

Both margins are **smaller than the single-seed collapse basins the project itself documented**:
seed 44 swung clora_k2048 by ~40pp, dora_r8 by ~50pp, lorawd_wd0p5 by ~30pp
(`CONCLUSIONS` §6.1, `matrix-campaign-results`). A ~1pp win under a known ~10–40pp seed variance is
**not a win**; it is noise. The paper's own limitation #1 says single-seed is "NOT fine for any
head-to-head ranking claim" — yet Claim 2 IS a head-to-head ranking claim. This is internally
inconsistent.

Compounding it (the fairness gaps, all already known but load-bearing here):
- LoRA+wd is **r32**; LoRA/DoRA/CorDA/LoRA-Null are r16. It has 2× the adapter capacity of half
  the field (C3).
- **Only LoRA got the wd knob.** No other method was given weight decay. "The simplest magnitude
  control wins" is untested against "the same magnitude control applied to any other method."
- LoRA+wd sits at fdelta 0.394 (inside the claimed sweet-spot band); LoRA sits at 0.623 (outside).
  But that is a statement that *wd moved LoRA to a better operating point*, which is Claim 2's
  mechanism — it is NOT evidence that wd-LoRA beats a *differently-tuned* fancy adapter. A fancy
  adapter tuned to fdelta≈0.39 is the missing comparison.

Required before Claim 2 can say "wins/surpasses":
1. Seeds 43/44 on the ~6–8 headline cells (C4) with error bars. If the 1pp win is inside the CI,
   the verb becomes **"matches"** — which the docs already concede is the honest verb (§3.4) but
   the abstract/title still say "surpasses/wins."
2. A param-matched LoRA+wd **r16** control AND wd applied to ≥2 frontier fancy adapters (C3), so
   the claim is "wd helps everyone and geometry adds nothing on top," not "the one method we
   happened to regularize won."

Until both land, the strong claim is **not supported by the evidence present** and must be softened
to "LoRA+wd lands on the frontier at far lower engineering cost." (Directly answers the brief's
question (a): NO, the strong claim needs the fairness/param-matched work first.)

---

## [BLOCKER] B6 — Direct numeric contradictions across the four documents. Pick one source of truth and reconcile.

A reviewer who cross-checks your own tables against your own text will find these. Each is a
credibility hit:

1. **Math LoRA+wd operating point.** `table_main_math.tex` says **49.1 / 24.4 / fdelta 0.359 at
   lr3e-4**. `key_numbers.md` §4, `CONCLUSIONS` §3.2, and `02_figures` all say **50.6 / 24.6 /
   fdelta 0.399 at lr5e-4**. These are different cells. Which is it?
2. **Magnitude scale.** The narrative (`01`) quotes `‖ΔW‖` = "72–1395" and "≈72 at lr1e-3"
   throughout §(e)/appendix; `CONCLUSIONS` §0 and `key_numbers.md` say the real field `fdelta`
   ranges 0.05–3.7 and the "72–1395" scale is obsolete. `01_project_narrative.md` has **not been
   corrected** and still presents the wrong scale as fact in its appendix. One of your four
   core docs is quoting a deprecated axis.
3. **R² for the LR-proxy result.** Brief and `01` say **0.75 vs 0.35**; `key_numbers.md`,
   `CONCLUSIONS` §2.5, and `fig7` say **0.74 vs 0.32**. Small, but it means the number was never
   locked. (Also: `key_numbers` §2 lists r(LR)=-0.57 giving R²=0.32, consistent; `01` is the
   outlier.)
4. **Correlation headline.** `01` leads with pooled **-0.87**; everything current says **-0.86**.
   My recomputation: **-0.858**. Round consistently to -0.86.
5. **DoRA math point.** `02_figures` and `key_numbers` §4 list DoRA math 33.3/25.2; `table_main_math.tex`
   omits it. `02` flags this ("mismatch #2") but it is unreconciled.
6. **"7 methods × 7 LRs = 49" vs "6 methods."** `02_figures` line 13 says "6 methods × 7 LRs, n=49"
   — but 6×7=42, not 49. n=49 requires 7 series (the LoRA-Null-pooled-into-LoRA counts as an extra
   7). The arithmetic in the figure doc is wrong on its face.

Required: designate `key_numbers.md` as the ONLY source of truth (it already claims to be), then
**correct `01_project_narrative.md`'s appendix and §(e) to the fdelta scale**, fix the math table,
and make every quoted number trace to one recomputed value. This is mechanical but mandatory — a
paper that contradicts its own tables gets desk-rejected.

---

## [IMPORTANT] I1 — Near-circularity is under-defended; the ANCOVA "control" is thin.

The docs correctly identify near-circularity ("bigger ΔW perturbs more") and correctly say "lead
with the residual/ANCOVA test." But the ANCOVA as it stands is a weak shield:
- It is **6 methods, single seed, n≈7 per method**. An F(5,42)=8.3 driven entirely by one method
  (SC-LoRA) is not "geometry adds nothing" — it is "geometry adds nothing for 5 methods we happened
  to include and something for the 6th, on one seed."
- The residual test shows method *intercepts* are ~0, but does not test whether method changes the
  *slope* — a reviewer will ask for a slope-interaction term (does any adapter bend the curve, not
  just shift it?). Report the interaction ANCOVA, not just intercepts.
- "Method identity adds nothing beyond `‖ΔW‖`" would be far stronger with a held-out predictive
  test: fit retention~log‖ΔW‖ on 5 methods, predict the 6th, report RMSE vs a method-aware model.
  A pure in-sample R² bump from 0.74→0.87 is exactly what adding 5 free intercepts always does.

Required: add the slope-interaction ANCOVA and a leave-one-method-out predictive check. This is the
non-circular core of the paper (§2.6 says so) and it is currently one under-powered F-test.

---

## [IMPORTANT] I2 — Qwen is 13/112 cells and mostly LoRA; the "two-model replication" claim is not yet earned, and math ANTI-replicates.

The honesty framing ("in progress, supporting") is present and good. But note two things a reviewer
will seize:
- Qwen-**math** currently gives **r = +0.67 (wrong sign!)**, ns. The paper must not bury this. A
  positive correlation (bigger update → *better* retention) on one of four model×domain cells is a
  real anti-result; if it persists after the sweep fills in, the "law" is Llama-CS-specific plus a
  partial replication, not a universal law.
- The Qwen-CS slope is **-34.8 pp/decade** vs Llama's -10 to -15. "The law replicates" is doing a
  lot of work when the slope is 2–3× steeper. Replication of *sign and monotonicity*, yes;
  replication of the *law* (a quantitative relationship), not yet. Say the former.

Required: hold the "two-model" language until ≥5 Qwen adapters × both domains are in, and report the
Qwen-math sign explicitly as an open anti-replication, not omit it.

---

## [IMPORTANT] I3 — The "sweet-spot band [0.31, 0.62]" is post-hoc and self-fulfilling for LoRA+wd.

The band is defined as where adaptation is near-max and retention near-ceiling, then the paper
observes LoRA+wd (0.394) "lands in it for free." But the band was *drawn from the same 49 points*
that include LoRA+wd. This is circular: you cannot define the target region from the data and then
celebrate that your protagonist hits it. Also, LoRA's own best point (0.623) is *outside* the band
by the paper's own numbers, yet LoRA is the reference "on-curve" method — so the band excludes a
method the paper calls well-behaved.

Required: either derive the band from an independent criterion (e.g., a held-out utility function
specified a priori) or present it descriptively ("adaptation and retention cross here") without the
"wd lands in it for free" causal spin.

---

## [IMPORTANT] I4 — "Prior art already showed magnitude↔CF (CLoRA)"; the stated delta is not yet demonstrated.

Roadmap attack surface #6 is correctly identified. But the claimed delta — "method-freeness across
8 adapters × 2 domains × 2 models" — is, per B1/B5/I2, currently: 6 adapters (2 problematic),
1.5 domains (math sparse + Qwen-math anti-replicating), 1.3 models (Qwen 13/112). The delta over
CLoRA is real in principle but **not yet instantiated by the evidence present.** If the paper ships
before the 2×2 completes, the honest delta shrinks to "we show it's method-free across 6 adapters on
Llama-CS," which a reviewer will call incremental over CLoRA.

Required: either finish the breadth that makes the delta real, or reframe the contribution as the
*measurement methodology* (the fallback in §7.7 — the LR-sweep-as-instrument + the fair `‖ΔW‖_F`
axis), which is defensible with current data and is genuinely novel relative to CLoRA.

---

## [IMPORTANT] I5 — Retention "core" = only 2 benchmarks; base-ceiling uncalibrated for 3 of 5.

- "Core retention" is the mean of just BBH-AO and MMLU-Pro. The headline retention axis is 2
  benchmarks, and the base ceiling (26.0) exists only for those two. MMLU/ARC/TruthfulQA have **no
  base ceiling** (`key_numbers` §7), so "broad" retention percentages are uninterpretable and the
  per-benchmark slopes (fig5) cannot be normalized. This is cheap to fix (5 eval-only runs, C5) and
  must be done before any camera-ready retention percentage.
- TruthfulQA is "flat/immune" (slope -0.5, r=-0.10) — but with no base ceiling and MC-style scoring
  near chance, "immune" may just mean "already at floor/uninformative." Verify it is not a
  dead-benchmark artifact before headlining "truthfulness is untouched."

---

## [IMPORTANT] I6 — The LoRA-Null labeling bug is not merely cosmetic; it contaminates the "LoRA" series and the ANCOVA.

`CONCLUSIONS` §6.10 calls it a cosmetic legend fix ("pooled law identical either way"). Not quite:
- The **"LoRA" ANCOVA residual (+0.79)** and the LoRA within-method r (-0.97) are computed on a
  series that is **7 LoRA + 7 LoRA-Null = 14 points mislabeled as one method.** Verified: the
  lrsw "lora" group has 14 rows. So the LoRA residual and the "LoRA is on the law" claim are
  actually a LoRA∪LoRA-Null average, and LoRA-Null's own residual is never reported.
- If LoRA-Null (a data-aware null-space init — i.e. a *geometric* method) is silently averaged into
  the vanilla-LoRA baseline, that flatters the "geometry inert" story by construction.

Required: fix the labeling, report LoRA and LoRA-Null as separate series with separate residuals,
regenerate fig0/fig2 and the LoRA robustness count, before any submission. This is a BLOCKER-adjacent
IMPORTANT — it changes reported per-method statistics.

---

## [NICE] N1 — arXiv IDs all unverified; one is impossible.

`01` §(b) and `CONCLUSIONS` §8 flag every citation `[VERIFY]` and note `2603.02224` has an
impossible (March-2026) date. Verify all IDs (OPLoRA 2510.13003, CorDA 2406.05223, SC-LoRA, CorDA++
2506.13187) before submission; a wrong flagship citation is an easy reviewer jab. Mechanical.

## [NICE] N2 — No compute/carbon/repro statement, no per-cell wall-clock table.

For a paper whose method IS a sweep, reviewers increasingly want the total run count and cost. You
have it (288 Llama cells, Qwen 6–10h/cell). A short repro/compute appendix is cheap credibility.

## [NICE] N3 — "Wake-up call to a field that ships an adapter every week" tone.

The framing is combative. That is fine IF the evidence is airtight; with B1–B6 open it reads as
over-claiming. Keep the aggressive framing in reserve for the "strong" tier (all experiments done);
for the "minimum defensible" tier, lead with the measurement contribution, not the polemic.

## [NICE] N4 — DoRA math best point at lr2e-5 (33.3/25.2) is suspicious.

DoRA's math "best" is at the *lowest* LR with GSM8K 33.3 — that looks like an under-adapted point
being reported as an operating point, not a genuine optimum. Check it is not a
divergence/under-training artifact before it appears in any table.

---

## Cross-cutting verdict on the brief's four questions

- **(a) Is the strong claim supported, or does it need the fairness/param-matched work first?**
  **It needs the work first.** The "LoRA+wd surpasses fancy adapters; their wins are LR" claim is
  NOT supported by the evidence present: the margin is sub-noise on a single seed (B5), it is
  never demonstrated that any fancy adapter *wins* at any LR (B3), CorDA — the key test case — is
  excluded (B1), and the one significant off-curve result is calibration-confounded (B4). The
  *law* (Claim 1) is supported; the *consequence* (Claim 2) and *diagnosis* (Claim 3) are not yet.

- **(b) Unsupported claims / missing controls:** "8 adapters/one curve" (really 6) [B1];
  param-matched LoRA+wd r16 + wd-on-every-method [B5]; reproduce-the-win-then-dissolve-it [B3];
  eval-matched calibration [B4]; separate LoRA-Null series [I6]; slope-interaction ANCOVA + LOO
  predictive test [I1].

- **(c) Figures/numbers that don't exist but are needed:** the fixed-vs-swept-LR "gotcha" figure
  [B3]; CorDA back in fig0/fig2/table [B1]; error bars (seeds 43/44) on headline cells [B5];
  base ceilings for MMLU/ARC/TruthfulQA [I5]; a corrected fig0/fig2 with LoRA-Null split out [I6];
  Qwen panels once ≥5 adapters land [I2].

- **(d) Statistical rigor:** single seed on ranking claims [B5]; n≈7/method, one-seed ANCOVA
  driven by one method [I1]; near-circularity defended only by an in-sample R² bump [I1]; no CIs
  anywhere on the headline table. All must be addressed for a ranking claim; the law itself is
  robust at n=49.

- **(e) Honesty gaps:** the "eight" vs six mismatch [B1] is the biggest; exploded/duplicate CorDA
  rows still in the shipped data [B2]; four docs disagreeing on the math table and the magnitude
  scale [B6]; "law replicates on Qwen" while Qwen-math anti-replicates [I2]; a post-hoc sweet-spot
  band presented as a free win [I3].

---

## TOP-10 BLOCKERS (the 10-line summary requested)

1. "Law across 8 adapters" is really 6 — CorDA is EXCLUDED from every headline figure/table and,
   where present, sits OFF the curve; LoRA-Null is silently pooled into LoRA. Fix the count or the claim.
2. The shipped `campaign_summary.jsonl` still contains the CorDA residual-save explosion
   (fdelta=515, ret=0) and duplicate rows for every CorDA LR — the pipeline's own bug is in the data.
3. Claim 3 ("wins are an LR artifact") has NO figure and never reproduces any fancy adapter's win at
   any LR — you cannot debunk a win you never showed. Build the gotcha exhibit or delete the claim.
4. The one significant result (SC-LoRA -4.15pp off-curve) is confounded by nq_open↔academic-eval
   calibration mismatch; strike all "data-aware inits forget more" language until C2 runs.
5. "LoRA+wd wins the Pareto" is a +0.8–1.5pp margin on a SINGLE seed, under a documented ~10–40pp
   seed-collapse variance — it is noise; the paper's own limitations contradict this ranking claim.
6. LoRA+wd is r32 with the only wd knob in the field; needs param-matched r16 control + wd given to
   ≥2 fancy adapters before "surpasses" is fair.
7. Direct numeric contradictions across the four docs: math table 49.1/24.4 vs 50.6/24.6; magnitude
   scale 72–1395 (uncorrected in `01`) vs fdelta 0.05–3.7; R² 0.75/0.35 vs 0.74/0.32. Reconcile to
   one source of truth.
8. Qwen is 13/112 cells, mostly LoRA, and Qwen-MATH ANTI-replicates (r=+0.67, wrong sign) — "two-model
   replication" is not yet earned; report the anti-replication, don't omit it.
9. Base ceilings missing for MMLU/ARC/TruthfulQA → "broad" retention and per-benchmark slopes are
   uninterpretable; near-circularity is defended only by an in-sample R² bump (add LOO predictive +
   slope-interaction test).
10. Bottom line on the brief's question (a): the LAW (Claim 1) is defensible NOW; the CONSEQUENCE and
    DIAGNOSIS (Claims 2–3, the "wake-up call") are NOT supported by the present evidence and require
    the fairness/param-matched/seed/CorDA work first. Ship the measurement-methodology framing if you
    must submit before that work lands.
