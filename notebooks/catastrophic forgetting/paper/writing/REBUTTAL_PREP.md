# REBUTTAL_PREP — Author's-eyes-only

*Internal. Not for submission. Anticipates the top reviewer objections to the current
`paper.tex` (round-4 manuscript, post claim-renumbering) and prepares the honest defense we can give
TODAY, plus which pending experiment retires each risk. Grounded in `paper.tex`, `04_critique.md`,
`05_review_notes.md`, and `data/key_numbers.md` (single source of truth; sections 14–15 for the
split-LoRA-Null recompute). Numbers here are quoted from `key_numbers.md`; never invent evidence in
a rebuttal.*

**Claim numbering (renumbered 2026-07-02, presentation order):** Claim 1 = Mechanism (magnitude
law), **Claim 2 = Diagnosis (LR artifact)**, **Claim 3 = Consequence (LoRA+wd corollary)**. Older
entries below that say "Claim 2 (LoRA+wd)" refer to what is NOW Claim 3 — read accordingly.
Line-number citations (LNNN) refer to the round-1 manuscript and have drifted; anchor on section
labels instead.

> **UPDATE 2026-07-17 (post-freeze).** Headline numbers are now `key_numbers.md` **§18 FINAL FREEZE +
> §19 POST-FREEZE ADDENDUM** — pooled **r=−0.847, n=1035**, 6 model×task families, 8 methods, 3–5 seeds —
> **superseding every §14–15 reference below** (the n=49 single-seed numbers are historical). Framing is
> "magnitude relation (flat-then-falling with a knee)", not "Magnitude Law". Two consequences for this doc:
> (i) **O3 is RESOLVED** — E4 eval-matched calibration puts SC-LoRA at **+0.92pp ABOVE the relation**
> (n=20) vs −3.39pp nq_open-calibrated; the deviation was a calibration-set artifact, not method geometry
> (§18.3). (ii) New ammunition for every "geometry adds little / your ANCOVA is thin" objection (O5, O9):
> the **nested ΔR² ladder (§19.1)** — family FE R²=0.390 → **+0.395 magnitude** (F≈1890) → **+0.017
> geometry** (F=30) → **+0.006 method** (n=1034, run-level) — an out-of-the-box quantification that
> magnitude explains ~23× the variance geometry does, now multi-seed and multi-family.

**Pending-experiment key (used in every "resolves it" line):**
- **B4** — eval-matched calibration re-run + sensitivity arm (nq_open vs academic-eval calibration).
- **B5a** — param-matched LoRA+wd control (r16 and r32) + wd knob applied to ≥2 fancy adapters.
- **B5c** — seeds 43/44 on the ~6–8 headline cells (error bars).
- **CorDA nq_open re-run** — clean re-eval + 0-step self-check.
- **CorDA++** — strongest 2025 geometric method, param-matched.
- **Qwen 2×2** — ≥5 adapters × {CS, math} + higher-LR math cells.
- **base-ceiling calibration** — 5 eval-only no-FT runs for MMLU / ARC / TruthfulQA.

Reviewer-facing priority is set by two axes: how surely a competent hostile reviewer *finds* it,
and how much it *damages* acceptance if we have no answer. Ordered most-severe-first.

---

## O1 — "You claim eight adapters and one law, but the figures show six, and the one adapter that would actually test 'geometry is inert' — CorDA — is missing."

**The objection (reviewer's voice).** *"The abstract, intro, and conclusion all say 'eight
adapters.' The hero figure and the ANCOVA show six visible series. CorDA — a flagship data-aware
method and precisely the case that could falsify your 'geometry is inert' thesis — is excluded from
every headline exhibit, and LoRA-Null is silently pooled into LoRA by a labeling bug. You are
selling method-independence while hiding the two methods most likely to break it. This is not a
footnote; it is the load-bearing claim."*

**Severity / likelihood.** Highest. A sharp reviewer finds this in ten minutes by counting series in
fig0. If unanswered it reads as cherry-picking and taints the whole paper's credibility, not just
Claim 1's scope. This was the #1 BLOCKER in `04_critique.md` (B1).

**Prepared response (honest, today — UPDATED 2026-07-02 second pass).** The LoRA-Null half of this
objection is now **retired**: LoRA-Null is its own series in every figure, table, residual, and
robustness count (the labeling bug was fixed at the analysis level; pooled law numerically
identical, see `key_numbers.md` §14). The manuscript does not say "eight … one curve" in any
load-bearing sentence. It uses the updated coverage sentence (intro §sec:intro): *"across the seven
of eight adapters we can assess (LoRA, LoRA-Null, LoRA+wd, DoRA, MiLoRA, CLoRA, SC-LoRA), six lie
on a single ‖ΔW‖_F curve; SC-LoRA is the one provisional below-curve deviator, and CorDA is
withheld pending a calibration-fairness fix."* The Limitations section states plainly that the
study assesses seven of its eight adapters (six on-curve) and explains CorDA was mis-calibrated on
wikitext-2 and is being re-run on nq_open. The "eight" that survives is the *study design* (8
arms), not the *evidence*. We frame CorDA's absence as a declared, reason-given withholding — the
honest move — not a silent drop. We do not claim method-independence *over CorDA*; we claim it over
the 7 we can assess.

**Which pending experiment resolves it.** CorDA nq_open re-run + B4 fair calibration. Expected
outcome (good case): CorDA returns to the curve → we restore the full 8th series to fig0/fig2/table
and the coverage sentence becomes "eight of eight." Expected outcome (bad case): CorDA stays
below-curve after fair calibration → we report it as a *second* off-curve method alongside SC-LoRA
and retreat the claim to "geometry is inert for calibration-free inits; data-aware inits carry a
residual we trace to calibration↔eval mismatch." Either way the word "eight" can finally sit next to
"curve." CorDA++ is future work, not needed to close this.

**Residual risk.** Even in the good case, a reviewer can say the headline evidence *at submission
time* was six adapters; the fix is post-hoc. And if CorDA lands off-curve, the clean "geometry is
inert" story is permanently softened to "inert for calibration-free inits" — a real, if honest,
weakening of the thesis.

---

## O2 — "Every ranking claim rests on a single seed, under a variance you yourself measured in the tens of points."

**The objection.** *"You concede a prior three-seed matrix showed seed swings of tens of points on
individual cells. Your entire Claim 2 (LoRA+wd on the frontier) and Claim 3 (best-LR-not-shared) are
ranking claims read off ONE seed (42). A +1.5pp adaptation / +0.8pp retention 'edge' under a
±tens-of-pp seed variance is not a result — it is noise dressed as a finding. Your own Limitation #1
says single-seed is 'not fine for any head-to-head ranking claim,' yet Claims 2 and 3 are exactly
that."*

**Severity / likelihood.** Very high, near-certain. This is the classic empirical-ML kill-shot and
it is internally documented (`04` B5, `05` item 6). Damaging because it directly attacks the two
"wake-up-call" claims that make the paper interesting.

**Prepared response (today).** We do not stake the paper on the ranking. The manuscript already
demotes the verb to "matches or edges," never "dominates" (§sec:pareto L542–552), and reframes Claim
2 as a *corollary*: the result "is not that LoRA+wd ranks first, it is that a subspace-free magnitude
knob reaches the frontier at all." That reframing survives seed noise because "reaches the frontier"
is a qualitative statement, not a 0.8pp delta. The *law* (Claim 1) is separately robust: n=49 across
7 series, r=−0.86 pooled / −0.92 on-curve, p=3×10⁻¹⁵ — a 49-point correlation is not a single-seed
ranking and does not inherit the collapse-basin variance. Limitation #1 (L560–570) states the seed
gap unflinchingly and pre-commits that error bars (seeds 43/44) are required before any per-method
delta is headlined.

**Which pending experiment resolves it.** B5c seeds 43/44 on the ~6–8 headline cells. Expected
outcome: if the LoRA+wd edge is inside the CI, the verb stays "matches" (which the docs already
concede is honest) and Claim 2 is unharmed under the corollary framing; if it survives seed
averaging, we earn back a stronger verb. Either way, error bars convert "single-seed noise" from a
fatal objection into a stated, bounded uncertainty.

**Residual risk.** Until seeds land, a reviewer can refuse to credit *any* cross-method comparison,
including the qualitative "reaches the frontier" — because "on the frontier" is itself read off one
seed's frontier. This is the cheapest desk-reject insurance to buy (see triage).

---

## O3 — "Your one significant result is confounded by a calibration/evaluation mismatch you admit you can't resolve." — **RESOLVED 2026-07-17 by E4**

> **RESOLUTION (E4 eval-matched calibration, `key_numbers.md` §18.3).** SC-LoRA calibrated eval-matched
> sits **+0.92pp ABOVE the relation** (mean residual, n=20) vs −3.39pp under nq_open calibration (n=24):
> the old −4.15pp deviation was a **calibration-set artifact, not method geometry**. This is the "good
> case" predicted below — the one-curve story got *stronger*. The text below is kept as the historical
> record of the objection and the pre-E4 defense.

**The objection.** *"The single statistically significant finding in your entire ANCOVA is SC-LoRA
at −4.15pp (p=0.006). But SC-LoRA, CorDA, and LoRA-Null calibrate on nq_open (factoid QA) while you
measure retention on academic/reasoning benchmarks. The one 'interesting' deviation you have is
exactly the one most likely to be a calibration-set artifact. You cannot simultaneously claim
'geometry is inert, all on one curve' AND 'SC-LoRA is a real off-curve deviator' AND 'the deviation
may be a calibration artifact.' Pick a lane."*

**Severity / likelihood.** High. A methods-literate reviewer will spot the confound immediately
(`04` B4, the docs' self-declared "biggest kill-shot"). Damaging because it sits on the exact result
the ANCOVA figure rings.

**Prepared response (today).** We already picked the lane: SC-LoRA's deviation is presented as
**provisional and confounded**, never as a geometric finding. Every mention carries the caveat —
fig2 caption ("provisional and confounded by the calibration mismatch," L357), §sec:ancova text
(L334–337), and Limitation #3 which names this "the biggest open question" (L580–590). We explicitly
embargo all "data-aware inits forget more than their budget" language until B4 lands. The honest
framing in the body is: "one method's residual is nonzero; we cannot yet say whether that is geometry
or calibration." Note this confound *cuts in our favor* for Claim 1 — if the sole deviator is an
artifact, the "one curve" story gets *stronger*, not weaker.

**Which pending experiment resolves it.** B4 eval-matched calibration + sensitivity arm. Expected
outcome (likely): SC-LoRA (and LoRA-Null, CorDA) return toward the curve under eval-matched
calibration → the cleaner result "the law is method-free once calibration is fair," and the ANCOVA
becomes 6/6 on-curve. Expected outcome (bad case): the residual persists → we then have a *real*
geometric penalty for data-aware inits and must partially retract "geometry is inert." B4 is the
single highest-leverage experiment because it decides which of these two papers we are writing.

**Residual risk.** If B4 shows the residual is real geometry, the thesis takes a genuine hit for the
data-aware family — and CorDA (O1) likely lands off-curve too, compounding it. We are honest that
this is unresolved, but "unresolved" on your one significant result is itself a reviewer talking
point.

---

## O4 — "LoRA+wd wins because you gave it 2× the rank and a regularizer no other method got."

**The objection.** *"LoRA+wd is r32; LoRA, DoRA, CorDA, LoRA-Null are r16 — it has twice the adapter
capacity of half the field. And weight decay was applied to LoRA ONLY. 'The simplest magnitude
control wins' is untested against 'the same magnitude control applied to any other method.' You
haven't shown magnitude beats geometry; you've shown the one method you regularized and gave extra
capacity to won. A fancy adapter tuned to ‖ΔW‖≈0.39 is the missing comparison."*

**Severity / likelihood.** High, near-certain from any fairness-minded reviewer (`04` B5, `key_numbers`
§13). Damaging to Claim 2 specifically.

**Prepared response (today).** The *law* framing sidesteps this entirely and the manuscript says so
(Limitation #2, L571–578): the magnitude law is a statement about ‖ΔW‖, not about ranks — LoRA+wd
(0.394) and every other adapter are plotted by the update magnitude they actually produce, and they
all interleave on one curve regardless of rank. The mechanism paragraph (§sec:pareto L483–495) frames
LoRA+wd's advantage as "it carries the smallest update in the comparison (‖ΔW‖=0.394)" — i.e. it wins
by *landing at a good magnitude*, which is Claim 2's whole point, not by capacity. We openly flag that
the *strong* form of Claim 2 needs a param-matched control. We never claim r-matched superiority in
the current draft.

**Which pending experiment resolves it.** B5a — param-matched LoRA+wd (r16 and r32) + wd knob given
to ≥2 frontier fancy adapters. Expected outcome: if wd helps every method equally and geometry adds
nothing on top, the claim upgrades to "wd helps everyone, geometry adds nothing" — far stronger and
immune to the capacity objection. If a fancy adapter tuned to ‖ΔW‖≈0.39 matches LoRA+wd, that too
*confirms* the law (magnitude, not method, is what matters) — so B5a is win-win for Claim 1.

**Residual risk.** Until B5a lands, the strong "surpasses" form of Claim 2 is unsupported and a
reviewer can hold that against the paper's framing even though the body already retreats to "matches/
edges." If B5a shows a fancy adapter *beats* LoRA+wd at matched magnitude, Claim 2's "lowest
engineering cost" selling point weakens.

---

## O5 — "Your law is near-circular: a bigger update perturbs the model more, by construction."

**The objection.** *"'Retention falls as ‖ΔW‖ grows' is close to a tautology — of course a larger
weight perturbation degrades the model more. You've dressed up a definitional relationship as an
empirical law. Your only non-circular defense is an in-sample R² bump from 0.74 to 0.87 after adding
five free per-method intercepts — which is exactly what adding five free parameters always does. And
your ANCOVA is single-seed, n≈7 per method, with its significance driven entirely by one method."*

**Severity / likelihood.** High. A statistically sophisticated reviewer will make both halves of
this (`04` I1, `05` item 10). Damaging because it attacks the *interpretation* of the core result.

**Prepared response (today — UPDATED 2026-07-02 second pass: both deferred analyses are now RUN,
results favorable).** The manuscript explicitly concedes the raw correlation is "near-tautological"
and rests the claim on a three-part battery (§sec:ancova), not on the correlation:
(i) **Intercepts:** ΔR²=0.13 from six free per-method intercepts, F(6,41)=7.05 — significant but
driven entirely by SC-LoRA; on-curve intercepts jointly n.s. (F(5,35)=1.79, p=0.14).
(ii) **Slopes (slope-interaction ANCOVA, now run):** among the six on-curve adapters the slopes are
statistically indistinguishable (F(5,30)=0.28, p=0.92; per-method slopes −8.1…−11.7 pp/dec); the
full-pool interaction (F(6,35)=9.32) is again pure SC-LoRA (slope −26.0).
(iii) **Leave-one-method-out (now run):** the pooled law fit on six series predicts the held-out
on-curve adapter at 1.7–3.6 pp RMSE (mean 2.5) — about the in-sample accuracy of a method-aware
model (2.2 pp); held-out SC-LoRA mispredicts by −7.4 pp, consistent with its provisional status.
The in-sample-R²-bump objection is therefore answered with out-of-sample prediction. The remaining
answer to "of course bigger perturbs more" is unchanged: geometry papers claim precisely that
*where* you perturb lets you decouple adaptation from forgetting — our result is that it does not,
and *that* is not tautological.

**Which pending experiment resolves it.** The two re-analyses are done (`analysis_a1_a4.py`,
`key_numbers.md` §15). What remains is B5c seeds to de-risk the "single-seed ANCOVA" half.

**Residual risk.** Circularity is partly a framing fight that never fully dies — some reviewers will
maintain that any ‖ΔW‖↔retention curve is "obvious." The LOMO check materially helps but the
philosophical objection lingers.

---

## O6 — "This is incremental over CLoRA, which already linked magnitude to forgetting — and Lee et al. scoop the sweep."

**The objection.** *"CLoRA already connected update magnitude to forgetting; its penalty is an
indirect magnitude control. The concurrent Lee et al. already show LoRA variants converge once each
is tuned at its own learning rate. So your 'law' is known and your 'sweep-as-instrument' is
concurrent prior art. Stripped of the breadth you don't yet have (CorDA, seeds, Qwen 2×2), the real
delta is 'method-free across six adapters on one model, one seed' — incremental."*

**Severity / likelihood.** Medium-high. A reviewer who knows the literature will raise it (`04` I4).
Damaging to the novelty case, though not to correctness.

**Prepared response (today).** The manuscript carves the delta precisely (§sec:related L182–195): our
contribution is *not* "magnitude matters" (we credit CLoRA explicitly). It is three things CLoRA and
Lee et al. cannot claim: (i) magnitude is the *method-free* governing variable across a zoo of
adapters, not one method; (ii) once ‖ΔW‖ is controlled, geometry is causally close to inert (the
ANCOVA) — a claim structurally unavailable to a single-method paper; (iii) the reported per-method
wins carry the ingredients of an LR artifact a sweep dissolves. On Lee et al. specifically, the
manuscript positions them as *corroborating adaptation-side* evidence and stakes our novelty on the
*retention* side, which they do not study (L139–145, L323–325). The measurement contribution (fair
‖ΔW‖ axis + sweep-as-instrument) is genuinely novel relative to both and is the guaranteed-defensible
fallback if the rankings shift.

**Which pending experiment resolves it.** Qwen 2×2 completion + CorDA re-instatement make the breadth
delta *real* (8 adapters × 2 domains × 2 models) rather than aspirational. B5a/B5c make the
"geometry adds nothing on top of magnitude, for everyone" claim airtight, which is the sharpest
distinction from CLoRA. Expected outcome: with breadth filled, "incremental over CLoRA" collapses;
without it, we fall back to the measurement-methodology framing (defensible with current data).

**Residual risk.** Concurrency with Lee et al. is a timing hazard we cannot fully control — a
reviewer may still perceive overlap regardless of the retention-vs-adaptation distinction. And if we
ship before breadth lands, the incrementality charge has partial merit.

---

## O7 — "The Pareto margin you build a claim on is smaller than your own noise floor."

**The objection.** *"LoRA+wd's edge is +1.5pp adaptation over SC-LoRA and +0.8pp retention over DoRA
(Table). Retention margins in your LR-artifact table are sub-1pp. These are below the seed variance
you document. A sub-noise margin cannot support 'lands on the frontier of every elaborate adapter.'"*

**Severity / likelihood.** Medium-high. Overlaps O2 but is narrower — it targets the *margin* even if
one grants the single seed. Certain to be raised.

**Prepared response (today).** We explicitly decline to lean on the margin. §sec:lr (L437–442): "the
retention margins here are sub-1pp and within single-seed noise; the load-bearing point is not the
size of the gap but that *no fixed-LR advantage survives a symmetric sweep*." And §sec:pareto's honest-
verb paragraph (L542–552) reframes the whole claim away from margin: "not that LoRA+wd ranks first,
it is that a subspace-free magnitude knob reaches the frontier at all." The claim is deliberately
constructed to be true even at zero margin — being *on* the frontier, not *above* it, is what the
magnitude law predicts. So a sub-noise margin is consistent with, not damaging to, the framing.

**Which pending experiment resolves it.** B5c seeds 43/44 puts a CI on the margin; the claim does not
need the margin to be positive, only to be non-negative within noise, which "matches/edges" already
allows. B5a additionally tests whether the frontier position is capacity-driven.

**Residual risk.** A reviewer may argue that a sub-noise margin means we cannot even assert LoRA+wd
is *on* the frontier vs slightly inside it — the qualitative claim still consumes some seed-42-
specific luck. Minor, but real.

---

## O8 — "Your 'two-model replication' is 13 of 112 cells and your math arm anti-replicates with the wrong sign."

**The objection.** *"You headline 'two base models,' but Qwen is ~13/112 cells, almost all LoRA. Worse,
Qwen-math gives r=+0.67 — the WRONG sign: bigger update, *better* retention. One of your four
model×domain cells actively contradicts the law. And the Qwen-CS slope is −34.8 vs Llama's −10 to
−15, so even where it 'replicates' it's 2–3× steeper — you've replicated a sign, not a law."*

**Severity / likelihood.** Medium-high. A reviewer who reads §sec:law will catch the +0.67. Damaging
if it looks buried; neutral-to-good if reported openly.

**Prepared response (today).** We report the anti-replication openly and draw no Qwen ranking —
this is a strength, not a liability, if framed as scientific honesty. Abstract (L74–76) and §sec:law
(L299–308) state Qwen is "a partial replication (~13/112 cells)," that CS-LoRA "replicates the sign
and monotonicity, not yet the quantitative law" (naming the −34.8 vs −10/−15 slope gap), and that
"the Qwen math LoRA sweep does *not* yet replicate (r=+0.67, n.s., p=0.21)" — with the diagnosis that
the higher-LR cells reaching the forgetting regime are not yet run. We never claim "two models"
without the "partial" qualifier. The p=0.21 matters: the +0.67 is *not significant* on n=5, so it is
"no signal yet," not "significant contradiction."

**Which pending experiment resolves it.** Qwen 2×2 completion (≥5 adapters × both domains + higher-LR
math cells). Expected outcome: the higher-LR math cells push Qwen into the forgetting regime; if the
sign flips negative, the anti-replication dissolves and we earn the two-model claim. If it persists,
the law is honestly scoped to "Llama-CS + partial replication," not universal.

**Residual risk.** The +0.67 sign is genuinely uncomfortable regardless of significance — a hostile
reviewer will quote it as "the law fails on the one out-of-distribution test you ran." If the
completed sweep does not flip it, "universal law" is off the table for good.

---

## O9 — "Your ANCOVA is underpowered: single seed, n≈7 per method, and its whole significance rides on one deviator."

**The objection.** *"F(6,41)=7.1 sounds decisive until one notices it is single-seed, n≈7 per method,
and driven almost entirely by SC-LoRA — the one method you also say is confounded (O3). Remove
SC-LoRA and you have 'geometry adds nothing for six methods I happened to include, on one seed.'
That is not a fairness test; it is an underpowered F with one influential point."*

**Severity / likelihood.** Medium. A statistics-focused reviewer raises it (`04` I1). Overlaps O5 and
O3 but is distinct: it attacks the *power and leverage* of the specific F-test.

**Prepared response (today — UPDATED 2026-07-02 second pass).** We disclose the weaknesses in-text,
and the "driven by one method" fact *supports* our thesis: six of seven adapters have residuals
indistinguishable from zero (+0.99, +0.60, +0.06, +1.04, +0.09, +1.37, all n.s.) — the F is
significant *because* of the one deviator we already flag as provisional/confounded. So "remove
SC-LoRA and geometry adds nothing" is precisely our claim, not a rebuttal to it. The in-sample-only
half of the charge is now retired: the slope-interaction ANCOVA and the leave-one-method-out
predictive check have been run (see O5; `key_numbers.md` §15) and both come out method-free for the
on-curve six.

**Which pending experiment resolves it.** B5c seeds 43/44 (raises effective n, puts CIs on
residuals); B4 resolves whether the one influential point (SC-LoRA) is real or an artifact. Together
these retire the "underpowered, one-point-driven" charge.

**Residual risk.** n≈7 per method on one seed is simply thin; even with seeds 43/44 we reach n≈21 per
method, still modest. A reviewer wanting a multi-seed × multi-method factorial will not be fully
satisfied.

---

## O10 — "Your sweet-spot band is post-hoc and self-fulfilling — and it excludes the very method you call well-behaved."

**The objection.** *"The band ‖ΔW‖∈[0.31,0.62] is drawn from the same 49 points that include LoRA+wd,
then you celebrate LoRA+wd (0.394) 'landing in it for free.' Circular. Worse, plain LoRA's own best
point (0.623) falls *outside* your band — yet LoRA is your reference on-curve method. You defined the
target from the data and then congratulated your protagonist for hitting it."*

**Severity / likelihood.** Medium-low. Real but narrower; a careful reviewer raises it (`04` I3, `05`
item 9). Damaging mainly to the rhetorical "for free" spin, not to the core result.

**Prepared response (today).** The manuscript already labels the band descriptive, not causal:
fig8/§sec:pareto call it "the descriptive sweet spot … (read off these points, not fitted)"
(fig:budget caption L713–717) and `key_numbers` §6 tags it `[EXTERNAL: design choice; not a fitted
CI]`. The band is presented as an annotation of where the rising-adaptation and falling-retention
curves cross, not as an a-priori target LoRA+wd was scored against. We can further defuse by stating
the band is illustrative and that the load-bearing claim (LoRA+wd carries the smallest update,
‖ΔW‖=0.394) is a raw measured fact independent of the band. That LoRA's 0.623 sits at/past the right
edge is *consistent* with the story (un-regularized LoRA over-shoots; wd pulls it back) — we can turn
the apparent embarrassment into supporting evidence.

**Which pending experiment resolves it.** No experiment strictly needed; this is a re-analysis/
framing fix (derive the band from an a-priori utility criterion, or present it purely descriptively).
B5c seeds would let us put uncertainty on the crossover location. Lowest-cost of all the items.

**Residual risk.** Minimal once framed descriptively — but if we ever restore causal "lands in it for
free" language, the circularity charge is valid and easy to make stick.

---

## O11 — "Biderman et al. (TMLR 2024) already tested weight decay as a forgetting mitigation and found it DOESN'T work — your Claim 3 contradicts published evidence." *(ADDED 2026-07-02; WILL come up)*

**The objection (reviewer's voice).** *"'LoRA Learns Less and Forgets Less' (arXiv:2405.09673, TMLR
2024) explicitly compared LoRA against classical regularizers as forgetting mitigations and found
that weight decay 'appears to learn and forget as much as full finetuning' and even deteriorates at
longer training. Your headline practical claim is that weight decay is the simplest sufficient
forgetting control. You cite the paper but you are contradicting its experiment. Who is wrong?"*

**Severity / likelihood.** High likelihood (the paper is famous; any PEFT-literate reviewer knows
it), medium damage — because the contradiction is only apparent, and the manuscript now defuses it
in-text (§sec:related, "one point of apparent tension").

**Prepared response (today).** Nobody is wrong; the two experiments sit at different points of the
same law, and the magnitude law explains both. Verified specifics of their setup (checked against
arXiv:2405.09673v2, §4.5/Figure 4): they applied weight decay to **full finetuning** (all ~7B
parameters), at coefficients **5e-5 and 1e-4**, at a **single previously-tuned learning rate**, and
compared against LoRA. At those coefficients the penalty barely moves ‖ΔW‖ — so under the magnitude
law their finding ("learns and forgets like unregularized full FT") is exactly what we would
predict. We apply weight decay **0.3** (3,000–6,000× their coefficient) to the **adapter matrices
only**, and we verify directly that it bounds the thing that matters: LoRA+wd carries the smallest
measured ‖ΔW‖ in our sweep (0.394) and sits dead on the retention curve at the high-retention end.
Their design also evaluates at one learning rate per method — which is precisely the single-rate
comparison our Claim 2 shows can manufacture or hide differences; without a per-method sweep, a
regularizer's effect on the trade-off cannot be separated from its effect on the effective learning
rate. Bonus corroboration, not contradiction: their headline result (LoRA forgets less *and learns
less* than full FT) is a coarse-grained instance of our two-edged budget — less magnitude, less
forgetting, less adaptation.

**Which pending experiment resolves it.** None needed for the defusal (it is an analysis-level
reconciliation, already in the manuscript). B5a (wd knob on other adapters, param-matched) would
further show the wd effect is generic magnitude control, not a LoRA-specific trick.

**Residual risk.** A reviewer could ask for the direct experiment: full-FT + large weight decay
under our sweep, to show their regime joins our curve. That is out of scope (we study PEFT), but
"out of scope" must be said gracefully; the honest line is that our claim is about bounding ‖ΔW‖ of
*adapters*, where the coefficient that does so without killing adaptation demonstrably exists.

---

## O12 — "Why re-run every adapter yourself? Just compare against the numbers the original papers report." *(ADDED 2026-07-03)*

**The objection (reviewer's voice).** *"You spend ~1,700–2,000 GPU-hours re-training eight
adapters when every one of these methods already published results on Llama-2-7B. Why not simply
tabulate their reported numbers against your LoRA+wd? Re-running everything invites porting bugs and
looks like you're manufacturing a home-field advantage."*

**Severity / likelihood.** Medium. A pragmatic reviewer raises it; damaging mainly to the
efficiency/necessity framing, not to correctness.

**Prepared response (today).** A fair cross-method comparison of *forgetting* simply cannot be
assembled from the literature, because the adapter papers do not measure retention on a common axis.
Of the competitors, **only CLoRA reports our retention benchmarks (BBH + MMLU-Pro)**; CorDA/SC-LoRA/
LoRA-Null report closed-book factoid-QA exact-match (TriviaQA/NQ/WebQS), MiLoRA reports only a
WikiText loss proxy, and PiSSA/DoRA report **no forgetting metric at all** (see the fairness audit,
Q2 table). Their "retention" numbers therefore cannot be placed on one axis — so re-running every
adapter under a single identical harness is a *precondition* for a fair forgetting comparison, not a
convenience, and is itself part of the measurement contribution (now stated in §sec:setup, "Why a
common harness is a prerequisite, not a convenience"). On the *adaptation* axis a cross-paper
comparison is narrowly valid (GSM8K broadly; CS-8 via MiLoRA/DoRA), but every competitor reports a
**single fixed LR** (2e-5; MiLoRA 3e-4) vs our best-of-7 — comparing their one point to our swept
best is asymmetric and structurally favors us, which is exactly the single-LR practice Claim 2 is
about (Exhibit A for the LR-artifact diagnosis). Two guards against the "home-field / porting-bug"
half: (i) we audited every port against its reference and pass a 0-step ΔW=0 self-check on the
residual-init arms; (ii) we validate our accuracy scale against the canonical anchor — plain LoRA
scores BoolQ 69.97 vs canonical 69.8 and CS-8 79.1, *between* the published Llama-2-7B LoRA baselines
DoRA 77.6 and CLoRA 79.9 (§sec:setup "Scale validation"). And where cross-paper comparison *is*
clean — CLoRA — it independently corroborates the magnitude law from a competitor's own data (LoRA
forgets on BBH/MMLU-Pro; an L2/wd baseline recovers retention; the 𝔽 proxy our fdelta follows tracks
forgetting across methods; §sec:law "External corroboration", App. CLoRA cross-check).

**How the base-harness gap is handled.** CLoRA's base BBH (34.91) exceeds our answer-only base
(33.10) — a harness-config gap — so we never merge raw numbers 1:1; we compare *relative degradation
from each paper's own base* and use CLoRA as triangulation. Stated explicitly in-text and in the
appendix table caption.

**Which pending experiment resolves it.** None needed — this is a framing/positioning point, now
made positively in the manuscript. Qwen 2×2 + CorDA re-instatement further widen the systematic set.

**Residual risk.** A reviewer could still want a literature meta-table for adaptation; we give the
CLoRA anchor and the scale-validation, and defer a broader adaptation meta-comparison as out of the
retention-focused scope.

---

## O13 — "Your ΔW magnitude for DoRA is mismeasured — you omit DoRA's magnitude vector, so DoRA is plotted in the wrong place." *(ADDED 2026-07-03)*

**The objection (reviewer's voice).** *"You compute ‖ΔW‖ from PEFT's get_delta_weight, which for
DoRA returns only B·A·scaling and drops DoRA's magnitude-vector rescaling that DoRA applies at
forward time. So DoRA's x-coordinate on your hero curve is wrong — DoRA is one of your six on-curve
adapters, and you've plotted it at the wrong magnitude. How can the 'geometry is inert' claim survive
a measurement error on one of the six?"*

**Severity / likelihood.** Medium. A DoRA-literate or code-reading reviewer finds it; damaging if
unacknowledged because it touches the ANCOVA membership.

**Prepared response (today).** We disclose it precisely as a **lower bound**, not a silent error
(Limitations, new item "DoRA's magnitude coordinate is a lower bound"). DoRA's *retention* (y-axis)
is measured correctly — evaluation runs the true DoRA forward pass, magnitude vector included. Only
the *‖ΔW‖ x-coordinate* is under-measured, because get_delta_weight omits the forward-time magnitude
rescaling. So DoRA may sit slightly to the **right** of where we plot it. Critically, the direction
of the correction cannot break the on-curve conclusion: DoRA currently sits **on or above** the curve
(spline residual +1.37 pp, n.s., key_numbers §5/§14), and moving it to a larger ‖ΔW‖ — where the
curve predicts *lower* retention — only *increases* its positive residual. The correction pushes DoRA
further above the law, never below it. We do not overclaim the other direction either: we do not
assert DoRA would move a specific amount, only that its plotted magnitude is a lower bound.

**Which pending experiment resolves it.** A corrected recompute of DoRA's true
ΔW = m·(W0+BA)/‖W0+BA‖_c − W0 from the DoRA checkpoints (needs the 7 checkpoints + GPU). The
checkpoints were **not retained**, so a clean re-run of the 7 DoRA cells is required; deferred to
camera-ready (visible %TODO in the manuscript). Because the correction direction is retention-favorable
for DoRA, it is not on the acceptance-critical path.

**Residual risk.** Until the recompute lands, a reviewer can note the x-coordinate of one on-curve
adapter is approximate. Bounded and disclosed; the sign argument (correction can only help DoRA's
on-curve status) contains the damage.

---

## Priority triage

**The single experiment that most de-risks acceptance: B4 (eval-matched calibration).** Per the
project docs it is the "highest-leverage pending experiment" (Limitation #3, `04` B4). It decides
which paper we are writing: if the data-aware inits (SC-LoRA, CorDA, LoRA-Null) return to the curve
under fair calibration, the thesis upgrades to its cleanest form — *"the law is method-free once
calibration is fair,"* 6/6 (and ultimately 8/8) on-curve — and simultaneously retires O1 (CorDA),
O3 (the confounded significant result), and the influential-point half of O9. If they stay off-curve,
we learn that *now*, internally, and reframe to "inert for calibration-free inits" before a reviewer
forces the retraction on us. No other single experiment touches this many objections.

**The cheapest desk-reject insurance: seeds 43/44 (B5c).** ~6–8 headline cells on two extra seeds
converts our most-certain and most-damaging generic objection — "single seed, sub-noise margin"
(O2, O7), which every empirical-ML reviewer raises reflexively — from a potential desk-reject into a
bounded, error-barred caveat. It is far cheaper than the full 2×2 or CorDA++ and directly satisfies
the paper's own pre-committed promise in Limitation #1. Ship nothing claiming a per-method delta
without it.

**Sequencing.** B4 first (decides the thesis), B5c in parallel (cheap, unblocks all ranking
language). The two re-analyses behind O5/O9 (slope-interaction ANCOVA + leave-one-method-out
predictive check) are **DONE** (2026-07-02, `analysis_a1_a4.py`; favorable — see O5) and are in the
manuscript. Qwen 2×2, B5a, CorDA++ strengthen but are second-tier for acceptance vs desk-reject.
