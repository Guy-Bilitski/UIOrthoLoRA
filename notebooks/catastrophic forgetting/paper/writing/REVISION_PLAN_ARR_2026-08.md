# Revision plan against the 2026-08-08 review pipeline

Source: `comprehensive-review-08082026/` (5 steps, blind second pass, final verdict
**Overall 3.0 = Findings**, Soundness 3.0, Excitement 3.5, Confidence 4).
Target: **Overall 4**.

---

## 1. Bottom line

The review is good and mostly right. It does not attack a single measurement. Every
statistic the reviewer recomputed reproduced, and the reviewer says so at length. What
holds the paper at 3.0 is two things, and neither is an experiment we failed to run:

1. **Four published papers we do not cite**, one of which (Shuttleworth et al., NeurIPS
   2025) reaches a conclusion in direct tension with ours using a mechanism we never
   measure. This is pure writing.
2. **The headline claim is written wider than the instrument that produced it.** Our
   geometry block is three coordinates against a fixed 256-cut; the abstract says
   "geometry". Most of the fix is reporting, and the parts that are not are cheap.

Everything else in the review is Tier 2/3 bookkeeping. The reviewer states explicitly
that the score moves to 3.5 to 4 if items 1 and 2 are addressed, so the path is
already conceded.

The one thing that could genuinely cost us is that we cannot recompute geometry at a
different spectral cut, or compute the spectrum of `W0+dW`, because **no 7B adapter
checkpoints survived the fleet evacuation**. Section 4 handles that.

---

## 2. What the review gets right, ranked by score movement

### 2.1 The literature block (highest value, pure writing)

Verified independently, all three are real and published:

| Paper | Venue | Status for us |
|---|---|---|
| Shuttleworth, Andreas, Torralba, Sharma, *LoRA vs Full Fine-tuning: An Illusion of Equivalence*, arXiv:2410.21228 | **NeurIPS 2025** (poster 115207, OpenReview `PGNdDfsI6C`) | **Must cite. The one that hurts.** Intruder dimensions: new leading singular directions of the *updated* matrix, near-orthogonal to pretrained ones. Forgetting localises there, and they scale their singular values down as a causal intervention. Covers PiSSA. Reports larger learning rate and larger alpha giving more intruder dimensions and more forgetting at near-identical task accuracy, which is our 5.2 with a geometric mechanism attached. Also reports that rank-stabilised high-rank LoRA comes to resemble full fine-tuning, which bears on our rank result and on the full-fine-tuning offset. |
| Zhang et al., *The Primacy of Magnitude in Low-Rank Adaptation*, arXiv:2507.06558 | **NeurIPS 2025** (poster 115720) | **A published ally we are not claiming.** Argues update magnitude is the fundamental driver and that spectral initialisation works through magnitude amplification, treating learning rate, scaling factor and initialisation as levers on one quantity. That is our RQ2, on the adaptation axis. |
| Huang, Cheng, Wang, *Mitigating Catastrophic Forgetting in LLMs with Forgetting-aware Pruning*, arXiv:2509.08255 | **EMNLP 2025** (`2025.emnlp-main.1108`) | **A second published ally.** Task-vector-to-pretrained-weight ratio predicts forgetting, and they build FAPM on it. Independent, published, relative-magnitude account on the retention axis. |

Consequences for the text:

- §1 "the factors that cause, or at least track, forgetting are still not established"
  is **wrong as written** and has to go.
- §2 "None of these measures a common magnitude axis across methods or sweeps the
  learning rate per method" needs narrowing to what is still true.
- Our delta survives whole: **none of them puts the adapter family on one retention
  measurement with per-method rate tuning.** Zhang and Huang are corroboration from
  two other directions, which makes the retention axis a *sharper* delta, not a weaker
  one.

This reframing is also better rhetoric. "Nobody knows what governs forgetting" is a
weak opening that invites a reviewer to find someone who does. "Independent published
work points at magnitude on the adaptation axis and on the pruning side; we are the
first to test it on the retention axis across the adapter family with tuning
equalised" is a claim nobody can knock down.

### 2.2 The claim does not match the instrument (the soundness hit)

`rq1_stats/04_ladder_ci.py` line 27: `GEO = ["e_top_w", "lspec", "stable_rank_w"]`.
The commonality shape block is `e_top` + stable rank. Neither includes `e_bot` or
either input-side share, and **`ein_top` is the coordinate in Table 3 with the real
between-method range** (SC-LoRA 0.48 against 0.07 to 0.09 for the plain arms). So the
fingerprint that discriminates designs in Figure 2a and the block that scores +0.017
are not the same object, and the abstract joins them in one sentence. That is a fair
hit and it is the difference between Soundness 3.0 and 3.5.

Four sub-items, in decreasing order of how much they cost us:

- **Input-side shares excluded from the ladder.** Fixable from the existing store in
  under a day (`ein_top_w`, `ein_bot_w` are already in `adapter_metrics*.jsonl` and are
  already printed in the grand table generator).
- **No recovery check.** We never show that the ladder *would* find a geometry effect
  of a known size. Answerable by simulation on the real covariates, no new data.
- **The 256-cut is never varied.** Not recomputable without the adapters. See §4.
- **The adapted-matrix spectrum is never computed.** Not recomputable without the
  adapters. See §4.

The counter-argument we already hold and do not deploy: method identity adds only
+0.006 after size and geometry, which bounds any *between-method* geometry feature.
It says nothing about *within-method, run-to-run* variation, which is the form the
intruder-dimension effect takes. Say both halves of that out loud.

### 2.3 Three accountings of the magnitude-to-geometry ratio

| Accounting | Ratio | Where |
|---|---|---|
| Nested ladder | **23x** | Table 7, quoted in abstract and contributions |
| Single predictors: stable rank +0.116 vs log F∆ +0.420 | **3.6x** | Table 9 |
| Three-block with KL drift: shape +0.031 vs magnitude +0.033 | **near-tie** | App. C |

The reviewer is right that the reader only meets the largest of the three. Note the
three are not the same estimand (single-predictor R² is not an incremental ΔR²), and
we should say so, but the honest move is to give the range in the body and stop
quoting 23x bare. Two specific repairs:

- The ladder forces **one common geometry slope across six families** whose own partial
  correlations run −0.595 and −0.666 on Llama against −0.004 and +0.073 on Qwen.
  Pooling slopes that heterogeneous attenuates the geometry term. **Refit with
  family-varying geometry slopes and report it next to the pooled 0.017.** Cheap, and
  magnitude will almost certainly still win by a wide margin. The point is much better
  made by a specification that cannot be accused of suppressing the alternative.
- We attribute the three-block reversal wholly to block overlap, but the shape block
  also grows from 2 to 5 measures and n falls from 1,034 to 911. As written the
  attribution is asserted, not shown. Either show it or soften it.
- Put the **1,078 of 2,000** number in §5.2, not just as an appendix pointer.

### 2.4 The full-fine-tuning offset (the sharpest catch in the review)

§5.2 currently reads: "even full fine-tuning traces the same flat-then-falling shape,
4 to 9 points below the LoRA-family curve", inside a paragraph arguing that the
magnitude relation is not a recipe artifact. Read plainly, that is **a 4 to 9 point
retention gap at matched magnitude**, which is:

- larger than our 3.05-point random-direction penalty, the number we present as the
  size of a direction effect, and
- several times the 1 to 4 point per-method residuals we describe as small.

It is the largest non-magnitude effect anywhere in the paper and we currently read it
as confirmation. Limitations 2 scopes the +0.017 ceiling to the LoRA family, which is
the right scope, but the abstract's unscoped final sentence is not covered by it.

Honest handling makes the paper stronger, not weaker: at fixed magnitude, being
low-rank versus full-rank is worth 4 to 9 points, which is exactly the boundary of our
claim. Give it a paragraph. It also connects directly to Shuttleworth (rank-stabilised
high-rank LoRA resembling full fine-tuning), so one paragraph discharges two review
items at once.

### 2.5 Weight decay 0.3 provenance (needs a decision, not an experiment)

The review is right that "decay 0.3" appears once, in App. A.1, with no provenance.
**The repo answers the question and the answer is the uncomfortable one.**
`handoff/11_WEIGHT_DECAY.md` line 13 records a preliminary fast-scale sweep over
wd ∈ {0, 0.05, 0.1, 0.3} with retention 23.4 / 24.4 / 26.6 / **28.4**, and line 37
plans to extend the grid "if 0.3 is the sweet spot". So 0.3 **was** read off a decay
sweep scored on retention. The reference arm is tuned on one axis the comparators
never got.

We have three real answers and should give all three:

1. **Disclose the provenance in one sentence.** Trying to finesse this is the only way
   it becomes fatal.
2. **Knob identity adds at most ΔR² = 0.006 once F∆ is known**, so a different decay
   slides the arm along the shared curve rather than above it. We hold this already
   and do not deploy it at the operating-point battery, which is where the headline
   sentence comes from.
3. **New, and this is the strong one: rerun the battery with a different reference
   decay, from existing data.** The two grid families carry LoRA+wd at five decay
   values with full rate coverage:

   | family | wd=0 | 0.1 | 0.2 | 0.3 | 0.5 |
   |---|---|---|---|---|---|
   | frc (Llama CS grid) | 26 | 22 | 26 | 36 | 21 |
   | frm (Llama math grid) | 18 | 27 | 27 | 27 | 18 |

   So on both Llama settings we can re-issue the head-to-head with wd = 0.1 or 0.2 as
   the reference and show the verdicts do not move. That converts W2 from a disclosed
   asymmetry into a tested and immaterial one, for free. (The four sweep families carry
   wd = 0.3 only, so this is a robustness check on two settings, not all six. Say so.)

### 2.6 Tier 2, all cheap and all correct

- **Power qualifier in the abstract.** "In 25 Holm-corrected comparisons, no adapter is
  significantly better" carries no power qualifier while Table 5 puts max MDE at 49.9
  points (Qwen CS). The force of the sentence comes from the count and several of the
  25 could not have detected anything. Add the qualifier; we already have the MDE table.
- **Drop "universal" from the fragility ordering.** W = 1.000 gives p = 4.4e-4 only by
  treating six correlated families as independent raters. Four are Llama; the CS sweep
  and CS grid are the same model and task. The same agreement gives p = 0.0074 over
  four distinct model-by-task settings and p = 0.11 over two models. The result
  survives at the setting level; the word does not. This is the one place our own
  clustering discipline is not applied, which is what makes it embarrassing.
  Reviewer's own addition, worth checking before someone else does: our slopes are
  ceiling-normalised by base score, not by base minus chance. MMLU-Pro is 10-way with
  Llama base 19.0 against a ~10 floor, so 9 points of true headroom divided by 19.0.
  Renormalising could move the ordering. A few lines of code.
- **PiSSA.** Reported as a losing arm (−7.1, 90% CI [−8.96, −5.28]) from five runs at
  one unswept rate, which is the artifact §5.2 warns against, and we already exclude it
  from the geometry comparison for spanning no magnitude range. Mark the sparse rows as
  not tuning-controlled, or drop it from the equivalence table as we dropped it from
  the geometry one. Same table also issues verdicts at n=2 and n=9 on the same visual
  footing as n=157.
- **Name the near-tautology objection.** Update magnitude, KL drift and retention are
  all measures of how far the model moved from base, on a general-domain probe set.
  Our rescaling and random-direction controls are the right answer, but the objection
  is never stated, so the reader has to reconstruct our defence. One paragraph.

---

## 3. Where the review is wrong or overstated

Three places. Two of them we can refute with our own data, which is worth doing
explicitly in the revision because it turns a weakness into evidence of care.

### 3.1 Koubbi et al. is mischaracterised. Do not cite it as described.

The review asks us to cite "Koubbi, Hernandez and Boussard, mean-field LoRA
forgetting, ICML 2026" as deriving "a phase transition in forgetting with respect to
perturbation norm", a theoretical prediction of our knee.

arXiv:2402.15415 is **"The Impact of LoRA on the Emergence of Clusters in
Transformers"** (Koubbi, Boussard, Hernandez, Feb 2024). It is a mean-field
token-clustering paper. It shows clusters under a perturbed attention matrix stay
close over short horizons and diverge over long ones, depending on the magnitude of
the parameter difference. It is not about catastrophic forgetting, it measures no
retention, and I found no ICML 2026 version.

**Action: do not add it on the review's description.** If we want it, one of us reads
the current version first and we cite what it actually says, or we leave it out. A
mischaracterised citation in a paper whose selling point is careful bookkeeping would
cost more than the citation is worth.

### 3.2 The Qwen k/v limb is structurally true and inferentially wrong

The review's claim: Qwen's attention k/v matrices have only 512 output directions, so
top-256 and bottom-256 exhaust them, giving a 50/50 partition on Qwen against a
6%-tail contrast on Llama, and this is "a live candidate explanation" for our own
finding that residual geometry is about zero on both Qwen families.

**The structural half is exactly right and I verified it in our data.** For a Qwen run,
per-matrix `e_top + e_bot`:

```
k_proj   n=28  e_top=0.5616  e_bot=0.4384  sum=1.0000
v_proj   n=28  e_top=0.5096  e_bot=0.4904  sum=1.0000
q_proj   n=28  e_top=0.0990  e_bot=0.0550  sum=0.1540
```

Exactly 1.0000 for k and v. The partition is degenerate there, no argument.

**The inference does not survive our own weighting.** The adapter-level coordinates are
Frobenius-weighted over matrices, and on Qwen the k/v matrices carry almost none of
the update:

```
Qwen  : F^2 share of k/v among adapted matrices, n=60 runs
        mean 0.022, median 0.002, max 0.237
Llama : mean 0.262, median 0.244
```

Two per cent of the energy on average, two tenths of a per cent at the median. A
degenerate partition on 2% of the mass cannot drive the aggregate coordinate, so it
cannot explain the Qwen null.

**Action:** recompute the Qwen residual geometry association with k/v excluded and
report the two numbers side by side. If they agree, which the energy shares say they
will, we have killed the limb with one line and demonstrated the instrument is not
fragile in the way alleged. Cheap, and it comes entirely from `permatrix_qwen/`.
(The generator `37_geometry_battery_table.py` already notes the 512-direction fact in
its caption, so we knew; we just never showed it does not matter.)

### 3.3 "The instrument is unvalidated" overstates what is missing

We do have a validation of the *outcome's* sensitivity to direction: the
random-direction control shows same-magnitude updates in random directions land 3.05
points below the curve, so retention *can* detect a direction effect. What is missing
is narrower: a demonstration that **our three coordinates** would register such an
effect. Worth saying in the reply, and worth closing with the recovery simulation
(§5, Track B4) rather than conceding the wider version.

---

## 4. Hard constraint: what we can and cannot recompute

**No 7B adapter checkpoints survived the fleet evacuation.** Confirmed:
`handoff/41_EVACUATION_2026-07-17.md`, restated in `MISSING_EXPERIMENTS.md`,
`/scratch/cf_models` empty, `results/geo_drift/base_svd/` empty, zero
`adapter_model.safetensors` anywhere on this box. The only surviving checkpoints are
the 21 DeepSeek-284B adapters in `results/ds_adapters_evac/`, which have no retention
evals and support no retention claim.

| Review item | Recomputable from what we have? |
|---|---|
| Ladder with input-side shares | **Yes.** `ein_top_w` / `ein_bot_w` are in the adapter metric store |
| Family-varying geometry slopes | **Yes.** Reuses `corr_common.build()` |
| Qwen k/v exclusion check | **Yes.** `permatrix_qwen/`, 369 files |
| Recovery / power check for the ladder | **Yes**, by simulation on the real covariates |
| Broad-aggregate headline correlation | **Yes.** `retention_broad` is in `m1_master.csv` |
| Alternative reference decay (2.5 item 3) | **Yes**, on the two grid families |
| Bootstrap on F∆ vs spectral / Frobenius norm gap | **Yes** |
| **Cut sensitivity at 64 / 512 / 1024** | **No.** Needs `dW`, that is, the adapters |
| **Intruder dimensions, spectrum of `W0+dW`** | **No.** Same reason |

So the two items that need the checkpoints are exactly the two items the reviewer put
at the top of the geometry weakness. That is the decision in §6.

**One partial route worth considering, flagged as a proxy.** Intruder dimensions are
new leading directions of `W0+dW` that are near-orthogonal to the pretrained top
subspace, which in our recorded coordinates corresponds to large `spec` together with
low `e_top`. We already have both per matrix in `permatrix/`, and the base singular
values are re-derivable by re-downloading Llama-2-7B and redoing the 160 SVDs on this
box, a few hours of CPU and no GPU. That would give a **proxy intruder score** per
adapter that can enter the ladder. It is an approximation and would have to be labelled
one. My recommendation: hold it in reserve as a rebuttal asset, do not build the
revision on it, and prefer the real measurement in §6 Option B if compute is approved.

---

## 5. The plan

### Track A. Writing only, no compute. This is the 3.0 to 3.5 move on its own.

| # | Item | Where | Size |
|---|---|---|---|
| A1 | Add and reconcile Shuttleworth, Zhang, Huang. Rewrite §1 "still not established" and §2 "none of these measures". Reposition as converging evidence with the retention axis as our delta | §1, §2 | ~20 lines |
| A2 | Scope the abstract's last sentence. Keep the finding, bound it to the LoRA family and to the coordinates measured | abstract | 2 lines |
| A3 | Power qualifier on the 25-comparison sentence | abstract, §6 | 2 lines |
| A4 | Drop "universal" from the ordering claim, state it at the setting level | abstract, §1 contributions | 3 lines |
| A5 | Full-fine-tuning offset gets its own paragraph, tied to Shuttleworth's rank-stabilisation result | §5.2 | ~10 lines |
| A6 | Weight decay 0.3 provenance, one honest sentence plus the ΔR² ≤ 0.006 answer | §3.1 or App. A.1 | 3 lines |
| A7 | Mark PiSSA and the other sparse rows as not tuning-controlled, or drop PiSSA from the equivalence table | §5.1, Table 4 | 2 lines |
| A8 | Name the near-tautology objection and point at the controls that answer it | §5.2 | ~6 lines |
| A9 | State the 1,078 of 2,000 number in the body; "land about 1.3 points above the curve" for the rescaled adapters | §5.2 | 2 lines |
| A10 | Relabel Lee et al. from concurrent work to prior work, keep the delta sentence | §2 | 1 line |
| A11 | Broader impact paragraph: barrier-lowering positive, plus the benchmark-choice dual-use note | after Limitations | ~6 lines |
| A12 | Licences and terms for models and datasets, package versions, AI-assistance disclosure, DOIs and Anthology links for all 23 refs, archival versions for the 8 preprints that have them | App. A, references.bib | mechanical |
| A13 | Cite the base models and benchmarks (Llama-2-7B, Qwen2.5-7B, DeepSeek-V4, WikiText-103, MMLU, ARC-Challenge, TruthfulQA, lm-evaluation-harness) | throughout | mechanical |

### Track B. Statistics from the frozen store. No new training.

| # | Item | Answers | Effort |
|---|---|---|---|
| B1 | Refit the ladder with the geometry block extended to include both input-side shares and `e_bot` | W3's sharpest limb | half a day |
| B2 | Refit with family-varying geometry slopes, report next to the pooled 0.017 | W4 limb 1 | half a day |
| B3 | Alternative reference decay (0.1 / 0.2) head-to-head on frc and frm | W2 | 1 day |
| B4 | Recovery check: inject a geometry effect of known size into the real covariates, show what the ladder recovers as a function of effect size | "unvalidated instrument" | 1 day |
| B5 | Qwen residual geometry with k/v excluded, reported next to the full number | the Qwen limb of W3 | 2 hours |
| B6 | Pooled correlation under the broad aggregate, applying RQ4 to our own headline | reviewer's "nice touch", closes the loop | 1 hour |
| B7 | Bootstrap the F∆ vs spectral norm vs Frobenius gap (0.420 / 0.349 / 0.348) | turns a ranking into a result | 2 hours |
| B8 | Fragility slopes renormalised by base minus chance | Tier 2 item 8 addendum | 2 hours |
| B9 | Print the six per-family slope vectors instead of ranges in Table 6 | verifiability of the ordering | 1 hour |
| B10 | Reconcile DoRA 19.1 vs 19.2 and SC-LoRA ±1.9 vs ±1.8 across Tables 1/3/14; fix App. A.5's "160 adapted matrices" (Llama 160, Qwen 140, confirmed in the metric store); derive or drop Table 11's last column | bookkeeping credibility | half a day |

Note on B1 and B2: both are small edits to `rq1_stats/04_ladder_ci.py`, which already
carries the bootstrap machinery. Neither is a rewrite.

**Risk to state plainly:** B1 and B2 can move the number. If including the input-side
shares lifts the geometry step materially, the abstract has to change. I would rather
we find that than a reviewer, and the paper survives either way, because the finding
is then "magnitude dominates, and here is the one geometric coordinate that adds
signal", which is a more interesting paper than the current one.

### Track C. New compute. Optional, and it is the difference between 3.5 and 4.

**The geometry validation slice.** Retrain a small Llama-2-7B commonsense slice with
adapters kept this time, then compute what we cannot compute now.

- Minimal: 3 methods (LoRA+wd, MiLoRA, SC-LoRA) x 4 rates x 1 seed = 12 runs.
- Preferred: 3 methods x 7 rates x 2 seeds = 42 runs, which is enough magnitude span
  to fit a within-slice ladder.
- Cost estimate from App. A.7's ~2,000 GPU-hours over 1,035 runs: roughly 2 GPU-hours
  per run including evals, so 25 to 85 GPU-hours. On rented A100s that is a small
  amount of money and one to three days wall clock, most of it setup.

What it buys, in the order the reviewer asked for it:

1. **Cut sensitivity.** Recompute `e_top` / `e_bot` at 64, 256, 512, 1024 and show the
   coordinate and its association with retention are stable, or find that they are not.
2. **Intruder dimensions, measured properly.** Compute the spectrum of `W0+dW`, count
   new leading directions near-orthogonal to the pretrained top subspace, and enter the
   count as a ladder coordinate on the slice. This is the single item that decides
   whether W3 survives.
3. A direct bridge to Shuttleworth, measured on our own runs rather than argued.

If the intruder count adds nothing beyond magnitude, we have killed the strongest
objection to the central claim. If it adds something, we have found the one geometric
structure that matters and the paper gets better. Either outcome is publishable, and
both are better than the current position of having no answer.

---

## 6. Page budget

Step 1 of the pipeline recommended a **desk reject on length**: content ends p.9
l.729, Limitations at l.730, against an 8-page limit. The abstract is 224 words against
200. This is not negotiable and it constrains everything above.

Track A adds roughly 55 lines, about half a page. So we need about **1.5 pages of
cuts**. Candidates, in the order I would cut them:

1. §3 Methodology. Guy has already marked this "still too elaborated" twice in the
   source. A unified-protocol statement plus a pointer to App. A saves most of a page.
2. §5.1 detail into App. B, which already exists (`app:head2head`).
3. The cost and efficiency clause at the end of §5.2, which is a Table 11 pointer and
   Table 11's last column is the one the reviewer wants derived or dropped.
4. §5.3, the tightest of the four result subsections relative to its claim.

A trim also helps the reviewer's other complaint about balance: 21 appendix pages, 42
appendix pointers, and the primary supporting exhibit for **all four** research
questions currently sits outside the body.

---

## 7. Decisions I need from you

1. **Compute for Track C, yes or no, and how much.** This is the one that decides
   whether we are aiming at 3.5 or 4. My recommendation is yes at the 42-run size. If
   no, we scope the geometry claim to the three coordinates and the 256-cut explicitly
   and accept that a reviewer may still hold W3.
2. **How far to scope the headline.** "To first order, what tracks forgetting is the
   size of the update, not its geometry" is the paper's identity. My proposal is to
   keep the sentence and bound it: within the LoRA adapter family, and across the
   geometry coordinates we measure. Your call on the exact wording, since this is the
   sentence everyone quotes.
3. **How to disclose weight decay 0.3.** The repo says it was read off a retention
   sweep. I think we state that plainly and then show it does not matter three ways
   (§2.5). Confirm you want the plain version rather than a softer one, and confirm my
   reading of `handoff/11_WEIGHT_DECAY.md` matches your memory of how the choice
   actually got made.
4. **Which ARR cycle we are targeting**, so I can size Track C against the deadline.
5. **Whether to reframe §1 and §2 as converging evidence** rather than an open
   question. I think this is strictly better positioning, but it changes the opening
   paragraph of the paper and that is yours to approve.

---

## 8. Already applied

Minor, verified, no regeneration mismatches:

- "The Retention benchmarks" to "Retention benchmarks" (abstract).
- "Kalajdzievski (2024) fit" to "fits" (§2).
- Figure 1 caption now says 448 of 456 with the 8 out-of-frame runs accounted for. The
  generator's own docstring already said this; the caption did not.
- "LLaMA-2-7B" to "Llama-2-7B" (Table 13 row).
- "Qwen-2.5-7B" to "Qwen2.5-7B" across every paper-facing table, figure and generator.
  Data keys in the master CSVs are untouched, and all three affected exhibits
  regenerate byte-identically (grand table self-check passes, geometry battery and
  lrsweep tables diff clean, `fig_geometry_detect.pdf` rebuilt).
- "Metamath" in the bibliography was already correct in the current source.

**Not synced to Overleaf.** The local mirror has uncommitted changes and the sync rule
is pull first, re-apply your edits, then push. Say the word and I will do the pull and
push together with whatever we agree above.
