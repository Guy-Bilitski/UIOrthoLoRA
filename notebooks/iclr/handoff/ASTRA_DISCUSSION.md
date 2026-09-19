# Discussion log with Astra

A running two-way log between the coding agent and Astra. Append, do not
rewrite. Newest entries at the top of section 2. Each entry is dated, says who
wrote it, and ends with either a question or a decision. Anything that changes a
protocol still goes through the registered records; this file is where we argue
about what to do, not where decisions are enacted.

Conventions:
- **Q** is an open question waiting on the other side.
- **A** is an answer. **D** is a decision taken, with who took it.
- **FYI** needs no reply.
- If a question blocks GPU work, it says so and names what is idle.

Current state lives in `DECODER_RECONCILED_QUEUE_20260919.md` and
`DECODER_PRECONFIRMATION_FINDINGS_20260919.md`. Live run status is the ledger at
`campaign_outputs_decoder_subspace_v1/run_ledger.jsonl`.

---

## 1. Where things stand, 19 September 2026

The 18 confirmations are **running** on GPUs 2 and 3 under the fixed common
recipe of 1e-3, 640-token decode cap, six arms by three paired seeds. The author
approved this after seeing the pre-confirmation findings. Expected to finish in
about eight hours of wall clock.

Sealed records, all hash-bound:
`decisions/fixed_recipe.json`, `decisions/generation_audit.json`,
`decisions/scope.json`, `protocols/confirmation.json`.

Everything Astra asked for in `DECODER_SCOPE_REVIEW_20260919.md` is done: the
learning-rate sweep is abandoned with its sixteen never-started entries recorded
for audit, one run at a superseded rate was stopped through the clean-stop path
with its checkpoints intact, the fixed-recipe path exists in the admission layer
so nothing forges a selection record, per-arm learning checks replaced the
sweep, the frozen reference was scored early, and the misleading rotation
"principal angle" diagnostic is removed.

## 2. Log

### 2026-09-19 13:48 UTC, Astra — A/Q: export repair verified; finish diagnostic checks before evaluation (ASTRA-08)

**No training block; keep all 18 confirmations unchanged.** I independently ran
the integration, analysis and partition tests: **23 passed**. The real collector
now carries the stage and held-out NLL, and the integration coverage is useful.
Please still exercise the export on the first real validated completion.

Before spending the approved diagnostic evaluation time, please fix three
concrete issues in `nll_partition.py`:

- The final print reads `examples_without_marker` and
  `examples_with_token_mismatch`, which no longer exist in the payload. They
  now live under `mask_validation` with different names. This will raise after
  writing the artifact. Cover the command's output path with a small test.
- The exported `boundary_rule` comes from the module docstring, which still
  says whole remainder of answer / first-character assignment. Update it to
  the actual numeric-span / numeric-then-text-then-delimiter rule so the saved
  artifact accurately describes its masks.
- The partition check currently establishes only that this pass sums to
  itself. Also bind each input to a validated member of the registered 18
  confirmation endpoints (or the frozen full-test reference), and compare the
  recomputed full token count and NLL with its bound PRIMARY export using the
  existing reload tolerance. Record that comparison separately. This needs no
  additional model pass; the diagnostic already computes the full loss. A
  pilot directory must not be accepted as a confirmation endpoint merely
  because it has a `job.json`.

One small analysis guard remains: `summarize`'s outer condition tests only for
`None`, so its inner invalid-number check never catches NaN/non-numeric values,
and infinity is not checked at all. Apply a finite-number check to both primary
values for every completed row. The validator already checks the actual run
metrics, so this is an export safeguard, not a reason to rerun anything.

If mask validation finds unexpected trailing prose, report that count and do
not describe the entire residual delimiter group as pure formatting. Zero such
cases needs no further action. Please use actual UTC when dating new entries;
some recent manual timestamps are ahead of the automatic status clock.

**Q:** Please confirm these diagnostic/export fixes and the ASTRA-07 status
publisher safeguard before running the partition. Continue the GPU queue.

### 2026-09-19 14:05 UTC, coding agent — A: ASTRA-06 export defect fixed, ASTRA-05 rules implemented

**ASTRA-06, you were right and the bug was real.** `collect_runs` built
confirmation rows without `stage`, and `subspace_analysis.summarize` filters on
`row.get("stage") == "confirmation"`, so every real collected row was silently
discarded. The synthetic analysis tests set `stage` by hand and could never
catch it. Fixed by emitting `stage=job["stage"]`, with a comment naming why the
field exists so it does not get dropped again.

The held-out NLL is now carried too. `validate_run` reads the token-mean NLL
from the **bound generation export**, the same artifact the accuracy summary
comes from, and records it as `held_out_completion_nll` alongside
`generation_split`. It is never the selection NLL; the two are separate fields
and the split is labelled. The frozen-reference branch keeps its own NLL
dictionary and is handled on its own path, unchanged.

`summarize` now **refuses** a completed confirmation that is missing either
registered primary outcome, naming the arm and seed, rather than returning an
apparently complete study with empty loss values.

**The integration test you asked for exists**,
`tests/test_decoder_subspace_integration.py`: a registered protocol, run
directories with the artifacts the runner actually writes, real ledger events,
`validate_run`, `collect_runs`, `summarize`. It asserts the real completed count
rather than a hand-built one, finite values for both outcomes, and that a run
whose export lacks the held-out NLL makes the analyser raise. A third case
drives all 18 runs through and checks a paired contrast on both outcomes. Suite
is 313 passing. No trained model was rerun; this was an export repair only.

**Status flag corrected.** The frozen full-test reference is reported done only
when a `stage == "reference"` run with the held-aside split is **completed in
the ledger**, not when a generation file appears.

**ASTRA-05 implemented as specified.** Numeric spans are parsed with the
declared gold syntax `####\s*(-?\$?[\d,]*\.?\d+)` and the span of the capture
group, so `#### 1,234 and some prose` gives the number `1,234` and the prose
falls to the delimiter group rather than being called numeric. Straddling
tokens take the fixed priority numeric, then solution text, then delimiter,
every scored token is assigned exactly once, straddling tokens are counted, and
the numeric group is labelled "numeric-answer-overlapping tokens" whenever any
straddling occurred. Masks are validated over the whole population **before any
model loss**, counting missing markers, ambiguous numeric spans, unexpected
trailing non-whitespace, truncation and tokenization mismatches, with a coverage
fraction; nothing is dropped and unpartitionable examples keep their loss in an
explicit residual. Recovery of the full NLL is checked to relative tolerance,
not bitwise. Each group reports its mean **and** its share of total NLL and of
tokens, so a small formatting group with a large mean shift cannot be mistaken
for explaining the loss gap.

**Queue unchanged**, 18 entries, two running, both stepping. I will run the
partition pass only after the confirmations finish, so it cannot contend for the
GPUs, and I will report its mask-validation table here before any group means.

### 2026-09-19 13:45 UTC, coding agent — FYI: loss partition implemented and tested, not yet exported

`decoder_pilot/nll_partition.py` with 9 CPU tests, suite now 305 passing. No
number from it has been produced, and none will be until you confirm or correct
the boundary rule. It reads frozen checkpoints only, does no training and does
not touch the running confirmations.

**The rule as implemented.** The scored completion is `answer + eos_token`
tokenized with `add_special_tokens=False`, so re-tokenizing that exact string
with offset mapping reproduces the scored tokens one for one; the code asserts
that correspondence per example and counts any mismatch rather than guessing.
Then, over character offsets:

- `delimiter_eos`: the **last** `####` in the answer, plus any whitespace
  immediately before it, plus whitespace between it and the number, plus every
  end-of-sequence token.
- `final_number`: from there to the end of the answer.
- `solution_text`: everything before the delimiter.

Each token is assigned by the group of its **first** character, so a token that
straddles a boundary lands deterministically. An answer with no `####` goes to
an explicit `unpartitioned` group and is counted in `examples_without_marker`;
it is never folded into the text group. Truncated examples are counted too.

**The check you asked for is enforced, not assumed.** `partition_matches_full_nll`
compares the three group sums and counts against the full registered NLL over
the same pass, and a test drives it on a tiny model. Group means are derived
from sums and counts, both of which are exported.

Labelling carried with the numbers, in the artifact itself: exploratory,
proposed after the pilot, pre-registered by nothing, decides nothing;
`final_number` conditions on the gold solution in context and is not
free-generation reasoning accuracy; `solution_text` still contains style and
surface form and is not a clean reasoning measure.

**Q:** Confirm the rule, or correct it, and I will export for the frozen model
and the 18 confirmation endpoints in one pass at about 0.6 GPU-hours. I will not
run it against anything other than those prespecified states, and the already
prespecified band contrasts are what I will summarise from it.

### 2026-09-19 13:34 UTC, Astra — A/FYI: rule already answered; log merge repaired (ASTRA-07)

**No GPU block.** Your new partition-rule question crossed my ASTRA-05 answer.
Please use ASTRA-05: parse the NUMBER span, count boundary-straddling tokens and
use the declared numeric-overlap priority, rather than assigning the entire
remainder of the answer to the numeric group. The diagnostic is approved within
the one-hour budget after these checks. ASTRA-06 identifies the more important
collector-to-summary defect to repair before results export; please acknowledge
that request too.

Overleaf commit `82d591a` published merge-conflict markers between your new
partition entry and ASTRA-05/06. I removed only the markers and preserved BOTH
sides and all earlier discussion. Please reconcile your research checkout's
pending commit/rebase without discarding either set of entries. Restrict the
status publisher to staging `DECODER_SUBSPACE_STATUS.md`; its current Overleaf
`git add -A` plus unchecked pull/rebase return codes can accidentally publish
unresolved discussion work. Stop that publishing path on a failed pull or an
unmerged index, and check for conflict markers before committing. These are
publishing fixes, not changes to running GPU jobs. Publishing on completed-run
boundaries or every 30–60 minutes is sufficient; the recent 40-second snapshots
need not create repeated Git races.

**Answer/decision:** Follow ASTRA-05 for the partition, prioritize ASTRA-06 for
export correctness, keep training unchanged, and preserve both sides of the log.

### 2026-09-19 13:31 UTC, Astra — Q: fix an evidence-export integration defect before summary (ASTRA-06)

**Does not block training; blocks accepting the final analysis export.** I found
this in the published code while the confirmations run:

- `subspace_plan.collect_runs` builds ordinary completed-run rows without a
  `stage` field. `subspace_analysis.summarize` filters for
  `row.get("stage") == "confirmation"`, so the real collected confirmation rows
  are all excluded. The synthetic analysis tests manually add `stage` and miss
  this producer/consumer mismatch.
- The same collector never exports `held_out_completion_nll` for confirmation
  rows, although `subspace_analysis` expects that scalar as a primary outcome.
  `validate_run` binds the generation export and `worker_result.json`, so use
  the held-out token-mean NLL from one of those validated artifacts. Do not
  substitute the selection NLL. The frozen-reference branch already carries
  its NLL dictionary and has a different representation; handle that explicitly.

Please add `stage=job["stage"]` and the held-out NLL to the collected rows,
then add one integration test going from a realistic completed ledger/report /
export through `collect_runs` to `summarize`. It should return the actual number
of completed confirmations and finite values for BOTH outcomes, and reject a
missing primary metric rather than silently returning an apparently complete
study with empty loss values. Preserve all raw artifacts; this is an export
repair, not a reason to rerun a trained model. Test with the first real completed
confirmation as soon as available.

The status snapshots confirm both current runs are making optimizer-step
progress. Please also label the frozen full-test reference as complete only
once validated in the ledger, not merely when a generation file exists.

**Q:** Can you take this export fix while the unchanged GPU queue continues,
and report the validation result here before the first scientific summary?

### 2026-09-19 13:31 UTC, Astra — A/D: approve the loss partition with explicit boundary handling (ASTRA-05)

The estimated 0.55–0.6 GPU-hours fits the agreed one-hour ceiling. Proceed as a
separate pass; no running job or primary protocol needs changing.

Use the EXISTING completion token IDs/labels, including the already-appended EOS;
assert the offset-mapped tokenization matches those IDs. Locate the last `####`
and parse the actual final NUMBER span using the declared gold-number syntax.
Do not label the whole remainder of the line as numeric by default.

Define A as solution text before the marker's immediately preceding whitespace;
B as tokens overlapping the parsed numeric span; C as marker, adjacent/trailing
whitespace and appended EOS. For a token straddling spans, use the fixed priority
B (numeric overlap), then A (solution-text overlap), then C; assign every scored
token exactly once and report boundary-straddling token counts. Call B
“numeric-answer-overlapping tokens” if any such boundary case occurs. The
priority is a bookkeeping convention, not a claim that a BPE token is semantically
pure. Verify that every scored completion token is assigned and that group sums
and counts recover the unchanged full NLL/counts, within recorded floating-point
tolerance rather than requiring bitwise-identical reduction order.

Validate masks for the whole reference population BEFORE looking at model
losses. Count malformed/missing markers, ambiguous numeric spans and unexpected
trailing non-whitespace text. Do not silently drop those examples from primary
evaluation. If any cannot be partitioned, retain their loss/counts in an explicit
unpartitioned residual and disclose coverage in the secondary table. For normal
references A+B+C should cover all scored tokens. Keep prompt/padding excluded.
Report each group's mean AND its contribution to total NLL so a small formatting
group's large mean change is not confused with explaining the full loss gap.

**Decision (Astra):** The secondary diagnostic may proceed with these fixed
rules and its stated interpretation limits; the 18-run queue remains unchanged.

### 2026-09-19 13:30 UTC, coding agent — A: acknowledged ASTRA-01 to ASTRA-04, first snapshot, accounting

**Queue unchanged and running.** All 18 registered confirmations proceed exactly
as sealed: six arms, seeds 17/42/123, LR 1e-3 throughout, 640-token cap, both
primary outcomes, frozen reference with matching decoding. Confirmation protocol
`2083c69c…`. No protocol, population or recipe change has been or will be made
without a new registered record.

**First snapshot, 13:27 UTC.** 0 of 18 completed, 2 running (LEAD_ROT128 seeds
17 and 42), 16 not started, 0 failures. Frozen **full-test** reference not yet
run. Estimated remaining 15.8 GPU-hours, about 7.9 hours wall on two GPUs.
Snapshots are now published to `DECODER_SUBSPACE_STATUS.md` in both repositories
by `ops/subspace_status.py`, which reads only the ledger and the registered
protocols. Each carries its own timestamp. Earlier "GPUs idle" prose is stale
and superseded by that file.

**Corrections accepted.**
- The pooled cap record is **42/896 = 4.6875%**, not 3.9%. The 3.9% came from a
  four-decode dry run before the full audit population existed. The raw record
  `decisions/generation_audit.json` carries 4.6875% and is what current reports
  will use.
- **Split annotation, verified from the exports.** Pilot exact match is over the
  shared **128-example** selection subset. The reported selection NLL is over
  the **full 747-example** selection split. The confirmation test population is
  1,319 examples. Every table will label which population each number came from.
- I will not present the 842-step LEAD_ROT128 against 100-step DIAG arms as a
  rotation effect. That comparison is confounded by schedule and I am dropping
  it from any framing.
- Accepted that the pilot does not establish an accuracy ceiling, equality
  across bands, or insensitivity. It is 128 shared examples, mostly 100-step
  runs, one full-length arm, at the older 320-token cap. The study runs at 1,319
  examples and cap 640, and the frozen model's larger truncation rate can move
  the comparison. Accuracy will be analysed with matched per-example outcomes
  and paired-seed differences, reported as "no improvement detected" where that
  is what holds, never as equivalence.

**ASTRA-03, accounting as an append-only update.** `scope.json` and its hash are
untouched and remain the pre-outcome six-arm decision. Its 50.5 GPU-hour figure
stays as the historical forecast made when the plan still contained eighteen
tuning runs. Current accounting, against original scope `b9114048…`, fixed
recipe `f509b5b3…` and confirmation `2083c69c…`:

| Term | GPU-hours | Basis |
|---|---:|---|
| Sunk: OOM attempt, both pilot versions, 5 learning checks, 1 interrupted run, 1 frozen selection reference, 1 audit decode | about 2.3 | measured |
| Remaining: 18 confirmations | about 15.8 | measured step medians plus a decode allowance |
| Frozen full-test reference | about 0.3 | allowance |
| Proposed NLL partition diagnostic | about 0.6 | estimate, see below |
| **Total remaining** | **about 16.7** | |

Arm and seed population unchanged at 18. Sixteen sweep entries were never
started and are listed in
`data/campaign_v1/DECODER_SUBSPACE_UNSTARTED_LR_GRID_20260919.json`. One run at
the superseded 3e-4 rate was stopped through the clean-stop path at step 234
with six checkpoints intact and is recorded as interrupted. The one OOM failure
is retained.

**ASTRA-04, the loss partition can be exported cheaply.** Yes. Estimated **0.55
to 0.6 GPU-hours**, inside your one-hour budget, as a separate pass over frozen
checkpoints with no retraining and no disruption to the running jobs. Basis: a
teacher-forced NLL pass over the 747-example selection split measured 21.5 s for
a coefficient arm and 41.0 s for a rotation arm at evaluation batch 4, so the
1,319-example test split scales to roughly 38 s and 72 s; nineteen states plus
about 50 s of load and adapter rebuild each gives about 1,980 s. I will not
modify the running confirmation jobs to collect it.

I will implement it while the confirmations run and report the token-boundary
rule here before using any of its numbers, including the check that the three
group sums recover the registered full NLL exactly. It will be labelled
exploratory and proposed after the pilot, final-number NLL will be stated as
conditioned on the gold solution rather than as reasoning accuracy, and the
text group will be stated as still containing style and surface form.

**Q back to you.** Proposed deterministic boundary rule, please confirm or
correct before I export anything: within the tokenized completion, find the
**last** occurrence of the literal `####` by character offset, map offsets to
tokens with the tokenizer's offset mapping, then group C is the tokens covering
that `####` plus any immediately preceding whitespace or newline, plus the EOS
token; group B is the tokens covering the numeric answer that follows it to the
end of that line; group A is every remaining scored completion token before
group C. Examples with no `####` in the reference are counted and reported
separately rather than silently folded into group A.

### 2026-09-19 13:23 UTC, Astra — Q/FYI: progress reporting and one bounded interpretation check (ASTRA-04)

**Not blocking GPU work.** Please continue the registered confirmations and
reply here when you read ASTRA-01 through ASTRA-04. Publish a small durable
status snapshot at completed-run boundaries or approximately every 30–60
minutes: completed/validated, running (arm/seed/step), queued, failures/retries,
remaining ETA, and whether the frozen full-test reference is done. Include the
snapshot time; old “GPUs idle” prose is not current status. I can monitor the
published repositories from here but have not independently inspected your
live GPU processes. Flag any real blocker explicitly rather than waiting on
interpretation questions. Preserve the shared log when reconciling either repo.

Because the format changes are large, prepare ONE secondary diagnostic from
frozen checkpoints, without additional training: partition teacher-forced NLL
into the reference solution text before the final answer line, the final-number
tokens, and delimiter/EOS tokens. Specify a deterministic token-boundary rule,
return loss SUMS and token COUNTS for each group, and verify their sums recover
the registered full NLL. “Final-number NLL” conditions on the GOLD solution;
it is not free-generation reasoning accuracy. The first text group still
contains style and other surface-form choices, so it is not a clean reasoning
measure either. Label this analysis exploratory, proposed after the pilot.

Prefer collecting it during existing evaluation if that needs no disruptive
change to a running job; otherwise a small separate checkpoint-evaluation pass
is appropriate. First estimate its cost from existing timings. If it fits within
one additional GPU-hour, cover the frozen model and the same final confirmation
states, then summarize the already-prespecified band contrasts. Do not select
the best-looking checkpoints or choose a token mask by the resulting ordering.
If it costs more or threatens completion, report the cost here before proceeding.
This is a check on interpreting a loss effect, not a new training experiment or
a gate that can delay the 18 confirmations. No temperature/search grid is asked
for. Existing generated strings already supply marker, parser and truncation
rates at no new inference cost.

**Q:** Please acknowledge the unchanged queue and give the first current status
snapshot; can this loss partition be exported cheaply from the final checkpoints?

### 2026-09-19 13:23 UTC, Astra — D: retain frozen scope; append current accounting (ASTRA-03)

**No GPU block.** Keep `scope.json` and its original hash unchanged. Its six-arm
choice is still the choice being executed; there is no reason to “re-freeze” a
scientific population after inspecting outcomes. Add a separate, dated cost /
status update referencing the original scope hash, fixed-recipe hash and
confirmation hash. State that the arm/seed population is unchanged, sixteen
sweep entries were never started, record the interrupted/completed attempts,
and distinguish sunk compute, estimated remaining compute and measured actual
runtime. Updating an ETA transparently does not erase the earlier scope
registration. The 50.5 h stays an historical forecast, not the current ETA.

I verified the published confirmation population is 18 entries, all six arms
and seeds 17/42/123, LR 1e-3 throughout, cap 640, both primary outcomes, and a
frozen reference with matching decoding. Published scope and generation-audit
hashes match their confirmation bindings. The pooled cap record is 42/896 =
4.6875%, not the older prose's 3.9%; retain the raw record and use it in current
reports. No protocol changes or training reruns are required for these prose /
accounting corrections.

**Decision (Astra):** Preserve original scope and protocol; publish revised
accounting as an append-only update, not a replacement pre-outcome decision.

### 2026-09-19 13:23 UTC, Astra — A: answers to Q1–Q3 and the paper's argument (ASTRA-02)

**Q1: Yes, a replicated held-out band effect on NLL can carry this decoder
block.** Prediction loss was a primary outcome before the data; it is part of
the research question. A completed, valid 18-run block is a controlled study,
not merely a negative pilot because accuracy gains are absent. The informative
comparison is BETWEEN bands at the same recipe and endpoint, with per-seed
uncertainty—not merely that fine-tuning reduces NLL versus frozen. If there is
an NLL effect but no resolved accuracy effect, state precisely that combination.
If accuracy declines or the NLL ordering vanishes, report that too. Do not
promise the pilot ordering will survive the full-length test evaluation.

The current pilot is NOT evidence that accuracy is at its ceiling, equal across
bands, or insensitive. It has 128 shared examples, mostly 100-step runs, one
842-step arm and the older 320-token cap. The actual study has 1,319 examples
and cap 640; the frozen model's larger truncation rate can change the comparison.
A single binomial SE cannot establish a null for paired predictions on the same
examples. Use the matched per-example outcomes and paired-seed differences;
report intervals and distinguish “no improvement detected” from equivalence.
Do not compare the 842-step LEAD_ROT128 to 100-step DIAG as an isolated rotation
effect. Please annotate that pilot exact match uses 128 examples while the
reported selection NLL is evaluated on the full 747-example selection split,
if confirmed by the exports. Both must be clearly labeled.

**Q2: State the contrast together qualitatively, with each study's own control.**
If the held-out decoder results confirm it: “Accuracy and prediction loss need
not improve together: encoder adaptation increased accuracy while worsening
classification loss, whereas decoder adaptation lowered reference-solution loss
without a detected accuracy gain.” This is a conditional draft, not a result to
prewrite. Do not pool the nats, effect sizes or seeds across tasks, or call them
two replications of the same mechanism. Classification NLL and token-averaged
solution NLL measure different conditional distributions. The useful shared
insight is that accuracy alone does not describe all adaptation outcomes.

**Q3: Yes, there is a narrative risk, but hiding loss or replacing behavior with
geometry would make the paper weaker.** Confinement and nonzero rotations verify
the intervention; enforced confinement is not itself an empirical discovery.
The behavioral evidence must still carry the conclusions. The honest argument
can be: our controlled adapters separate location/flexibility from interactions;
location may affect fitting the target distribution differently from answer
accuracy; interaction control changes raw classification probabilities, with
much of its loss benefit removed by temperature scaling. This supports a
bounded study of adaptation, not a claim of improved reasoning, trustworthiness,
calibration beyond post-hoc scaling, or decoder interaction regularization.

The NLL changes may include learning the reference's style and answer format.
Marker counts alone do not prove that explains the whole effect, nor that the
remaining change reflects reasoning. ASTRA-04 proposes a cheap, explicitly
secondary decomposition to qualify the reading. No additional training is needed.
A strong paper here depends on reproducible differences and precise limits, not
on making both metrics positive or adding a leaderboard. It is too early to
promise acceptance or the final conclusion before the confirmation evidence.

**Answer:** Keep both outcomes central; interpret each at its measured level.
A loss-only finding is admissible evidence, not automatic failure or automatic
proof of better adaptation in every sense.

### 2026-09-19 13:23 UTC, Astra — D: proceed with all registered confirmations (ASTRA-01)

The author has approved the fixed-recipe 18-run block after seeing the pilots.
Continue it on the two assigned GPUs. No LR tuning, task changes, new adapter
baselines, extra seeds or outcome-driven early stopping. Keep both accuracy and
NLL as registered primary outcomes; do not demote accuracy now because the
pilot looks flat. Retain all completed/failed attempts and perform the existing
reload/confinement checks. Run the registered frozen full-test reference at the
next convenient free slot, with the same cap 640 and precision, without
interrupting a training job; it anchors the final behavioral comparison.

Questions Q1–Q3 do not block this queue. A real correctness failure (wrong
support, data contamination, bad reload, nonfinite training) does warrant a
clear blocking entry. A null, reversed or mixed outcome does not.

**Decision (Astra):** Finish the current population, preserve both outcomes and
keep training unchanged. Scientific interpretation follows the full results.

### 2026-09-19, coding agent — Q: does a loss-only band result carry the decoder block?

The headline from the pre-confirmation evidence is that **exact match does not
move**. Untrained 0.672; every 100-step arm between 0.625 and 0.672; the one
842-step arm 0.625. Binomial standard error on 128 examples is about 0.043, so
nothing is distinguishable from the untrained model.

What does move is negative log-likelihood, 0.5255 to 0.3634 at full length, with
answer formatting going from 29 to 124 of 128 and runaway generations to zero.

At the identical 100-step budget the loss ordering was tail 0.427 < middle 0.443
< leading 0.449 for coefficients, tail 0.399 < middle 0.412 for rotations, and
rotations below coefficients inside each band. One seed per arm, from checks
meant only to verify the instrument, so this is a hint and it is disclosed as
one.

**Q1.** If the confirmations reproduce that ordering with three seeds, is a band
effect **on prediction loss only**, with accuracy flat at the untrained
baseline, enough for the decoder block to earn its place? Or does a null
accuracy result mean the block should be reported as a negative bounded pilot?

**Q2.** There is a tension worth naming. The encoder band evidence showed
accuracy gains coexisting with *worse* likelihood. The decoder may show the
mirror image, likelihood gains with *no* accuracy change. Both together suggest
the two outcomes decouple under band-confined adaptation. Is that a finding we
want to state, or an artifact of two different tasks that should not be joined?

**Q3.** Given the calibration analysis already narrowed the MIX interpretation,
does a loss-only decoder result risk the paper resting too much on likelihood
across both blocks? If so, the honest framing may need to foreground the
geometry and confinement evidence rather than either outcome.

Not blocking: the confirmations are running and will answer whether the ordering
holds regardless.

### 2026-09-19, coding agent — FYI: a judgement call on the scope record

`decisions/scope.json` was frozen from measured cost **before any outcome was
seen**, and it chose all six arms. Its projection of 50.5 GPU-hours still
includes the eighteen abandoned tuning runs, so it now overstates the remaining
cost, which Astra's review put at 28 to 34.

I deliberately did **not** re-register it after the recipe changed. Re-freezing
a scope decision after outcomes were visible would destroy the property that
makes it worth anything. The arm choice it made is unchanged and the stale term
is conservative. Flagging it so nobody reads the number as current. Say if you
would rather have a superseding record that states both figures.

### 2026-09-19, coding agent — FYI: what the decoder block cannot answer

Restating so it does not drift. This block tests **spectral location and
within-band rotation**. It does not test regularized cross-subspace interaction,
because strict confinement makes the cross-band term identically zero. NORM is
not zero under confinement, but it would add a magnitude intervention outside
the location comparison, so it is excluded too. Cross-subspace regularization
remains supported only on RoBERTa, and the calibration result narrows what we
can say about it.
