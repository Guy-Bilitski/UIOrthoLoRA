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
