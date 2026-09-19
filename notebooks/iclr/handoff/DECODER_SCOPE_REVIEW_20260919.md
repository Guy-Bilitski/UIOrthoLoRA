# Review of decoder progress and compute priorities — 19 September 2026

## Current author instruction

The author asks that the experiments answer the subspace/interaction question
and that time not be spent optimizing competing adapters. Reasonable shared
settings are sufficient, with additional checks only when needed for a valid
experiment. This revision supersedes the 18-run LR sweep in the earlier plan.
The writing agent proposed that sweep; the coding agent implemented it as asked.

Reviewed research `4d7eb8bc` (including the implementation and measured pilots)
and Overleaf `c25f8bc`. The 08:07 UTC review brief reports tuning running on GPUs
2/3, not completed confirmations. This review inspected published artifacts and
source, not live GPU processes. Nothing here claims the queue has been stopped.
The operator must reconcile the current ledger before changing the queue.

## Scientific assessment

Keep the six strict-band conditions and three paired confirmation seeds. The
implementation changes the selected spectral band, holds support size and
parameter count fixed within a family, freezes the output head/backbone, and
starts at zero effective update. It measures accuracy and completion NLL on a
modern decoder. The published timing report also supports actual rotation and
confinement. This is a relevant extension of the encoder evidence, rather than
an unrelated adapter benchmark. The fresh-model reload, token masking, KV-cache
and memory fixes are useful work and should be preserved.

The extension addresses **spectral location and within-band flexibility**. It
does NOT test regularized cross-subspace interactions on a decoder. Those remain
supported only on RoBERTa, and their mechanism is not settled. The earlier
feedback specifically asks about modern-decoder interaction control as well as
tail adaptation; the new block addresses that request partially, not entirely.
Do not claim that adding a decoder automatically resolves the review or ensures
ICLR acceptance. Its value is a controlled finding, which may be a meaningful
band difference or limited sensitivity under the tested recipe, plus honest
integration of the existing evidence. The study tests whether location matters;
it must not be framed as confirming that all directions are interchangeable.

The existing temperature analysis substantially reduces the apparent special
probability-quality benefit of MIX: the post-scaling MIX-minus-NORM means are
about -0.0023 on RTE and -0.0092 on MRPC, with wide paired intervals crossing
zero. This supports a large role for confidence scaling; it neither proves
identical distributions nor settles layer allocation versus interaction as a
causal mechanism. The paper must state this, irrespective of decoder results.

## Compute decision: use a fixed common recipe

**Do not schedule further LR-search runs solely to complete the old grid.**
Preserve completed runs and their metrics. Let an in-flight short run finish and
save normally, or use the existing clean stop path if the operator judges that
useful; do not kill processes or corrupt checkpoints. Do not start another rate
because the old queue or admission code expects one.

For confirmations not yet started, use **LR 1e-3 for every band and both
families**, the already exercised timing-pilot rate. This is a reasonable
starting recipe with demonstrated numerical operation in both tail families;
it has not been established as optimal for any band. Keep 842 steps, effective
batch 16, revised microbatch 2 × accumulation 8, bf16, zero weight decay and the
same schedule and data order pairing. Do not choose the rate by which band wins.
A common rate also removes the prior per-family LR difference from the secondary
rotation comparison; the larger rotation parameter count still remains.

Before committing the full remaining matrix, use existing LR-1e-3 observations
where available. For arms without such evidence, at most one short 100-step
check per missing arm at seed 31415 is enough to check finite gradients/loss,
nonzero updates, early loss movement and active rotations. Keep the final-run
schedule horizon explicit when doing a shortened check. There are at most four
missing arms after the two tail pilots; no full 842-step tuning endpoints are
required. Do not demand a validation-accuracy gain at 100 steps or tune until one
appears. If there is a concrete numerical/optimization failure, first diagnose
it; a documented common LR reduction can be justified. Band ranking or an
unfavorable scientific outcome is not a reason to change settings. Later flat
curves or uniformly weak adaptation must limit conclusions about a null effect.

Compute the **frozen-model selection reference now**, on the same prescribed
128-example subset, and reuse step-0 NLL where compatible. Accuracy of 63–67%
after a timing pilot does not demonstrate improvement without that reference.
This costs inference, not another training job. Keep the final full-test frozen
reference and all 18 confirmations, both outcomes, the prescribed selection-only
length-cap check and the complete held-out test population. Reuse completed
checkpoints for any needed decode audit; do not train six extra full-length runs
just to supply that audit. Pin a common generation cap before test evaluation.

No required check above selects a best adapter. These checks establish that the
instrument operates and that any null comparison has an interpretable learning
context. The conclusion is conditional on this recipe, support size and budget,
not on the best possible optimization of each subspace.

## Preserve the record and change the gate correctly

The current `subspace_plan.py` insists on at least two rates and a complete
validated grid before confirmation. Add an explicit **fixed-recipe** path with
its own decision provenance; do not invent tuning completions or forge an LR
selection record. Freeze a new protocol version and record the scope change,
completed/in-flight/unused pilot runs and whether their outcomes were already
seen. The new common LR is a design choice, not a retrospective best-of-grid
selection. Existing tuning seed 31415 remains pilot evidence, never a renamed
confirmation seed. Preserve all raw files and old protocol hashes.

If the entire old grid or confirmations have already finished by the time this
instruction is read, do not repeat completed work to implement a cleaner story.
Report the actual state, retain the existing registered recipe and disclose its
selection. If confirmations have already begun, reconcile the frozen population
before any amendment rather than silently mixing old/new recipes in one table.
No additional sweep is authorized by unused budget.

## Remaining cost

Measured revised step times are 1.169 s DIAG and 3.252 s ROT128. The remaining
18 confirmations require about **9.3 GPU-hours of training**, using nine runs
per family at 842 steps. Using the slower family for every run gives 13.7 h.
The earlier full-generation assumption of 1,300 s per state is an extrapolated
allowance, not a measured full-test runtime: 19 states cost about 6.9 GPU-hours.
Keeping the prior conservative 6 h allowance for setup/NLL/reload/geometry and
25% contingency gives about **28–34 GPU-hours**, before adding already spent
tuning and any outstanding brief checks. Roughly one day on two continuously
available GPUs is a reasonable working projection, subject to actual progress
and decoding time. Preserve three seeds and full test scoring before spending
on parameter search. Count the initial OOM attempt and both pilot versions as
sunk compute; do not describe every budget term as measured.

## Interaction question and the remaining week

Do not add MIX to strict-band runs: its cross-band term is zero. NORM is NOT
zero under confinement; it would be an additional magnitude intervention, which
is outside the current location comparison. Preserve that distinction.

Complete and inspect the six-arm decoder evidence first. If the author wants a
modern-decoder claim about interaction control too, the relevant next block is
our practical UNREG/MIX/NORM construction on the SAME model/task, with three
paired seeds. It is a separate scoped decision, not part of this queue and not
an adapter leaderboard. Its penalty must actually change the intended geometry;
a limited norm-matching calibration for NORM would answer a scientific control
question and cannot honestly be replaced by arbitrary coefficients while still
claiming matched update size. No launch or new grid is requested here. Do not
add this block automatically merely because compute was saved.

The immediate writing work is to integrate calibrated results, the held-aside
band outcomes and their actual test-scoring provenance; narrow the generality
and mechanism claims; resolve the unsupported GPT-2 table uncertainties; then
add the decoder method/results. Better evidence and precise claims matter more
than maximizing the number of experiments. Positive, negative and mixed decoder
outcomes all belong in the paper; none alone establishes acceptance.

## Small corrections for the coding/writing agents

The review brief was corrected to avoid saying that question B's mechanism is
settled, that calibrated methods are equivalent, or that absolute pilot accuracy
proves adaptation. Also use 'negative log-likelihood (NLL)' rather than 'mean
likelihood' for the recorded loss.

One diagnostic-only code issue: `subspace.rotation_activity` computes
`acos((trace(R) - (q-2))/2)` and labels it a principal angle. At q=128 this is not
a general rotation-angle formula; several independent rotation planes make the
number misleading or clamp it to pi. Remove/rename that field or compute the
proper eigen-angle summary if needed. The existing distance-from-identity and
within-band off-diagonal energy are sufficient here. This does not require
retraining or invalidate the confinement/active-rotation checks.
