# Pre-confirmation findings — 19 September 2026

> **Correction, 19 September 14:30 UTC.** Section 2 below concluded that
> "training works, and it does not improve answer accuracy". That conclusion was
> drawn on a 128-example subset of the **inner selection** split, which is carved
> out of GSM8K **train**. The registered outcome is measured on the official
> **test** split, a different and harder population: the first validated
> confirmations score 0.489 there against 0.625 to 0.672 on the subset. The
> frozen model's 0.672 on the subset is therefore **not** a baseline for the test
> numbers, and the accuracy claim below is **not established on the registered
> population**. The frozen full-test reference at the same 640-token cap is
> registered and scheduled; until it lands, the sign of the accuracy effect is
> unknown. The loss, formatting, confinement and rotation observations are
> unaffected, since each compares like with like. See
> `DECODER_SUBSPACE_STATUS.md` and the 14:30 entry in `ASTRA_DISCUSSION.md`.

All six arms now have evidence at the common rate 1e-3, plus one full-length
842-step run and the frozen-model reference. This is what we know before
committing about 30 GPU-hours to the 18 confirmations. **A decision is needed.**

## 1. The measurements

Every state decoded on the same seeded 128-example inner-selection subset, same
prompts, decoding and scorer. Seed 31415 throughout, never a confirmation seed.

| State | Steps | Exact match | NLL | Update norm | `####` used | Cap hits |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN, untrained | 0 | 0.672 | 0.5255 | 0 | 29/128 | 10.2% |
| LEAD_DIAG | 100 | 0.664 | 0.4487 | 0.0127 | 61/128 | 10.2% |
| MID_DIAG | 100 | 0.625 | 0.4427 | 0.0128 | 71/128 | 4.7% |
| TAIL_DIAG | 100 | 0.625 | 0.4268 | 0.0120 | 77/128 | 2.3% |
| MID_ROT128 | 100 | 0.656 | 0.4123 | 0.0131 | 94/128 | 3.9% |
| TAIL_ROT128 | 100 | 0.672 | 0.3991 | 0.0118 | 105/128 | 1.6% |
| LEAD_ROT128 | **842** | 0.625 | **0.3634** | 0.0373 | 124/128 | 0.0% |

Every run passed P0 equivalence, exact zero insertion, band confinement to
within 3e-12 of total energy, and a reload reproducing its recorded NLL.
Rotations were active in every ROT128 arm and exactly at identity in every DIAG
arm. The full-length run's step-0 NLL is 0.5255, identical to the frozen
reference, which confirms zero insertion end to end on real weights.

## 2. What this means

**Training works, and it does not improve answer accuracy.**

- NLL falls monotonically, 0.5255 to 0.4487-0.3991 at 100 steps and 0.3634 at
  842. Most of the gain arrives by step 211 and then plateaus.
- The model learns the required answer format: the `####` marker rises from 29
  of 128 untrained to 124 of 128 at full length, and runaway generations fall
  from 10.2% to zero.
- Exact match does not move. Every state sits between 0.625 and 0.672 against a
  frozen baseline of 0.672, and the binomial standard error on 128 examples is
  about 0.043. Nothing here is distinguishable from the untrained model, and the
  one full-length arm is nominally the lowest.

The honest reading is that completion-only fine-tuning on GSM8K worked solutions
teaches this instruction-tuned checkpoint the answer *format* and the reference
solutions' *likelihood*, not better arithmetic reasoning. The model was already
near its accuracy ceiling for this decoding setup before any adaptation.

**The loss outcome does separate the arms, in an orderly way.** At the identical
100-step budget and rate: within the coefficient family the ordering is tail
0.427 < middle 0.443 < leading 0.449, within the rotation family it is tail
0.399 < middle 0.412, and rotations beat coefficients inside each band. That is
one seed per arm and the checks exist only to verify the instrument, so it is
motivation, not evidence. It is stated here because it bears on whether the
confirmations are worth their compute, and that is a disclosure, not a result.

## 3. Consequence for the registered outcomes

We registered exact match and NLL as co-primary. On this model and task:

- **Exact match is insensitive.** A band comparison on accuracy will compare six
  arms that all sit at the untrained baseline. A null there would say almost
  nothing about subspace selection, and it must not be written as showing that
  all subspaces are interchangeable.
- **NLL is sensitive and ordered.** A three-seed band comparison on NLL, read
  together with the confinement and rotation diagnostics, would be a real
  controlled result.

This is the "uniformly weak adaptation" case the scope review anticipated. It is
a scientific limitation of the model/task pair, and it is not permission to hunt
for a task where accuracy moves.

## 4. The decision

**Option A, run the 18 confirmations as registered.** About 30 GPU-hours, done
inside the window. Yields a three-seed band and rotation comparison on NLL with
full geometry, plus an honest statement that accuracy did not move for any band.
The accuracy table is still reported, with every seed, as a null.

**Option B, stop before the confirmations.** Saves the compute and reports the
decoder block as a bounded pilot: the instrument works, is strictly confined,
rotations are active, and the model does not adapt on accuracy. Weaker: no
three-seed band result at all, and the single-seed loss ordering above would
remain unconfirmed and unusable.

**Option C, change the task or the recipe to make accuracy move.** Explicitly
ruled out by the scope review, and it would be outcome-driven. Not recommended.

**Recommendation: Option A.** The loss outcome moves, is ordered, and directly
addresses subspace location, which is the question. One full-length arm cannot
establish that accuracy is flat for every band, and finding out is the study.
The cost fits. The requirement is that the write-up leads with what the evidence
supports, a band effect on prediction loss under a fixed recipe, and states
plainly that accuracy did not improve over the untrained model for any arm.

Both GPUs are idle. Nothing further will start without a decision.

## 5. Decode budget

The pooled cap-hit rate across the audited decodes is 3.9%, above the 1%
threshold, driven by the untrained reference at 10.2% against 0 to 4.7% for
trained arms. The confirmation and reference budget is therefore 640 tokens for
every arm, pinned before any test evaluation.
