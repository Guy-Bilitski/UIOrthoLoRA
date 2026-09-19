# Review brief: what we are running, what we are not, and why

For review. Written 19 September 2026, while the decoder tuning grid is running
on the two authorized GPUs. It states the research question, maps every piece of
evidence to the part of the question it answers, names what is deliberately not
being run, and gives the schedule against the deadline.

## 1. The research question, split in two

> How does the choice of pretrained spectral subspace, and regularization of the
> interaction between subspaces, affect fine-tuning and its results?

That is two separable questions, and they need different instruments:

- **A. Subspace selection.** Does it matter *which* band of the pretrained
  spectrum an adapter is confined to, and does allowing rotation *within* that
  band change the answer?
- **B. Mixing regularization.** Does penalizing the interaction between the
  leading and trailing subspaces change what training does and what it yields?

**These cannot be answered by the same runs.** Question A needs strict
confinement to one band. Under strict confinement the cross-band interaction
term is identically zero, so a mixing penalty added to a confined arm is either
vacuous or silently changes the intervention. That is why the decoder block
answers A only, and why B is answered by the completed encoder block. Building a
bands x families x losses grid would be the obvious mistake here; we are not
doing it.

## 2. Evidence map

| Question | Instrument | Status |
|---|---|---|
| B. Mixing regularization, raw effect | 18 RoBERTa confirmations, UNREG/MIX/NORM x RTE/MRPC x 3 seeds | **Complete and frozen** |
| B. Does the benefit survive calibration? | Temperature scaling of all 18 frozen endpoints, one inner-fitted temperature per run | **Complete**, and it changes the reading |
| A. Subspace selection, encoder | 21 RoBERTa band/head confirmations, lead/mid/tail x diag/rot64 x 3 seeds | **Complete and frozen** |
| A. Does the encoder band ordering carry to a clean split? | Held-aside evaluation of all 21 frozen endpoints | **Complete** |
| A. Subspace selection, modern decoder | 6 arms, lead/mid/tail x DIAG/ROT128, Qwen2.5-1.5B-Instruct on GSM8K, 3 seeds | **Running now** |

### What the completed evidence already says

- **The mixing benefit is largely a calibration effect.** Raw held-out negative
  log-likelihood favoured MIX over NORM by 0.96 nats on RTE and 0.36 on MRPC.
  After a single inner-fitted temperature per run the gap is 0.00 on both tasks,
  within seed noise. MIX trains less over-confidently; after correcting for that
  the three arms are indistinguishable on likelihood. This weakens the practical
  case for MIX as a probability-quality method. It does not erase the geometric
  intervention, the raw-loss finding, or the difference in how the update is
  allocated across modules, and it says nothing about mechanism. This result is
  preserved as it came out.
- **The encoder band ordering does not fully carry.** Tail rotation beat tail
  diagonal on the inner split, but on the held-aside split the difference is
  -0.008, that is, gone. Band location still separates from the head-only
  control by about 0.10 accuracy. Accuracy gains coexist with worse likelihood.
- **A disclosure.** Every band run had computed aggregate scores on the locked
  split at training time and stored them in its validation record. They were
  never exported and never used for any decision, and the CPU re-scoring
  reproduces them exactly, but the provenance is disclosed rather than hidden.

So question B is answered, including a partly negative answer, and question A is
answered on an encoder. The gap that remains is **A on a modern decoder**, which
is what the reviewers asked for and what is running now.

## 3. What is running

Six arms, one per band and family, on the square query and output projections of
all 28 layers of Qwen2.5-1.5B-Instruct, fine-tuned on GSM8K with completion-only
cross-entropy.

| Band, descending singular-value index | Coefficients only | Coefficients plus partial rotations |
|---|---|---|
| Leading [0, 512) | LEAD_DIAG | LEAD_ROT128 |
| Middle [512, 1024) | MID_DIAG | MID_ROT128 |
| Tail [1024, 1536) | TAIL_DIAG | TAIL_ROT128 |

Every update is strictly confined: `Delta = U_B H V_B^T`. Coefficients start at
zero and rotations at identity, so all six arms start from the identical
effective model. No ambient scalers, no leading or complement identity core,
everything else frozen including the embeddings and the output head.

Both primary outcomes are registered in advance and reported together:
**generated-answer exact match** on all 1,319 held-aside test questions, and
**completion-token mean likelihood** of the reference solutions. Neither
substitutes for the other, and a lower token loss is not treated as evidence of
better reasoning.

The measured timing pilots confirmed the instrument does what it claims: updates
stayed inside their band to within 2.5e-12 of total energy, and within-band
off-diagonal energy was 8.3e-12 for the diagonal family against 5.3e-02 for the
rotation family. The rotations are genuinely active, which is the precondition
for the flexibility contrast to mean anything at all.

## 4. What we are deliberately not running, and why

| Not run | Why |
|---|---|
| MIX or NORM on the decoder | Strict band confinement makes the cross-band interaction term zero. The penalty would be vacuous, or it would change the intervention and stop answering question A. |
| A bands x families x losses grid | It multiplies cost without separating the two questions, and it does not fit the window. |
| CENTER, the Haar-expected shrinkage control | Prepared and tested, but it refines question B, which already has its answer. Deferred by the author's priority. |
| LoRA and PiSSA comparisons | A best-adapter claim is not our question. They would add arms without addressing subspace selection. |
| A second model or a second task | One integration, done properly, inside the window. |
| Extra penalty sweeps, per-band learning lengths, rank sweeps | Not in the registered budget, and each one is a chance to tune toward a preferred outcome. |

## 5. How the decisions are constrained

Every decision rule was sealed before the runs it governs, and each is
mechanically enforced by the admission layer rather than trusted:

- **Learning rate.** One shared rate per family, chosen on the arithmetic mean
  across that family's three bands of inner-selection likelihood at the fixed
  endpoint. Choosing on a single band would let the recipe favour one location.
  Ties take the smaller rate. Per-band preferences and any dependence of the
  band ordering on the rate are reported, never silently substituted.
- **No test-set contact before the end.** Tuning runs have generation disabled
  outright. Admission refuses a tuning job that would generate.
- **Decode budget.** Fixed by a cap audit on a shared 128-example inner subset,
  counted from raw token boundaries and never from the answer parser, never
  adjusted by which band is winning.
- **Scope.** Frozen from measured cost before any outcome was seen.
- **Checkpoint identity.** Band bounds are hashed into the checkpoint
  fingerprint, because the leading and middle diagonal arms otherwise share
  parameter names and shapes and a wrong-band reload would succeed silently.
- **Every seed reported, nulls kept.** No arm is dropped to balance a table.

## 6. Cost and schedule

Measured, not estimated: 1.169 s per optimizer step for the diagonal family and
3.252 s for the rotation family, both fitting in about 16 GiB after the
microbatch revision that the first pilot forced.

| Stage | Runs | Wall clock on two GPUs | Expected finish, UTC |
|---|---:|---:|---|
| Timing pilots | 4 | done | 19 Sep 08:00 |
| Tuning | 18 | 5.5 h | 19 Sep 13:35 |
| Decode audit at the six selected-rate endpoints | 6 | 0.5 h | 19 Sep 14:05 |
| Learning-rate and decode-budget decisions | CPU | 0.25 h | 19 Sep 14:20 |
| Confirmations | 18 | 8.5 h | 19 Sep 22:50 |
| Frozen-model reference | 1 | 0.3 h | 19 Sep 23:10 |
| Validation and evidence export | CPU | 1 h | 20 Sep 00:15 |

Projected total is 50.5 GPU-hours against a 96-hour ceiling, so all six arms are
retained and the predeclared diagonal-only fallback is not needed. The training
cutoff is the morning of 24 September, so the decoder evidence package lands
about four days early, and the final 48 hours stay reserved for analysis and
writing.

If the GPUs become intermittent, the predeclared reduction is to drop the
rotation family, keeping three bands by three seeds. That keeps question A's
location half intact and costs 29.4 GPU-hours. It is chosen from cost, never
from outcomes.

## 7. What could change the conclusions, honestly

- **Uniformly weak adaptation would make a null band difference uninformative.**
  The timing endpoints reached about 63 to 67 per cent exact match after only
  100 steps, against a frozen model that has not yet been scored, so there is
  adaptation to measure. The frozen reference run settles how much.
- **Three seeds is three seeds.** Every paired interval is labelled nominal and
  exploratory. Similar means with wide intervals will not be reported as
  equivalence.
- **A spectral-location effect includes the optimization consequences of that
  location.** This design does not isolate an abstract geometric capacity
  independent of the optimizer and the task, and the write-up should not claim
  it does.
- **Scope limits.** One 1.5B decoder, one task, query and output projections
  only. The singular value decomposition is of the instruction-tuned starting
  checkpoint, not a raw pretraining-only one. Split isolation does not establish
  that the model never saw GSM8K.

## 8. Requested review

Specifically: is the split between questions A and B the right one, is excluding
the mixing penalty from the confined decoder arms correct, and is the balanced
across-band learning-rate rule the right way to keep band location the only
primary changed factor? Manuscript integration stays with the author and the
writing agent; this document is evidence and plan only.
