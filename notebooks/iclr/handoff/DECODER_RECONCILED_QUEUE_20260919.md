# Reconciled queue after the scope review — 19 September 2026

Response to `DECODER_SCOPE_REVIEW_20260919.md`, which replaces the 18-run
learning-rate sweep with a fixed common recipe. Reports the actual ledger state
before any amendment, as instructed. No confirmation has been started.

## 1. What the ledger actually contains

Nothing was killed, deleted or rewritten. Every raw file and old protocol hash
is preserved.

| Run | Protocol | Rate | State | Disposition |
|---|---|---|---|---|
| TAIL_DIAG, 100 steps | timing v1 | 1e-3 | completed, validated | kept as pilot evidence |
| TAIL_ROT128, 100 steps | timing v1 | 1e-3 | **failed, CUDA OOM at step 24** | kept as the record of the memory limit |
| TAIL_DIAG, 100 steps | timing v2 | 1e-3 | completed, validated | learning check for TAIL_DIAG |
| TAIL_ROT128, 100 steps | timing v2 | 1e-3 | completed, validated | learning check for TAIL_ROT128 |
| LEAD_ROT128, 842 steps | tuning | 3e-4 | **interrupted at step 234** | superseded rate, stopped cleanly |
| LEAD_ROT128, 842 steps | tuning | 1e-3 | running | at the common rate, becomes LEAD_ROT128's learning check |
| FROZEN, selection subset | reference | none | completed | the early anchor, see section 3 |
| MID_ROT128, 100 steps | learning checks | 1e-3 | running | missing-arm check |

Sixteen entries of the old grid were **never started**. They are recorded in
`campaign_outputs_decoder_subspace_v1/tuning_queue.drained.txt` and will not be
scheduled: `LEAD_ROT128@3e-3`, all three `MID_ROT128` rates, all three
`TAIL_ROT128` rates, and all nine `DIAG` rates.

The interrupted 3e-4 run was stopped through the engine's existing clean-stop
path, not by killing a process. It saved six checkpoints and its observations
intact, and the ledger records it as interrupted with the reason. Its outcome
was seen only as a partial training curve; it was never scored on the test split
and it enters no decision.

**Outcome exposure, stated plainly.** Before the recipe changed, the only
outcomes seen were: the two tail pilots' 100-step selection NLL and subset
accuracy, and the partial curve of the interrupted 3e-4 run. No band comparison
at a common recipe, and no test-split number of any kind, was available. The
common rate 1e-3 was not chosen from these; it is the rate the pilots already
used and the review prescribed.

## 2. The reconciled plan

**Recipe.** One common learning rate, 1e-3, for every band and both families,
declared as a design choice. It is not a tuned or best-of-grid rate and is not
established as optimal for any band. The admission layer now has an explicit
fixed-recipe path so nothing has to forge a learning-rate selection record.

**Learning checks, not a search.** One short 100-step check per arm that lacks
evidence at the common rate. TAIL_DIAG and TAIL_ROT128 are covered by the v2
pilots; LEAD_ROT128 by the in-flight 842-step run. Three remain: MID_ROT128,
LEAD_DIAG and MID_DIAG. A check verifies only that the instrument operates,
namely finite loss, a nonzero update, band confinement, a reproducing reload and
active rotations where the family has them. It cannot demand an accuracy gain
and cannot change the rate.

**Unchanged.** All 18 confirmations, six arms by three paired seeds, 842 steps,
effective batch 16 as microbatch 2 by accumulation 8, both primary outcomes, the
full 1,319-example test population, and the frozen-model reference.

## 3. The frozen reference changes the reading, and it was worth demanding

Scored on the same prescribed 128-example inner-selection subset, identical
prompts, decoding and scorer.

| State | Exact match | Selection NLL | Cap hits | `####` marker used |
|---|---:|---:|---:|---:|
| FROZEN, no adapter | 0.672 | 0.525 | 10.2% | 29 / 128 |
| TAIL_DIAG, 100 steps | 0.625 | 0.427 | 2.3% | 77 / 128 |
| TAIL_ROT128, 100 steps | 0.672 | 0.399 | 1.6% | 105 / 128 |

**The 63 to 67 per cent accuracy from the pilots demonstrates no adaptation
gain.** The untrained model already scores 0.672 on this subset, and the trained
diagonal arm is below it. The review was right to insist on this number before
any accuracy claim.

What 100 steps did change is real but narrower: negative log-likelihood falls
from 0.525 to 0.399, the model learns the required answer format, with the
`####` marker rising from 29 to 105 of 128, and runaway generations fall from
10.2% to 1.6%. So the instrument learns. It has not yet moved the outcome the
study registered as primary.

**Consequence for interpretation.** If accuracy is still flat at the full 842
steps, a null band difference on accuracy will be weak evidence, and we will say
so rather than report it as showing that subspace choice does not matter. The
likelihood outcome and the geometry diagnostics would still separate the arms,
and that is the honest framing. The in-flight 842-step run is the first evidence
on whether the full schedule moves accuracy at all; its subset decode is the
next thing to inspect.

**Cap decision.** The frozen model hits the 320-token cap on 10.2% of outputs
against 1.6 to 2.3% for trained arms. A cap that truncates the reference far
more than the arms it anchors is not a fair common budget, so the confirmation
and reference budget should be 640 tokens. That is pinned before any test
evaluation and applied identically to every arm and to the reference.

## 4. Cost

Measured: 1.169 s per step for the diagonal family, 3.252 s for the rotation
family, both fitting in about 16 GiB.

| Item | GPU-hours |
|---|---:|
| 18 confirmations, 842 steps, nine runs per family | 9.3 |
| Same, charging the slower family to every run | 13.7 |
| Full test generation, 19 states | about 6.9 |
| Setup, selection NLL, reload and geometry allowance | 6 |
| With 25% contingency | **28 to 34** |
| Three short learning checks | about 0.4 |
| Already spent, including the OOM attempt and both pilot versions | about 1.2 |

Roughly one day on two continuously available GPUs. Sunk compute is named as
sunk, and not every term above is measured: the 1,300 s per full test state is
an allowance extrapolated from a 128-example decode, not a measured full-test
runtime.

## 5. What is still required before confirmations

1. Finish the three learning checks and the in-flight 842-step run.
2. Seal the fixed-recipe record over all six arms.
3. Pin the 640-token decode budget through the cap audit.
4. Re-register the scope decision against the fixed recipe and register the 18
   confirmations.

Held for the author's go-ahead, as instructed. The frozen reference finding in
section 3 is the thing worth reading before that decision, because it bears on
whether the study can support an accuracy claim at all.
