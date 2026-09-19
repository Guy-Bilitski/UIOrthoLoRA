# Five-day plan for the two assigned GPUs — 19 September 2026

The author has confirmed **two GPUs for five days in a row**. This document
re-plans against that, costs every option from measurements taken in this
campaign rather than estimates, and proposes an order. **Nothing below beyond
the already-registered block is launched.** The scope review is explicit that no
additional block is authorized by unused budget, so each proposal needs Astra's
assessment and the author's approval before registration.

## 1. The budget is smaller than 240 GPU-hours

Five days times two cards is 240 GPU-hours, but the five days end on
**24 September**, which is the existing writing cutoff, and the plan reserves the
final 48 hours for analysis and writing. So:

| | |
|---|---:|
| Nominal, 5 days x 2 GPUs | 240 GPU-h |
| Usable before the 22 September writing reserve | about 124 GPU-h |
| At a realistic 80% duty cycle | **about 100 GPU-h** |
| Already committed: 8 remaining confirmations + the loss partition | 8 GPU-h |
| **Free for new work** | **about 92 GPU-h** |

The binding constraint is the deadline, not the hardware. Any new block must
finish with enough margin to be validated, exported and written up.

## 2. Measured unit costs

From this campaign, not extrapolated:

| Unit | Cost |
|---|---:|
| 842-step run, coefficient arm, including the full 1,319-example test decode | 0.62 h |
| 842-step run, rotation arm, including the full test decode | 1.10 h |
| Full test decode alone | 0.36 h |
| 100-step learning check with the subset decode | 0.15 h |
| Selection-split evaluation pass | 0.01 h |

## 3. Committed work, already registered

| Item | Status |
|---|---|
| 18 subspace confirmations | 10 complete, 8 to run, about 7.4 GPU-h |
| Frozen reference, selection subset and full test | complete |
| Exploratory loss partition over 19 states | approved, about 0.6 GPU-h, runs after the block |

Finishing tonight. Nothing about this changes.

## 4. Proposals, in recommended order

### Proposal A: the interaction block on the same decoder and task

**This is the gap the paper actually has.** Interaction regularization is
supported only on RoBERTa, and the calibration analysis showed most of MIX's
apparent benefit there was confidence scaling. The scope review names this as
"the relevant next block" and requires it to be a separate scoped decision.

Strict band confinement cannot test it, because the cross-band term is
identically zero. This block therefore uses the **practical, non-confined**
spectral adapter already implemented and tested in `decoder_pilot/adapters.py`:
UNREG, MIX and NORM with the leading identity core, tail 512 and ambient
scalers, on the same Qwen2.5-1.5B-Instruct and GSM8K, same splits, prompts,
decoding and scorer, three paired seeds.

NORM must be norm-matched to MIX rather than given an arbitrary coefficient,
otherwise a matched-update-size claim cannot honestly be made. The existing
CENTER-style matching rule and the calibration machinery already implement this.

| Stage | Runs | Cost |
|---|---:|---:|
| Learning checks, one per arm at the common rate | 3 | 0.5 h |
| MIX target endpoint plus the NORM grid and at most two refinements | 4-6 | 2.0 h |
| Confirmations, 3 arms x 3 seeds, full test decode | 9 | 9.5 h |
| **Total** | **16-18** | **about 12 GPU-h** |

Deliverable: the first modern-decoder evidence on interaction regularization,
with both registered outcomes and the same geometry diagnostics, which can be
read directly against the RoBERTa result and its calibration caveat.

### Proposal B: a registered learning-rate sensitivity probe

Our fixed recipe costs about nine accuracy points against the untrained model.
That is a finding, but it leaves one question open that a reviewer will ask: is
the band comparison, and any null in it, an artifact of an operating point that
over-fits the reference solutions?

A **pre-registered** probe answers it: all six arms at a single lower rate of
3e-4, one seed, reported whatever it shows. It is a sensitivity check, not a
search, and it cannot change the registered recipe or the 18-run population.

| | Runs | Cost |
|---|---:|---:|
| Six arms, one seed, full test decode | 6 | about 5 GPU-h |

If the degradation disappears at the lower rate, that is an important limitation
to state. If it persists, the main block's reading is considerably strengthened.

### Proposal C: nothing else

Explicitly **not** proposed, and I would advise against each even with capacity:

- **More seeds on the subspace block.** The paired contrasts already resolve to
  about 0.004 in accuracy, seventeen times finer than the encoder block. More
  seeds buy almost nothing.
- **A larger model.** Qwen2.5-3B would need a new sealed download, a new SVD
  cache and a microbatch of 1, and float32 masters plus the spectral buffers sit
  uncomfortably close to 24 GiB. It is days of work and real risk against a
  deadline that coincides with the hardware window.
- **A second task.** The scope review warns specifically against task hunting,
  and a weak or mixed result is a scientific limitation rather than licence to
  look elsewhere.
- **Any learning-rate sweep or penalty grid** beyond Proposal B's single probe.

## 5. Proposed schedule

| When | What | Cards |
|---|---|---|
| 19 Sep evening | Finish the 8 remaining confirmations | both |
| 19 Sep late | Loss partition over 19 states, then the evidence export | one |
| 20 Sep | Proposal A if approved: checks, norm calibration, then confirmations | both |
| 21 Sep | Proposal A finishes; Proposal B if approved | both |
| 22 Sep morning | All GPU work stops; validation and export complete | none |
| 22-24 Sep | Analysis and writing only | none |

Total GPU demand if both proposals are approved is about **25 GPU-hours against
roughly 92 available**, so the schedule has margin of more than three to one.
That margin is deliberate: it absorbs a failed run, a memory surprise on a
different adapter shape, or a decision to stop early, without touching the
writing reserve.

## 6. What I need

1. Astra's assessment of Proposals A and B, in particular whether the
   non-confined practical adapter is the right instrument for A and whether B is
   a defensible sensitivity check rather than a disguised search.
2. The author's approval before anything is registered.

Until both, only the committed block in section 3 runs.
