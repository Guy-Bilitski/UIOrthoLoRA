# Review of the five-day experiment plan — 19 September 2026

Reviewed research `5c19f95f` and Overleaf `ad5a392`, the published evidence,
the proposed practical runner, and the original ICLR feedback. This is a
research recommendation and preparation brief, not a claim that new runs have
been authorized or launched. The current 18-run study continues unchanged.

## Assessment

Proposal A, decoder interaction control, is the right main addition. Proposal B,
one lower-rate check, is defensible as a secondary robustness analysis. However,
the current plan is not yet a complete five-day plan: it understates the time
available, calls extrapolated practical-adapter costs measured, omits important
launch details, and rejects every second task for the wrong reason.

With the additional time, I recommend one specified answer-choice task on the
same decoder, alongside A. It tests the same subspace question while making
the evaluated loss directly concern the answer. No adapter leaderboard, broad
LR search, task search or larger-model migration is needed for this plan.

The current evidence is useful but changes the paper's interpretation. The
frozen GSM8K reference scores 0.5838, while the three completed LEAD_ROT128
runs average 0.4905; their solution NLL is lower (0.4450 versus 0.5934).
These are findings for that condition, not yet every band. Shorter outputs and
greater marker use accompany the change; they do NOT establish that lost
reasoning was caused by formatting, nor that the learning rate caused it.
Worsening accuracy with improving held-out solution NLL is not by itself
evidence of statistical overfitting. Keep both outcomes and the negative result.

## Timing and budget

The official full-paper deadline is **25 September 2026, 23:59 AoE**, which is
26 September 11:59 UTC. The previous internal GPU cutoff, **24 September
morning**, already leaves about two days for final integration. Subtracting
another 48 hours from that cutoff to mandate a 22 September stop double-counts
the writing reserve. Start writing now while the GPUs run.

Source: [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

Five full days on two cards are 240 nominal GPU-hours. From this review to
24 September morning, about 220 nominal GPU-hours remain before our internal
cutoff, or roughly 175 at an 80% planning allowance. These are availability
ceilings, not a target to consume. The core work should finish earlier.

| Work | Planning allowance, GPU-hours | Basis |
|---|---:|---|
| Finish current block and approved partition | 8–10 | Published remaining estimate; refresh from live ledger |
| Practical decoder interaction block, including calibration | 15–25 | Provisional; its actual runner/penalty must be timed |
| One answer-choice subspace replication | 12–24 | Provisional; same model, short targets, no long decode |
| Six-run lower-rate sensitivity check | 5–7 | Same strict-band workload as current measured runs |
| Total, with approximately 20% contingency | about 50–80 | Planning range, not measured consumption |

Do not label the practical block's 12-hour estimate measured: the quoted unit
costs come from confined DIAG/ROT128 runs. Ambient scalers, the leading core,
projector buffers and regularization change the work and memory. Measure a
short real practical run before sealing its full execution budget. Reconcile
completed, in-flight and queued entries once when issuing the revised plan;
the latest prose and older status/export snapshots describe different times.

## 1. Complete the registered GSM8K block

Retain all six bands/families, seeds 17/42/123, LR 1e-3, 842 updates and cap
640. Finish the 19-state exploratory NLL partition and its reference-export
agreement checks. Publish the final collector/summary output, per-example
outcomes and paired contrasts. Leading-minus-tail within each family remains
the main location contrast; rotations are a separate contrast with different
parameter counts. A null accuracy difference does not establish equivalence.

The reported 0.004 accuracy precision is not established for the full study by
the published partial export. That export contains four completions and only
one paired seed for its between-band contrast. Even a narrow interval for a
completed leading-band rotation contrast would not establish precision for
all other contrasts. Three seeds remain a sensible allocation; justify this
by the study design and the value of task coverage, not a universal power claim.

## 2. Proposal A: decoder interaction control — recommended

Use the same Qwen2.5-1.5B-Instruct, GSM8K splits, 56 q/o projections, tokenizer,
prompts and scoring. Use the practical spectral construction with ambient
scalers, leading core I, tail 512 and no rotations. This is the correct
instrument: unlike strict confinement, it permits cross-subspace updates.

Compare **UNREG / MIX / NORM**, each at seeds **17, 42, 123**: nine confirmations.
All scientific settings are shared except the explicit penalty. Proposed fixed
recipe: LR 1e-3, 842 updates, effective batch 16 (microbatch 2 x accumulation
8), zero weight decay, the existing schedule/precision and generation cap 640.
Keep the existing practical initialization e=d=0.01, h=0.01, documenting that
it has a small NONZERO initial update. This is not the zero-insertion instrument.
Score that common inserted-but-untrained state once alongside the existing
frozen model; reuse the latter only after verifying identical evaluation.

Use the previously proposed MIX coefficient **1e-3 per side**, with exactly
the declared sum over modules. This is a fixed design value, not an optimized
dose. At calibration seed 31415, obtain one MIX target endpoint. Match NORM's
pooled relative total-update norm using the existing three-dose grid and at most
two deterministic refinements, using norms only. Freeze the selected dose for
all nine confirmations. Report actual confirmation mismatches; a pilot match
does not guarantee a per-seed match. No penalty search by accuracy or NLL.

Reuse those calibration runs for learning/timing checks where possible, rather
than adding redundant full endpoints. Check finite training, measurable learned
updates, reload agreement and whether the penalty changes the intended geometry.
Failure to separate geometry limits the interpretation; it is not permission to
search test results for a better coefficient.

Make **accuracy and held-out solution NLL co-primary**, with MIX-minus-NORM
the main penalty contrast. Include UNREG, the two initial references, total and
learned update norms, actual cross-block energy, and module allocation. This
tests the effect of the regularizer beyond the tested global-size control. It
does not isolate interaction suppression from the layer allocation it induces.
Do not compare these arms with strict-band arms as a matched architecture ablation.

### A real implementation blocker for A, not for the active queue

`decoder_pilot/pilot.py` defines `train` twice (currently lines 304 and 397).
Python uses the second definition, which does not call `resolve_configuration`
or the training admission check. The registered implementation above it is
shadowed. The CLI supplies `None` for scientific flags it expects the protocol
to fill; the active function does not fill them. A normal registered invocation
cannot execute the intended contract as written. The old `plan.py` also still
requires LR-selection records and labels NLL secondary.

Repair or replace this path with a fixed-recipe, three-arm protocol and an
end-to-end CPU test from CLI parsing through job construction/admission and
collector/summary. Do not fabricate an LR-selection record or inherit the old
LoRA/PiSSA population. Reuse the proven reload/memory handling in the current
subspace runner. Only then time the practical path. Its adapter algebra being
tested does not certify this CLI. The current `subspace_runner.py` queue is
separate and is unaffected by this finding.

## 3. One answer-choice replication — recommended addition for discussion

I recommend **CommonsenseQA on the same Qwen checkpoint**, using the existing
six strict-band arms and three paired seeds: 18 confirmations plus one frozen
reference. This changes the task, not the question or adapter comparison.
It gives us a decoder setting where answer accuracy and answer probabilities
refer to the same finite set of choices, without a reference rationale's style
dominating the evaluated loss. It remains a constrained-choice task, not a
second demonstration of free-generation reasoning.

Pin this one dataset and protocol before running any task outcomes, preserve
the complete GSM8K result, and report this task whatever it shows. A blanket
ban on any second task misreads the earlier warning against searching tasks
until one gives a favorable result. This is a disclosed follow-up motivated by
the metric ambiguity and limited task coverage already observed.

Concrete preparation brief:

- Use the official 9,741 training examples, with a fixed, disjoint 10% inner
  selection split. Reserve the full **1,221-example public validation split**
  for final evaluation and call it held-out validation, not the official test.
- Freeze the prompt, canonical A–E answer tokens and scorer. Verify the five
  labels' tokenization at the answer position on CPU before registration.
  Train only on the answer-label token with ordinary full-vocabulary CE;
  mask prompt and optional EOS. No rationale training and no trainable head.
- Primary outcomes: accuracy from the largest of the five choice logits and
  NLL of the correct choice after normalizing those five logits. Also retain
  full-vocabulary label NLL and total probability mass on the five choices, so
  improved output formatting is not confused with improved choice discrimination.
- Reuse tail/band size 512, rotation size 128, q/o scope, LR 1e-3, 842 updates,
  effective batch 16 and seeds 17/42/123. One fixed recipe; no per-band tuning.
  Pin source hashes, split IDs, prompt length/truncation policy and exact budget.
- Run the frozen anchor and short instrument checks before the full matrix.
  A poor scientific result is reported, not a reason to replace the dataset.
  Export all five choice logits on inner and held-out splits for cheap later
  calibration and paired-example analysis.

Dataset source and split sizes: [official dataset card](https://huggingface.co/datasets/tau/commonsense_qa/blob/main/README.md).
The training/scoring design above is our proposed controlled protocol, not the
dataset author's prescribed benchmark recipe. Do not add another interaction
matrix to this task automatically. Time it before committing; 12–24 GPU-hours
is an allowance, not a hardware measurement.

## 4. Proposal B: retain as a bounded secondary check

Six strict-band runs at **3e-4**, paired to existing seed **17**, with the same
842 updates and evaluation, are a reasonable sensitivity check. This is the
exception to avoiding LR work that now has a concrete purpose: assess whether
the location/rotation comparisons depend on this particular operating point.
It is not selecting the best rate. Pin the six runs now, publish both rates and
both outcomes, and do not extend into a grid or replace the original study.

Call it a prospectively specified FOLLOW-UP after seeing the original results,
not an independent preregistered confirmation of an unseen-test hypothesis.
One seed supports a limited sensitivity observation, not a general claim of
robustness. Preserved degradation at one lower rate would not establish its
mechanism or optimality. This block comes after the two main additions above.
Existing learning curves can already describe optimization without new training.

## 5. What happens to each review concern

| Feedback concern | Planned response | Limit that must remain explicit |
|---|---|---|
| Modern decoder evidence | Finish spectral-location study; add practical interaction study and one answer-choice replication | One compact decoder, not large-model scaling |
| Accuracy versus probability fit | GSM8K baseline + partition; answer-level choice metrics; existing encoder temperature analysis | Token NLL is not reasoning accuracy; marker/length changes do not prove mechanism |
| Is MIX just a size penalty? | Independently calibrated NORM, plus actual per-seed norm and geometry reports | Matching is approximate; describe measured errors |
| Layer concentration alternative | Show module profiles and narrow mechanism claims | Neither global NORM nor CENTER matches layer allocation; this concern is not experimentally closed |
| Modern adapter baselines | Explain the controlled question and use our strict support interventions | No claim of best PEFT performance; PiSSA/DoRA benchmarking is outside scope |
| SVD/setup cost | Reuse recorded setup, storage, memory, throughput and merge measurements | Distinguish one-time setup from per-step costs |
| Reproducibility | Fix practical CLI; specify initialization, orthogonal map, coefficients, splits and generation | Tested primitives alone are not a tested experiment pipeline |
| Historical GPT-2 comparison | Recover exact run/recipe/aggregation provenance or remove unsupported uncertainty/results | Do not invent missing settings or claim matched imported protocols |

I would not add a hurried layer-allocation controller to this core plan. It
needs a separately validated matching method. CENTER is already available as
a possible simpler-regularization control, but it does not answer the exact
layer-concentration question. If the paper insists that suppressed interaction
itself is the unique mechanism, that stronger claim requires another control;
the recommended paper makes the narrower intervention claim instead.

## Execution and writing schedule

- **19–20 September:** finish and export the current study. Repair/test the
  practical path and prepare the answer-choice protocol. Draft the existing
  calibration and held-out encoder updates concurrently.
- **20–22 September:** after the author's scope decision, run the practical
  interaction block and the single answer-choice matrix. Share only the two
  assigned GPUs; timing checks precede the new full queues.
- **22–23 September:** bounded LR follow-up, validation, final evidence exports
  and any clearly justified recovery. Target finishing core training by the
  22nd; do not make that an artificial mandatory cutoff.
- **24 September morning:** freeze final evidence. Use the remaining submission
  window for the paper, appendix, references, costs, provenance and PDF checks.

The writing work is substantial: integrate the already delivered temperature
analysis and held-out encoder band results, present the decoder's accuracy/loss
tradeoff honestly, retain the new adapters as the experimental contribution,
and replace universal capacity/mechanism claims with the measured conclusions.
Keep the approved abstract/introduction unchanged until concrete replacement
wording is reviewed. These additions strengthen relevance and interpretation;
they do not guarantee acceptance or require a favorable outcome.

**Next step for the coding agent:** prepare the corrected, timed-ready protocols
and CLI fixes now, with the tables above as the proposed scope. Keep the active
18-run queue intact. New GPU blocks remain proposals for the author's decision;
do not interpret a review request as permission to launch all optional work.
