> **Latest author clarification — avoid LR sweeps:** Read
> `DECODER_SCOPE_REVIEW_20260919.md`. Keep the six subspace conditions and three
> seeds; use a reasonable fixed common recipe with brief learning checks.
> The old 18-run LR grid below is superseded for pending work. Preserve all
> completed work and reconcile the live ledger before changing the queue.

# Decoder subspace study — reviewed plan, 19 September 2026

## Question and scope

Does changing the selected spectral subspace of the starting checkpoint change
adaptation accuracy or prediction loss? Does allowing rotations within that
subspace change the answer? These are the new decoder study's questions.

Use our adapters as controlled instruments. **No LoRA, PiSSA, DoRA, OFT, CENTER,
new encoder training, second decoder, or second task in this queue.** Preserve
existing implementations. This plan supersedes the active run matrix, commands,
cost estimates and fallback in `DECODER_PILOT_DESIGN_20260919.md` and earlier
handoffs. Their descriptions of completed preparation remain historical records.

The decoder block does not establish a cross-subspace interaction-penalty result.
That question has its own RoBERTa experiment. Strict confinement makes the
cross-band interaction penalty zero; adding MIX/NORM to this block would either
be vacuous or change the intervention. Do not build a bands × families × losses
grid. Accuracy and task loss are both outcomes, not interchangeable measures.

## Model, task and intervention

Keep the prepared **Qwen2.5-1.5B-Instruct / GSM8K** inputs. This gives a decoder
with an existing solution-generation interface and avoids spending the week on
another integration. It is a bounded 1.5B decoder extension, not evidence at 7B+
scale. The SVD is of this starting, instruction-tuned checkpoint; do not describe
it as the raw pretraining-only checkpoint. Do not switch model/task after seeing
which gives a preferred band ordering. Prior model exposure to GSM8K is not
established by our split isolation, so do not claim contamination-free reasoning.

Adapt q_proj and o_proj in all 28 layers: 56 square 1536 × 1536 matrices. Freeze
all other parameters, including embeddings and the language-model output head.
This scope gives equal spectral dimensions across modules and reuses the sealed
SVD cache. It does not justify a claim about every projection or all model weights.

| Band, descending singular-value indices | Coefficients only | Coefficients plus partial rotations |
|---|---|---|
| Leading [0, 512) | LEAD_DIAG | LEAD_ROT128 |
| Middle [512, 1024) | MID_DIAG | MID_ROT128 |
| Tail [1024, 1536) | TAIL_DIAG | TAIL_ROT128 |

The planned confirmation matrix is **6 arms × seeds {17, 42, 123} = 18 runs**.
Use q = 128 for rotations on the last 128 directions of EACH band, preserving
the earlier rotated fraction 64/256. Coefficients are signed additive updates,
not relative multipliers of singular values. Every update is strictly
`Delta = U_B H V_B^T`. DIAG has diagonal H; ROT128 rotates the designated sub-block
on both sides. No ambient scalers and no additive leading/complement identity
core. Initialize coefficients to zero and rotations to identity. Every arm must
start at the same effective pretrained model.

Within a family, band comparisons have the same parameter count, support size,
initial state and optimization recipe. The rotation family has more parameters
than DIAG, so its comparison tests added flexibility, not parameter efficiency.
From the current parameterization the expected trainable counts are 28,672
(DIAG) and 1,863,680 (ROT128); verify actual inventories before registration.
Partial rotations do not provide a free core over the entire 512-direction band.
If the selected family learning rates differ, DIAG-versus-ROT also differs in
that recipe component; do not attribute that contrast solely to rotation. The
primary location comparisons remain matched within each family.

Evaluate the frozen starting decoder once with identical prompts, NLL and
answer scoring. This costs inference only and anchors the amount of adaptation.
There is no new classifier and no separately tuned baseline method.

## Shared training recipe and bounded tuning

Reuse pinned model revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and GSM8K
revision `740312add88f781978c0658806c59bc2815b9866`. The prepared split is 6,726
training / 747 inner-selection / 1,319 official test examples, split seed 271828.
Use worked solutions as completion-only next-token cross-entropy targets; mask
prompts and padding. Keep the existing chat prompt and answer parser fixed.

Start with max_length 640, effective batch 16, AdamW with zero weight decay,
linear decay to zero, zero warmup and gradient clipping at 1.0. Float32 adapter
masters and the existing bf16-autocast path are the starting implementation.
Use **842 optimizer steps**, approximately two passes, with the same paired
batch stream across arms; record actual examples/tokens rather than claiming
exactly two epochs. No task-specific head, rank sweep or per-band training length.
Choose the same microbatch/accumulation settings for all arms after memory timing;
reduce microbatch and increase accumulation together if necessary. Apply any
required gradient checkpointing consistently across arms.

Tuning is necessary here to avoid testing a single arbitrary learning rate:

1. Before tuning, run two 100-step timing/implementation pilots, one tail DIAG
   and one tail ROT128, at seed 31415 and LR 1e-3. They do not select a band or
   supply confirmation evidence. Measure the full pipeline, not just a step.
2. The full tuning grid is **{3e-4, 1e-3, 3e-3} × 3 bands × 2 families = 18
   runs**, each at seed 31415 and the same 842-step endpoint. No test evaluation
   or full generation at every tuning checkpoint. Retain all curves and failures.
3. For EACH family, choose ONE shared LR minimizing the arithmetic mean of the
   three bands' inner-selection token NLL at step 842. Exact ties choose the
   smaller LR. This avoids choosing the recipe on the leading or tail band
   alone and keeps band location the only primary changed training factor.
4. Show the per-band LR sensitivity from this grid. If rankings depend on LR,
   report that dependence. Shared-LR confirmations estimate performance under
   a balanced common recipe; they do not prove the best attainable capacity of
   each subspace. Do not hide a band-specific preferred LR or silently retune it.
5. If the entire grid is numerically invalid or updates do not learn because of
   an implementation fault, fix it before confirmation and retain the record.
   If it simply produces weak adaptation or mixed accuracy/loss effects, that is
   a scientific limitation, not permission to hunt for a favorable task or result.
   Any necessary recipe/grid revision is written down before confirmations and
   applied symmetrically; no open-ended sweep is included in the budget.

Keep full-selection NLL at steps 0, 211, 421, 632 and 842 and the training loss
trajectory. The **fixed 842-step endpoint is primary**. Selected checkpoints can
be retained, but do not decode and compare extra test endpoints by default.
Pilot generation uses a fixed, seeded 128-example subset of INNER selection,
shared across arms: the frozen model, the two timing endpoints, and the six
endpoints at the selected family LRs. This checks formatting, length and an
accuracy learning signal without multiplying full autoregressive evaluations.
A lack of accuracy improvement is reported, not used to drop an arm.

Greedy decoding starts with max_new_tokens 320. Before confirmation, if more
than 1% of outputs across this prescribed selection audit hit the cap, use 640
for EVERY arm and update the measured budget. Audit the 640 cap too and disclose
remaining truncation. Count cap hits from raw token/EOS boundaries, not from the
answer parser. Never adjust decoding by test accuracy or by which band wins.
Use no external calculator. Save exact prompts, all generated answers and parser
fallback/format-failure rates; do not exclude failed outputs.

## Evidence and interpretation

After the recipe and run population are frozen, evaluate the frozen model and
all 18 confirmation endpoints on all 1,319 test examples. No test scoring during
tuning. Primary outputs are generated final-answer exact-match accuracy and
completion-token mean NLL on reference solutions. Store per-example loss sums,
token counts and answer correctness with stable IDs so both aggregate outcomes
can be independently reconstructed. Reference-solution NLL is not calibration
of the final answer and does not establish improved reasoning representations.

Report each seed, mean/SD and paired band differences. Leading minus tail within
each family is the primary location contrast for both outcomes; middle-versus-
leading/tail and within-band rotation differences complete the descriptive view.
Label any paired n=3 confidence intervals as nominal and exploratory; do not
pool examples or six arms as independent training seeds. Similar means with
wide intervals do not establish equivalent accuracy, loss or adaptation capacity.

Read these together with learning curves, update Frobenius norms, strict support
checks and within-band off-diagonal energy. With zero coefficients, initial
rotation gradients are zero by construction; verify they become active once
coefficients move. If rotations never make a meaningful off-diagonal update,
ROT-versus-DIAG cannot support an empirical claim about effective rotation.
Off-band energy near zero is an enforcement check, not a discovery. Norms and
layer allocation are diagnostics; no additional norm-matching experiment is
included. Any spectral-location effect can include the optimization/sensitivity
consequences of that location; this study does not isolate an abstract geometric
capacity independent of the optimizer and task.

Useful evidence can be a reproducible band difference, a loss/accuracy tradeoff,
or limited band sensitivity while meaningful adaptation occurs. Uniformly weak
adaptation makes a null band difference less informative. Report it honestly.

## Compute budget and reduction rule

The available resource is **two assigned RTX 3090-class 24 GB GPUs**, subject to
confirming the actual current devices. No GPU throughput for this new family is
measured yet. The older 28–30 GPU-hour estimate was for a different study and
must not be reused as a measurement or promise.

Use one independent run per GPU. Our planning ceiling is **96 GPU-hours total,
including a 25% contingency**, approximately 48 hours on two continuously
available GPUs. This is a proposed scheduling ceiling, not an assertion that
continuous access has been allocated. Reserve an additional day for code/pilot
work and preserve the final 48 hours for analysis/writing. The working training
cutoff is the morning of 24 September from the existing plan; confirm the actual
paper deadline before launching. Reduce the available budget if allocation is
intermittent or starts late.

Full study: 18 tuning + 18 confirmation runs × 842 steps, plus two timing
pilots × 100 = **30,512 optimizer steps**. Full test generation is 19 model
states including frozen. Budget selection NLL, pilot/subset decoding, SVD/setup,
geometry, serialization and fresh-process reloads separately.

For illustration only, if both families average 5 s/step, the full training is
42.4 GPU-hours. With 30 min per full test generation (9.5 h total) and 12 h for
all other overhead, the 25%-buffered total is about 80 GPU-hours. At 8 s/step the
same assumptions give about 112 GPU-hours and fail the ceiling. These are
scenarios, not measurements. Replace every term with timings on the assigned
GPUs, using the slower family where appropriate and allowing for long decoding.

Before the tuning/confirmation population is frozen, use this reduction rule:

- If the full measured projection fits the smaller of the 96 GPU-hour ceiling
  and the time available before the cutoff, keep all six arms.
- Otherwise retain **all three DIAG bands × three seeds = 9 confirmations**,
  plus all three LRs × three DIAG bands = 9 tuning runs. This keeps the direct
  location question and drops the secondary flexibility question. Use the same
  training length and full test evaluation; do not replace seeds with one-seed
  coverage or silently cut the test set. Charge both completed timing pilots
  and other sunk costs to this reduced budget.
- Freeze this choice from cost/implementation evidence before confirmation
  outcomes. If even the reduced study cannot fit, report the concrete measured
  obstruction and revised options; do not substitute an encoder benchmark or
  launch an incomplete matrix. Complete paired location sets in the run order.

No extra penalty block is budgeted this week. Do not spend an apparent surplus
until the decoder exports, analysis and writing are secure.

## Coding work and acceptance checks

Reuse `notebooks/iclr/campaign/band.py` (`BandConfig`/`BandLinear`) algebra inside
`notebooks/iclr/decoder_pilot/`. The current decoder practical adapter contains
ambient scalers and a leading identity core and CANNOT answer this location
question by relabeling arms. Adapt placement rather than calling the RoBERTa
helper that unfreezes its classifier. Preserve existing executed/archived code.

The coding agent has since pushed research commit `0fddf62e` and Overleaf
`56e4e6a`: registration/admission, a frozen-model reference path and 211 reported
passing CPU tests now exist. Preserve and adapt this work; it still registers
the superseded five-arm benchmark. `DECODER_EXECUTION_PROTOCOL_20260919.md` and
`data/campaign_v1/DECODER_DECISIONS_20260919.proposed.json` are NOT current launch
instructions. Do not seal that old proposed decisions file.

Extend explicit band/flexibility CLI/config, frozen protocol/run admission,
checkpoint reconstruction, correct three-band diagnostics and export analysis.
Replace one-probe-arm LR selection with the balanced across-band rule above;
register both accuracy and task loss as primary outcomes. Remove the norm-dose
calibration dependency from this study, not the existing historical implementation.
Hash model, SVD cache, module inventory, band boundaries, q, recipe, data splits
and source into the protocol/checkpoint identity. Test wrong-band reload rejection,
zero insertion, frozen backbone/head, support confinement, rotation activation,
forward/delta/merge equivalence and fresh-model reload on all six arms.

Two practical hazards found in the prepared runner must be addressed for timing:
`load_model_and_tokenizer` sets `model.config.use_cache=False`; explicitly enable
KV caching for generation and restore the training setting, rather than relying
on a possibly different generation_config default. The current generation call
has no bf16-autocast context although training does: explicitly pin and measure
generation precision, use it consistently for frozen/trained models, and validate
merged evaluation under that precision. Also avoid keeping two full
GPU model copies alive during reload validation. Profile logits/NLL memory with
the 151,936-token vocabulary at max_length 640; the old rough activation estimate
is not a memory guarantee. Freeze consistent memory settings after profiling.

Make dense geometry checks sparse in time (initial/final plus essential pilot
validation), and add a mode that skips generation during tuning. Do not remove
correctness checks to meet a runtime claim. Count setup, evaluation, reload and
failed/retried work in the cost report. Save small adapter states; do not duplicate
base weights or SVD caches per run. Reuse immutable cached inputs via manifests
in a NEW `campaign_outputs_decoder_subspace_v1` output namespace.

First deliver the CPU-tested implementation, exact six-arm/reduced registries,
launch commands and timing worksheet. GPU use follows the actual assigned-device
record; discovering an idle GPU is not authorization. The preparation request
itself is not a request to silently start the full confirmation queue. Preserve
other users' jobs and existing research outputs. Publish scoped code/handoff
commits to `ortho_new`, without force pushing; update Overleaf planning files,
but leave scientific manuscript integration to the writing agent/author.

## Writing already required

Independently of decoder outcomes, integrate the delivered held-aside band scores
and temperature-scaling analysis. The raw MIX–NORM loss gap largely disappears
after inner-fitted temperature scaling; the paper must reflect that. Correct the
band test-scoring provenance disclosure. Resolve the unsupported historical GPT-2
table uncertainties before submission. Do not treat the current PDF/supplement
as already containing these new exports. Then add the decoder recipe/results
and limit each claim to the experiment that supports it. The coding agent supplies
the evidence package; the author/writing agent owns these manuscript changes.

## Locations and source checks

- Research: `Guy-Bilitski/UIOrthoLoRA`, branch `ortho_new`.
- Paper: Overleaf project `6aa54397e58b10444b0fa2aa`, `neurips_2026.tex`.
- Server checkout reported by the coding agent:
  `/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914`.
- Existing sealed inputs: `campaign_outputs_decoder_v1/inputs/`; verify their
  manifests instead of downloading or changing revisions.
- Completed evidence, under the handoff's `data/`:
  `temperature_logits_20260916/`, `temperature_analysis_20260916.json`,
  `band_held_aside_20260917/`, `run_records_20260919/`.
- `EXPERIMENT_PREPARATION_STATUS_20260919.md` records delivered code/tests and
  known evidence gaps; its old proposed run matrix is superseded here.
- Primary external references checked: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  and [GSM8K repository](https://github.com/openai/grade-school-math). Pin the
  prepared revisions, not the current web page versions.

This document fixes the preparation design and conditional budget rule; it is
not a report of new training results or measured GPU feasibility.
