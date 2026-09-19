Prepare the focused decoder experiment for our ICLR paper. Our question is:
does the choice of spectral subspace in the starting model affect adaptation
accuracy or prediction loss, and does within-subspace rotation change that?
We are not making a best-adapter claim. Do not add or run LoRA/PiSSA comparisons,
CENTER, another model/task, or a new interaction-penalty sweep.

Fetch latest `ortho_new` from `git@github.com:Guy-Bilitski/UIOrthoLoRA.git` and
Overleaf project `6aa54397e58b10444b0fa2aa`. First read
`notebooks/iclr/handoff/DECODER_SUBSPACE_STUDY_20260919.md` (also at the Overleaf
root), then `EXPERIMENT_PREPARATION_STATUS_20260919.md`. The new subspace plan
supersedes every older decoder matrix, launch command, cost and fallback.

Reuse the prepared Qwen2.5-1.5B-Instruct/GSM8K pipeline and pinned inputs. The
server checkout was reported at
`/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914`, with sealed
inputs under `campaign_outputs_decoder_v1/inputs/`. Verify the current state.
Use an isolated checkout/output namespace; preserve other projects and evidence.

Implement leading/middle/tail thirds of q_proj/o_proj in all 28 layers, each with
DIAG and ROT128 variants: six arms, three paired seeds {17,42,123}, 18 final runs.
Reuse `campaign/band.py` algebra in `decoder_pilot/`; its RoBERTa placement helper
cannot be used unchanged. Require Delta = U_B H V_B^T, zero initial coefficients,
identity rotations, no ambient scalers or outside-band core, and a frozen output
head/backbone. Bands are [0,512), [512,1024), [1024,1536); rotate the last 128
coordinates within each band. The current practical decoder arms are not a
valid substitute. Score the frozen decoder as an inference-only reference.

Use the plan's common 842-step completion-only SFT recipe. Prepare two short
timing pilots and bounded tuning: three LRs × three bands × two families, one
separate tuning seed. Select one shared LR per family by mean inner-selection
NLL across its three bands. Keep all LR curves; never select on test outcomes.
Measure generated-answer accuracy AND held-out completion-token NLL on all 1,319
test examples at the fixed final endpoint. Export paired differences, per-example
outputs, learning curves, update norms and confinement/rotation diagnostics.
No equivalence claim from similar means or inactive adapters.

Reuse and adapt the registration/admission and reference-evaluation code just
pushed in `0fddf62e`; it still describes the superseded benchmark. Do not use
its old proposed decisions JSON or execution commands. Replace the selection
rule and register both accuracy and loss as primary outcomes. Complete band-aware
checkpoint metadata/reconstruction.
Test zero insertion, frozen weights, confinement, active rotations after the
zero-core start, merge/forward agreement and fresh-model reload, including
wrong-band rejection. Explicitly manage generation KV caching, avoid two full
GPU models during reload, and profile vocabulary-logit memory. Add generation-
skipping for tuning and sparse dense-geometry checks. The plan specifies the
selection-only decode audit and common length-cap rule.

Only two assigned 24 GB GPUs are available. Prepare a measured timing worksheet
including tuning, all evaluation/generation, reloads and 25% contingency; target
at most 96 GPU-hours and completion before the existing September 24 cutoff.
Confirm actual allocation/deadline. If the full matrix will not fit, use the
predeclared DIAG-only fallback: all three bands × three seeds, retaining balanced
tuning and full test evaluation. Choose from timing before confirmation results.
Do not silently shorten training, drop seeds or return to encoder benchmarks.

Start with implementation and CPU tests. Deliver scoped pushed commits, tested
launch commands and the timing/registration plan; full confirmations are not to
be silently launched by this preparation request. Use GPUs only under the actual
assigned-device authorization. No new permission is needed for reversible CPU
preparation. Keep the approved manuscript intact; deliver evidence and a concise
handoff for the author/writing agent. Existing temperature and band exports are
already complete and should be reused, not rerun.
