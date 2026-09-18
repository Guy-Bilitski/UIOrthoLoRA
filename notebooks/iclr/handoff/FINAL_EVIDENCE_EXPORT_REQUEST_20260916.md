# Final paper evidence export request

> Superseded evidence status (2026-09-16): the bundle delivered at Overleaf
> `5510161` supplies the requested learned/module records, manifests and partial
> band export. See `PAPER_FINALIZATION_STATUS_20260916.md` and the independently
> generated `data/final_evidence_analysis.json`. Statements below about absent
> exports describe the earlier snapshot. The band block and a separately requested
> one-time held-aside evaluation remain pending.

Internal handoff for the author/coding agent; not submission material.
Project: `6aa54397e58b10444b0fa2aa` only. Do not modify another project or shared
research code. This requests existing artifacts and reporting, not new training.

## What the writing checkout currently has

Overleaf `5eedd7a` and the `origin/ortho_new` handoff inspected at research revision
`98d9ce5` expose the same 36 focused-study rows: 18 calibration/refinement plus
18 confirmation. They contain no BAND outcomes or learned-since-insertion columns.
The latest scoped results commit visible in that handoff is `525156e`.

The writing pass now leads with the focused study, includes full per-seed task
scores and paired contrasts, and preserves the historical six-task GLUE table.
The final full-paper integration needs the results below if they already exist.

## Required handoff

1. **Band × flexibility results and coverage.** Export all validated outcomes for
   the six leading/middle/tail × diagonal/partial-rotation arms, plus same-seed
   original-backbone head-only references. Include RTE, MRPC if run, and any already
   completed LoRA comparator under its own protocol. State completed, running,
   failed, and unlaunched cells; retain unfavorable results and failed gates.
   Per row: condition, task, seed, fixed budget, fixed accuracy and MRPC F1,
   selected checkpoint and its metrics, pooled norm, within-band off-diagonal and
   off-band energies, trainable inventory, learning-health status, source/config
   revision, run ID, validation ID/hash. Undefined zero-update fractions stay
   missing, not zero. Supply the registered numeric off-band tolerance.
2. **Learned-since-insertion diagnostics.** For the 18 focused confirmation runs,
   export pooled and equal-module LL/LT/TL/TT energies/fractions and update norms
   for `W_eff(final)-W_eff(insertion)`, separately from the existing total delta.
   Include the initial-state/checkpoint IDs and validation provenance. Export
   per-module block energies and pretrained squared Frobenius norms if already
   saved, so allocation can be checked without treating near-zero ratios as
   meaningful large effects. Do not estimate learned energies by subtracting
   squared total and initial norms.
3. **Executed protocol and compact evidence.** Include the immutable manifests
   covering each distinct arm/task recipe: optimizer groups and learning rates,
   weight decay, scheduler/warmup, clipping, dropout, batch/accumulation, sequence
   length, initialization, precision, module placement, head, checkpoint selection,
   split definition/sizes, source and model/tokenizer revisions. Include repaired
   tokenizer-parity/learning-gate records, invalidation records, and the original
   backbone probe reference with dataset/mask/denominator details. Current source
   defaults are not a substitute for an executed run manifest.
4. **Evaluation status.** State whether official held-aside validation has been
   evaluated for the frozen recipes/checkpoints. If already evaluated, export it
   separately from inner-selection scores. If untouched, say so; do not choose
   checkpoints or tune using it merely to populate the paper.

A scoped commit on the existing `ortho_new` branch and a mirror to the authorized
Overleaf project, or a readable artifact bundle path, is sufficient. Preserve
invalidation exclusions and per-run SHA provenance. No credentials, other-project
results, or raw private paths belong in anonymous paper artifacts.

## What is not being requested

No automatic extension to five seeds, new random-frame/centered-scaler campaign,
full fine-tuning, modern-backbone experiment, or new retention benchmark. These
are limitations of the current study, not prerequisites created by the writing
pass. Do not retune the frozen MIX/NORM doses because confirmation pairs miss 5%.
The missing abstract scratchpad is optional; the paper already has a full abstract.

## Integration rules already fixed

Use `BAND_PRESENTATION_SPEC_20260916.md`, published before review of band
confirmation outcomes. Keep tasks, protocols, fixed versus selected endpoints,
and historical versus fresh scores distinct. The tail pilot is not a superiority
result. A partial band block is exploratory; a complete block requires all its
declared arms/seeds. If any requested evidence is unavailable, report that fact
and its location/status instead of reconstructing unsupported claims.
