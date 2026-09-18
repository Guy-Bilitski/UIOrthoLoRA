# Final evidence integration — 2026-09-17

The manuscript and supplementary archive are ready for the author's full review.
The verified package meets the checked document-format requirements for ICLR
2027. The remaining scientific checks below are not represented as completed.
This is not a claim of acceptance readiness or a completed conference submission.

## What the paper now says

We introduce adapters that separate where a model can change its weights from
how directions in that region can interact. The completed studies show:

1. Strict tail coefficients raise RTE inner-selection accuracy from 59.77% for
   a trained head alone to 70.01%. Every paired seed improves; the mean gain is
   10.24 points, with a descriptive paired 95% interval [8.92, 11.56].
2. Tail rotations reach 71.02%, with seed changes of -2.61, +2.01 and +3.61 points
   relative to diagonal coefficients. They create off-diagonal structure but do
   not show a consistent accuracy gain. Both tail variants have worse task NLL
   than the head alone. All leading/middle controls and null results remain.
3. In the separate practical leading-plus-tail study, interaction control reduces
   held-aside NLL by 44% and 48% versus the selected size penalty, with mean
   accuracy differences below half a point. Calibration correction remains open.

Band results use inner-selection examples; practical results use the original
18-run held-aside evaluation. Historical GLUE and generation results retain
their own protocols and provenance labels. They are never pooled.

## Evidence verified and integrated

- Synchronized Overleaf evidence at `a88118e` and read the latest research
  handoff on the existing research branch. All 21 registered RTE band/head
  confirmations and both separately labeled timing pilots are present.
- Joined all 21 runs to the frozen checkpoint ledger, protocol SHA, validation
  hashes, recipe settings and source revisions. The ten recorded source
  revisions have identical campaign trees. All 46 recorded Python files were
  recovered from Git and checked against manifest SHA256 and Git blob identities.
- Recomputed every band mean/SD and all 18 registered paired contrasts for both
  accuracy and NLL. Added the main tail/head table, complete appendix tables and
  a figure showing every seed for accuracy and loss.
- Rechecked all 18 practical held-aside exports, per-example metrics, 864 module
  records, dose frontiers and invalidation exclusions. All 39 invalidated runs
  remain excluded. Fixed and best checkpoints and the two timing pilots remain
  separate. No raw measurement was changed.
- Preserved all 60 historical GLUE means and their SDs, with MRPC accuracy and
  the full table in the main paper. Existing algebra, allocation, calibration
  frontier and power-sensitivity checks pass.
- Fixed the analyzer's obsolete assumption that a run and its representative
  recipe must share the same whole-repository revision. The replacement checks
  exact experiment-source identity; it does not bypass provenance validation.

## Deliverables and validation

- `neurips_2026.tex`: sole manuscript, anonymous ICLR 2027 style.
- Local review PDF: `../build/finalization_20260917/neurips_2026.pdf`.
- Anonymous attachment: `supplement/analysis_artifact.zip`, also in the build
  directory. Attach this ZIP to the reviewer panel and the eventual submission.
  No public hosting URL is invented or required to access an attached supplement.
- Remaining export request: `FINAL_EXPORT_CLOSURE_REQUEST_20260917.md`.

The PDF has 38 pages total, nine main pages, the complete historical GLUE table
on page 8, resolved references, no overflow or missing-glyph errors, and anonymous
grayscale text. All main pages and new band tables/figure were visually checked.
Underfull spacing and Tectonic's bibliography-rerun notice remain nonfatal.
The local build is verified; Overleaf's own compiler remains the canonical build.

All 12 source/analysis checks pass in the standalone supplement. A direct ZIP
read verified every one of its 158 distributed file hashes and an exact match
to the manuscript. The ZIP includes the recorded experiment implementation,
13 representative recipe manifests, data, figures and regeneration scripts.
Two source-file project identifiers are anonymized, with original and distributed
hashes preserved. It is not a self-contained training replay: checkpoints,
prepared dataset tensors and original per-run validation reports are absent.

Source SHA256: `10e488454771d97c54832c7327430dc8904c5a852f9a7bb3e3c221fe747e5335`.
PDF SHA256: `f659c90b9202c0580437188c8b0f57f89923abfb45c4df5d75b07f32910962d5`.
ZIP SHA256: `1f75e18de81aa6dcef05b8e3587fb4a4686982b85e5747a3b0c60d8e8494f716`
(1,478,102 bytes).

## What remains before calling the evidence fully closed

The latest server handoff explicitly says the 21-run band block's held-aside
evaluation has not run. Temperature-scaling logits/results and complete training
curves are also absent from both Git and Overleaf. The author has been asked to
forward the concrete export request to the coding agent. No new training is
needed for the two priority checks: frozen band evaluation and temperature scaling.
Existing per-run manifests/reload-validation reports should accompany the exports.
The paper states endpoint findings and leaves calibration and convergence open.

CENTER, a second-task band replication, and a fresh-protocol LoRA comparison are
not in the bundle. They are additional studies, not silently missing rows from
the completed 21-run registration. The current claims do not depend on them.
Historical broad-GLUE run/count provenance remains explicitly qualified.

## Conference actions for the authors

The official [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
set the abstract deadline at September 18, 2026, 23:59 AoE and the paper/supplement
deadline at September 25, 2026, 23:59 AoE. The submission limit is nine main pages;
this PDF meets it. The required AI-use statement is included.

The author list and OpenReview profiles must be finalized by the abstract
deadline. Author eligibility, reciprocal-reviewing obligations and the submission
form declarations require author completion in OpenReview; no portal record was
inspected or submitted in this session. Upload the anonymous PDF and ZIP only,
not this internal repository's planning, review or handoff documents.

No GPU job, checkpoint selection, new evaluation or conference submission was
performed during this integration.
