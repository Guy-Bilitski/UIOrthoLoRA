# Paper finalization status — 2026-09-16

Historical record. The current evidence integration and submission-package status
are in `PAPER_FINALIZATION_STATUS_20260917.md`; the strict-band block is now 21/21
and the current supplement includes its recorded implementation.

## Current: completed section-by-section writing revision

The latest abstract revision explicitly introduces our new adapter designs.
It reports the comparison in plain language: roughly halved prediction loss,
with average accuracy differences below half a point. Seed counts, calibration
and allocation percentages remain in the experiment section. The introduction
also identifies the adapter designs among our contributions.

The active manuscript is **Controlling Interaction in Spectral Fine-Tuning**.
At the author's request, every main section now follows the completed comparison:
controlling leading–tail interaction gives 44%/48% lower held-out task loss than
the tested size penalty, with measured mean accuracy differences +0.12/-0.33
points. The introduction explains the gap, the methods explain the purpose of
each construction, and the experiments lead with the behavioral result.

The new paired-seed figure shows all 18 fixed-checkpoint outcomes. The subsequent
sections report the size frontier, module allocation, identity structure and
masked-token probe. The six-task historical GLUE table remains in the main paper.
Repeated qualifications are consolidated in one scope section; the incomplete
strict-tail comparison is retained in Appendix B.19. The unfavorable probe and
archived score outcomes remain visible. `SECTION_READTHROUGH_20260916.md` records
the purpose and revisions of each section.

Writing and validation are complete for the author's reading. The underlying
research queue is unchanged: `TAIL_ADAPTATION_EVIDENCE_HANDOFF_20260916.md` remains
the export/experiment contract. No GPU job, held-out evaluation or training run
was initiated by this writing revision.

Current local validation build:
`../build/plain_abstract_20260916/neurips_2026.pdf` — 35 pages total, nine main,
with the full GLUE table on page 8. All source/data/algebra/figure/PDF checks pass;
all main pages were visually inspected. Every displayed equation and numeric
tabular body is byte-identical to the pre-revision manuscript. Raw and derived
evidence files are unchanged.

The refreshed `supplement/analysis_artifact.zip` is 1,294,992 bytes,
SHA256 `a586a38ce7cff0a978252df0d47d44b80e783f5698121ff632c6d938995a7354`. Its manuscript matches the validated source exactly. All analysis
checks pass in the standalone build; a fresh ZIP extraction passes every file
hash. The new figure and generator are included in the artifact audit.

Overleaf's compiler remains canonical. This is the validated local review copy.
The supplementary archive is prepared for attachment; an anonymous public
hosting link is still absent. Internal author document, not submission material.

## Historical revision record

## Completed in round 3

- Reframed the claim around predicted mixing behavior, finite-training residuals
  and the observed magnitude frontier; title now **Magnitude and Mixing in Spectral Fine-Tuning**.
- Added all held-aside task-NLL contrasts, with MIX lower in every seed; confidence
  calibration and causal attribution remain open, with inner-selection logits requested.
- Added within-task frontier fits (12.1×/8.5× norm spans) as descriptive sensitivity,
  not a causal bound; added an explicit counterexample to forced norm-flow invariance.
- Added PSOFT coordinates/Gram distinction, conditional parameter-count annotations,
  a notation guide, threshold/identity reporting fields and proper hypothetical power
  sensitivity. Consolidated repeated caveats to retain nine main pages.
- Refreshed the standalone supplement; the next panel needs the ZIP as well as PDF.

## Retained from round 2

- Addressed review_panel_v3 with an item-by-item response in REVIEW_PANEL_V3_RESPONSE.md.
- Reframed the title/abstract around magnitude, mixing and confinement; distinguished
  observations from pretrained-frame specificity, performance and retention claims.
- Audited 36 clean focused rows, all 18 confirmations and 864 module records.
  All 39 invalidated run IDs remain excluded. Reproduced signed matching errors,
  sample SDs, paired nominal intervals and total/learned pooled/equal-module fractions.
- Added leading identity alignment (MIX learned 84.67% RTE, 89.50% MRPC) and
  post-review module-norm sensitivity at both requested thresholds. The cross gap
  survives restricting both arms to the same non-small modules; a causal mechanism
  is not established by this descriptive subset analysis.
- Integrated the separate held-aside export at a63b1f2. Recomputed all 18 sets of
  per-example predictions, accuracy/F1/loss, label/ID joins and request/manifest
  hashes, and checked loading records against original fixed-checkpoint metrics.
  No checkpoints, doses or seeds were selected on held-aside outcomes.
- Preserved all historical GLUE task means/SDs in main text; removed the unmatched
  Avg6 display, restored imported parameter counts and marked configuration-derived counts as distinct from missing historical inventories.
- Added prior-method coordinate derivations, cost qualifications and primary-source
  citation corrections. Corrected panel suggestions that would overstate the math,
  identity interpretation, distributional null or statistical power.
- Replaced incomplete band numerical tables with a compact pending-block note.
  The newer ledger freezes 8/21 endpoints; the older outcome CSV has only three
  seed-42 confirmations plus two distinct timing pilots. No coverage is inferred
  beyond its timestamped record. Existing training allocation is unchanged.
- Prepared a sanitized standalone analysis archive with CPU checks, file hashes,
  explicit source/derivative provenance and no private collaboration documents.

## Checks and review material

All required table/algebra/review/manuscript checks pass, along with focused,
final-evidence and review-v2 analyzers. The local PDF passes: 35 pages total, nine
main pages, historical GLUE on page 8, resolved references, no overfull boxes or
missing glyphs, anonymous grayscale text. Underfull spacing and Tectonic's known
bibliography-rerun notices remain nonfatal; the build is not called warning-free.

- Revised paper: neurips_2026.tex (Overleaf root).
- Review response: REVIEW_PANEL_V3_RESPONSE.md.
- Concrete coding-agent export request: REVIEW_V3_EVIDENCE_HANDOFF_20260916.md.
- Supplement builder: scripts/build_analysis_artifact.py.
- Deliverable validated supplement: supplement/analysis_artifact.zip (also copied to the local build directory).

## Remaining dependencies

1. Export inner-selection logits for temperature scaling, plus centered/raw scaler moments from all 18 existing frozen checkpoints;
   the available identity analysis is not a substitute for these measurements.
2. Supply the historical retention sweep inventory/counts. The appendix reports
   40 displayed conditions and retrospective exclusion, not an invented run count.
3. Complete and refresh the already registered band block, including head-only
   references, before drawing band/flexibility conclusions. Its prospective locked
   evaluation request remains unchanged; it is not yet evaluated as a full block.
4. Attach the prepared supplement to the actual review/submission or provide an
   anonymous endpoint. The archive is analysis-only, not a training/checkpoint release.
5. Historical broad-GLUE run/count provenance remains qualified. Optional CENTER,
   random-frame, second-probe and modern-backbone experiments have not been launched
   and are not represented as results or mandatory prerequisites for the narrow claim.

No training, remote job changes or public release occurred in this paper revision.
Author review and actual conference submission remain separate actions.
