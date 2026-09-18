# ICLR manuscript review guide

## Current author review — 2026-09-17

The introduction is awaiting the author's approval. In `neurips_2026.tex`, blue
is proposed wording, gray is replaced wording, and black is unchanged. The
abstract and first two introduction paragraphs are unchanged. The follow-up adds
result/protocol references, classifier and loss definitions, and concrete leading
and middle results. At the author's request, the first main experiment now
contains all seven conditions in its result table and explains the training and
evaluation split beside it. The main design and interpretation changes, added
table rows, and a detailed protocol paragraph in the appendix are blue; replaced
text is gray. Other later content is unchanged.
The gray experiment paragraphs show the wording from the preceding review round.
This request supersedes the clean-only review instructions below. Review macros
are inline; the historical `review_markup.tex` remains inactive.

For a clean preview, define `\ICLRCleanBuild` before the manuscript. Audit either
PDF with `python3 scripts/audit_manuscript.py --review --pdf /path/to/neurips_2026.pdf`.
The clean preview still must have nine main pages; the review copy includes the
old paragraphs and can be longer. Local builds are under
`../build/introduction_review_followup_20260917/{review,clean}/`. The previous supplementary
ZIP has not been repackaged for this pending prose revision.

## Earlier clean-manuscript guide — 2026-09-14

Authorized Overleaf project: `6aa54397e58b10444b0fa2aa` only.
Original snapshot: `ebf8e3c3705c2a26e43960030ac31fb74bec29a0`.
Status, 2026-09-14: clean manuscript revised against the NeurIPS reviews and the supplied review panel. Additional experiments and release confirmations remain outstanding. For a fresh GPU session, start with `GPU_HANDOFF.md`.

## Reading the paper

- The existing Overleaf root `neurips_2026.tex` now contains the entire manuscript directly: all sections, proofs, tables, and figures. Edit this file; no project-settings change is needed.
- All active text is plain. Approved additions are unwrapped; cuts, author notes, and review environments are absent. Former modular sections and compatibility entries are inactive snapshots under `archive/modular_2026_09_13/`.
- Original wording is archived in `review/original_main.tex` and Git history. `review_markup.tex` is inactive; it no longer controls the PDF.
- Direct author edits always take priority. Before every push, fetch this project's bridge, reconcile author changes, compile, fetch again, and use a normal fast-forward push. Never force-push.
- No other project's checkout, MCP configuration, remote, or shared implementation is modified.

## Scientific focus

The paper studies confinement, within-tail flexibility, ambient scaling, and leading–tail interaction. UIOrthoLoRA and its simpler counterparts are experimental tools, not candidates in a claim of best-adapter performance.

The central evidence is the mixing intervention, interpreted through effective-update geometry and magnitude. The module-level analysis separates leading, cross, and tail energy: cross share falls under two-sided control, while leading share rises in four task means. Table 1 includes A/B/C with seed SDs; Figure 2 includes the dimension-only reference. The appendix exposes every seed and selected step, block fractions, and logged time/memory. The fixed-basis identities concern representability without assuming task-tail alignment. Unified bounds distinguish tail-only confinement from practical block decoupling; commutator and random-projector identities characterize the scaler penalty without proving spectral specificity. Initialization/identity and multi-cutoff reporting are explicit, but their trained-model measurements remain pending.

The six-task GLUE table stays in the main experiments, grouped by source protocol without winner highlighting. Generation remains supporting appendix evidence. The earlier retention section and table are preserved in inactive `archive/retention_2026_09_14.tex`; the active paper excludes them because configuration selection, matched adaptation and checkpoint-level geometry are unverified.

Table 2 has one MRPC accuracy column, following the author's 2026-09-14 confirmation that the original broad-evaluation values are accuracy. Avg6 includes all six displayed task means. The separate mixing intervention still reports MRPC F1. Unreconciled historical parameter counts are preserved in the data archive, not used in the GLUE display or a parameter-efficiency claim.

The official ICLR 2027 style is unchanged. Main text, proofs, experimental details, computational-cost analysis, and supporting evaluations form one clean manuscript. No pending GPU experiment is reported as completed.

## Research coordination files

- `GPU_HANDOFF.md`: portable context, implementation hazards, source fingerprints, allocated-resource gate, first-session commands and restart/artifact contract. The expanded training runner is not yet implemented.
- `EXPERIMENTS_REQUIRED.md`: canonical P0–P8 plan, including norm/scaler/random-projector/head controls, trajectories, common-frame baseline comparisons, modern replication, costs and same-checkpoint functional probes. The first expanded tranche is 66 confirmation runs plus calibration, subject to timing and allocation, not the superseded 24-run plan.
- `REVIEW_PANEL_RESPONSE.md`: comprehensive review disposition, completed versus pending evidence, primary-source checks and reasoned non-adoptions.
- `REVIEW_RESPONSE_PLAN.md`: earlier NeurIPS review ledger, updated with the current evidence gates.
- `data/legacy_mixing/PROVENANCE.md`: provenance of the 27 condition rows used in the nine paired task/seed comparisons.
- `data/legacy_layers/PROVENANCE.md`: the 1,296 module records supporting Figure 2 and the four-block table.
- `data/GLUE_PROVENANCE.md`: reported-score preservation, metric correction, and parameter-count discrepancy.
- `MANUSCRIPT_AUDIT.md`: current audit results and submission-readiness gates.

Important outstanding confirmations include exact legacy run configurations, broader-table seed/parameter-count conventions, generation-run provenance, retention configuration selection and partitions, software/model/data versions, and final author verification of the proofs and AI-use statement.

The inactive retention archive concerns a nine-task BIG-bench multiple-choice subset, not full BBH, and a strict negative-shift threshold. These historical checks do not establish a causal spectral-retention result. P8 states the evidence needed before restoring that table.

## Validation

From the project directory:

```bash
python3 scripts/build_mixing_tables.py --check
python3 scripts/analyze_archived_layers.py
python3 scripts/check_spectral_algebra.py
python3 scripts/check_review_additions.py
python3 scripts/audit_manuscript.py --pdf /path/to/neurips_2026.pdf
```

The generator checks six inline regions against archived summaries, module diagnostics, and reported GLUE values. The algebra checkers cover projection identities, mixing bounds, commutators, random-projector expectations, initialization, chordal/identity definitions, rectangular accounting, factor symmetries, SDs, cost rows and boundary cases. The audit also verifies numerical preservation, references, and the PDF/log. None reruns training. Generated values are editable in the single TeX file, but a changed number must be reconciled with the underlying record; do not blindly regenerate over an author's correction.

Compile only the active root into an output directory outside the source mirror. The main conclusion ends on page 9; Table 2 is on page 8; statements, references, and appendix bring the current PDF to 21 pages. Check unresolved references/citations, overflow, page count, and rendered pages after future edits. The manuscript remains anonymous; do not enable `\\iclrfinalcopy` for submission.

## Submission boundary

A clean PDF is not a claim that missing experiments or author verification are complete. For anonymous supplementary release, exclude `review/`, `archive/`, inactive review machinery, internal planning/audit documents, project identifiers, credentials, and local paths. Assemble a scoped export; do not upload the private collaboration repository wholesale.

The Git push updates this Overleaf project only. It is not an OpenReview submission and does not rename the Overleaf dashboard project.
