# GitHub handoff location

This directory is a portable snapshot under `notebooks/iclr/handoff/`, not the
live Overleaf checkout. Its enclosing Git remote is GitHub. The current entry
point is `EXPERIMENT_PREPARATION_HANDOFF_20260919.md`; see
`SNAPSHOT_PROVENANCE.md` for the source revision and transport-only differences.
Manuscript changes require reconciliation with the real Overleaf project.

> **Coding-session report — 19 September 2026 (late):** frozen-checkpoint work is done (`EXPERIMENT_PREPARATION_STATUS_20260919.md`: temperature scaling of the 18 practical endpoints, held-aside evaluation of all 21 band/head endpoints, 39 run records and trajectories). CENTER (`CENTER_PROTOCOL_20260919.md`) and the decoder pilot (`DECODER_PILOT_DESIGN_20260919.md`) are implemented and CPU-tested but NOT launched; they wait for the author's two GPU IDs and scope approval.

> **Current preparation status — 19 September 2026:** Start with
> `EXPERIMENT_PREPARATION_HANDOFF_20260919.md` and
> `review_feedback/20260919/assessment.md`. The author confirmed **two GPUs**.
> All 18 practical and 21 RTE band/head confirmations are complete; their
> protocols and immutable evidence remain unchanged. Prepare the remaining
> checkpoint evaluations and proposed controls/decoder pilot. New training
> proposals require the agreed scope and budget; older queue, title, completion
> and four-GPU statements below are historical. The approved abstract and
> introduction remain unchanged. This update does not change live server jobs.

# Project-local working instructions

This is the paper repository for Overleaf project **6aa54397e58b10444b0fa2aa only**. Another agent works on a different project; do not modify its branch, code, output paths, processes or GPU allocation.

The author's latest writing instruction is to build every section around the
completed interaction-control result, using simple explanations and direct
claims. See `SECTION_READTHROUGH_20260916.md`. The active title is **How Pretrained
Spectral Subspaces and Their Interactions Affect Fine-Tuning**. Lead with the question, the controlled
adapter construction, and held-out loss/accuracy; consolidate interpretation
limits in the scope section instead of repeating them throughout the narrative.
Present our adapter designs as contributions, as the author explicitly requested.
Keep the abstract's result in plain language; seed counts, calibration details
and block-energy percentages belong in the experimental report.
The separate tail/head/rotation experiment and export contract remain in
`TAIL_ADAPTATION_EVIDENCE_HANDOFF_20260916.md`; this writing revision does not
change the training queue. The 2026-09-17 integration now includes all 21 RTE
band/head confirmations, with the tail/head result in the main text and every
location and paired contrast in the appendix. Band outcomes are inner-selection
scores; the separate practical 18-run outcomes are held-aside. Do not pool them.
See `PAPER_FINALIZATION_STATUS_20260917.md` and
`FINAL_EXPORT_CLOSURE_REQUEST_20260917.md` for the remaining export work. Resource isolation,
validation and evidence-preservation requirements still apply.

Before GPU-related implementation or execution, read `GPU_HANDOFF.md`, `EXPERIMENTS_REQUIRED.md`, `REVIEW_PANEL_RESPONSE.md` and `MANUSCRIPT_AUDIT.md`. These files carry the research context from prior sessions. The **2026-09-15 current execution plan** at the top of the experiment plan overrides older launch counts and priorities: budget against two GPUs, protect the clean 18-run study, then test band location/flexibility within the author's deadlines. Locate the server's repaired, isolated implementation and durable invalidation/gate records; historical statements that no runner exists are not current server status. Obtain a dedicated research checkout as described in the handoff, not a shared-code modification. Server access does not imply every GPU is allocated; confirm device IDs, budget and persistent storage before launch.

The scientific goal is to study leading/tail singular components, adaptation geometry and regularization. UIOrthoLoRA is an experimental instrument, not a best-adapter claim. Never present planned runs, synthetic algebra checks or reviewer-suggested outcomes as measured model results.

The sole active manuscript is `neurips_2026.tex`, using ICLR 2027 style. Keep all manuscript content inline. The author's 2026-09-18 instruction authorizes reviewing and resolving all remaining blue/gray suggestions into clean text for external LLM feedback. The abstract and introduction are already approved; retain their wording. Accept the verified first RTE experiment, complete leading/middle/tail table and linked protocol appendix, and remove superseded text and review macros. Keep individual runs, paired contrasts and implementation details in the appendix. Define the classifier and prediction loss, and link each introductory study to its main result tables and protocol. Preserve the six-task GLUE table in the main paper. Its MRPC column is accuracy for all rows, confirmed by the author; the separate legacy mixing study reports MRPC F1. Do not conflate these datasets or overwrite archived CSVs with reruns. This clean feedback snapshot does not close the scientific checks listed in `FINAL_EXPORT_CLOSURE_REQUEST_20260917.md`.

Before every Overleaf push, verify the exact project remote, fetch and reconcile direct author edits, validate the changed content, fetch again and use a normal non-force push. Author edits take priority. No credential belongs in the repository or GPU artifacts. Internal planning/review/archive files are not anonymous submission material.

For paper/data changes, run `scripts/build_mixing_tables.py --check`, `scripts/check_spectral_algebra.py`, `scripts/check_review_additions.py`, and `scripts/audit_manuscript.py` with Python 3. The clean feedback snapshot must pass the audit without `--review`. Compile and audit the PDF after TeX changes; if a future author review reintroduces blue/gray markup, also compile and audit its clean preview. The PDF audit expects its matching log and resolved references; clean PDFs must retain nine main pages. It does not certify experimental validity. Preserve source provenance and generated-region corrections instead of blindly regenerating over author edits.

At the end of a GPU session, update the handoff and durable run ledger with source revision, executed commands, completed/failed/retry run IDs and artifact locations. No run is complete until saved state reloads and reproduces its metrics.
