# Project-local working instructions

This directory is the portable GitHub handoff under `notebooks/iclr/handoff/`, not a nested Overleaf Git checkout. Read `SNAPSHOT_PROVENANCE.md` for delivery details. The enclosing research repository's `origin` points to GitHub; never treat it as the Overleaf bridge. This snapshot includes hash-verified historical fixtures so the CPU manuscript audit does not need Overleaf Git history or credentials.

This is the paper repository for Overleaf project **6aa54397e58b10444b0fa2aa only**. Another agent works on a different project; do not modify its branch, code, output paths, processes or GPU allocation.

Before GPU-related implementation or execution, read `GPU_HANDOFF.md`, `EXPERIMENTS_REQUIRED.md`, `REVIEW_PANEL_RESPONSE.md` and `MANUSCRIPT_AUDIT.md`. These files carry the research context from prior sessions. The **2026-09-15 current execution plan** at the top of the experiment plan overrides older launch counts and priorities: budget against two GPUs, protect the clean 18-run study, then test band location/flexibility within the author's deadlines. Locate the server's repaired, isolated implementation and durable invalidation/gate records; historical statements that no runner exists are not current server status. Obtain a dedicated research checkout as described in the handoff, not a shared-code modification. Server access does not imply every GPU is allocated; confirm device IDs, budget and persistent storage before launch.

The scientific goal is to study leading/tail singular components, adaptation geometry and regularization. UIOrthoLoRA is an experimental instrument, not a best-adapter claim. Never present planned runs, synthetic algebra checks or reviewer-suggested outcomes as measured model results.

The sole active manuscript is `neurips_2026.tex`, using ICLR 2027 style. Keep all manuscript content inline, with no blue/gray revision markup. Preserve the six-task GLUE table in the main paper. Its MRPC column is accuracy for all rows, confirmed by the author; the separate legacy mixing study reports MRPC F1. Do not conflate these datasets or overwrite archived CSVs with reruns.

Before every Overleaf push, verify the exact project remote, fetch and reconcile direct author edits, validate the changed content, fetch again and use a normal non-force push. Author edits take priority. No credential belongs in the repository or GPU artifacts. Internal planning/review/archive files are not anonymous submission material.

For paper/data changes, run `scripts/build_mixing_tables.py --check`, `scripts/check_spectral_algebra.py`, `scripts/check_review_additions.py`, and `scripts/audit_manuscript.py` with Python 3. Compile and run the optional PDF audit after TeX changes. The PDF audit expects its matching log, nine main pages and resolved references; it does not certify experimental validity. Preserve source provenance and generated-region corrections instead of blindly regenerating over author edits.

At the end of a GPU session, update the handoff and durable run ledger with source revision, executed commands, completed/failed/retry run IDs and artifact locations. No run is complete until saved state reloads and reproduces its metrics.
