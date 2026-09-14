# GitHub delivery provenance — 2026-09-14

This handoff is delivered in the existing `ortho_new` branch of `Guy-Bilitski/UIOrthoLoRA`. Only new files under `notebooks/iclr/` are introduced. The other project's working tree, research code and 31 unpublished local commits are not modified or published by this delivery.

## Source snapshots

- Paper/experiment context: Overleaf project `6aa54397e58b10444b0fa2aa`, commit `79d3c1164c5ea3f14f6e89fc5363e8abd3535c90`.
- Numerical audit fixtures: the same Overleaf project's commit `9c8043c1b531f3afb27fa55ce101c8635736c938`, original paths recorded in `audit_baseline/manifest.json` with SHA-256 digests.
- Published research base: `8cd4a061dad86073f4ef52997ca8a9412465b194` on GitHub `ortho_new`. The six source fingerprints recorded by the paper audit match this revision exactly. Pin the actual delivery commit and any later modifications in new run manifests.

## Included

The scientific handoff, designated P0–P8 experiment plan, review-response ledgers, manuscript audit, provenance, the active single-file manuscript and its bibliography/style dependencies, all archived GLUE/mixing/layer CSVs, CPU analysis/tests, the original manuscript for context, and the inactive retention archive. These are ordinary files: no nested `.git`, submodule/gitlink, Git bundle, credential helper or Overleaf token is needed.

The full historical modular/review/draft archive and the supplied review-panel packet remain in the author workspace/Overleaf history; they are not needed for GPU execution. `REVIEW_PANEL_RESPONSE.md` preserves the actionable review context. References in older reading-guide prose to historical modular files refer to that source archive, not an active manuscript dependency here.

## Transport-only changes

- GitHub-specific startup notes in README, AGENTS and GPU_HANDOFF explain this directory and the enclosing research root.
- `scripts/audit_manuscript.py` uses the included hash-verified numerical fixtures when present, instead of requiring an unrelated Overleaf Git history. All original preservation, algebra/data and optional PDF assertions remain active.
- The historical fixture files themselves are byte-for-byte exports of the pinned baseline. Their checksum manifest is checked before parsing any numbers.

The active TeX, bibliography, experiment design, CSV data and other CPU scripts are unchanged from the exported paper revision. No model training was added or claimed by this delivery. CPU checks pass in this directory without Overleaf Git history; they do not implement the expanded training runner.

## Work and synchronization

Start with `GPU_HANDOFF.md`, then the experiment plan. Update the run ledger and handoff as implementation/pilots proceed. Keep raw new measurements separate from the legacy CSVs. The GitHub remote is not an Overleaf destination: live manuscript updates require the separate authorized bridge and reconciliation of the author's latest edits. Do not publish private coordination files wholesale as anonymous supplementary material.
