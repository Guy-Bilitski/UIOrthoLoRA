# Manuscript audit — review-panel revision, 2026-09-14

Internal author document. Scope: Overleaf project `6aa54397e58b10444b0fa2aa` only. This is not an anonymous submission artifact.

## Verdict

The current manuscript is a coherent, clean, nine-page main-paper draft with the full GLUE results in the main experiments and all manuscript content in `neurips_2026.tex`. The equations and reported reanalysis pass the checks below. This is not a declaration of submission readiness: missing causal controls, modern-model geometric replication, and historical run provenance remain real research/reproducibility gaps. No GPU training was performed in this revision.

The paper studies spectral components through parameterizations and regularization. UIOrthoLoRA is an instrument, not a best-adapter claim. The rewrite does not treat leading singular directions as proven stores of knowledge, or tail directions as noise.

The supplied review panel has been mapped to completed changes and open experiments in `REVIEW_PANEL_RESPONSE.md`. `GPU_HANDOFF.md` carries the full context into a fresh server session, including source fingerprints and the explicit fact that the expanded training runner is not yet implemented. `EXPERIMENTS_REQUIRED.md` is the sole P0–P8 design; its expanded first tranche supersedes the earlier 24-run plan.

## 1. Requests addressed

| Request | Result |
|---|---|
| Restore GLUE experiments | Main Section 5.5 / Table 2 includes both RoBERTa backbones and every originally reported task mean and standard deviation. |
| One TeX file | All prose, equations, proofs, tables, schematic, and block-figure data are inline in `neurips_2026.tex`. No manuscript `input`, `include`, or external image dependency remains. Bibliography and official style remain standard support files. |
| Use nine pages substantively | Main argument concludes on page 9, with two main figures and two main tables. Added block-allocation reanalysis and a proved characterization of the scaler penalty, not invented experiments or altered margins. |
| Full self-audit | Source structure, numerical preservation/reaggregation, theorem checks, implementation correspondence, metric compatibility, citations, anonymity, page limit, and rendered pages checked. Remaining evidence failures are not marked passed. |

The former modular editing files were moved to `archive/modular_2026_09_13/`, with an explicit inactive notice. They are not compiled. Earlier original wording remains in Git history and `review/original_main.tex`.

## 2. Numerical evidence

### Mixing intervention: reproducible reanalysis

The archive contains 27 A/B/C summaries and their 1,296 module records. Every run has the same 48 named attention matrices. RTE, MRPC, CoLA, and STS-B have seeds 42 and 17; SST-2 has seed 42. Smoke tests and C-only reruns are excluded. Every shared numerical summary field reconciles with its 48 module records; the largest absolute discrepancy is approximately 1.39e-17.

New block-allocation percentages are derived from per-matrix records before averaging:

| Task | Cross A → C | Tail A → C | Leading A → C |
|---|---:|---:|---:|
| RTE | 43.41 → 27.45 | 10.20 → 28.24 | 46.39 → 44.30 |
| MRPC | 43.25 → 10.08 | 10.19 → 26.27 | 46.56 → 63.65 |
| CoLA | 43.75 → 21.09 | 10.51 → 21.93 | 45.74 → 56.98 |
| STS-B | 43.46 → 13.71 | 10.28 → 34.02 | 46.26 → 52.27 |
| SST-2 | 43.33 → 17.21 | 10.15 → 27.06 | 46.52 → 55.74 |

This is the important finding promoted to Figure 2: cross interaction falls, while the leading fraction rises in four task means. It supports the distinction between block decoupling and tail confinement in the practical leading-plus-tail construction. It does not imply that absolute leading energy increases while the total update shrinks.

The reanalysis handles the recorded denominator mismatch: leakage ratios use the ambient delta norm, whereas off-tail ratios use the coordinate norm. The manuscript's reconstruction normalizes in the recorded coordinate frame. The maximum inferred squared-norm mismatch is 2.64e-4, consistent with small finite-precision deviations; no exact runtime orthogonality is presumed.

At run level, cross share, relative update magnitude, off-tail ratio, and left/right drift fall in all nine A/C pairs. At module level, cross share falls in 410/432 pairs, off-tail ratio in 332, magnitude in 429, and left/right drift in 260/256. These are dependent within-run observations, not 432 independent seeds. Task score decreases in seven pairs, ties in one, and increases in one. Task-mean changes range from -1.0321 to -0.2284 points. Claims are descriptive, without statistical equivalence or significance claims.

One-sided regularization lowers raw left mixing and raises raw right mixing in all nine pairs. Factor rescaling can produce that pattern without changing the effective matrix, so it is not called proof of compensatory learning through the right side.

Table 1 now includes all three conditions, with sample SD across available seeds (no invented dispersion for SST-2's single seed). B has the highest mean score on RTE, MRPC and CoLA without stronger cross suppression, so a necessary performance/geometry tradeoff is not asserted. Figure 2 includes A/B/C and a separate dimension-only N reference. Appendix Tables 4 and 5 expose all 27 seed-level endpoints, their validation-selected steps and four-block fractions. Unequal steps, one penalty strength, n≤2 and validation-conditioned selection remain limitations; reporting more columns does not solve them.

### GLUE: important correction, unresolved provenance

All 60 means and 60 standard deviations are preserved. On 2026-09-14, the author confirmed that the original broad-evaluation MRPC values are accuracy. This supersedes the prior audit's F1 assignment from the archived metric description and local training helper; no new raw accuracy outputs were recovered. Table 2 now has one MRPC accuracy column, consistent with [RandLoRA's source table](https://arxiv.org/html/2502.00987v2). Avg6 is recomputed from all six displayed means, yielding spectral values 85.50/84.68 on base and 88.22/88.52 on large. The rotation ordering still reverses across backbones. The separate archived mixing-intervention MRPC F1 results remain unchanged. See `data/GLUE_PROVENANCE.md` for the correction trail.

The baseline rows are imported, not uniformly retuned. Five source runs are not equivalent to the original spectral protocol's reported six seeds. The complete final six-seed artifacts could not be independently reconstructed from the partial/tuning CSVs found locally. Historical parameter counts also do not reconcile with the stated rotated configuration, most notably RoBERTa-large. Counts were removed from the GLUE display and preserved in `data/glue_reported.csv`; see `data/GLUE_PROVENANCE.md`. This is an open author-artifact gate, not an allegation or a fabricated correction.

### Supporting evaluations

The generation numeric body is unchanged from the previous approved draft. The earlier uncertainty corrections remain: Houlsby-style BLEU 0.6 and FourierFT ROUGE-L 0.1, checked against [FourierFT Table 3](https://arxiv.org/html/2405.03003v1). Its Trainable header explicitly excludes frozen storage; it lacks matched geometric diagnostics and verified spectral-run seed manifests.

The former retention section and exact numeric table are preserved in inactive `archive/retention_2026_09_14.tex`, not compiled. The active Appendix E explains why they are excluded from current evidence: selection rules, matched adaptation, replication and same-checkpoint geometry are not established. Its earlier threshold/subset checks remain historical provenance, not proof of a retention benefit. The audit verifies that archiving preserved every original table number. Restoration requires P8's retention-study gate.

### Available resource measurements

Appendix Table 6 now lists all 27 archived `total_train_time` and `peak_gpu_memory` entries, converted to minutes and GiB. The times sum to 14.9755 training-call hours. Inspection of the runner shows that this clock includes training, in-training validation and checkpoint work; peak memory is CUDA maximum allocated memory measured after training following a pre-training reset. Neither field measures setup peak, total process/reserved memory, post-training diagnostics or total project compute. Per-run hardware attribution and matched LoRA timings remain unverified, so this is disclosure, not a controlled efficiency benchmark.

## 3. Mathematical and implementation audit

- Complete square SVD, declared cutoff, nonzero pretrained tail, and effective delta are explicit. Rectangular projectors cover the full complements, including unmatched null directions. Repeated singular values and basis dependence of diagonal coefficient families are acknowledged.
- The two projection-error identities were checked for arbitrary targets; they require no assumed task-tail alignment. Full rotations span the free core, whereas partial rotations do not. Neither identity predicts optimization success or generalization.
- All four leading-plus-tail block identities and norm bounds were checked. Bounded scaler/core conditions are explicit for both the tail-only and general-core limits. Zero cross mixing can coexist with a nonzero leading block.
- Ranked-subspace stability uses a **sufficient** strict separation condition, not an incorrectly claimed necessary one. A two-dimensional tail crossing remains a counterexample to unconditional ranked stability.
- The training loss now sums the per-matrix Frobenius penalties, matching `compute_total_mixing_loss`, rather than leaving the number-of-modules convention unspecified.
- Added and proved the projector-commutator identity: the left penalty equals half the squared commutator norm, or a graph-Laplacian quadratic form in the ambient scales. Its nullspace is constant on connected components; only a connected projector graph forces a globally scalar zero-mixing scaler. Connectivity is not claimed as measured in the experiments.
- Added the expected Haar-projector penalty for a fixed independent scaler: k(d-k)/((d-1)(d+2)) times its squared centered norm. Penalty-only gradient flow conserves the mean. A common nullspace or an expectation identity does not imply identical finite penalties or joint training endpoints.
- Computed the documented recipe's initial effective delta and its absolute Frobenius norm/leading fraction. These are analytical values, not recovered checkpoints or relative effect sizes. Defined identity alignment and signed residuals without claiming they can be inferred from existing scalar ratios.
- Defined chordal distance and overlap with the correct normalization by min(k,d-k); squared chordal distance averages squared sines. Added multi-cutoff/gap/tie reporting and a separation-qualified perturbation citation. Largest-angle drift remains supplementary.
- Reciprocal factor scaling and the tail-only core-scaling identity were checked. Absolute block bounds are not normalized-fraction guarantees; divergence of the core violates the bounded-factor hypothesis.
- The dimension-only allocation reference is 4/9 leading, 4/9 cross, 1/9 tail at the experimental cutoff. This is not a fitted null or a significance test, and the expectation of a norm ratio is not the square root of an expected energy fraction.
- Complexity covers one-time dense SVD, full frozen bases, stored trainable entries versus manifold dimension, forward products, rotations, regularization/diagnostics, and merging. It does not invent measured throughput/memory improvements.

The NumPy tests pass 100 projection/core cases, 300 unified block-bound and commutator cases, and 72 complete rectangular decompositions, plus zero/coordinate-aligned/disconnected/crossing boundary cases. The additional review suite checks 6,000 Haar projectors in each of three small dimensions against the analytical expectation, mean conservation, initialization, chordal/identity identities, sample SDs and all cost rows. Numerical tests are consistency checks, not a substitute for proofs, model experiments or final author review.

## 4. References, presentation, and build

Related work explicitly distinguishes SORSA's singular-factor orthonormality penalty from confinement to a fixed pretrained span and ambient-scaler mixing control. SVDiff and SVFT are credited as antecedents; PiSSA/MiLoRA initialization is not equated with continued confinement. The cited geometry studies are not dismissed or claimed to lack interventions.

Primary-source checks now also cover Spectral Adapter, PSOFT, KaSA, OFT/BOFT and feature-scaling precedents SSF/IA3. The Spectral Adapter leading replacement is mapped algebraically into the same block frame, without identifying it with the additive tail-core family. Updated Spectral Adapter/VeRA/AdaLoRA/CorDA conference metadata. The canonical LoRA/VeRA protocols differ from the imported RandLoRA reference rows, so replacement published numbers are not treated as matched experiments. Earlier LoRA/model/benchmark corrections remain; retention-only entries are no longer active citations. The bibliography resolves all 31 cited keys. Source links and exact distinctions are in `REVIEW_PANEL_RESPONSE.md`; this is a targeted audit, not a claim to have replicated cited work.

The official [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) allow nine main pages; references, appendix, and specified statements are outside the limit. The official style and margins are unchanged. The main conclusion ends on page 9; the six-task GLUE table is on page 8. The compiled document has 21 pages including statements, references, and appendix. Reproducibility, AI-use, and ethics statements follow the main-paper boundary.

Source checks pass: one manuscript file, 59 unique labels, balanced braces/environments, no missing citation/reference keys, no active blue/gray review macros, no enabled final-copy switch, and no credentials/private project identifiers in the active manuscript/PDF. The figures and text use grayscale; official review line numbers remain gray. Rendered main pages and appendix tables were inspected for readability. The final log has no overfull boxes, undefined citations/references, missing glyphs, or TeX errors. Underfull spacing and Tectonic's repeated bibliography-rerun notices remain; a successful build is not misreported as completely warning-free.

## 5. Evidence still required before submission

1. Recover the original final GLUE summaries/checkpoints, configuration manifests, and parameter-count convention; also verify the supporting generation/retention provenance.
2. Run magnitude-matched, centered-scaler, initialization-decay, random-projector and both head-only controls, with shared initial effective weights where applicable and prespecified paired seeds. Normalized placement changes alone do not establish a magnitude-independent or spectral-specific learning benefit.
3. Test equal-dimensional leading/tail/mixed/random references and the coefficient/rotation/scaler contrasts before making claims of tail preference or isolated rotation benefit.
4. Obtain representative spectral-baseline geometry and one modern generative-backbone replication under a common protocol. Reframing does not remove this external-validity concern.
5. Collect persistent checkpoint diagnostics if using “dynamics”; the current paper makes endpoint-geometry claims only. Include initialization/identity residuals, multi-cutoff/chordal drift, same-checkpoint behavior probes and controlled setup/storage/runtime costs with complete release metadata.

The designated implementation/run specification is `EXPERIMENTS_REQUIRED.md`. The author must review the revised proofs, interpretation, source records, and AI-use disclosure before submission. The private collaboration archive must not be uploaded wholesale as anonymous supplementary material.

## Reproduce the checks

```bash
python3 scripts/build_mixing_tables.py --check
python3 scripts/analyze_archived_layers.py
python3 scripts/check_spectral_algebra.py
python3 scripts/check_review_additions.py
python3 scripts/audit_manuscript.py --pdf /path/to/neurips_2026.pdf
```

The last command expects the corresponding TeX log alongside the PDF. Generated-region checks preserve ordinary prose edits; if an author changes a generated number, reconcile the source record explicitly before regenerating.

The final 2026-09-14 revision also compiled in an independent clean output directory. All 21 pages matched the primary build in extracted text and rendered pixels, despite the documented nonfatal Tectonic rerun notices.
