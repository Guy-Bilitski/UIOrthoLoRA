> **Current preparation status — 19 September 2026:** Start with
> `EXPERIMENT_PREPARATION_HANDOFF_20260919.md` and
> `review_feedback/20260919/assessment.md`. The author confirmed **two GPUs**.
> All 18 practical and 21 RTE band/head confirmations are complete; their
> protocols and immutable evidence remain unchanged. Prepare the remaining
> checkpoint evaluations and proposed controls/decoder pilot. New training
> proposals require the agreed scope and budget; older queue, title, completion
> and four-GPU statements below are historical. The approved abstract and
> introduction remain unchanged. This update does not change live server jobs.

# Manuscript audit — review-panel revision, 2026-09-14

## Current section-by-section writing revision — 2026-09-16

The subsequent abstract revision states adapter ownership explicitly and replaces
the detailed seed/calibration/block-percentage passage with a plain-language
comparison. The introduction is aligned with this contribution statement.
Current validation build: `../build/plain_abstract_20260916/neurips_2026.pdf`.
The earlier section-wide validation below remains applicable to unchanged content.

The title is **Controlling Interaction in Spectral Fine-Tuning**. The author's
latest instruction centers every section on the completed interaction-versus-size
comparison, expressed in simple language. The main experiment leads with held-out
prediction loss and accuracy, followed by the size sweep, update allocation and
probe behavior. The strict-tail experiment remains separately documented in the
appendix. `SECTION_READTHROUGH_20260916.md` maps every section to its purpose.

All source/data/algebra/figure audits pass. The final PDF has 35 pages total,
nine main pages and the complete historical GLUE table on page 8. References are
resolved; no overflow or missing-glyph errors were found. All main pages were
rendered and visually inspected. The new grayscale outcome plot shows every
training seed, verifies its CSV against the frozen manifest, and reproduces
exactly from its generator. All displayed equations, aligned equations and
numeric tabular bodies remain byte-identical to the preceding manuscript.
Raw and derived evidence files are unchanged.

Validation build: `../build/interaction_story_20260916/neurips_2026.pdf`.
The refreshed standalone archive passes its analysis suite and, after fresh
extraction, every distributed file hash. Its manuscript equals the current source.
No new task evaluation, experiment, shared implementation change or GPU action
occurred. Earlier revision descriptions below are historical.


## Current author-directed tail-adaptation refactor — 2026-09-16

The abstract was subsequently revised at the author's request to state the
completed finding directly. Independent recomputation from all 18 frozen
held-aside rows gives MIX versus NORM mean-loss reductions of 43.9603% (RTE)
and 47.7679% (MRPC), with mean accuracy differences +0.1203 and -0.3268 points.
The abstract rounds these to 44%/48% and +0.12/-0.33 points and explicitly places
them in the practical leading-plus-tail experiment. It makes no strict-tail,
rotation, accuracy-equivalence or calibrated-loss claim. The pending comparisons
and mechanism qualifications remain in the main text. Only the abstract changed
in the manuscript; the new PDF and refreshed supplementary archive pass their
checks. Current validation build: `../build/abstract_results_20260916/`.

The active title is **Adapting in the Spectral Tail**. The author explicitly
reopened writing and recentered it on strict tail capacity, within-tail rotations,
and the behavior of controlled leading–tail interaction. Adapters precede
diagnostics; a new main table leads with held-aside loss/accuracy. Existing data
and proof content remain, while the full bounds and detailed inner-selection
table move to the appendix. `TAIL_ADAPTATION_EVIDENCE_HANDOFF_20260916.md` records
the revised priorities and the central missing export: paired tail/head/rotation
confirmations and learning curves. Neither the narrative refactor nor a clean
compile establishes those empirical results. Historical status notes follow.

Validation completed for this refactor: 36 PDF pages, nine main pages, historical
GLUE on page 8, 98 unique labels and 31 cited keys. All required source/algebra/data
audits and the new behavior/band-snapshot generator pass. The PDF has no undefined
references, overflow or missing-glyph errors and retains grayscale text. Main
pages 1, 2, 5, 6, 8 and 9 were rendered and visually inspected. The regenerated
sanitized archive passes every hash and analysis check after extraction to a
fresh directory without Git history. Raw experimental exports are unchanged.

Internal author document. Scope: Overleaf project `6aa54397e58b10444b0fa2aa` only. This is not an anonymous submission artifact.

## Current review-v3 update — 2026-09-16

Supersedes presentation details in the dated notes below. The current title is
**Magnitude and Mixing in Spectral Fine-Tuning**. REVIEW_PANEL_V3_RESPONSE.md maps
all roadmap items and corrects overstatements in the simulated review. The paper
separates penalty-only predictions from finite task-trained endpoints and reports
the one-seed magnitude frontier explicitly. New analysis reports all held-aside
NLL contrasts, within-task frontier fits, and hypothetical paired-t power
sensitivities with an independent numerical check. PSOFT coordinates, conditional
parameter counts, notation and the metric specification are clarified. No new
training or held-aside fitting is performed; requested inner-selection logits
are documented in REVIEW_V3_EVIDENCE_HANDOFF_20260916.md. The refreshed analysis
supplement must accompany the PDF in the next reviewer handoff.

## Current review-v2 update — 2026-09-16

This section supersedes dated experiment-availability statements below. See
`REVIEW_PANEL_V2_RESPONSE.md` for every MUST/SHOULD disposition and corrections to
the simulated panel's algebra/statistics; see `PAPER_FINALIZATION_STATUS_20260916.md`
for the current handoff. New evidence at `a63b1f2` supplies all 18 frozen held-aside
evaluations and an eight-checkpoint band freeze ledger. Held-aside predictions,
labels, metrics, loading records, checkpoint identities and file hashes reproduce.
The older band outcome CSV still contains three confirmations and two pilots;
freeze coverage is not a substitute for missing outcome rows.

The revised paper reports signed norm errors, total/learned equal-module cross,
identity alignment and matched-subset small-norm sensitivity. MIX's learned leading
identity fractions are 84.67% RTE / 89.50% MRPC. They qualify a directional mechanism,
not demonstrate scaler homogenization. Held-aside MIX–NORM accuracy differences
are +0.12 / -0.33 pp with wide nominal paired intervals; neither task benefit nor
equivalence is established. The historical six-task table stays in main text with
all task means/SDs; its unmatched Avg6 display is removed. Incomplete band numerical
tables are withheld pending the complete registered block. Theory/prior-method
coordinates, costs, citations and retrospective retention exclusion are clarified.

Current compile: 32 total pages, nine main pages, full historical GLUE on page 8.
All required source/data/algebra/PDF audits pass, including the new review analyzer.
A sanitized standalone analysis supplement is prepared and tested separately from
the private archive. Centered scaler exports and the full historical retention
sweep inventory remain requested; complete block B and actual supplementary
attachment/hosting remain outstanding. No training or shared implementation changes.
The Overleaf compiler on the current project head remains canonical.

## Historical status notes (superseded where noted above)

## Current analysis update — 2026-09-16, final export integration

The export at `5510161` resolves the earlier learned-delta and protocol handoff
requests. `scripts/analyze_final_evidence.py` independently joins all 18 fixed
confirmation checkpoints/validation hashes and reaggregates their 864 module
records. All three views (initial, total and learned-since-insertion), both
weightings, and pooled norms agree to a maximum absolute error of 7.22e-16.
Both invalidation records exclude all 39 corrupted-campaign IDs. The original
probe CE recomputes to 2.095626057818126 over 4,001 masked tokens. Tokenizer parity
reports 181 exact examples; this session checked the record, not tokenization again.

The manuscript now reports learned leading allocation and absolute module-energy
concentration. Pooled MIX leading fractions are 45.2% RTE and 65.5% MRPC after
subtracting insertion; equal-module fractions are 67.3% and 64.7%. The latter
prevents a blanket task-dependent-placement claim. Equal-module initialization
corrections can reach 6.6 and 13.4 pp; initialization is not called negligible.

Band presentation was fixed before confirmation inspection, after disclosure of
the tail pilot. The new appendix includes 3/21 confirmations and two separate
pilots at the 07:04 UTC export, all planned contrasts and pending cells. No band,
rotation or head comparison is claimed complete. Official splits remain untouched;
`LOCKED_EVALUATION_REQUEST_20260916.md` specifies the separate frozen evaluation.

See `PAPER_FINALIZATION_STATUS_20260916.md` for current readiness. The September 14
verdict and detailed review history below are retained as historical records,
not current experiment availability. The canonical PDF is Overleaf's compilation;
local PDFs are validation builds. No GPU job or shared research source changed.

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
