# Response to review panel v2 — 2026-09-16

Internal author record for Overleaf 6aa54397e58b10444b0fa2aa; not anonymous submission material.
Panel source: `../reviews/review_panel_v2/`. Reviewed manuscript was 9d09a88 (30 pages).
The packet is a simulated review, not an ICLR decision. Its recommendations are
assessed against the source and the evidence, rather than accepted by vote.

## Changes and evidence

The manuscript title is narrowed to **Magnitude, Mixing, and Confinement in
Spectral Fine-Tuning**. The main contribution is a measured comparison of two
penalties, with algebra delimiting possible interpretations, rather than a causal
account of why the pretrained frame matters. The six-task historical table remains
in the main paper under the author's standing instruction; all 60 task means and
SDs remain intact. A shared Avg6 column is removed from the display (source values
remain). Imported parameter counts are restored, while unreconciled spectral
inventories are marked explicitly rather than filled with configuration-derived
counts. Basis storage and arithmetic costs accompany the score discussion.

### New results from evidence already available

- Identity alignment: MIX total leading-energy-weighted identity fractions are
  84.67% RTE and 91.38% MRPC; learned fractions are 84.67% and 89.50%.
  Non-identity leading residuals are 6.94% and 6.89% of all learned energy.
  This qualifies “leading modification”: most of the leading energy is compatible
  with an identity shift, but the data do not prove scaler constancy or a mechanism.
- Small-module sensitivity: using the same modules in each pair, selected by both
  total relative norms being at least 1e-3, learned NORM–MIX equal-module cross gaps
  remain 30.03–33.61 pp RTE and 36.35–38.02 pp MRPC across seeds. All thresholds,
  counts, values and paired gaps are reported; the analysis is labeled post-review.
- Held-aside evaluation arrived at a63b1f2 while this revision was underway.
  All 18 frozen checkpoints join to the request, original validation identities,
  loading checks and per-example records. Recomputed accuracy/F1 match exactly;
  losses agree within float32 aggregation tolerance. On RTE, UNREG/MIX/NORM
  accuracy is 71.72/71.36/71.24%; on MRPC it is 86.19/85.46/85.78%.
  MIX–NORM accuracy differences are +0.12 pp (nominal CI -3.92,4.16) and
  -0.33 pp (-3.89,3.24). They do not establish equivalence or a task advantage.
  The held-aside export is separate from inner-selection and historical scores.
- The band freeze ledger now contains eight of 21 checkpoints. Partial numerical
  tables/figure are replaced by the registered pending-block note. Prior outcomes
  remain in their timestamped raw data and audited analysis JSON. No favorable
  pilot or seed is substituted, and no completed-block conclusion is claimed.

`scripts/analyze_review_v2.py --write/--check` regenerates the identity,
small-module sensitivity, held-aside seed and paired tables from the frozen data.
It checks module identity-residual accounting, checkpoint/report joins, every
per-example prediction and label set, and the evaluation manifest/CSV hashes.

## MUST-item disposition

| Item | Disposition | Manuscript evidence / remaining dependency |
|---|---|---|
| M-1 Shape-invariance interpretation | Addressed, with corrected algebra | Section 4: fixed-core joint rescaling gives delta -> t² delta and squared-norm penalty -> t⁴; fractions stay invariant. This is an available shrinkage direction, not an optimizer guarantee. Section 5 interprets the observed residual movement. |
| M-2 Frame specificity / identity / scalers | Identity analysis addressed; scaler export pending | Main Section 5.3 and identity appendix report all arms under both delta definitions; centered E/D moments requested in REVIEW_V2_EVIDENCE_HANDOFF_20260916.md. CENTER/random-projector training is optional, not claimed done. |
| M-3 Equal-module robustness and dimension reference | Addressed | Table 1 adds total equal-module cross and dimension-only row; figure adds reference line; main text adds learned equal-module shares with the correct normalization. No isotropy claim. |
| M-4 Historical table | Addressed within standing author constraint | Table stays in main text. Shared Avg6 removed; imported source counts restored, unknown spectral inventories marked; source dispersion convention and configured basis-memory cost captioned. No invented matched-cost comparison. |
| M-5 Prior-method coordinates / S_L=0 | Algebra addressed for representative families; new experiment deferred | Related work and coordinate table derive representative method placements, including singular-value scaling in LoRA-XS and left/right orthogonal conventions. Strict-band jobs do not test the practical scaled S_L=0 corollary; this distinction is now explicit. No new training launched. |
| M-6 Citations / SVFT / HRA | Addressed | Four venue records corrected from primary sources; HRA added; SVFT off-diagonal variants explicitly credited alongside the exact approximation identity. |
| M-7 Addressee | Addressed prospectively | Introduction names researchers evaluating regularizers and matched-norm controls; it does not invent a prior published equivalence claim. No task benefit is assigned to block allocation as an optimization target. |
| M-8 Reachable artifact | Packaged and tested; external delivery still required | `supplement/analysis_artifact.zip` contains sanitized data/scripts, a source/distributed hash manifest and a CPU verifier; it passes standalone checks without private Git history. It must be attached to the actual review/submission; no public URL or checkpoint availability is fabricated. |
| M-9 Excluded retention evidence | Partially addressed; inventory requested | Appendix E gives 40 displayed conditions across two named models/two tasks and states retrospective exclusion after outcomes appeared in an earlier draft. Displayed conditions are not misreported as training-run counts; full sweep count/IDs remain unavailable. |
| M-10 Statistical wording / signed errors | Addressed, MDE label declined | Signed delta_s added while absolute e_s remains the registered tolerance measure. Removed isolated directional accuracy commentary. Nominal interval half-widths are called imprecision, not power-based MDEs. No retrospective retuning/widening of the matching criterion. |
| M-11 Partial band bulk | Addressed | Numerical incomplete tables/figure removed from manuscript pending the full block; registration, latest 8/21 freeze count and archived outcomes retained. |

## SHOULD and minor-item disposition

| Items | Response |
|---|---|
| S-1 More drift/bound/cutoff diagnostics | Existing-record export requested where available; ranked-subspace measurements cannot be reconstructed from block fractions. No stability result is asserted from their absence. |
| S-2 Small-module sensitivity | Completed for both suggested floors, both delta views and identical module subsets; all paired seed values retained. |
| S-3 Head-only | Already registered in block B; do not duplicate jobs. Remains necessary before assigning task benefit to the adapter. |
| S-4 Merge interference | Optional follow-up, not automatically run. Define task heads and total/learned delta and insertion handling prospectively; a merged accuracy alone would not identify geometry causally. |
| S-5 Calibration variability | Added observed CV diagnosis; one-seed target does not guarantee a 5% per-pair match. No assertion that matching was mathematically impossible and no outcome-conditioned tolerance change. |
| S-6 Scaling costs | Added per-matrix 2.25/4/64 MiB bases at d=768/1024/4096 (two bytes), relative storage/arithmetic and float32 caveat; no unmeasured hardware usability threshold. |
| S-7 Probe comparison | Reports MIX's mean above both UNREG and NORM on both tasks, with single-corpus limitation. |
| S-8 Second probe | Optional extra evaluation; single 256-example corpus remains explicitly scoped. |
| S-9 Sequential forgetting/merging bridge | Prospective extension, not an inference from parallel single-task checkpoints. Existing RTE/MRPC adapters do not constitute sequential continual learning. |
| S-10 Chordal metric naming | Existing formula is a normalized projection-Frobenius/chordal distance; terminology retained, no new empirical claim. |
| S-11 Distributional null | Declined as unsupported by a point expectation alone. A valid orientation null should preserve chosen update singular values/rank and specify sampling; no claim of “indistinguishable from isotropy” is made. |
| Other minor wording | Explained cutoff at first experimental use; clarified initialization and identity subtraction; fixed RTE/MRPC ordering; removed undefined health-flag claims from main text; marked heatmap as medians only; clarified rank-q family; archive score range now -1.03 to -0.23; training/probe/held-aside splits distinguished. |

## Corrections to the packet that affect the revision

1. **Scaling is not an unchanged update.** M-1's instruction says joint scaler
   rescaling leaves the effective update unchanged while its norm falls. With a
   fixed core the update scales by t². Keeping it unchanged requires compensating
   core scaling in the tail-only case, a different identity with different limits.
   A shape-preserving direction does not make NORM structurally unable to change
   shape; the observed frontier itself disproves that universal assertion.
2. **A constant-scaler fit does not reproduce the full block profile.** It fits an
   LL/TT ratio with a free core norm but predicts exactly zero cross energy. MIX
   has nonzero cross energy. High identity alignment is compatible with, not a
   unique proof of, scaler homogenization.
3. **Do not mix total and learned weightings.** The roadmap's example pairs RTE
   learned MIX 10.0 with total NORM 46.3 and gives MRPC NORM approximately 46;
   the correct learned equal-module pairs are 10.035/46.251 and 7.687/45.061.
4. **Point-null proximity is not isotropy or its absence.** The panel itself
   records distinguishability in a synthetic isotropic model; neither that model
   nor its simulated SD is an empirical null test of these trained updates.
5. **The proposed MDE is an observed t-interval half-width.** A power-based MDE
   requires a target power, alternative and variance model. We keep descriptive
   CIs and do not attach an incorrect power label or infer equivalence.
6. **Small modules need not contribute zero cross fraction.** Direct recomputation
   is necessary; the review's 48/live-count rescaling is not justified for small
   nonzero norms. The actual matched-subset result is reported.
7. **LoRA-XS uses Sigma in its left fixed factor.** Its literal published formula
   is C_JJ=Sigma_J R (or an absorbed/reparameterized core), not simply R.
8. **Eight frozen band endpoints are not a full updated outcomes bundle.** We use
   the newer freeze count only for coverage; we do not fabricate new per-arm
   metrics by extending the older five-row export.
9. **Identity fields were already exported.** The panel reviewed a manuscript that
   omitted them, not the available JSONL. This revision computes them now. Scaler
   vectors remain a separate missing export; “not exported” is scoped accordingly.
10. **Preserve registration and resource scope.** Recalibrating to confirmation
    outcomes or widening tolerance post hoc would change this completed study.
    Six CENTER confirmations do not include the separate calibration budget.

## Primary-source checks

- [SVFT: update form and sparsity patterns](https://arxiv.org/html/2405.19597v1), sections 3–4.
- [LoRA-XS: fixed A=U_r Sigma_r and B=V_r^T](https://arxiv.org/html/2405.17604v2), section 3.2; [ECAI 2025 acceptance](https://ecai2025.org/accepted-papers/).
- [OFT: orthogonal and rescaled variants](https://arxiv.org/html/2306.07280v3), sections 3.2–3.5.
- [HRA: right multiplication by a reflection chain](https://arxiv.org/html/2405.17484v2), section 3; [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html).
- [Shuttleworth et al., NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/ff541950d1e885af90f523571564a401-Abstract-Conference.html).
- [RandLoRA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/382b95a6580cfb0b5ca33c74b4e0e770-Abstract-Conference.html).
- [FourierFT, ICML 2024](https://proceedings.mlr.press/v235/gao24o.html).
