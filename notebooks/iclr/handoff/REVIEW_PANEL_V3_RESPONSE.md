# Response to review panel v3 — 2026-09-16

Internal author record for the isolated ICLR paper; not anonymous supplement material.
Reviewed snapshot: f9ff3d2, 32 pages. Source: ../reviews/review_panel_v3/.
This is a simulated panel, not an ICLR decision. Its 6/10 aggregate and polarized
individual decisions motivate inspection; they do not certify correctness or acceptance.

## Revision disposition

| Roadmap item | Action and evidence |
|---|---|
| M-1 Predicted versus empirical content | Rewrote abstract, introduction, Section 4.2 and Section 5.2. The scalar limit predicts zero cross and identity leading structure; declared scalar initialization already has zero mixing penalty. Joint task training produces the measured departures. The magnitude frontier and residual finite-penalty geometry are the empirical observations. Added an explicit factor-gradient counterexample to the panel's stronger entailment assertion. |
| M-2 Title scope | Retitled **Magnitude and Mixing in Spectral Fine-Tuning**. Confinement remains a theoretical distinction; the incomplete band block is still explicitly pending and carries no empirical conclusion. |
| M-3 Parameter counts | Table 2 now displays configuration-derived adapter-only counts, marked by dagger: UILinLoRA/UIOrthoLoRA 0.086/0.479M base and 0.221/1.008M large. The formula is modules × (256 + 2d + 2q²). These are not recovered inventories for the historical score checkpoints; task heads/bases are excluded. All original task means and SDs and source counts remain in CSV. |
| M-4 Post-treatment conditioning | Main Section 5.3 and sensitivity appendix explicitly identify attained norms as post-treatment variables. Threshold restriction tests sensitivity to small modules; it does not control allocation. A prospectively calibrated per-module-norm intervention is the missing design. We do not assert that a centered-scaler arm alone identifies a direction-only effect. |
| M-5 PSOFT | Primary-source checked asymmetric factors and internal scales. Added C_JJ=(diag(alpha) R diag(beta)−I) Sigma_J to the coordinate table; principal support remains fixed under vector relaxation. Distinguished its column-Gram preservation condition from our support bound. Numerical support/Gram checks pass. |
| S-1 Held-aside loss | Reported all six paired NLL contrasts, their seed values, means/SDs and nominal CIs; highlighted the ordering in abstract/main text. MIX means 1.229/0.395 nats versus NORM 2.193/0.755 and UNREG 2.429/0.871 (RTE/MRPC), lower in each pair. Temperature scaling is explicitly pending inner-selection logits and must not fit official held-aside labels. |
| S-2 Historical/fresh score gap | Explicitly contrasts historical base RTE 76.2–78.7% with fresh held-aside 71.2–71.7%. Recipe, rotation, seeds and checkpoint selection differ. The gap is neither an estimate nor a bound on selection optimism. |
| S-3 Power sensitivity | Added a proper hypothetical two-sided paired-t calculation: n=3, alpha=.05, Gaussian independent seed differences, 80% power. Required effect is 3.2640 times assumed difference SD. All accuracy/F1, task-NLL and probe contrasts are tabulated using observed SDs as uncertain scenarios. Noncentral-t calculation independently checked by normal/chi-square quadrature. Not achieved power, an exclusion bound, or an equivalence test. |
| S-4 Artifact access | Refreshed supplement/analysis_artifact.zip with this paper, new analysis, hash manifest and standalone verifier. Give the panel the ZIP together with the PDF. The previous ZIP already existed; a PDF-only review could not audit it. No public hosting or conference submission has been performed. |

## Further reviewer comments

- Shortened repeated scope language; the conclusion now gives a concrete reporting action.
  The empirical section names the one calibration seed and one MIX dose; neither
  the whole frontier nor the MIX dose response is claimed replicated.
- Recomputed descriptive log-norm frontier fits from unrounded CSV values. RTE
  slope −2.9052 and MRPC −2.8999 pp per log norm imply illustrative 8% shifts of
  0.2236/0.2232 pp. Maximum fit residuals are 0.8780/0.3008 pp. No fitted slope is
  called a causal or Lipschitz bound, and no million-fold extrapolation is made.
- Figure 2 distinguishes confirmation by triangle shape and fill, and its caption
  gives the matching-error ranges. No confirmation points are dropped or retuned.
- Added a notation guide, named the NORM penalty explicitly, clarified learned
  delta/per-seed sensitivity ranges, and neutralized the first-submission wording
  while retaining that the sensitivity thresholds were chosen after inspection.
- The reusable report now includes identity residuals, per-module raw values,
  cumulative energy and threshold-count curves, and both stated thresholds.
  It explains approximate-SVD tolerance and diagnostic cost. Cumulative energy
  alone does not recover relative-norm thresholds.
- Reduced archive claims to contextual observations; n=1/2 and SD up to 5.63 pp
  appear in the main text. The lower score-change endpoint remains labeled by the
  archive's limited replication, not promoted to a population estimate.
- Added GaLore's changing gradient subspaces and GOFT/qGOFT in the coordinate
  appendix; distinguished DoRA's column-wise weight norm from our update norm.
- Added the multi-adapter serving distinction without claiming bases must be
  duplicated: pretrained bases can be shared, whereas separately cached merged
  weights consume a dense copy per task. No measured deployment advantage claimed.
- Exact GPT-2 module/dimension inventory, historical GLUE count conventions and
  retention sweep count/IDs remain unavailable. Additional decimals, MRPC F1 for
  accuracy-only records, or missing execution configurations are not fabricated.
- LoRA geometry, per-module matching, CENTER/random-projector arms, MIX dose sweep,
  further calibration seeds, cutoff diagnostics and retention batteries remain
  research opportunities, not experiments secretly implied by this writing pass.
  The existing band/head-only study is not duplicated or reallocated here.

## Corrections to panel inferences

1. **The trained contrast is not forced by norm invariance.** Radial flow is true
   for freely optimized delta under a pure squared-norm objective. In factor
   coordinates, delta-dot = −2 beta J Jᵀ delta, generally nonradial. The new exact
   2×2 example changes cross share under a magnitude-only factor update. Task loss
   and AdamW add further reasons not to assert invariance of trained endpoints.
2. **The MIX scalar limit is conditional, not the measured endpoint.** Projector
   connectivity is not established for the experiment. Penalty-only flow does
   predict a scalar limit for connected graphs, but it cannot predict measured
   10%/4.5% cross shares or 84.7%/89.5% identity alignment exactly. Nor is cross
   share a linear measure of percent progress toward a fixed point. The actual
   scalar initialization already lies in the zero-penalty set.
3. **Strict band is not the same constant-scaler ablation.** Setting E=D=I with
   S_L=I leaves a leading identity core. Existing strict-band jobs omit that core
   and start at zero delta. Neither is a trained centered-scaler control.
4. **The 13× span combines tasks.** Within-task spans are 12.1008× RTE and
   8.4693× MRPC. A pooled extrema ratio is not either task's dose response.
5. **A fitted slope is not a bound.** One-seed dose variation changes the whole
   trained endpoint. The descriptive 0.22 pp calculation neither neutralizes a
   protocol deviation mathematically nor proves a magnitude-independent effect.
6. **Historical derived counts do not recover provenance.** They are useful
   conditional cost annotations, but do not dissolve the missing-inventory issue.
7. **Temperature scaling does not settle causal attribution.** It must be fitted
   off the held-aside labels. A remaining NLL gap could reflect other training
   effects; an erased gap would support a logit-scale account, not prove a unique
   mechanism. The review's within-3% norm statement is also approximate: ratios
   of arm-mean norms differ by about 3.1% RTE and 3.9% MRPC.
8. **Historical/fresh differences do not bound selection optimism.** Protocol and
   arm changes can act in either direction; naming confounds does not create a bound.
9. **Serving arithmetic is not adapter storage.** The 192× arithmetic ratio is
   valid for the stated dense branches; assigning it to per-task trainable storage
   is false, and shared bases need not be replicated per task.
10. **Panel reproducibility language is not a verification of the experiment.**
    Its PDF arithmetic checks cannot independently establish untouchedness or raw
    checkpoint validity. This revision audits exported records without claiming a
    new training/checkpoint reload. Fixed reviewer settings also do not eliminate
    stochastic review variation, so score changes are not a controlled causal estimate.

## Primary sources checked

- PSOFT, [version 3, equations 6–8 and geometry condition](https://arxiv.org/html/2505.11235v3).
- Temperature scaling, [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), section 4.
- [GaLore, ICML 2024](https://proceedings.mlr.press/v235/zhao24s.html) and its [changing gradient subspaces](https://arxiv.org/html/2403.03507v2).
- [GOFT/qGOFT](https://arxiv.org/abs/2404.04316), ICML 2024.
- [DoRA column-wise weight decomposition](https://arxiv.org/html/2402.09353v3).
- [SciPy noncentral-t definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html); numerical power independently cross-checked.
