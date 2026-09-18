> **Current preparation status — 19 September 2026:** Start with
> `EXPERIMENT_PREPARATION_HANDOFF_20260919.md` and
> `review_feedback/20260919/assessment.md`. The author confirmed **two GPUs**.
> All 18 practical and 21 RTE band/head confirmations are complete; their
> protocols and immutable evidence remain unchanged. Prepare the remaining
> checkpoint evaluations and proposed controls/decoder pilot. New training
> proposals require the agreed scope and budget; older queue, title, completion
> and four-GPU statements below are historical. The approved abstract and
> introduction remain unchanged. This update does not change live server jobs.

# Additional experiments for the ICLR spectral-adaptation paper

## Latest author instruction — tail-adaptation narrative, 2026-09-16

Follow `TAIL_ADAPTATION_EVIDENCE_HANDOFF_20260916.md` before the dated overrides
below. The paper now centers strict tail capacity and within-tail rotations.
The head-only/tail-diagonal/tail-rotation block and its loss/accuracy trajectories
are essential evidence. Refresh existing outcomes, prioritize missing registered
core runs, and retain every original arm/pilot/failure in the ledger. CENTER is
secondary; existing-checkpoint calibration may proceed without delaying this
block. Do not alter registrations or resource limits, or relabel practical
leading-plus-tail results as strictly confined measurements.

## Author priority override — 2026-09-16, after v4

The writing is frozen. Follow `EVIDENCE_PRIORITY_HANDOFF_20260916.md`: existing
checkpoint logits/temperature scaling first, CENTER calibration and six
confirmations next, then head references; inspect any surviving LoRA checkpoint
without new training. Defer remaining strict-band launches and module-norm-matched
training. The dated execution plans below remain historical registration context.
No GPU scheduler was changed by this paper session; the resource owner must
reconcile the live queue and current device/storage budget before acting.

Project: `6aa54397e58b10444b0fa2aa` only. Created 2026-09-13; expanded after the review panel on 2026-09-14.
Updated 2026-09-15: the deadline-scoped plan below takes priority. The server agent reports that the repaired-input campaign is underway; this local document is not a live run ledger.
This is the designated experiment plan, kept with the Overleaf source for author access. It is an internal planning document, not a completed-results section or a submission artifact.

Start a fresh GPU session with `GPU_HANDOFF.md`. Review disposition and reasons for non-adoption are in `REVIEW_PANEL_RESPONSE.md`. The review packet is a supplied simulated panel, not an actual ICLR decision. No proposed outcome below is assumed true, and no training launch is implied by this document.

## Current execution plan — 2026-09-15, two-GPU budget

The author has **4 days to the abstract and 11 days to the full paper**, measured from this planning request. Budget against **two assigned RTX 3090 24 GB GPUs** throughout; four available GPUs accelerate the same queue. These are author-supplied planning windows, not independently verified conference deadlines. Record their exact local timestamps in the server ledger.

This section supersedes the older execution order, nine-arm first tranche, 66-run launch target, and five-seed target below. Those sections remain a methodological reference/backlog, not jobs to launch automatically. Preserve the clean focused study's already registered protocol; additions get a separate, versioned registration and source snapshot. Do not amend active jobs to implement the new band experiment.

### Scientific target

1. **Location:** how does useful adaptation differ when it is restricted to equally sized leading, middle, or tail singular bands?
2. **Flexibility:** within each band, what changes when diagonal coefficient tuning gains partial rotations?
3. **Interaction:** does suppressing leading–tail mixing change update geometry beyond the tested magnitude-only control?

The ongoing UNREG/MIX/NORM campaign addresses question 3 in the practical, unrotated, scaled adapter. It does not by itself answer questions 1–2. The new strict-band study answers those questions in a separately identified family. Together they connect confinement, flexibility and interaction; they are not a single fully crossed causal experiment. In particular, this budget does not test all pairwise leading/middle/tail regularizers or prove that the bands have fixed semantic roles.

### Evidence reset: mandatory carry-forward

The author relayed the server agent's tokenizer incident report: 39 new-campaign runs were invalidated, derived results withdrawn, inputs rebuilt, independent tokenization parity checked on 181 examples, and short learning gates passed. The reported repaired bundle is `preparation_20260915T0707Z`. These are **server-reported checks, not independently rerun here**. Verify their durable manifests and gate artifacts before resuming; the ledger, not this timestamp, determines current execution state.

No corrupted run, fitted dose (including beta=27.5), MRPC frontier, or claimed 10x separation has evidentiary standing. Keep their invalidation records and exclusion checks. Restart calibration from the clean study's registered initial grid, not corrupted dose estimates. The legacy NeurIPS records are a separate archive; the reported incident was new-campaign-only. Do not overwrite either archive or silently revalidate excluded IDs. The manuscript's withdrawal of the invalid focused results must remain in place until clean evidence is validated.

### One queue, in priority order

| Priority | Work | Confirmation runs | Gate / purpose |
|---|---|---:|---|
| A — protect | Finish clean RTE + MRPC UNREG/MIX/NORM calibration and confirmation | 18 total in the existing campaign | 2 tasks × 3 arms × 3 seeds; question 3 |
| A — alongside | Implement reporting and analyze saved checkpoints | No new training | Per-seed matching, task learning, spectral allocation; separately time diagnostics |
| B — next | RTE: 3 bands × 2 flexibility settings × 3 seeds, plus original-backbone head-only references | 18 + 3 = 21 new | Questions 1–2; start only after correctness and timing gates |
| C — replicate | The identical band/flexibility/reference matrix on MRPC | 21 new | Second-task replication; follows A and B |
| D — comparator | LoRA rank 8 on RTE + MRPC, 3 seeds | 6 new | Common-frame external comparator, not proof of a shared mechanism |

The minimum planned confirmation package is **39 runs** (18 existing-campaign + 21 new), excluding calibration, smoke, timing and invalid runs. With MRPC replication it is 60; with LoRA it is 66. This last number is coincidental: it is **not** the older nine-arm/full-FT 66-run suite. Subtract genuinely validated, manifest-compatible runs when estimating remaining work. Do not restart a completed clean run merely to satisfy a new folder convention.

### A: finish the clean study without moving its goalposts

- Retain the repaired tokenizer/input gates, registered objectives, module set, fixed-step endpoints, calibration seed, confirmation seeds and matching rule. The reported focused grid has 10 entries (MIX and UNREG on each task plus three NORM doses per task), with bounded, norms-only refinement; verify the actual clean registration rather than reconstructing settings from this prose.
- Advance each task to confirmation when its own calibration is resolved. A failed match is a reportable outcome, not a reason to search indefinitely. Do not delay the other task behind a global calibration barrier.
- Freeze the chosen task-specific dose before confirmation. For each paired confirmation seed, report the actual matching error `e_s = abs(n_NORM,s - n_MIX,s) / n_MIX,s`, using the **registered** norm definition and tolerance; a zero denominator is undefined. Show every pair, not just a mean norm or calibration match. Do not retune using confirmation geometry.
- Report task scores and loss reduction with the geometry. Preserve accuracy and F1 with explicit labels on MRPC. A geometrically restricted model that fails to learn is not evidence of useful controlled adaptation.
- Keep full dose frontiers, failures and nulls. Three-seed summaries include individual values, sample SD and paired differences; neither modules nor checkpoints are independent seed replicates. Task equivalence and generic spectral specificity are not established by this three-arm study.

### B/C: a small, controlled band-by-flexibility study

Use RoBERTa-base and the same 48 square attention projections, clean task bundle, splits and fixed per-task training budget as the focused study. Register seeds `{42,17,123}`; reserve calibration seed 31415. Freeze a finite, symmetric pilot/tuning budget before confirmation. Use a common declared recipe across band locations within each flexibility setting; if flexibility settings need distinct learning rates, give them the same search budget and report it. Never tune a location more because its result is disappointing.

For each 768-dimensional projection, use the saved pretrained SVD with descending singular values. Define equal-sized bands using zero-based, half-open indices:

- Leading: `[0:256)`; middle: `[256:512)`; tail: `[512:768)`.
- Strict update: `Delta = U_B H V_B^T`, added to the unchanged pretrained matrix. **No ambient diagonal scalers and no additive fixed leading/complement core.** All backbone parameters outside this update remain frozen; train the same task head in every condition.
- Diagonal setting: `H = diag(h)`, with 256 signed trainable coefficients.
- Partial-rotation setting: retain the same 256 coefficients, and rotate the last 64 directions **within the chosen band** on both sides. Thus `H = blockdiag(diag(h_first_192), R_U diag(h_last_64) R_V^T)`. The ambient rotated indices are `[192:256)`, `[448:512)`, or `[704:768)` respectively. Use the same tested orthogonal-map implementation in all three bands. This is partial flexibility, not a free dense 256-by-256 core or a new rotation method claim.
- Initialize `h=0` and rotations to identity so every condition starts at the exact same pretrained effective backbone. Validate nonzero coefficient gradients and usable rotation gradients after coefficients move: zero rotation gradients at the all-zero core are expected initially, but persistent inactivity is a failed gate. Do not silently change initialization to make a run work.
- Include three paired **original-backbone head-only** runs per task. They share initial head state, batch order and task budget with all six conditions. Reuse only an already validated run with identical relevant manifests. These references do not substitute for a frozen-insertion head control in the practical study, whose initial effective model differs.

Tests before launch: band indexing, complete delta reconstruction, no off-band energy except numerical tolerance, equal initial logits, forward/merge agreement, active gradients, correct trainable inventory, save/reload and a learning smoke test. Implement and test in an isolated source snapshot; do not hot-patch the active campaign. If the existing adapter cannot represent this exact family, report the implementation gap and ETA instead of relabeling a different parameterization.

The primary comparison is task adaptation across the six conditions and versus the frozen-backbone references. Report update norm and within-band off-diagonal energy as well. Zero off-band energy is an enforcement check, not an empirical discovery. Location comparisons have equal trainable capacity within a setting; rotations intentionally add capacity, so their effect is not a capacity-matched claim. Neither equal band dimension nor identical step budgets imply matched update magnitude. Do not infer universal superiority of a band from one task.

### Measurements integrated into existing artifacts

Do not restart ongoing runs just to add logging. Use their saved initial/intermediate/final states; record absent snapshots as missing. Preserve the original fixed-step and validation-selected endpoint distinction.

1. **First:** finish the registered final-checkpoint metrics and per-seed match table for A; these must not wait on an expensive new SVD sweep.
2. **Next:** measure all nine leading/middle/tail block energies in the fixed equal-thirds frame, both absolute and normalized. Retain total-from-pretraining and learned-since-insertion deltas, pooled energies and per-module summaries. The old binary cutoff is leading+middle versus tail: summing the appropriate nine blocks must reproduce its four blocks. Do not silently redefine legacy `major/medium/minor` implementation labels.
3. **Then, where saved states and time permit:** plot training progress at 0/10/25/50/75/100%, task score/loss, norm, block allocation and subspace drift. Existing additional snapshots may be retained. At zero delta, energy fractions are undefined. Time expensive drift/SVD diagnostics separately; endpoints take priority over dense trajectories.
4. Reuse existing training-loss, gradient-norm, clipping and failure logs when available. Spectral/subspace stability is not optimizer stability. Without replicated optimization measurements, describe geometry and task learning, not a demonstrated training-stability mechanism or knowledge retention.

Deliver three compact analysis outputs: band × flexibility task results; clean MIX/NORM/UNREG geometry-versus-achieved-norm with per-seed match status; and, if checkpoints support it, spectral trajectories. Calibration dose curves are exploratory; distinguish them visually from confirmation points.

### Deadline and capacity gates

Treat day 0 as this request. These are scheduling targets, not promised runtimes:

| Window | Required focus | Cutoff |
|---|---|---|
| Days 0–1 | A continues; prepare reporting and B's CPU tests/smoke/timing | Do not displace A for unvalidated new code |
| Days 1–3 | Complete and audit A; run B if its measured finish fits | By day 3, freeze the evidence used in the abstract |
| Day 4 | Submit an abstract supported by validated observations | B may continue for the full paper; no pending result becomes an abstract finding |
| Days 4–8 | Finish B, then C; D only if the schedule still fits | If C cannot fit, consider the smaller D instead; record the scope choice using runtime, not favorable outcomes |
| Days 9–11 | Freeze new launches; extract, validate, write and audit the full paper | Reserve at least the last 48 hours for analysis/writing, not a new sweep |

On **two GPUs**, A owns both by default until its queue is clear. A spare assigned slot may run B only when no ready A job is delayed; CPU implementation/reporting proceeds alongside. On **four GPUs**, protect two slots for A and use the other two for ready A work or validated B, whichever advances the declared deadlines. Do not reserve idle devices while useful priority work is ready. Preserve per-task calibration dependencies.

If allocation drops, checkpoint and resume only this project's jobs on the explicitly remaining assigned devices. Recompute the queue immediately; do not touch another project's processes. Resource changes never justify bypassing an input, learning or artifact gate.

Preserve the server's stage-specific storage authorizations as well as its GPU limits. This plan grants no extra disk quota and does not make calibration and confirmation allowances interchangeable. Forecast retained checkpoints, shared SVD references and diagnostic artifacts before admitting B/C/D; use compact reproducible states, and request a resource decision if a complete block cannot fit. Do not delete invalidation evidence or another campaign's artifacts to create capacity.

Ask the server agent to return, after the next timing gate:

- Valid completed/remaining counts, actual clean registration and input/source fingerprints, and exact deadline timestamps.
- Measured per-run setup/training/evaluation/diagnostic times and peak VRAM, including a diagonal and rotated band pilot; retained-storage forecast against each authorized output root. No memory-fit assumptions from parameter counts alone.
- Remaining GPU-hours and finish ranges under **two and four GPUs**, including serial calibration/refinement, reload/extraction and a stated contingency allowance. `remaining GPU-hours / GPU count` is only a lower bound, not an ETA.
- Which complete blocks A/B/C/D fit before day 3 and day 9; update this forecast on failures, stalls, calibration decisions and GPU reassignment.

Never remove a difficult seed or a losing condition to hit a deadline. If a block cannot finish, retain its partial data as exploratory and defer the complete claim. Reduce breadth before weakening correctness or hiding variability. Four GPUs should first buy earlier completion and replication, not extra hypotheses.

**Deferred beyond this deadline-scoped queue:** CENTER/DECAY_INIT/RANDPROJ, extra penalty arms, five-seed expansion, full FT, additional PEFT baselines, random adapter frames, pure-tail mixing replication, new generative/retention campaigns and dense diagnostic sweeps. They remain scientifically useful but are not prerequisites for reporting the narrower three-question study honestly. Their absence limits claims of pretrained-frame specificity, cross-model generality and retention; reframing does not erase those limitations.

## Expanded methodological reference and backlog (2026-09-14)

The sections below retain the earlier design details and reviewer context. The current execution plan above governs scope, order and run counts; do not launch the expanded suite from the older instructions below.

### Objective and priority

The principal question is whether suppressing leading–tail interaction changes the geometry of adaptation beyond the effects of making the effective weight update smaller. A second question separates the available spectral subspace from flexibility inside it. Training trajectories are measurements added to these runs, not a separate benchmark campaign.

Existing results remain evidence about the documented practical adapter. They must not be relabeled as new tail-only experiments. UIOrthoLoRA and its simpler counterparts are experimental instruments: the goal is to understand spectral components and regularization, not to establish a best adapter. Comparisons below test geometric explanations and their scope, not a leaderboard claim. Following the NeurIPS reviews, a representative spectral-baseline comparison and modern-backbone mechanistic replication are important parts of the expanded plan, after the matched control and verified diagnostics.

## Environment information needed when the GPU is ready

- Access method, workspace path, GPU model/count/VRAM, explicitly assigned device IDs, and maximum GPU-hours or wall-clock allocation.
- Whether model/dataset downloads are allowed, existing cache locations, disk quota, and a persistent location for checkpoints and results.
- Any scheduler/queue rules and other jobs that must not be interrupted.
- Any archived configurations/checkpoints for the original runs, especially the second SST-2 ablation seed if it exists.

Create a dedicated code checkout/environment and outputs under an `iclr_6aa54397` namespace. Pin the code commit and imported module paths. Never modify shared `src/peft`, change the other agent's branch, reuse its output folders, or assume an idle GPU is assigned to this project. Keep Overleaf credentials out of the training environment unless separately needed and authorized.

## P0 — correctness and provenance gate

Complete before expensive training. Most checks are small CPU tests; use a short GPU smoke test for the actual backbone.

1. Define `Delta_total(t) = W_effective(t) - W_pretrained` and `Delta_learned(t) = W_effective(t) - W_effective(0)`. Record both when insertion changes the initial model.
2. Exercise the two actual update forms explicitly:
   - Ideal tail-only: `Delta = E U_T H V_T^T D`.
   - Practical leading-plus-tail: `Delta = E (U_L S_L V_L^T + U_T H V_T^T) D`, with `S_L=I` for the documented legacy form.
3. Verify the implemented forward difference, merged-weight difference, and diagnostic delta agree within dtype-appropriate tolerances. Test disabled adapters, merge/unmerge, save/reload, rotations on/off, scalers on/off, and leading block on/off.
4. Save the original SVD bases. Do not reconstruct them independently at every checkpoint. Check `LL+LT+TL+TT` reconstructs the full delta and squared Frobenius energies sum correctly; test square and rectangular layers using complete complementary projectors.
5. Test known cases: a strict tail update, a leading-only update, a cross-only update, zero update, scalar rescaling, and a tail singular-value crossing. Zero-update fractions are undefined, not zero observations.
6. Check factor symmetries: `E -> aE, D -> D/a` leaves either effective update unchanged while changing raw mixing magnitudes. For the freely scalable tail-only core, test `E,D -> aE,aD; H -> H/a^2`. Log the factor/core norms that qualify the theorem.
7. Repair the old diagnostic API mismatch and task-metadata writer in the isolated implementation. Compute mixing metrics even when their loss coefficient is zero. Do not reuse the historical placeholder-zero baseline logs.
8. Check initialization and gradients. All conditions within a contrast must have the same initial effective model. Prefer zero-delta initialization for the ideal hierarchy only after confirming usable gradients; do not initialize every multiplicative factor to zero. Keep the practical legacy initialization as a separate contrast, with its initial offset measured.
9. Reconcile broad GLUE run provenance before any ranking/efficiency claim. Obtain final per-seed summaries, exact configurations, and parameter inventories. The author confirmed on 2026-09-14 that the original broad-evaluation MRPC values are accuracy, matching the imported RandLoRA table; Table 2 now uses one accuracy column and recomputed Avg6. The separate mixing-intervention records use MRPC F1. Evaluate and save both metrics in future common-protocol runs, with explicit metric names. The old rotated parameter counts conflict with the stated recipe; see `data/GLUE_PROVENANCE.md`. Preserve reported records separately from reruns.
10. Test the scaler-penalty commutator identity against the implementation, including disconnected/coordinate-aligned projectors and scalar scalers. Confirm that training sums penalties over adapted modules (48 in the current RoBERTa-base study); do not accidentally average them or reuse lambda after silently changing the convention. The current manuscript includes CPU tests of the exact identity, not a measured connectivity claim for pretrained models.

Deliverables: automated test report, exact resolved configurations, package/model/dataset revisions, initialization-delta report, and a smoke-run artifact that can be reloaded to reproduce the metrics.

## P1 — mixing, shrinkage, and scaler homogenization (central experiment)

### Question and stage gates

Does the pretrained-projector penalty change effective geometry beyond direct update shrinkage, generic scaler homogenization, and head-only learning? Review-panel issues M-1/M-2/M-3 and DA M1 motivate this expanded design. This section supersedes the earlier four-condition/24-run plan. Run IDs, not letters, are authoritative; legacy A/B/C remain unregularized/left-only/two-sided everywhere.

Backbone: RoBERTa-base; query, key, value, attention-output in all 12 layers (48 square matrices); practical leading-plus-tail form with leading core identity, k_val=256, k_vec=0. Same backbone, head initialization, batch order, schedules, dropout, insertion offset, and checkpoint rules within each task/seed contrast. Diagnose and record any unavoidable differences.

Start with RTE and MRPC. Separate calibration seed: 31415. First confirmation seeds: {42,17,123}; target five confirmation seeds by adding {2021,1054} for the decisive contrasts, subject to the timed GPU allocation. Three seeds are a first stage, not adequate evidence for equivalence. Do not stop adding seeds based on which outcome favors the hypothesis.

### Conditions (nine per task/seed)

| Run ID | Trainable components | Added objective / reference |
|---|---|---|
| P1_UNREG | Practical spectral adapter + task head | No explicit intervention penalty; declared common AdamW decay remains fixed |
| P1_LEFT | Same | Historical left-only mixing penalty |
| P1_MIX | Same | Historical two-sided mixing penalty, starting at lambda_E=lambda_D=1e-3 |
| P1_NORM | Same | Effective-delta Frobenius penalty, calibrated to P1_MIX's achieved norm |
| P1_CENTER | Same | Centered-scaler quadratic, calibrated to the same norm range |
| P1_DECAY_INIT | Same | L2 distance of scalers from their actual initialized values, not an assumed identity initialization |
| P1_RANDPROJ | Same SVD-based adapter | Two-sided mixing loss using fixed independent Haar rank-k projectors instead of pretrained projectors |
| P1_HEAD_BASE | Only identical task head | Original pretrained backbone; adapter absent |
| P1_HEAD_INIT | Only identical task head | Practical adapter frozen at its shared insertion state |

The two head references answer different questions because the practical insertion delta is nonzero. P1_HEAD_INIT shares the initial effective model with P1_UNREG/MIX; P1_HEAD_BASE measures the unchanged pretrained backbone. Neither is a norm-matched treatment arm.

Nine arms × two tasks × three seeds = **54 confirmation runs**, plus calibration. Include the two early geometric comparators in P5 in this same first tranche (LoRA-8 and full FT: **12 more**). Thus the first expanded tranche is **66 unique confirmation runs**, not a GPU-hours promise. Five seeds for all eleven conditions would be 110 runs total (44 beyond the first tranche). Extending all eleven to five tasks at three seeds is 165 runs total. These are alternative coverage targets, not additive promises; record the chosen budget before confirmation.

### Exact nuisance controls

For module l, Delta_l means W_effective,l minus the original W_pretrained,l, including insertion offset.

- NORM: beta times the mean over modules of ||Delta_l||_F^2 / ||W_pretrained,l||_F^2. Do not substitute weight decay on factors. A learned-since-insertion variant is a separately labeled sensitivity test.
- CENTER: gamma times the sum over modules of c_l (||epsilon_l-mean(epsilon_l)1||_2^2 + ||delta_l-mean(delta_l)1||_2^2), with c_l=k_l(d_l-k_l)/((d_l-1)(d_l+2)) for square layers. This matches the expected unnormalized random-projector penalty for a fixed scaler; it does NOT imply a fixed random training run has that loss or endpoint. Treat sides separately for unequal dimensions.
- DECAY_INIT: gamma times the sum of ||epsilon_l-epsilon_l(0)||_2^2 + ||delta_l-delta_l(0)||_2^2. The original s0 is 0.01 or 0.1, not 1. This is an explicit loss, distinguishable from decoupled AdamW weight decay.
- RANDPROJ: choose Q_l and R_l once per run as independent Haar rank-k projectors; sum half-squared commutators ||E_l Q_l-Q_l E_l||_F^2/2 and ||D_l R_l-R_l D_l||_F^2/2. Keep the adapter itself in the original SVD basis. Evaluate ALL outcomes in the original pretrained SVD frame, as well as optionally in the penalty frame. Save projector seeds and bases. One random frame per training seed confounds orientation and training variation; use at least three orientation draws at a fixed calibration seed to check orientation sensitivity, and state the limitation if confirmation has only one draw each.

A penalty-only gradient-flow calculation preserves the scaler mean; it does not force the scalar core amplitude to shrink or guarantee ranked stability. Joint task optimization may behave differently. Log mean and centered norm separately instead of interpreting raw mixing norms as a collapse test.

### Calibration, dose response, and matching

1. Resolve loss aggregation before calibration: legacy MIX/LEFT sum over 48 matrices. Do not silently average or reuse coefficients after changing conventions.
2. On the separate calibration seed, sweep MIX and LEFT lambda in {1e-4,1e-3,1e-2}, including unregularized zero once; log all tried settings, failures and their reasons. Use 3–5-point log grids for NORM/CENTER/DECAY_INIT/RANDPROJ, expanding only by a fixed rule if no norm overlap is found.
3. Match actual endpoint norms, not coefficients. Predeclare ±5% tolerance in pooled relative Frobenius norm sqrt(sum_l ||Delta_l||_F^2 / sum_l ||W_pre,l||_F^2), AND display the per-module relative-norm distribution. Also report the equal-module mean rho_F used by the legacy tables; they are different aggregates.
4. Choose hyperparameters using only the norm target and the calibration/validation protocol, never cross share, drift, retention, or test performance. Freeze grids, matching rule, common endpoint, metric primary/secondary choices and run list before confirmation. If confirmation norms fail the tolerance, report a failed match and the full frontier; do not choose a new favorable pair after seeing geometry.
5. If global matching conceals different layer allocation, add a per-module target control calibrated from MIX's pilot norms. Label it separately. Frobenius matching does not hold operator norm, stable rank, initialization contribution, or task performance fixed.
6. Show geometry versus achieved magnitude across the entire pilot grid. For a confirmatory dose-response claim, repeat the two noncentral strengths of MIX and LEFT on the three first confirmation seeds: 2 strengths × 2 arms × 2 tasks × 3 seeds = **24 additional runs**. Pilot-only curves remain exploratory.
7. Per-step norm projection is a secondary robustness arm, not the default NORM arm. If used, predeclare its target trajectory from calibration, implement it at the effective-delta level, verify forward equivalence and optimizer-state handling, and account for the insertion offset. Never rescale an endpoint after training and call it a magnitude-matched training control.

### Endpoints, statistics, and decision rules

Primary endpoint: fixed optimizer-step budget within each task, selected before confirmation. Secondary: best validation checkpoint by the same declared primary task metric. Save both and do not pool them. Save 0%,1%,5%,10%,25%,50%,75%,100% checkpoints as specified in P3.

Use an inner training/calibration split for hyperparameter selection and a locked evaluation split; do not use official GLUE test labels if unavailable. If reporting official validation after selecting on it, label it selection-conditioned. MRPC: save accuracy AND F1; declare accuracy primary for the new common-protocol benchmark and record F1 for continuity with legacy mixing runs.

Per task, report every seed, mean/sample SD, paired differences and intervals with the resampling unit explicitly the training seed. Three/five-seed bootstrap intervals have limited resolution; do not convert them into equivalence or reliable cross-task population claims. No pooled nine-sign p-value: shared tasks, data and models do not establish independent signs. Never use module count as n. Predeclare any practical task-loss tolerance and multiplicity treatment before testing.

Primary geometric outcome: effective total-delta p_cross at the common cutoff and step. Co-primary interpretability reports: norm matching, p_LL/p_TT, learned-since-insertion fractions, identity alignment, scaler means/spreads and chordal drift. Task score and the P8 behavior probe must accompany geometric outcomes. New largest-angle results remain supplementary.

- MIX differing from NORM only rules out this tested magnitude-only control.
- MIX differing from CENTER/DECAY_INIT and RANDPROJ at matched norm supports a more specific role for the pretrained projector, but not universal spectral or retention superiority.
- Similar outcomes favor generic regularization as a sufficient explanation within the tested regime.
- Identity alignment near one with reduced scaler nonuniformity supports the fixed-core explanation; it does not by itself show task irrelevance.
- All outcomes, including nulls/reversals, are reportable. The hypothesis is not a required result.

## P2 — idealization-to-practice bridge and within-tail flexibility

### P2a: remove the leading core

Repeat P1_UNREG/P1_MIX/P1_NORM on RTE, seeds `{42,17,123}`, with the leading core removed: **9 additional runs**, with new calibration if needed. Use distinct P2_TAIL_* run IDs. Match initialization across these three conditions. Report the initial offset when comparing this family with P1; removing a branch changes more than its endpoint spectrum.

This tests whether the practical findings also hold for the tail-only family covered by the confinement theorem. If it is not run, the paper must keep the theorem and the empirical family explicitly distinct.

On attained tail-only checkpoints without ambient scaling, directly check the sufficient no-crossing condition sigma_k(W_pre) > ||Sigma_T+H||_2. Do not apply that condition to the leading-plus-tail or coordinate-scaled form. Record both true and false cases; failure of a sufficient condition is not proof of instability. For all applicable checkpoints, instantiate Proposition 2: save e, d_s, s, h_s, mu, nu, four block operator norms, four right-hand-side bounds, and slack ratios. Mark zero denominators undefined. Compare norms per module before averaging rather than multiplying averaged factors.

### P2b: confinement versus expressivity

On RTE and, resources permitting, MRPC, compare fixed-tail diagonal coefficients, partial within-tail rotations (`k_vec=64`), full-tail rotations or a dense tail core, coordinate-relaxed tail adaptation, and standard LoRA. Use the same pretrained reference cutoff and tail dimension. Three paired confirmation seeds for the contrasts promoted to the main paper.

The exact matrix approximation theorem concerns an unrestricted tail core. A 64-dimensional rotation inside a 256-dimensional adapted tail does not instantiate that entire family. Report actual parameter count, attainable rank, initialization, memory, and runtime. Do not describe the hierarchy as parameter-matched unless it is.

To distinguish the location of the available subspace from its capacity, compare equal-dimensional **tail, leading, and random** subspaces using the same inner parameterization and training recipe. Include a middle-spectrum control if affordable. These controls are required before claiming the tail has a special empirical advantage over the leading spectrum; otherwise scope conclusions to the chosen tail instrument.

Include an equal-dimensional **mixed leading-and-tail** selection, not just a middle-spectrum interval: half the selected directions from each end is one prespecified construction. Hold coefficient initialization and initial effective delta fixed where possible; do not confound location with initializing leading directions at much larger singular values. State any remaining initialization mismatch.

Use a coefficients/rotations/scalers factorial contrast on the same selected subspace: fixed diagonal core, rotated core, scaled diagonal core, and scaled rotated core. Match optimization and initialization, report the extra capacity separately, and test the rotation-by-scaling interaction. This directly addresses whether the functional changes come from scalers, rotations, or both. A larger representable family is not a prediction that every trained instance will perform better.

Use a prespecified subset of contrasts, not a full Cartesian sweep. The exact number of P2b runs depends on the GPU budget; freeze it after the pilot and before examining confirmation outcomes.

### P2c: measure alignment rather than assume it

For a task-dependent target proxy `G`, measure its diagonal-tail, off-diagonal-tail, and outside-tail Frobenius energies and the two approximation residuals in Proposition 1. An initial task gradient is a local proxy; a full-fine-tuning delta is a trained proxy, not a known task optimum. Reuse the six RTE/MRPC full-FT runs already counted in early P5, or archived verified full-FT checkpoints; do not count or rerun the same references twice. Compare equal-dimensional spectral and random subspaces. Compute the theoretical best projection errors exactly from the same `G`; distinguish these from the loss and geometry attained by trained constrained models. This tests empirical alignment without pretending that the projection identity itself requires an alignment assumption.

## P3 — spectral trajectories, recorded during P1/P2

Save diagnostics at 0%, 1%, 5%, 10%, 25%, 50%, 75%, and 100% of optimizer steps, plus the best validation checkpoint. Store the pretrained bases once and each adapter/head state persistently. Do not delete the selected trajectory checkpoints at run completion.

Per module/checkpoint, record:

- Step, examples/tokens processed, learning rates, task loss, regularization terms, validation metric, and head displacement.
- Effective total and learned-since-insertion deltas; Frobenius/operator norms, stable rank, and reconstruction error.
- Absolute `LL`, `LT`, `TL`, `TT` energies and fractions; `p_cross=p_LT+p_TL`, `p_off=p_LL+p_cross`. Preserve norm ratios separately for comparison with old tables.
- `mu_E`, `nu_D`, Frobenius mixing norms, `||E||_2`, `||D||_2`, `||H||_2`, and factor-normalized overlap diagnostics with explicit zero-factor handling.
- Largest principal angle, mean squared principal-angle sine/projector distance, reference boundary gap, and evidence of singular-value crossings.

Required review-driven measurements (do not infer these from legacy ratios):

- Initialization: save Delta_init, its Frobenius/operator norms relative to W_pre, and signed coordinate blocks or recoverable checkpoints. Compute fractions independently for Delta_total and Delta_learned. Do not subtract block energies to obtain learned energy; cross terms matter.
- Fixed-core explanation: eta_I=(tr C_LL)^2/(k||C_LL||_F^2), the identity-projected residual C_LL-(tr C_LL/k)I, and the diagnostic Delta_total-mean(E)mean(D)U_LV_L^T. This latter subtraction is a hypothesis-specific reference, not generally the exact leading contribution. Save the exact E U_L V_L^T D term separately.
- Scalers: signed means, RMS magnitude, centered L2 norms, and centered/RMS ratios. CV=SD/abs(mean) is undefined near zero; flag it and retain the stable RMS-normalized alternative. Record identity alignment as undefined if C_LL=0.
- Drift at k in {16,64,128,256,512} for d=768 (use prespecified proportional cutoffs for other shapes). Save the principal-angle spectrum or singular values of U_L^T U'_L; report largest sine, fraction of angles above 30 degrees, normalized overlap tr(PP')/k, and chordal distance ||P-P'||_F/sqrt(2 min(k,d-k)). State that this normalization averages over potentially nonzero angles, not all k angles. Repeat on the right. Label any other normalization distinctly.
- Gaps: sigma_k, sigma_(k+1), absolute/relative boundary gap, ||Delta||_2 and spectral crossings, per module. Tied cutoffs must be flagged. Do not use ||Delta||_2/gap as an unconditional Wedin bound or predicted equality; check the actual theorem's separation hypotheses before claiming a bound.
- Nulls: dimension-only fractions are a reference expectation, not proof of isotropy. Add repeated random-orientation reference measurements preserving the measured update's singular values/rank/norm; use both the actual and random frames without conflating this post-hoc diagnostic with RANDPROJ training.

Report both per-module summaries and pooled energy fractions. Keep the leading cutoff fixed across methods and checkpoints. Include the dimension-only reference: in a square 768-dimensional matrix with a 256-dimensional tail, isotropic expected `TT` energy fraction is `1/9`; a large off-tail fraction alone is not preferential leading adaptation.

Plot both step-aligned and magnitude-aligned trajectories. Do not call a timing difference a distinct learning phase without replicated evidence. If snapshots are unavailable or unreliable, omit a dynamics claim.

## P4 — targeted robustness, after the central result

- Inspect the pretrained-projector graph's near-null directions if interpreting the scaler penalty as enforcing uniformity. The exact connected-graph nullspace is scalar, but weak connections and finite penalties can permit approximate nonuniform solutions. A scalar-scaler/core-only control is an optional targeted follow-up, not a substitute for P1 magnitude matching.
- Distinguish P1_RANDPROJ (only the penalty projector changes) from a **random adapter-frame** control (the fixed U/V factors change). For the latter, compare random-frame unregularized and random-frame mixing-controlled runs on RTE, three seeds (six runs plus orientation calibration). Hold dimension and trainable capacity fixed and evaluate in the original pretrained SVD. To match the practical initial model, explicitly add the frozen correction Delta_init_spectral-Delta_init_random to the forward update, or use a declared zero-insertion paired contrast. Verify forward/merge/delta identity for the chosen form; its extra fixed correction changes the applicable block algebra and must not be hidden. Predeclare this choice before confirmation.
- Scalar-scaler/core-only follow-up: train a,b,H and the same head with E=aI,D=bI; compare at matched update norm where feasible. It tests whether learned nonuniform scalers are needed, not whether the tail or leading block is universally useful.
- Left-only and right-only penalties to test asymmetry in actual block allocation, not only factor magnitudes.
- A direct effective cross-block penalty as a factor-rescaling diagnostic. Declare it as an additional intervention, not silently as the original method.
- Diagnostic cutoff sensitivity and random-orientation nulls matched in rank, singular values, and norm. Preserve reference bases for the principal analysis.
- Same-checkpoint block suppression/reweighting with norm restoration, if needed to test functional sensitivity. Include sham interventions and report remaining rank/operator-norm changes. Post-training ablation is not proof that retraining could not compensate.
- Replicate the strongest mechanistic contrast on a generative backbone as specified in P6. Additional retention datasets remain lower priority unless we intend a functional-retention claim.

## P5 — representative methods as geometric comparators

Purpose: determine whether the observed block allocation and response to constraints are specific to our instruments or shared by established PEFT approaches. This is not an experiment to demonstrate that UIOrthoLoRA wins.

- **Early core comparators, run with P1:** LoRA rank 8 and full fine-tuning on RTE/MRPC, seeds {42,17,123}: **12 runs** plus method-appropriate tuning, already counted in the 66-run first tranche. Full FT means all backbone weights train, not just the 48 attention matrices; diagnose the common attention subset and disclose/update-norm-account for other trained weights separately. Do not call these methods norm-matched unless a separate matching intervention establishes it.
- **Expanded spectral/standard coverage:** PiSSA, MiLoRA, SORSA, AdaLoRA, DoRA on the same two tasks and three seeds: **30 additional runs** plus tuning. Add Spectral Adapter-R and PSOFT as the closest rotation comparators for claims about that family; KaSA and SVFT are targeted extensions, not an unbudgeted exhaustive requirement. Include both scaled diagonal and rotated instruments from P1/P2 where manifests match exactly.
- The purpose is one common-frame figure for existing methods, not importing more favorable baseline scores. Do not replace the historical Table 2 with different published numbers and describe it as a matched experiment. Original LoRA protocols can include MNLI-initialized adapters and median rather than mean aggregation; no intermediate-task initialization in the primary common-protocol comparison. A with/without-MNLI study is separate, with its additional data and compute declared.
- Give every method the same validation-search budget, data splits, token/step budget, evaluation frequency, and checkpoint-selection rule. Use method-appropriate search ranges rather than identical learning rates that can disadvantage one method. Preserve all tried configurations, including failed runs. Tune on validation only; report a held-out task endpoint separately where available.
- Match trainable budgets as closely as each parameterization permits and report actual counts, rank, frozen storage, and unavoidable residual mismatches. Do not equate a rank number, a tail dimension, and a count of singular coefficients.
- For decomposition-initialized methods, measure `W_effective - W_pretrained`, not just the trained factor product; account for their frozen residual and nonzero initial factors. Use the same pretrained diagnostic basis and cutoff for every method.
- Report `LL/LT/TL/TT` energies, norm, principal-angle diagnostics, task learning, and trajectories. Treat a baseline's spectral initialization and continued confinement as different properties.

The present six-task GLUE table deliberately remains a **subset** comparison: MNLI and QQP are absent. Do not invent a historical reason for their omission or call the average a full GLUE score. Adding one of them can test task/data-scale sensitivity, but does not replace P1/P2 or a modern mechanistic replication. Any claim of full GLUE coverage requires actually evaluating the full declared suite under a common protocol.

## P6 — generative-backbone mechanistic replication

Purpose: test whether the central geometry/regularization relationship survives beyond RoBERTa, not merely append a modern-model performance table.

Choose one accessible instruction-tuned model after inspecting GPU memory and its license/download permissions. The existing Llama-3.2-3B setting provides continuity; a 7B/8B backbone is an option if the allocated hardware permits. Choose one held-out generation/reasoning task, provisionally GSM8K, with an explicit training/validation/test separation, frozen prompting/decoding, and no test-based selection. These are provisional choices, not executed experiments or fixed resource promises.

Base replication matrix: scaled fixed core and scaled rotated core, each with unregularized, two-sided-mixing, and magnitude-only-matched training (six conditions), plus LoRA. Three seeds give **21 confirmation runs**, plus calibration. This crosses the rotation/scaler question with the mixing intervention. If extending a pretrained-frame-specific conclusion from P1, add centered-scaler and random-projector-penalty controls for the fixed core: two more conditions, **27 runs total**, not 27 additional. Calibrate them within the new backbone. Without these controls, limit replication claims to placement/magnitude, not spectral specificity. Reuse an unadapted base-model evaluation as a functional reference; if any output head is trained, include a head-only reference as well. If budget permits, transfer PiSSA/MiLoRA/SORSA comparisons from P5; otherwise keep the baseline-coverage limitation explicit.

Use complete rectangular projectors for non-square projections, fix the adapted module set across methods, and log layerwise as well as pooled geometry. Match effective update magnitude within this backbone rather than borrowing RoBERTa penalty coefficients. Keep partial-rotation size and tail dimension explicit. Include at least the central checkpoint sequence and preserve artifacts. Report null or reversed effects as limitations of the mechanism's scope, not as runs to discard.

## P7 — cost measurements, collected with every pilot and confirmation

Record SVD setup time and peak setup memory; frozen basis bytes; trainable and optimizer-state bytes; peak allocated/reserved training memory; warmup-excluded optimizer-step time and tokens/second; orthogonal-map construction and regularizer overhead; separate spectral-diagnostic time; merge time and peak merge memory; and merged versus unmerged inference latency at fixed batch/sequence lengths. Pin hardware, precision, attention implementation, compilation, gradient accumulation, and checkpointing. Use synchronized repeated measurements and report variation. Show first-task cost and amortized multi-task cost only if basis reuse is actually implemented. Compact trainable counts alone are not a measured memory or speed result.

## P8 — functional probes on the same checkpoints

Primary low-cost probe: neutral held-out masked-token cross-entropy using the adapted backbone and the original frozen RoBERTa MLM head; bypass the classification head. Restore and verify every pretrained MLM-head tensor, including its original decoder weights; never evaluate a newly initialized head. For full FT, input embeddings may change: explicitly untie a frozen copy of the original output decoder from the updated input embeddings, rather than silently changing the probe head. Confirm the unadapted reference reproduces pretrained MLM logits within dtype tolerance. Fix the corpus, tokenization, sample IDs, mask locations, sequence length and random seeds before comparing conditions. Save per-example losses/logits if permitted. Report masked-token cross-entropy and, if exponentiated, label it exponentiated masked-token loss, not autoregressive perplexity. Compare the pretrained and insertion-state references, all P1 arms, LoRA and full FT on the same examples.

Relate per-run changes to p_cross, p_LL, rho_F, identity alignment and chordal drift within task and matched-norm contrasts. A small cross-run correlation is exploratory and confounded by treatment and task; do not pool dependent modules or call correlation causal. A null or reverse relationship is an informative scope result. Optional secondary probes are fixed-sample CKA/representation drift or fresh-head transfer with a genuinely held-out task and its own training/evaluation split (not zero-shot transfer).

**Retention study gate:** the old Gemma/Llama sweep table is now inactive in `archive/retention_2026_09_14.tex`. Do not revive it as evidence without (1) complete sweep manifests and a stated selection rule; (2) held-out/paraphrase adaptation; (3) non-trained-example retention with severe-forgetting denominator and rates; (4) at least three seeds; (5) UILinLoRA, rotated, mixing-controlled, magnitude/scaler controls and LoRA with geometry on identical evaluated checkpoints; (6) adaptation-versus-retention frontiers or predeclared matched-adaptation comparisons; (7) exact adapted modules, frozen basis memory and compute. The nine-task BIG-bench subset is not full BBH. This is lower priority than the RoBERTa probe and P6 mechanistic replication.

## Artifact contract and execution order

Each run must save a manifest (source hash, dependencies, model/data revisions, task/splits, seeds, complete configuration, explicit update form, initialization, regularizer/matching rule, checkpoint rule, device, timestamps), raw checkpoint diagnostics, adapter/head state, and logs. Tables and figures must be generated from these artifacts with provenance embedded in the output. Save failed runs and exclusion reasons.

Execution order: P0 → timed calibration pilot → freeze compute/run plan → P1 plus early P5 LoRA/full-FT comparators → P8 probe and analysis → P2a and targeted location/flexibility contrasts → expanded P5 and P6. P3 logging and P7 timing run throughout. Use P4's random-frame/scalar-core follow-ups to resolve residual ambiguity before enlarging the benchmark suite. Schedule only on the allocated GPUs, not concurrently with another agent's jobs unless capacity was explicitly assigned.

A first expanded allocation covers calibration plus 54 P1 and 12 early P5 confirmation runs (66 total), with P3/P7 logging and P8 evaluation. It is not sufficient to close every reviewer concern; the five-seed target, dose-response confirmation, frame/core controls and broader replication require explicit additional allocation. If the GPU budget is smaller, register a reduced pilot as exploratory rather than silently omitting controls. Do not promise GPU-hours before timing the actual environment. Pending results must remain absent from manuscript findings.
