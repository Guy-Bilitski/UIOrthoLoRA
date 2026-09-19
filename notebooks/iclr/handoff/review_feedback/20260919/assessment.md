> **Latest author priority — modern decoder first:** Read
> `DECODER_FIRST_PRIORITY_20260919.md`. Prioritize the prepared decoder pilot,
> tuning/calibration and confirmation runs on the two assigned GPUs. Defer
> CENTER and further encoder controls. The coding agent has now delivered the
> temperature and band held-aside exports; verify/reuse them, do not rerun them.
> Older schedules below are superseded where they place controls first.

Assessment of the ICLR LLM feedback — 19 September 2026

My assessment is that this is a credible, carefully documented study with a real contribution, but it still has material acceptance risks. I would not call it a weak paper on the evidence available, and I would not call it scientifically finished. The main risks concern the interpretation, relevance and generality of the results. More runs help only if they answer those particular questions. The previous manuscript cleanup established document consistency; it did not close the outstanding scientific checks.

The feedback is useful but uneven. Its strongest criticisms are better grounded than its extravagant praise. It is reasonable to ask whether the prediction-loss improvement survives calibration, whether simpler regularization explains it, and whether it transfers beyond the original encoder. It is not justified to describe the existing evidence as an advance in trustworthiness, better representations, or an alternative to ensembles. Its statement that every proof was exhaustively verified is not independent mathematical certification.

The paper's strongest identity is a controlled study of adaptation geometry, using new adapter constructions as experimental instruments. The contribution is the separation of subspace location, within-subspace flexibility and interactions, together with the behavior measured under these interventions. Some mathematical ingredients are standard projection and norm arguments. Their value is in the framework and the questions they let us test; we should not rely on the feedback's claim that each ingredient is a major architectural invention.

**What the current evidence establishes**

| Question | Evidence we have | Remaining limit |
|---|---|---|
| Can fixed tail directions support useful adaptation? | Tail coefficients improve RTE accuracy over training the output classifier alone. | One task; currently reported on the inner-selection split. |
| Does changing the selected band or adding rotations consistently improve accuracy? | The complete leading/middle/tail study contains 21 confirmations, with similar average accuracies and no consistent rotation gain. | Three seeds do not establish equivalent adaptation capacity. Frozen-checkpoint held-out evaluation is still missing from the available exports. |
| Does the interaction penalty change training behavior? | The 18-run controlled comparison shows much less cross-subspace update energy and lower held-out NLL than the tested size penalty on RTE and MRPC. | The intervention also changes update allocation across modules and confidence. It does not isolate cross share as the sole cause. |
| Is this a broadly better PEFT method? | Historical GLUE and GPT-2 results provide supporting context. | Different tuning protocols prevent a clean method ranking; modern matched comparisons are absent. |

The experiment count represents substantial work. The 21 band confirmations, 18 practical confirmations and 27 legacy runs answer different questions. They are not 66 independent replications of the central loss result. Likewise, 1,296 module records do not replace seed or task replication. The expensive controlled work on a small model is scientifically useful; its depth does not by itself establish transfer to larger models.

**A useful new result from the existing records**

I computed calibration diagnostics without training or running any model. Both tasks use two classes and ordinary, unweighted cross-entropy. Therefore each archived per-example loss determines the probability of its true class: p(true class) = exp(-loss). The other probability is its complement. I checked the executed evaluator's hash against its manifest, reproduced the published accuracy and loss, verified predictions against the reconstructed probabilities, and independently checked ECE against the existing calibration analyzer.

These are descriptive means over the same three seeds, with 15 fixed equal-width confidence bins. Brier is the binary convention mean((p(class 1)-label)^2); smaller values are better.

| Task | Condition | NLL | ECE | Binary Brier |
|---|---|---:|---:|---:|
| RTE | UNREG | 2.429 | 0.265 | 0.267 |
| RTE | MIX | 1.229 | 0.225 | 0.243 |
| RTE | NORM | 2.193 | 0.270 | 0.270 |
| MRPC | UNREG | 0.871 | 0.118 | 0.123 |
| MRPC | MIX | 0.395 | 0.075 | 0.109 |
| MRPC | NORM | 0.755 | 0.116 | 0.125 |

MIX has lower ECE and Brier than NORM in all six task–seed pairs. On incorrect predictions, mean confidence falls from 95.4% to 89.4% on RTE and from 91.1% to 80.3% on MRPC. This supports a concrete interpretation: the trained MIX models are less confidently wrong. The loss improvement is not merely a numerical peculiarity of NLL. Nevertheless, RTE calibration remains poor, and these data do not show that the improvement requires spectral interaction control or that representations are better.

These diagnostics are post hoc, based on small fixed evaluation sets. ECE depends on binning; probabilities reconstructed from float32 losses inherit rounding and saturation. No temperature was fitted, and no test labels were used to select a model or a hyperparameter. The analysis and all source hashes are in `calibration_diagnostic.py` and `calibration_diagnostic.json`. I have kept these new scientific results out of the manuscript pending our discussion.

Temperature scaling remains the most informative inexpensive next check. Fit a separate positive scalar temperature for every one of the 18 frozen endpoints on its inner-selection predictions, then evaluate the same held-out examples. The existing analyzer already implements this separation. Missing selection predictions require checkpoint inference, not retraining. Temperature scaling is an established baseline for calibration; it changes probabilities while preserving argmax classifications. [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)

If the NLL gap mostly disappears, the honest conclusion is that MIX supplies a training-time confidence benefit that can largely be reproduced by inexpensive calibration. That would weaken the practical case, but would not erase the geometric intervention or the raw-loss result. If a meaningful gap remains, it rules out a single global logit scale as a complete explanation. It still does not prove better representations or identify a unique spectral mechanism.

The training horizon is also relevant: these are long, fixed-endpoint runs, and the selection rule emphasizes accuracy. Overconfident errors can grow after accuracy plateaus. If saved checkpoint histories permit it, evaluating a checkpoint chosen by inner-selection NLL would be a useful secondary practical control. The original fixed-endpoint comparison should remain intact and separately reported.

**Where the feedback needs correction**

1. PiSSA is a relevant comparator, but it is not a strict leading-subspace confinement experiment. It initializes two trainable low-rank factors using principal components and freezes the residual matrix. Because the factors remain trainable, their spans can move; confinement to the original leading directions does not follow. This is an inference directly from its parameterization. Our fixed leading/middle/tail study asks a different and cleaner location question. PiSSA itself examines spectral initialization choices, so our novelty must rest on strict support and interaction controls, not discovering that different spectral initializations can be tried. [PiSSA](https://arxiv.org/html/2404.02948v2)
2. Comparing the head-only NLL of 0.654 directly with MIX's 1.229 is invalid: these are different experiments and evaluation splits, with different adapter initializations. The within-band-study comparison still shows that accuracy gains coexist with worse NLL. We need the frozen head-only and band checkpoints scored on the same held-out RTE set before making the cross-study comparison.
3. SVD costs are already analyzed in Appendix C and referenced in the main text. The feedback later praises that very analysis. The remaining useful addition is measured setup time, peak memory and training throughput, not another complexity paragraph. Frozen full-basis storage and per-step work are at least as relevant as the one-time decomposition. A fast method for finding leading singular vectors is not automatically a solution for recovering the spectral tail.
4. Layer-wise concentration is a consequence of the intervention, not a pre-existing imbalance that invalidates the whole MIX-versus-NORM comparison. It is a competing explanation for how the intervention produces the benefit. That distinction matters: the overall effect is supported in this setting, while its internal mechanism remains unresolved.
5. Older models do not make controlled experiments worthless. However, claiming broadly applicable adaptation behavior from one encoder and two small tasks remains vulnerable. One well-designed decoder replication could be more valuable than many additional GLUE tables.

**The mechanism control I would discuss first**

Our own theory supplies a particularly useful competing explanation. The penalty encourages ambient scalers to become more uniform. Its expectation under random projectors is proportional to ordinary centered-scaler shrinkage. A CENTER arm penalizing deviations from the mean scaler, without the pretrained projector, is already supported by the recorded runner. It would take six confirmation runs across the two tasks, plus an independent calibration stage.

If CENTER reproduces MIX's behavior, the paper should explain the result as a useful regularization effect whose dependence on pretrained spectral geometry has not been established. If it does not, the geometric explanation becomes more credible. This control addresses a simpler alternative, but does not itself settle the module-concentration issue.

A direct response to the concentration criticism would use a NORM-style control that targets MIX's per-module update-size profile. Derive that target from an independent calibration seed, freeze the rule before confirmation, and report how closely it actually matches both total size and allocation. Do not choose modules or strengths using held-out losses. Do not treat post-training rescaling of existing NORM weights as equivalent to training under this control. This is a more involved experiment; its priority should depend on the calibration result and the causal claim we want to make.

The existing size match is also approximate: confirmation errors reach 8%, and five of six pairs exceed the intended 5% calibration tolerance. This is already disclosed in the paper. The very large cross-energy difference is unlikely to be explained by a small norm mismatch alone, but the observed geometric sensitivity analysis does not establish a corresponding bound on prediction loss. We should continue to describe a comparison with the tested size penalty, not an exact causal isolation at identical update norms.

**A feasible strategy with the confirmed two-GPU budget**

My preferred plan is staged. Reserve the last two days for analysis and writing. Pilot runtimes and memory must determine the training commitment; I have not verified live availability or launched any jobs.

| Priority | Proposed work | What it resolves | New training |
|---|---|---|---|
| First | Export selection/held-out predictions for all 18 practical endpoints; evaluate all 21 frozen band/head endpoints on held-out RTE. | Calibration and completion of the central subspace evidence. | None; inference only. |
| Next | CENTER on RTE and MRPC, with calibration independent of confirmation. | Whether simple scaler regularization reproduces the interaction result. | 6 confirmations plus calibration. |
| Preferred extension, after a successful pilot | One compact decoder and one generative task; UNREG, MIX, NORM, LoRA and PiSSA with three seeds. | Decoder transfer and matched practical baseline context in one experiment. | 15 confirmations plus tuning/calibration. |
| Fallback if decoder implementation consumes the budget | Matched LoRA and PiSSA on existing RTE/MRPC protocols. | Stronger local baseline evidence, while retaining an explicit architecture limitation. | 12 confirmations plus tuning. |
| Conditional | Per-module allocation control, small interaction-dose sensitivity check, or additional seeds. | Whichever uncertainty remains most important after the first results. | Agree the scope before launch. |

A concrete candidate is Qwen2.5-1.5B-Instruct, a 1.54B-parameter decoder with 28 layers and grouped-query attention. I would pilot a single GSM8K fine-tuning/evaluation protocol before committing, checking memory, learning signal, answer extraction and runtime without looking for a favorable MIX result. Use fixed prompts, completion-only training loss, deterministic decoding and the same scorer across methods. Report task success as well as held-out token loss. One compact decoder would provide a useful bridge, not establish large-model scaling or general reasoning gains. [Official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

The current controlled runner is specialized to RoBERTa classification; this extension is not a model-name swap. Adapting square query/output projections consistently across all methods would reduce porting risk, while the grouped-query key/value projections require attention to rectangular geometry. Reuse decompositions across conditions where valid. Include one-time SVD time, peak memory, trainable and frozen storage, and steady-state training throughput in the pilot. A 0.5B model would be cheaper, but a result near the task floor would say little; the pilot should establish that the task is informative.

Matched baselines mean the same examples, adapted modules, evaluation, seed pairing and a comparable tuning budget. They do not mean forcing every method to use our adapter's learning rate. Parameter counts should be reported rather than implying that equal rank makes these architectures equivalent. LoRA and PiSSA are the first choices; DoRA and OFT are useful optional coverage, not all mandatory additions this week. A fixed-basis comparator such as SVFT would also be scientifically relevant if implementation is ready.

**Reproducibility details and the historical generation table**

The orthogonality implementation is recoverable and now explicit: the executed code uses PyTorch's orthogonal parametrization with `matrix_exp`, identity initialization and default trivialization. Dense stored parameter counts do not mean the effective rotations were unconstrained.

The registered protocol fixes the two MIX coefficients at 0.001, also the value in the archived two-sided condition. Only NORM's dose is chosen by the size-matching search. I clarified this, but have not invented an original heuristic for why 0.001 was first chosen. Sensitivity to this value remains unestablished. A modest bracketing check is more useful than a new large sweep if time permits.

The identity leading core gives each leading direction the same inner coefficient. Replacing it by the pretrained singular values changes the family and its initial offset. All practical arms share the identity choice, so it does not create an imbalance between those arms. It does limit how far their result transfers to other initializations. I clarified the choice without claiming it is optimal or silently changing the method.

The GPT-2 issue deserves resolution before the final submission. Local E2E code, scores and some manifests exist, but I cannot bind the two table rows and their uncertainty estimates to complete run populations. In particular, the rounded UIOrthoLoRA central values closely reproduce the single archived `lr_0.05_svalues_256_svectors_30_seed_17_init_sigma_0.1_init_scaler_0.1` score file: BLEU 0.6877, NIST 8.7291, METEOR 0.4671, ROUGE-L 0.7162, CIDEr 2.4707. That is a provenance question, not proof that the row is wrong. The available manifests for other runs cannot establish the recipe for this row. We should recover its exact run IDs, aggregation, parameter counts and decoding settings; otherwise correct or remove the unsupported entries rather than fill gaps from generic code defaults. This decision is larger than a cosmetic edit and I have left its numbers unchanged.

I applied only verified minor manuscript clarifications: orthogonal parameterization, fixed MIX dose, identity-core meaning, the forward-path description, explicit GPT-2 protocol comparability limits, loss units, task labels on the intervals and the awkward rank-constrained sentence. The approved abstract and introduction remain unchanged. These edits do not claim that the open experiments have been completed.

My recommendation is to retain the present research question and invest the week in discrimination between explanations. The existing results justify saying that similar classification performance can arise from different allowed spectral directions, while interaction regularization changes probability quality and the distribution of updates. They do not yet justify saying directions never affect adaptation capacity or that reducing interactions itself causes a better representation. Calibration, one simple mechanism control, and a compact matched decoder study would materially strengthen the paper if the results support those extensions. No additional experiment guarantees acceptance, and unfavorable results should narrow the claims rather than trigger a search for a favorable benchmark.
