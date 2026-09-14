# NeurIPS reviews: changes and remaining evidence

Internal author document, 2026-09-13. Applies only to Overleaf project `6aa54397e58b10444b0fa2aa`. Do not include this file in an anonymous submission export.

Updated 2026-09-14: this file retains the original NeurIPS-review mapping. The subsequent comprehensive review is addressed in `REVIEW_PANEL_RESPONSE.md`; the sole current design is `EXPERIMENTS_REQUIRED.md` (P0–P8). Start a GPU session with `GPU_HANDOFF.md`, which includes implementation hazards, source fingerprints and the fact that the expanded runner still needs implementation/testing. All new GPU evidence remains pending.

## Scientific contract

This is a study of spectral components and interventions, not a claim that UIOrthoLoRA is the best adapter. The constructions vary fixed coefficients, within-subspace rotations, ambient scaling, and mixing control. Task performance measures whether useful adaptation remains possible and what an intervention costs; it is not the paper's primary ranking criterion. Reframing does not resolve missing empirical controls. No acceptance outcome is promised.

The author approved the first restructuring draft and requested removal of its blue/gray markup. The active manuscript now contains plain accepted text; old cuts and author notes are no longer compiled. Original research and old wording remain in `review/original_main.tex` and Git history. Subsequent edits in this pass are also clean, as requested.

## Reviewer concern to action

| Concern | Change made in the paper | Evidence still required |
|---|---|---|
| Missing SORSA; incremental adapter novelty (jDBz, EdNv) | Cite and distinguish the regularized objects in Related Work. State overlap rather than claim a new use of SVD. | P5: compare actual effective-update geometry with representative spectral methods under a common protocol. |
| LinLoRA originality (xv8J) | Explicitly identify it as an implementation label for a fixed diagonal-tail construction, with SVDiff and SVFT attribution. | No originality claim for this construction; obtain any additional specific antecedent the authors intended. |
| Alignment assumptions do not justify task performance (jDBz) | Use exact projection identities for arbitrary `G`, separating outside-tail and within-tail error. No Transformer generalization theorem or presumed task-tail alignment. | P2b/P2c: equal-dimensional location controls and measured task-proxy alignment. |
| Tail not justified against leading/mixed adaptation (EdNv) | No claim of intrinsic semantic importance or universal tail superiority. The tail is a selected experimental reference. | P2b: leading, tail, mixed leading/tail, and random with the same core parameterization. |
| Rotations versus scalers (xv8J) | Explain that ordering reverses between RoBERTa-base and large, and that mixing runs use no rotations. More expressivity is not a performance guarantee. | P2b factorial contrasts and P6 replication, explicitly including UILinLoRA. |
| Missing baselines and tuning fairness (all) | Label old imported comparisons; no state-of-the-art or statistical-superiority claim. Remove winner highlighting in supporting tables. | P5: LoRA, PiSSA, MiLoRA, SORSA, AdaLoRA, DoRA with equal search opportunity and geometric measurements. |
| Small/older models; modern-model effects unclear (all) | Separate RoBERTa mechanism evidence from supporting generation; archive the unverified larger-model retention comparison outside active evidence. | P6: one modern generative-backbone replication of the component/regularizer contrast, not merely another performance table; P8 governs any restoration of retention results. |
| Why omit MNLI/QQP? (jDBz) | Show all six evaluated tasks in main Table 2; name the omitted tasks. Avg6 includes MRPC accuracy, confirmed by the author for the spectral rows on 2026-09-14. | Historical selection reason is not established. Do not invent one. Add task-scale coverage only if needed for the declared scope. |
| Incomplete complexity analysis (EdNv) | Add Appendix C on setup, basis storage, trainable entries, forward/backward considerations, regularization/diagnostics, merging, and inference. Include a main-text summary. | P7: controlled runtime/memory measurements. Asymptotic analysis is not a measured efficiency result. |
| Claims exceed evidence; limitations (all) | Distinguish confinement, block decoupling, ranked drift, magnitude, and functional retention. Dedicated Discussion/Limitations and Conclusion. | P1 magnitude matching and head-only control; P3 trajectories if making a dynamics claim. |

## Confirmations needed before submission, not author notes in the PDF

1. **Exact legacy-run provenance.** The original implementation paragraph and inspected historical/current code include a leading identity core. The unified analysis covers that form. Supply archived resolved configurations/code hashes if the actual experiments used another form; do not relabel them as tail-only runs. The non-RTE ablation metadata writer stored an RTE dictionary, so that field is not authoritative for every task.
2. **Seed coverage and GLUE metric/count audit.** Mixing tables reproduce 27 condition rows and now reconcile with 1,296 module records. SST-2 has only seed 42 in the paired records. Supply a second seed if one exists. Separately recover final six-seed GLUE artifacts: local tuning/partial records do not reconstruct all reported summaries. The author confirmed on 2026-09-14 that the broad-evaluation MRPC values are accuracy; Table 2 now has one accuracy column and Avg6. The separate mixing-intervention MRPC F1 results are unchanged. The stated rotated recipe implies 479,232/1,007,616 adapter entries for base/large, not the reported 0.4M/0.6M; exact historical configurations/counting conventions must be resolved. Counts are preserved in `data/glue_reported.csv` but omitted from the main table. See `data/GLUE_PROVENANCE.md`.
3. **Generation records.** Verify the spectral runs' configurations, seeds, and uncertainty definition. The imported rows match Table 3 of Gao et al. (FourierFT); that primary source resolves two transcription errors: Houlsby-style adapter BLEU uncertainty is 0.6, not 3.6, and FourierFT ROUGE-L uncertainty is 0.1, not 0.4. Both are now corrected and the comparison provenance is cited. This supporting table is not a uniform baseline rerun.
4. **Retention provenance.** The inspected `tuner_knowledge/src/analysis/process_results.py` uses strict `sc_after - sc_before < -threshold` and filters non-trained examples. Its threshold-0.8 summaries support the strict definition, not an inclusive loss of eight out of ten answers. The launcher task lists and `bigbench_summary.csv` columns identify nine BIG-bench multiple-choice tasks, not full BBH. These corrections remain in inactive `archive/retention_2026_09_14.tex`, whose numeric table is preserved exactly; it is no longer active paper evidence. Restoration requires verified per-model partitions, selected configurations, common evaluated checkpoints, matched adaptation, replication and geometry. Do not infer retention rates from unverified shared denominators. See P8.
5. **Training and release metadata.** Reconcile exact software/model/data revisions, licenses, resolved training settings, initialization, hardware, and total compute. The analytical cost section follows the inspected implementation; the new 27-row logged-cost table discloses available measurements but cannot establish matched speed or total project compute.
6. **Author verification.** Review the revised proofs, references, statistical scope, and AI-use disclosure. The draft does not claim this verification has already occurred. Run anonymous-export checks before submission.

## Numerical handling in this pass

The mixing rows continue to be generated from the same CSVs; no GPU training or synthetic training results were added. The previously corrected SST-2 drift still comes from the named `mean_Drift_U` column, not the neighboring off-tail operator-ratio column. Supporting mean scores remain unchanged. Two generation-baseline uncertainty entries were corrected against the primary published table, as documented above. Formatting cleanup removes review macros and winner emphasis, not unfavorable observations. The old entries remain in the archived source and Git history.

The latest full audit adds a genuine reanalysis of saved module diagnostics, not new model runs: cross energy falls under two-sided control in all nine run pairs, but leading share rises in four task means. Figure 2 and the four-block appendix table expose this distinction. A proved commutator/graph-Laplacian characterization further clarifies what the scaler penalty constrains. The manuscript now occupies nine main pages with all content inline in the existing root file. `MANUSCRIPT_AUDIT.md` records checks, corrections, and the still-open empirical gates.

The canonical run specification is `EXPERIMENTS_REQUIRED.md`. Its P0–P8 labels separate correctness, magnitude/scaler/frame controls, component/location contrasts, trajectories, robustness, external comparators, modern replication, measured costs and same-checkpoint functional probes. The expanded first tranche comprises 54 P1 plus 12 early P5 confirmation runs (66 total), plus calibration and optional dose-response confirmation. It supersedes the old 24-run plan; timing and explicit GPU allocation gate the actual run list. It does not close every reviewer concern.

## Sources checked for positioning

- SORSA, version 6 (Cao and Song, 2025; first released 2024): https://arxiv.org/abs/2409.00055
- SVDiff: https://arxiv.org/abs/2303.11305
- SVFT: https://arxiv.org/abs/2405.19597
- PiSSA: https://arxiv.org/abs/2404.02948
- MiLoRA: https://arxiv.org/abs/2406.09044
- AdaLoRA: https://arxiv.org/abs/2303.10512
- FourierFT Table 3 and Section 4.2: https://arxiv.org/html/2405.03003v1
- BIG-bench / BBH distinction: https://arxiv.org/abs/2206.04615 and https://arxiv.org/abs/2210.09261

These citations resolve attribution, not the missing empirical comparisons. The internal ledger is for research coordination, not text to paste into a rebuttal as if all experiments had been completed.
