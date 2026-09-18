# Review v2: remaining evidence handoff

Internal author/coding-agent handoff for Overleaf 6aa54397e58b10444b0fa2aa only.
The 18-run held-aside export at a63b1f2 has arrived; do not rerun it.

## Existing-checkpoint export (highest priority; no new training)

For each of the 18 frozen P1 confirmation checkpoints in the locked-evaluation request, export the 48 modules' E and D scaler vectors, or their means, squared norms and centered squared norms with explicit definitions. Preserve fixed step, task/condition/seed, run ID, checkpoint SHA and validation SHA. Include both unnormalized centered norm and centered norm divided by total scaler norm (undefined for zero norm); raw scalers are factor-gauge dependent, so retain both sides and their means. We already have and will analyze total/learned identity alignment and identity-residual energy; do not recompute those merely because the review believed them missing.

If already saved, include attained factor mixing norms, operator norms needed for the bound, chordal distances/overlaps, and boundary singular values with their exact definitions and cutoff. Re-measurement at alternative cutoffs is useful but secondary to scaler export and completing the existing band block. This request does not authorize new GPU allocation or new training.

## Historical provenance

Provide the count and IDs of the original Gemma-3-12B-it / Llama-3.2-3B-it retention sweep runs underlying the archived TriviaQA/HotpotQA table. The manuscript archive has 20 method/capacity rows × two tasks = 40 displayed adapted conditions, not a verified run count. Outcomes were already displayed before exclusion, so the revision will describe the exclusion as retrospective. Supply any more precise chronology, selection rule, and run inventory; do not infer run count from table cells.

## Existing band block

The latest freeze ledger has 8/21 registered checkpoints. Continue the already authorized plan and refresh validated outcomes/coverage and the complete-block held-aside export when ready; preserve all failed/retry records and no tuning on held-aside results. Partial numerical band tables will leave the manuscript pending the complete block, but remain in the timestamped data.

## Optional experiments, requiring a separate scoped decision

CENTER (calibration plus confirmation) or a random-projector penalty distinguishes frame specificity from generic scaler homogenization. A practical S_L=0 arm with ambient scalers and mixing penalty tests the positive tail-confinement bound; the running strict-band experiment enforces confinement by construction and does not substitute for this test. These are not six-run requests including calibration: any new dose selection needs its own declared budget and disjoint calibration. Do not launch merely because the panel recommended them. Head-only references are already registered in block B. A future merge-interference experiment must specify which task head, total versus learned deltas, and insertion handling before evaluation; existing RTE/MRPC checkpoints do not constitute a sequential forgetting experiment.
