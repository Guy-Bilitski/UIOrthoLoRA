Fetch latest ortho_new and Overleaf 6aa54397e58b10444b0fa2aa. Read
notebooks/iclr/handoff/DECODER_SCOPE_REVIEW_20260919.md first. This is the latest
author clarification and supersedes the mandatory 18-run LR sweep.

Keep the six strict-band decoder arms, three paired seeds, frozen reference,
accuracy AND NLL, and the measured memory settings. For pending confirmations,
use the common LR 1e-3 already exercised by both tail pilots, plus only brief
learning checks for arms lacking evidence at that rate. Preserve completed runs;
stop dequeuing further rates solely to finish the old search grid. Reconcile
any already-running/finished confirmations before changing protocols, and do not
restart finished work to implement this revision. Retain all old evidence.

Add a truthful fixed-recipe registration path; the current gate requires a full
grid. Do not fabricate tuning completions or relabel pilot seeds. Freeze the
amendment and common decoding settings before test evaluation. Score the frozen
model on the selection subset now; absolute pilot accuracy does not show an
adaptation gain. Reuse existing checkpoints/observations wherever possible.

Publish the current completed/running/pending inventory, amended protocol,
remaining GPU-hours and then the full confirmation/evidence schedule. No LoRA,
PiSSA, new penalty sweep or automatic new experiment block. This study tests
location and within-band flexibility; a decoder cross-subspace regularization
claim remains a separate discussion. See the review for precise writing and
one diagnostic-only correction. Do not alter the approved manuscript.
