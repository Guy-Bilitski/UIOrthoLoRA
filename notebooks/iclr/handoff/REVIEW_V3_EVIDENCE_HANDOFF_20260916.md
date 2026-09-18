# Review-v3 evidence handoff — existing checkpoints first

The author's subsequent post-v4 priority change is recorded in
`EVIDENCE_PRIORITY_HANDOFF_20260916.md`. It supersedes the training queue below
and adds the executable logit-export contract. The temperature-fitting rule
specified here is preserved.

Internal author/coding-agent request. Do not change frozen doses, checkpoints,
seeds, registration records, held-aside labels, or existing GPU allocations.
This paper session launches no training and does not reprioritize running jobs.

## Needed export: probability calibration, all 18 frozen confirmations

Export inner-selection sample IDs, labels and the two class logits for exactly
the checkpoints in data/locked_evaluation_request_20260916.json. Include run ID,
checkpoint hash/step, split fingerprint, source revision, class ordering and
unweighted cross-entropy convention. Prefer float32 or float64 logits; no raw text.
Reproduce saved accuracy/F1/loss as a loading check. Export corresponding official
held-aside logits if already cached, and bind them to the existing per-example
records; do not change or select predictions using held-aside outcomes.

Analysis specified before this new export: one scalar T>0 per checkpoint, fitted
only on inner-selection NLL in log-temperature coordinates over [-6,6]. Report
boundary optima; do not enlarge the interval based on held-aside performance.
Freeze T, then compute held-aside NLL and top-label ECE in 15 fixed equal-width
bins over [0,1], with intervals ((j-1)/15,j/15] and zero included in the first.
Retain unscaled T=1 results, per-run/per-bin counts, confidence/accuracy summaries,
all paired contrasts and numerical reproduction errors. Positive T preserves
argmax predictions. This is an exploratory follow-up prompted by the already
observed NLL difference, not a new untouched primary test or causal mechanism test.

## Pending existing-checkpoint exports from round 2

REVIEW_V2_EVIDENCE_HANDOFF_20260916.md still specifies the centered/raw scaler
moments or vectors for all 18×48 modules. Export historical retention sweep
inventory/counts and exact broad-GLUE/GPT-2 module/dimension/count conventions if
recoverable. A configuration-derived count is not a replacement for an executed
inventory. Existing spectral diagnostics at additional cutoffs are useful when
available; do not impute them from one cutoff's four aggregate blocks.

## Training requests raised by the panel (not launched here)

Finish the already registered band/head-only block and its full exports under the
current resource owner. The previously planned LoRA geometry comparator is the
highest-priority cross-method addition if the remaining authorized budget permits;
report all four blocks on the same 48 matrices and cutoff, plus inventory and task
metrics. New per-module-matched, centered-scaler/random-projector arms, MIX dose
sweeps, and additional calibration seeds require a separate frozen design and
resource allocation. Neither a centered-scaler comparison nor a no-scaler arm
alone identifies a direction-only causal effect. Strict-band jobs also omit the
practical leading identity core, so they are not an E=D=I, S_L=I ablation.

## Reviewer delivery

The next panel must receive supplement/analysis_artifact.zip alongside the current
PDF. The panel reports that the archive was unavailable, so it could not verify the
available raw exported module records and hashes. The archive is sanitized and
analysis-only; it does not include model weights or the training runner.
