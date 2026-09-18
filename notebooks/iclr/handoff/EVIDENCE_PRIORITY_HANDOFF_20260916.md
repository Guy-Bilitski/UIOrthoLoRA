# Evidence priority after manuscript v4 — 2026-09-16

Historical priority record. The author's subsequent tail-adaptation refactor in
`TAIL_ADAPTATION_EVIDENCE_HANDOFF_20260916.md` supersedes the writing freeze and
strict-band deferral here. The temperature analysis/export contract remains
valid; CENTER is now secondary to the confined tail/head/rotation comparison.

Internal author/coding-agent handoff for this ICLR project. The author has frozen
the writing and redirected remaining effort to evidence. This supersedes the
September 15 instruction to finish strict-band before CENTER, and the training
priority in REVIEW_V3_EVIDENCE_HANDOFF_20260916.md. Completed registrations stay
immutable. This instruction does not grant additional GPU or storage resources.

## Queue and execution status

1. Export logits for the 18 frozen practical confirmations; run temperature scaling.
2. Calibrate CENTER, freeze its doses, then run RTE/MRPC × seeds 17, 42, 123:
   **six confirmations plus calibration**, not six total training jobs.
3. Complete/reuse identical-protocol head references, beginning with the three
   registered RTE original-backbone head-only seeds if still outstanding.
4. Inspect for a recoverable LoRA checkpoint and measure its geometry without
   delaying priority work. Do not launch new LoRA training by default.
5. Defer remaining strict-band launches and module-norm-matched training.

This paper session has not accessed the GPU server, changed its scheduler,
stopped jobs or launched training. The resource owner must reconcile the live
queue first: the eight-checkpoint freeze is a dated export, not current status.
Stop admitting queued band jobs; checkpoint interrupted owned jobs through the
existing controller. Preserve completed results, failed/retry records and the
registration. Record the stopping decision as post-review compute reprioritization,
not a completed block or an outcome-independent stopping rule. Use only assigned
devices/storage and the isolated repaired-input checkout; do not touch unrelated
processes. Return the live ledger and revised GPU-hour estimate.

## 1. Temperature scaling: export and analysis contract

Export `data/temperature_logits_20260916/` for exactly the 18 fixed endpoints in
`data/locked_evaluation_request_20260916.json`. Keep that request and its existing
evaluation immutable. Reuse cached held-aside logits. If missing, a deterministic
forward pass on the identical checkpoint/examples is an export/loading operation;
record provenance and reproduce existing predictions/losses. No retuning or
best-checkpoint substitution. No raw dataset text is needed.

`manifest.json` must contain these fields; angle-bracket values below describe
required metadata, not actual evidence:

```json
{
  "schema_version": 1,
  "request_json_sha256": "<SHA256 of existing locked request JSON>",
  "class_order": [0, 1],
  "loss_convention": "unweighted_mean_cross_entropy_no_label_smoothing",
  "export_source_revision": "<40-character commit>",
  "export_script_sha256": "<64-character SHA256>",
  "records": [
    {"run_id": "<existing run ID>", "file": "records/<run ID>.json", "sha256": "<file SHA256>"}
  ]
}
```

There must be 18 entries. Each referenced record contains:

```json
{
  "run_id": "<existing run ID>",
  "task": "rte",
  "condition": "P1_MIX",
  "seed": 17,
  "fixed_optimizer_step": 5670,
  "checkpoint_sha256": "<same hash as existing locked per-example export>",
  "validation_sha256": "<original validation-report hash>",
  "training_source_revision": "<generating training commit>",
  "input_manifest_hashes": {
    "model_directory": "<original prepared model manifest SHA256>",
    "probe_directory": "<original prepared probe manifest SHA256>",
    "task_directory": "<original prepared task manifest SHA256>"
  },
  "inner_selection": {
    "split_fingerprint": "<prepared inner split fingerprint>",
    "sample_ids": ["<stable source-namespaced ID>"],
    "labels": [0],
    "logits": [[0.0, 0.0]]
  },
  "locked_evaluation": {
    "split_fingerprint": "<existing locked fingerprint>",
    "sample_ids": ["<exact existing per-example ID>"],
    "labels": [0],
    "logits": [[0.0, 0.0]]
  }
}
```

The one-example arrays are schema illustrations, **not model results**. Export
every example in existing evaluation order, two finite float32/float64 class
logits per example. Include the prepared source-split identity, source-row mapping
and input-manifest hashes in export provenance: disjoint display IDs alone do
not prove source-data separation. Preserve paired sample order, labels and split
fingerprints across arms/seeds within each task. Columns are classes 0 and 1.
The analyzer reproduces original inner accuracy/F1/loss and locked predictions
and per-example loss, using the existing loading tolerance (absolute 0.0005,
relative 0.0001); retain all reproduction errors.

As specified before these new logits in the round-3 handoff: fit one T>0 per
checkpoint on **inner-selection NLL only**, log(T) in [-6,6]. Freeze T, then
evaluate locked NLL and top-label ECE in 15 fixed equal-width right-closed bins.
Keep T=1 results and every bin. The implementation solves the convex problem in
inverse temperature with these same bounds, reports boundary optima, and uses
T=1 for a completely flat objective. No held-aside fitting or bound expansion.

```bash
python3 scripts/check_temperature_scaling.py
python3 scripts/analyze_temperature_scaling.py --bundle data/temperature_logits_20260916 --out data/temperature_analysis_20260916.json
```

The analyzer rejects missing/incomplete input and writes no manuscript prose.
It reports every seed, means/sample SDs, nominal paired seed intervals, all three
between-arm contrasts before/after scaling and within-arm changes. Primary
follow-up: scaled MIX minus NORM NLL, separately per task. ECE/other contrasts
are descriptive. Retain nulls and reversals. This is exploratory after inspection
of unscaled held-aside losses; n=3 intervals are not multiplicity-adjusted or
equivalence tests. A surviving NLL gap supports a probability-quality difference
after this calibration procedure, not geometry as its cause or equal accuracy.

## 2. CENTER: freeze the design before new training

Keep the clean practical recipe, initialization, head/data-order seeds, optimizer,
dropout, preprocessing and task budget. RTE: initial scaler/coefficient 0.01,
5,670 steps. MRPC: 0.1, 2,760 steps. Leading core I, tail size 256, no rotations,
all 48 matrices and full ambient scalers. Change only the explicit regularizer;
do not center/constrain/reset the actual scalers. Use the planned **sum**, not mean:

    gamma * sum_l c_l * (sum_i (epsilon_li - mean(epsilon_l))^2
                         + sum_i (delta_li - mean(delta_l))^2)
    c_l = k_l * (d_l-k_l) / ((d_l-1) * (d_l+2))

Here d=768, k=512 on both sides. This is the fixed-scaler Haar-expected
unnormalized projector penalty, not an expectation of trained endpoints. Check
gradients, shift invariance, zero loss for constant scalers, the 48-module sum,
and short repaired-input learning/reload gates. Preserve actual scaler means.

Calibration seed 31415; targets are the **existing MIX calibration endpoints**
from `data/focused_norm/selection_final.json`:

| Task | Pooled total relative Frobenius target | Target run |
|---|---:|---|
| RTE | 0.07923554598047688 | 20260915T075556Z_53b69e612ef3 |
| MRPC | 0.04129567209296267 | 20260915T075556Z_4edea9e8b40d |

Evaluate gamma {0.0001, 0.001, 0.01} per task at the fixed endpoint. After all
three validate, select minimum absolute relative norm error, exact ties by
smaller gamma. Freeze if error ≤5%. Otherwise allow **at most four additional
distinct doses per task**, using only pooled endpoint norms:

1. Sort doses by gamma. Among consecutive pairs straddling target, select the
   pair with the smallest better-endpoint absolute relative error (tie: smaller
   lower gamma) and evaluate its geometric midpoint.
2. If no pair straddles, extend tenfold above the largest gamma when all norms
   exceed target, or tenfold below the smallest when all fall below. Restrict
   gamma to [0.000001, 1]; if that bound is reached, stop with a failed match.
3. Stop at the first added dose within 5%, otherwise at the cap. Select nearest
   valid dose by the same rule. Log all attempts/failures/retries. Do not consult
   score, geometry, drift, probe loss or held-aside outcomes for this selection.

No monotone response or successful match is assumed. This budgets 3–7 calibration
endpoints per task plus six confirmations: **12–20 endpoints total**, excluding
smoke tests/retries. If resources require a smaller design, record it before
outcomes and retain a failed-match fallback; never silently change the rule
mid-sweep. Freeze the full frontier, selected doses, executable loss, source/input
hashes and timing/resource ledger before confirmation.

Run both tasks and all three seeds regardless of outcomes, paired to existing
MIX/NORM/UNREG after verifying common task-path behavior. Report signed per-pair
norm errors; no confirmation retuning or deletion of failed matches. Freeze all
six new endpoints before their separately labeled held-aside evaluation. Return
existing runs.csv fields plus total/initial/learned block energies, pooled/equal
module summaries, 48 module norms, identity residuals, raw scaler means and
centered norms, probe CE, fixed/best task metrics and loss, inventories, source/
checkpoint/validation hashes and loading checks. Export logits if calibrated
CENTER NLL is to be compared.

CENTER reproducing the pattern supports generic homogenization as sufficient
within this regime; a null contrast is not proof of equality or universal frame
irrelevance. Divergence distinguishes these finite-penalty objectives, but norm,
module allocation and optimization can still differ. CENTER alone does not
establish pretrained-frame specificity; random-projector/frame controls address
a separate question.

## 3. Head references and existing LoRA

Three seeds cover **one task and one head-reference type**, not both tasks/types.
Reuse RTE `HEAD_BASE` only with identical inputs, head/data-order seeds, optimizer
and budget. It measures the original backbone. `P1_HEAD_INIT` instead freezes
the shared nonzero practical insertion state and directly tests whether training
the adapter helps beyond training its head. Do not substitute a strict-band
zero-insertion model for that control. Obtain timing/resource estimates before
expanding either control to both tasks.

For recovered LoRA: identify base-weight revision, task, protocol, checkpoint
rule, scaling, target matrices and seed; verify effective delta/reload metrics;
compute all four blocks and module norms in that exact pretrained SVD frame and
cutoff convention. Label a historical checkpoint as a separate instrument example,
not a matched fresh-campaign comparison. Table 2's imported LoRA numbers do not
establish that a corresponding local checkpoint exists. Report absence if none
survives; never attach another protocol's score to a recovered checkpoint.

## 4. Reviewer access and completed presentation items

The verified v4 `supplement/analysis_artifact.zip` is 1,250,242 bytes, SHA256
`781127cff2568c082c37bb3d2c92937ab1afd0eb9cb05a418c3d6b55bd2b425b`.
It contains sanitized analysis evidence, not weights/full training runner.
Supply an anonymous HTTPS download or the review system's actual supplementary
attachment; test a signed-out download and its SHA. No public URL is configured
or claimed here. Do not publish private handoffs, reviews or server paths.

At paper commit `a3dfd08`, Table 19 already gives 80%-power effect sensitivities
under explicit n=3 paired-test variance assumptions. The strict-band appendix
already withholds the partial numeric tables and retains a pending study note.
The writing and existing ZIP stay frozen while evidence is collected. Refresh
the supplement and make only necessary result/access/status edits when the
corresponding evidence or hosting endpoint arrives.
