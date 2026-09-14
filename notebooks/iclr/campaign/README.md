# Prospective ICLR campaign

Scope: `iclr_6aa54397`, on the existing `ortho_new` branch. The authoritative
design is `../handoff/EXPERIMENTS_REQUIRED.md`. This package now includes a
CPU-validated optimizer-step engine, pinned input preparation, budget accounting,
and a persistent P0 smoke controller/worker. The full production campaign is
**not yet implemented**: calibration expansion and confirmation orchestration
still need completion. Whole-run validation, gated throughput pilots, initial
magnitude-grid registration/admission and norm-only decisions are implemented.
No pretrained task outcomes are supplied by CPU tests.

Implemented and checked on CPU:

- Explicit practical leading-identity-plus-tail and ideal tail-only layers,
  diagonal/dense cores and partial identity-initialized rotations. Frozen original
  bases include full rectangular complements. Total and since-insertion deltas
  are separate, and insertion state persists through reload.
- Forward, merge, unmerge, disabled-adapter and checkpoint equivalence; explicit
  float32/float64 dense linear support. No quantized path or automatic device map.
- Nine P1 condition IDs and their exact objective primitives; module-summed
  mixing/centered/initial-decay/random-projector losses, module-mean relative
  squared effective-delta loss, and explicit failed norm matches. Haar draws use
  their own RNG. Baseline mixing diagnostics remain populated at zero coefficient.
- Original-frame block energies, factor bounds, initialization/identity residuals,
  scaler means/spreads, multi-cutoff principal-angle spectra, qualified gaps,
  rectangular accounting, orientation nulls and separate pooled/module summaries.
- Verified attention-module targeting, original-backbone and insertion-state
  head-only trainable sets, an all-parameters full-FT trainable set, and the
  original frozen MLM head with an independent output decoder.
- Collision-free manifests, exclusive JSON writes, an append-only locked ledger
  with terminal failures, new retry IDs and a reload-validation completion gate.
- Compact durable checkpoints: one immutable frozen reference, every trainable
  tensor, optimizer/scheduler state, Python/NumPy/Torch RNG and exact batch order.
  Atomic no-replacement writes and hashes detect partial/corrupted artifacts.
- Fixed-step AdamW with explicit warmup/decay, gradient accumulation weighted by
  example count, finite-gradient checks, step timing and memory fields. Initial
  dropout RNG is paired after method construction; resume restores saved RNG.
  All trajectory and improving validation checkpoints persist. The primary fixed
  endpoint and secondary best-validation endpoint are separate pointers.
- Standard additive LoRA with explicit rank/alpha, true full-FT displacement
  outside the common attention subset, and fresh RoBERTa reconstruction from
  saved architecture/bases without recomputing the original SVD.
- Independent checkpoint reproduction of task metrics, geometry and P8 probe
  outputs. A checkpoint report cannot authorize run completion; the ledger now
  requires whole-run P3/P7/P8 validation and the complete artifact contract.

Run from the dedicated research-repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m notebooks.iclr.campaign.preflight
```

The preflight saves a new immutable report below
`notebooks/iclr/handoff/data/campaign_v1/preflight/` on every invocation. It runs
the four authoritative checks, verifies the historical source fingerprints, and
tests the prospective implementation. Tiny randomly initialized model tests are
CPU integration checks, not the required pretrained RTE smoke run.

The dedicated environment uses Python 3.10.14; `requirements.lock.txt` records
all installed package versions. Import the delivered custom PEFT through `src/`.
The shared Python 3.13 environment fails RoBERTa import due to its TorchVision
operator mismatch. No shared package was changed. bitsandbytes is installed only
because the delivered custom PEFT imports it unconditionally; no quantization is
enabled or validated here.

Important implementation choices and remaining work:

1. P1 has `tail_size=256`, hence leading cutoff **512**, for 768-wide attention
   weights. Never interpret `k_val=256` as the leading cutoff. The recipe in
   `protocol.py` follows the manuscript table; the legacy experiments.py includes
   different large-model-search overrides. Final choices must be resolved during
   calibration and recorded before confirmation.
2. The campaign rotations start at identity, as the prescribed factorial contrast
   requires. The legacy layer initializes independent random orthogonal maps.
   Unrotated update/forward correspondence is tested directly against the legacy
   implementation. Rotated campaign initialization is a deliberate new protocol,
   not a claim of reproducing legacy rotated results.
3. `Resources.validate_training()` requires explicit devices, a positive budget
   or explicitly recorded completion-duration authorization,
   persistent directory/storage allowance, download policy and authorization
   provenance. The author authorized physical GPUs 2/3 through campaign
   completion. The proposed initial 50 GiB output allocation and public downloads
   were accepted via "Go ahead"; this interpretation was explicitly reported.
   See the immutable `RESOURCE_AUTHORIZATION_20260914.json` in handoff/data/campaign_v1.
4. Pinned inputs, disjoint selection/locked splits, fixed corpus masks, conservative
   global reservations, an owned-child supervisor and whole-run validation are
   implemented. Initial magnitude-grid selection is implemented; expansion admission,
   a frozen matrix and confirmation driver still need implementation.
   P3/P7/P8 must be verified on actual pretrained RoBERTa
   checkpoints, including real dtype/device placement and controlled P7 costs.
5. `spectral.regularization()` remains the deliberately unoptimized correctness
   reference. `regularizers.CachedRegularizer` caches fixed projectors, builds
   gradients only for the chosen intervention, and still logs actual left/right
   mixing at zero coefficient. Its losses and gradients match the reference on
   CPU. The worker must persist random bases in reference extras and build its
   cache on the assigned device. GPU overhead remains unmeasured.
6. Do not freeze the confirmation matrix before timed calibration. The enumerated
   66 runs are a specification count only, not scheduled, launched or finished.
   P2/P4, expanded P5 and P6 remain required/conditional as defined by the plan.
   The exact extension count, model and tuning budget remain unresolved.

See `../handoff/CAMPAIGN_STATUS_20260914.md` for the resource inventory and exact
continuation instructions. Legacy data and training-source files are unchanged.

## Engine and restart contract

`engine.run_steps()` takes prepared model/examples plus mandatory selection,
diagnostic, probe and regularizer callbacks; it returns `awaiting_validation` or
`interrupted`, never `completed`. Production calls must provide explicit resource
authorization, the physical GPU ID/UUID, a bounded reservation and a persistent
output root. `smoke.py` admits P0 smoke and explicitly gated throughput-calibration
jobs, intended for dedicated tmux sessions. Registered initial magnitude-grid
entries are also admitted after both validated timing pilots. Expansion and
confirmation jobs remain rejected until their admission is implemented. The only bypass is explicitly named
`synthetic_cpu_test`, used by tests with all GPUs hidden.

Validation selection uses accuracy, excludes step 0 and breaks ties by earliest
step; these choices must appear in the frozen protocol. Checkpoints retain their
observation paths and inherited history across retries. A resumed attempt uses a
new directory and can refer to the prior immutable reference. Never rerun into an
existing attempt directory. `validation.validate_checkpoint()` compares all
scientific outputs; only `diagnostic_seconds` is excluded from numerical equality.
Its report has `validation_scope="checkpoint"` and is intentionally insufficient
to satisfy the run-completion ledger gate.

`register_calibration.py` requires both completed throughput reports, explicit
per-task step budgets and explicit three-to-five-point nuisance grids. It seals
a new immutable protocol and does not launch training. `calibration.py` fixes
MIX/LEFT doses and a central MIX-1e-3 norm target; all three RANDPROJ orientations
must fit the +/-5% tolerance for a matched label. Selection minimizes the worst
orientation norm error, with lower coefficient breaking ties. Failed matches
retain the closest tested coefficient and complete frontier. Both outer doses
may be proposed for at most two fixed-rule expansion rounds; an expansion
proposal does not itself authorize execution. `calibration_io.py` only extracts
hash-bound, validated fixed-step total norms and per-module distributions from
the scientific ledger; task scores, block fractions and P8 outcomes are not
selection inputs. No experimental magnitude grid has been registered yet.

The current CPU suite includes exact uninterrupted-versus-resumed optimization,
dropout and sampler equivalence; failed checkpoint writes/corruption; selection
and fixed-endpoint separation; and full tiny-RoBERTa optimization/P3/P8/reload.
The fixtures use randomly initialized tiny models and synthetic examples. They
are not the required pretrained RTE smoke or calibration experiments.

## Preparation and persistent worker

`prepare_inputs.py` downloads exact files at immutable public Hub revisions into
the assigned campaign cache and creates separate immutable source copies. RTE
and MRPC each have a stratified inner selection split and a locked official
validation split. WikiText-2-raw-v1 test supplies 256 fixed masked examples; this
is held out from task adaptation, not claimed unseen during RoBERTa pretraining.
Every prepared tensor, ID list, mask and source file has a retained fingerprint.
The original MLM loader rejects missing, unmatched or newly initialized tensors.

`allocation.py` retains all leases and settlement events. Stale leases never
auto-expire. Only a controller observing its exact child exit can settle a lease;
overruns are charged in full. Failed artifacts and cache/source copies count
toward the 50 GiB allowance. Completion-duration authorization removes a global
numerical time cap, not per-worker deadlines, storage checks or GPU isolation.

`supervision.py` inspects only assigned-device aggregate telemetry and its own
Popen child/logs. It requests a checkpoint boundary before terminating its own
unresponsive child, with cleanup grace included in the reservation. A monitoring
failure is not proof of process exit and leaves the lease active. The controller
must itself be run persistently. `worker.py` stores the original bases and MLM
head, P0 forward/merge/disable checks, task/P3/P8 observations, step/cost records
and independent checkpoint reproduction. Successful workers still return
`awaiting_validation`, never scientific run completion.
