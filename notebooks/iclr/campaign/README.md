# Prospective ICLR campaign

Scope: `iclr_6aa54397`, on the existing `ortho_new` branch. The authoritative
design is `../handoff/EXPERIMENTS_REQUIRED.md`. This package now includes a
CPU-validated optimizer-step engine and checkpoint system, but is **not yet an
executable production campaign**: pinned data/model preparation, persistent
workers, monitoring, global budget accounting and calibration orchestration are
still pending. No new pretrained task-training outcomes are supplied by CPU tests.

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
3. `Resources.validate_training()` requires explicit devices, positive budget,
   persistent directory/storage allowance, download policy and authorization
   provenance. The author assigned physical GPUs 2 and 3. Remaining resource
   fields have not been assigned; no training has been launched.
4. The pinned model/data loaders, fixed inner/locked evaluation split and corpus
   masks, persistent worker/monitor, global resource reservations, whole-run
   validator, calibration selection, frozen matrix and confirmation driver still
   need implementation. P3/P7/P8 must be verified on actual pretrained RoBERTa
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
output root. There is no production CLI yet. The only bypass is explicitly named
`synthetic_cpu_test`, used by tests with all GPUs hidden.

Validation selection uses accuracy, excludes step 0 and breaks ties by earliest
step; these choices must appear in the frozen protocol. Checkpoints retain their
observation paths and inherited history across retries. A resumed attempt uses a
new directory and can refer to the prior immutable reference. Never rerun into an
existing attempt directory. `validation.validate_checkpoint()` compares all
scientific outputs; only `diagnostic_seconds` is excluded from numerical equality.
Its report has `validation_scope="checkpoint"` and is intentionally insufficient
to satisfy the run-completion ledger gate.

The current CPU suite includes exact uninterrupted-versus-resumed optimization,
dropout and sampler equivalence; failed checkpoint writes/corruption; selection
and fixed-endpoint separation; and full tiny-RoBERTa optimization/P3/P8/reload.
The fixtures use randomly initialized tiny models and synthetic examples. They
are not the required pretrained RTE smoke or calibration experiments.
