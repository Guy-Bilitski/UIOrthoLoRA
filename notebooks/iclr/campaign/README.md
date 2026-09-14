# Prospective ICLR campaign

Scope: `iclr_6aa54397`, on the existing `ortho_new` branch. The authoritative
design is `../handoff/EXPERIMENTS_REQUIRED.md`. This package is a new, partially
implemented runner foundation. It is **not yet an executable training campaign**.
No new task-training outcomes are supplied by the CPU tests.

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
4. The data/model loaders, persistent worker/monitor, optimizer-step loop,
   optimizer/scheduler/RNG checkpoint persistence, independent durable reload
   validator, calibration selection, frozen matrix and confirmation driver still
   need implementation and integration testing. No completion status may rely
   only on the current unit tests. P3/P7/P8 need verification on actual RoBERTa
   checkpoints, including real dtype/device placement and cost measurements.
5. All nuisance terms are currently materialized by `regularization()`, and its
   projector matrices are rebuilt on each call. This is a correctness reference;
   optimize with cached fixed projectors and relevant gradients before expensive
   training, rechecking objective/gradient equivalence and measuring overhead.
6. Do not freeze the confirmation matrix before timed calibration. The enumerated
   66 runs are a specification count only, not scheduled, launched or finished.
   P2/P4, expanded P5 and P6 remain required/conditional as defined by the plan.
   The exact extension count, model and tuning budget remain unresolved.

See `../handoff/CAMPAIGN_STATUS_20260914.md` for the resource inventory and exact
continuation instructions. Legacy data and training-source files are unchanged.
