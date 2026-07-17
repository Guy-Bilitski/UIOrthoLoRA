# Baseline-Init Fidelity Audit

Audit of the custom baseline re-implementations used in the catastrophic-forgetting
campaign, checked against the **actual published source repos** (fetched and quoted
during the audit, not recalled from memory). The campaign publishes these numbers, so a
silent deviation in any baseline invalidates the comparison.

**Scope:** baseline re-impls only — `corda_init.py`, `milora_init.py`, `sclora_init.py`,
`lora_null_init.py`, `data_aware_init.py`, and the C-LoRA path in `train_cs.py`.
UIOrthoLoRA (our method) and the other custom tuners are out of scope.

## Sources verified (quoted verbatim from the repos)

- CorDA `cordalib/act_aware_utils.py` + `cordalib/decomposition.py` (iboing/CorDA)
- SC-LoRA `scloralib/act_aware_utils_output.py` + `decomposition_two.py` (CoffeePot1206/SC-LoRA)
- LoRA-Null `adapterlib/decomposition.py` — `decompose_to_adapter2` (HungerPWAY/LoRA-Null)
- CLoRA `clora.py` (sutakori/CLoRA)
- CorDA + LoRA-Null READMEs (calibration-dataset defaults)

## Findings summary

| # | Method | Issue | Severity | Status |
|---|--------|-------|----------|--------|
| 1 | CorDA | KPA covariance calibrated on **wikitext-2**, but paper/repo default is **nqopen** (QA knowledge). Inconsistent with this codebase's own SC-LoRA/LoRA-Null, which correctly use nqopen | **HIGH** | Confirmed deviation |
| 2 | CorDA + LoRA-Null | Activation normalization uses `x.abs().max()` (max-of-abs); repo uses `torch.max(x).abs()` (abs-of-max). Differs for signed activations → per-sample reweighting of the covariance | **LOW–MED** | Confirmed deviation |
| 3 | CorDA | Damping convergence test uses spectral norm (`matrix_norm ord=2`); repo uses Frobenius (`torch.dist`). Same 0.05 threshold | **NEGLIGIBLE** | Note only |
| 4 | MiLoRA | bottom-r SVD split, loss-preserving | — | **Faithful** |
| 5 | SC-LoRA | output 2nd-moment, β/sign folding, top-r eigvecs, B=Qᵣ/A=QᵣᵀW | — | **Faithful** |
| 6 | LoRA-Null | null-dim = r, top-r SVD of null-projected W, A not frozen | — | **Faithful** |
| 7 | CLoRA | frozen orthonormal Pᵤ/Pᵥ penalty with ½ factors | — | **Faithful** |
| 8 | CorDA | cov-aware decomposition (Wc=W·C_fix, V·C⁻¹ adjust, smallest-r, UV-fuse, residual) | — | **Faithful** |

## Detail

### Finding 1 — CorDA KPA calibration dataset (HIGH)

`train_cs.py:214` loads **wikitext-2** for CorDA covariance, with a comment claiming it
is the "CorDA default KPA calib." The CorDA README states KPA *"samples questions from
QA datasets such as `triviaQA` and `nq_open`"*; the example uses
`--calib_dataset "nqopen" --calib_loader_size 256`. CorDA-KPA's entire mechanism is to
**freeze the directions most responsive to the calibration data** (the knowledge to
preserve) and train the rest. Calibrating on general LM text (wikitext) instead of QA
knowledge changes *which directions are frozen*, so the CorDA baseline preserves the
wrong subspace relative to the paper — and likely understates CorDA's retention on
QA-style benchmarks. This is the load-bearing finding: it makes the published CorDA
column not the paper's CorDA. The inconsistency is telling — `sclora` (D-) and
`lora_null` (calib) in the same file *do* use `nq_open`, so wikitext here looks like an
oversight, not a deliberate design.

### Finding 2 — CorDA / LoRA-Null activation normalization (LOW–MED)

`corda_init.py:25` and `lora_null_init.py:36` use `m = x.abs().max()` (max of absolute
values). The CorDA repo uses `input / torch.max(input).abs()` (absolute value of the max
element). For signed activations these differ (e.g. `[-10, 3]` → 10 vs 3), reweighting
each sample's contribution to the accumulated covariance and slightly perturbing the
eigen/SVD directions. Notably `sclora_init.py:44` uses `Y.max().abs()` (the *correct*
repo semantics) — so the distinction was understood but applied inconsistently. Impact
is bounded (a per-sample scalar), but it is a true deviation in two baselines.

### Findings 4–8 — Faithful (with evidence)

- **MiLoRA** (`milora_init.py`): plain SVD of W₀, adapter = bottom-r (smallest singular
  triples), residual = principal part, loss-preserving. Matches the paper; the docstring
  guards the PiSSA footgun. No calibration (correct).
- **SC-LoRA** (`sclora_init.py`): accumulation matches the repo exactly — D+
  `+= (1-β)·cov/c_size`, D- `-= β·cov/b_size`, per-sample `output/torch.max(output).abs()`,
  output outer product; then top-r eigvecs of symmetrized M, `B=Qᵣ`, `A=QᵣᵀW₀`,
  `W_res=W₀-QᵣQᵣᵀW₀`. `decomposition_two.py` confirms `eigvecs[:, -r:]` and the same
  B/A/residual.
- **LoRA-Null** (`lora_null_init.py`): repo's main `singular_aware` branch uses
  `U_[:, -r:]` (null-space dim **= r**, not a larger threshold) and the top-r SVD of the
  null-projected weight, and does **not** freeze A. CF defaults (`null_dim=None→r`,
  `freeze_a=0`) match this exactly. The docstring's "fidelity flag" about null_dim is
  unfounded — the repo also uses r. (A distinct `singular_aware_2` variant with A=0
  exists in the repo; CF correctly mirrors the main branch.)
- **CLoRA** (`train_cs.py` `CLoRARegularizer`): penalty `½‖A·Pᵥ‖²_F + ½‖Bᵀ·Pᵤ‖²_F`
  summed over modules ×λ, with frozen random-orthonormal `Pᵤ(out,k)`, `Pᵥ(in,k)`. Matches
  `clora.py` term-for-term (including the ½).
- **CorDA decomposition** (`corda_init.py`): `damp=0.01` doubling until error<0.05,
  `Wc=W@C_fix`, SVD, `V=(Vt@C_inv)ᵀ`, smallest-r for KPA, `W_res=W-U·diag(S)·Vᵀ`, UV-fuse
  `B=U√S / A=√S·Vᵀ`. Matches `cordalib/decomposition.py` exactly.

## Out of scope but worth a one-line sanity check

`residual_save.convert_saved_to_w0_relative` (used by all four residual methods): these
methods overwrite `base.weight=W_res`, but PEFT saves only the adapter. Eval correctness
depends on this conversion reconstructing W₀+ΔW. Not paper-fidelity (it's plumbing), but
a load-and-diff sanity check is cheap insurance before publishing.

## Recommendations (decisions for the authors — no code changed by this audit)

- **Finding 1 can change a published number.** Either (a) re-run CorDA with `nqopen`
  calibration to match the paper, or (b) if wikitext was intentional, document it
  explicitly as a deviation and fix the misleading comment at `train_cs.py:214`.
- Finding 2: optionally align `corda_init.py` / `lora_null_init.py` normalization to
  `torch.max(x).abs()` for exact repo parity; low impact, can be a footnote instead.
- Findings 4–8 need no action.

## How to re-confirm this audit independently

1. Run the self-tests (read-only, no training): `python milora_init.py`,
   `python sclora_init.py`, `python lora_null_init.py`, `python corda_init.py` — each
   prints loss-preserving recon error (~0) and method-specific invariants.
2. Re-fetch the repo lines quoted above and diff against the CF ports.
3. For Finding 1: confirm CorDA's README/example uses `--calib_dataset nqopen`.
