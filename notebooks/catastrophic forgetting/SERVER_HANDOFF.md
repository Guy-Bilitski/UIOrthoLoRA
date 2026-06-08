# Server Handoff — PEFT upgrade to 0.19.1 (read before running the ablation study)

**Audience:** the coding agent running on the GPU server (B200).
**Companion doc:** `instructions.md` (the ablation-study spec) in this same folder. This file records the environment/library work already done on the dev machine so you can reproduce it on the server and start the experiments.

**Status:** ✅ The vendored PEFT library in this repo has been upgraded from `0.17.2.dev0` → **`0.19.1`**, the custom **UIOrthoLoRA** tuner has been re-applied on top, and it has been smoke-tested (build / forward / backward / save / load all pass). These changes are committed on branch **`ortho_new`** and pushed to `origin`.

---

## 1. What this repo actually is

This repo is **not** a project that *depends on* PEFT — it **is a vendored fork of HuggingFace PEFT** (`src/peft/...`) with one extra tuner added:

- Custom tuner: `src/peft/tuners/uiortholora/` (`config.py`, `layer.py`, `model.py`, `bnb.py`, `__init__.py`)
- The experiment harness lives in `notebooks/tuner_knowledge/src/` (`train.py`, `eval.py`, `inference.py`, the `run_multiple_*.sh` launchers, etc.)

So "update PEFT" meant **re-vendoring the library source**, not `pip install -U peft`.

---

## 2. What was changed (the upgrade)

### Library bump
- The fork was based on upstream commit **`c15daaa5`** (the `0.17.2.dev0` window, just after `v0.17.1`; it already contained RoAd + Arrow + LoRA-GenKnowSub).
- The genuine fork delta vs that base was tiny and clean:
  1. the whole `src/peft/tuners/uiortholora/` directory, **plus**
  2. three registration edits (UIOrthoLoRA imports/`__all__` in `src/peft/__init__.py` and `src/peft/tuners/__init__.py`, and `UILINLORA`/`UIORTHOLORA` enum values in `src/peft/utils/peft_types.py`), **plus**
  3. two *accidental* fork artifacts that were **intentionally NOT carried over**: a stray blank line in `vera/layer.py`, and an accidental removal of `"VBLoRAConfig"` from `peft.__all__`.

### How the re-vendor was done
1. Saved `src/peft/tuners/uiortholora/`.
2. Replaced the entire `src/peft/` tree with upstream **`v0.19.1`**.
3. Restored `uiortholora/` and re-applied the 3 registration edits (using `register_peft_method`, which 0.19.1 still supports).
4. Refreshed build metadata: `setup.py`, `pyproject.toml` now report `0.19.1`; `src/peft/__init__.py` → `__version__ = "0.19.1"`.
5. Added `trl` to `requirements.txt` (the harness imports `trl.SFTTrainer/SFTConfig` but it was missing from requirements).

> Note: tuner removed by upstream between 0.17→0.19 (e.g. `bone`, superseded by `miss`) will show as deletions in the diff — that is expected and correct.

### Files changed (committed)
- `src/peft/**` — the full 0.17→0.19 library update + the 3 registration edits
- `setup.py`, `pyproject.toml` — version → 0.19.1
- `requirements.txt` — added `trl`
- `notebooks/catastrophic forgetting/SERVER_HANDOFF.md` — this file

**The UIOrthoLoRA tuner code itself (`layer.py`, `config.py`, `model.py`) was NOT modified** — it is treated as tested experiment code. It runs unchanged on 0.19.1.

---

## 3. Reproduce the environment on the B200 server

The dev machine used `uv` + a repo-local `.venv` (gitignored). On the server:

```bash
cd <repo root: UIOrthoLoRA>
git checkout ortho_new && git pull

# create venv (uv recommended; plain venv also fine)
uv venv .venv --python 3.12        # or: python3 -m venv .venv
source .venv/bin/activate

uv pip install -r requirements.txt # or: pip install -r requirements.txt
uv pip install -e . --no-deps      # editable-install THIS repo as `peft` (do NOT pip install peft from PyPI — that would shadow the fork)

python -c "import peft; print(peft.__version__)"   # must print 0.19.1
```

### Versions verified on the dev machine
| package | version | note |
|---|---|---|
| peft | **0.19.1** | this repo, editable install |
| torch | 2.12.0+cu130 | CUDA 13 wheels — Blackwell/sm_100 (B200) supported |
| transformers | 5.10.2 | **transformers v5** — newer than what 0.17 targeted; another reason for the bump |
| trl | 1.5.1 | needed by the harness |

**B200 GPU notes:**
- B200 is Blackwell (sm_100). Use a torch build with CUDA ≥ 12.8 (cu128) or cu130. `torch 2.12.0+cu130` works.
- `instructions.md` requires **bf16** everywhere — B200 has native bf16, good.
- A harmless startup warning appears from bitsandbytes: `Failed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'`. If you do 4-bit/8-bit (QLoRA-style) runs, `pip install kernels>=0.11.1`. For bf16 full-precision adapter training it is irrelevant.
- `transformers 5.x` + `trl 1.5.x` are very new; if a harness API (e.g. `SFTConfig`/`SFTTrainer` arg names, `DataCollatorForLanguageModeling`) raises, pin to a compatible older trl/transformers rather than editing the tuner.

---

## 4. Verification already performed (so you can trust the base)

On a synthetic model (256-dim `q_proj`/`v_proj`, `num_svalues_to_adapt=64`, `num_svectors_to_adapt=16`):

- `import peft` → `0.19.1`; `from peft import UIOrthoLoRAConfig, UIOrthoLoRAModel` OK.
- `PeftType.UIORTHOLORA` registered in `PEFT_TYPE_TO_CONFIG_MAPPING`.
- `get_peft_model(...)` builds (BaseTuner injection / `_create_and_replace` / `_create_new_module` paths all compatible with 0.19.1).
- Forward + backward OK; **all 10 trainable tensors receive gradients** (D, E, sigma, left_unitary, right_unitary).
- `save_pretrained` writes `adapter_model.safetensors` containing the rotation parametrization tensors.
- **Save→load roundtrip on identical base weights reproduces outputs to `max abs diff ≈ 2.4e-7`** (functionally exact).

**One benign warning to expect on `from_pretrained`:**
`UserWarning: Found missing adapter keys ... uiortholora_*_unitary.default.parametrizations.weight.original / .0.base`.
This is a known `torch.nn.utils.parametrize` quirk — the raw `weight.original` tensors differ but the effective orthogonal matrix is restored via the `.0.base` cache, so forward output is bit-for-bit correct. **Do not "fix" this by editing the tuner.** It does not affect training or eval correctness.

---

## 5. UIOrthoLoRA parameter mapping — answers to the `[CONFIRM]` items in `instructions.md`

`instructions.md` marks several names `[CONFIRM]`. Verified against the actual code (`src/peft/tuners/uiortholora/`):

| `instructions.md` concept | **Actual repo name** | Where |
|---|---|---|
| `k_val` (top boundary; # singular values adapted) | **`num_svalues_to_adapt`** (config, default 128) | `config.py:82` |
| `k_vec` (null-space rotation dim) | **`num_svectors_to_adapt`** (config, default 128) | `config.py:83` |
| scalers `D`, `E` | **`uiortholora_D`, `uiortholora_E`** (`ParameterDict`, trainable iff `use_de`) | `layer.py:43-44,108-109` |
| `D/E` on-off switch | **`use_de`** (config bool) | `config.py:94` |
| trainable singular values `σ` | **`uiortholora_sigma`** (one `Parameter` of size `num_svalues_to_adapt`) | `layer.py:42,104` |
| rotations `R_U`, `R_V` (`θ_L`, `θ_R`) | **`uiortholora_left_unitary`, `uiortholora_right_unitary`** (orthogonal-parametrized) | `layer.py:45-46` |
| major-tier "strip σ → identity" | buffer **`{adapter}_S1` = ones**, frozen, `requires_grad=False` | `layer.py:91` |
| SVD tier buffers | `U1/S1/Vt1` (major), `U2/S2/Vt2` (medium), `U3/S3/Vt3` (small/null) | `layer.py:90-102` |
| other knobs | `uiortholora_alpha`, `uiortholora_dropout`, `scaling_factor`, `enforce_sv_positive`, `initial_scaler`, `initial_sigma`, `init_uiortholora_weights` | `config.py` |

`major_component_size = r − num_svalues_to_adapt`. Two `PeftType`s exist: `UIORTHOLORA` (this tuner) and `UILINLORA` (a vestigial enum value; there is no separate `uilinlora` tuner dir — UILinLoRA is the `k_vec=0` configuration, i.e. ablation **A2**).

---

## 6. ⚠️ Gap you must close before generating ablation configs

`instructions.md` §5 assumes per-component **freeze flags** that **do not yet exist** in `config.py`:
`train_rotations`, `train_sigma_med`, `train_sigma_small`, `train_scalers`, `major_sigma (identity|original)`, `rotation_param (matrix_exp|cayley)`, `sigma_small_init`, `capacity_matched`.

Today the config exposes only: `use_de` (≈ `train_scalers` / A4), `num_svalues_to_adapt` (k_val), `num_svectors_to_adapt` (k_vec, → A2/A7), `initial_sigma`, `initial_scaler`, `enforce_sv_positive`, `scaling_factor`.

**Implementation task #2 in `instructions.md` is therefore still open:** add the freeze/switch flags to `config.py` + wire them in `layer.py` (and ensure freezing *excludes the param from the optimizer*, per §11). Also note:
- `uiortholora_sigma` is a **single** vector of length `num_svalues_to_adapt` spanning both the medium and small tiers — so "freeze `σ_med`" (A3) vs "freeze `σ_small`" means freezing **slices**, not separate parameters. Trace `layer.py`'s forward to confirm exactly which slice maps to which tier before implementing A3.
- `R_U=R_V=I` + "exclude `θ` from optimizer" (A1) means freezing the `left_unitary`/`right_unitary` parametrization params, not deleting the tier.

Confirm the tier↔sigma-slice mapping in `layer.py` (read the forward/merge math) before trusting A1/A2/A3 semantics — `instructions.md` §Do-not-fabricate applies.

---

## 7. Next steps for the server agent

1. Recreate the env per §3; confirm `peft.__version__ == 0.19.1`.
2. Implement the freeze/switch flags (§6) in `config.py`/`layer.py`.
3. Build the capacity calculator (`instructions.md` §4); read per-module dims from the model config (do **not** hardcode 4096).
4. Pick `(k_val*, k_vec*)` matching LoRA-r32 budget; record `P0`.
5. Generate configs A0–A8 (+ param-matched A1) × {cs, math} on LLaMA-2-7B; run via the shared trainer in `notebooks/tuner_knowledge/src/`.
6. Evaluate + produce tables/figures into `results/ablation/` per `instructions.md` §6–§8.
7. Run the §9 sanity checks (A0 reproduces head-to-head numbers; A1-vs-A0 is the headline rotation result).
