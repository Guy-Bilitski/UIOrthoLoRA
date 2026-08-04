# Catastrophic Forgetting in PEFT: What Governs Retention Loss?

Code and frozen results for an observational study of catastrophic forgetting under
parameter-efficient fine-tuning (PEFT). We train a frozen pool of ~1,035 LoRA-family
adapters (n = 1,034 after dedup in the pooled statistics) across two 7B base models
(Llama-2-7B, Qwen2.5-7B) and two adaptation tasks (Commonsense-170K, MetaMathQA math),
covering LoRA, LoRA+weight-decay, CLoRA, DoRA, PiSSA, MiLoRA, CorDA/CorDA++, SC-LoRA,
and LoRA-Null, swept over learning rate, weight decay, and seeds. Every adapter is
evaluated under one protocol for in-domain adaptation, retention (BBH + MMLU-Pro, plus
a broad suite), cross-entropy drift on base-model text, and update geometry relative to
the base weights' singular subspaces. The central observation: retention tracks the
update magnitude ||dW||_F; geometry metrics add little once magnitude is controlled.
A 284B-scale recurrence check (DeepSeek-V4-Flash) is included via `scripts/deepseek/`.

Note on an internal name: the method string `uiortholora` appears in
`train_cs.py`/`run_lib.py`. It is the project's internal name for its orthogonal-update
LoRA variant; the string is left untouched because the training code is frozen.

## Repo layout (release tree)

```
train_cs.py              shared trainer: every method = --method lora|clora|uiortholora + an init flag
eval_one_gpu.py          in-process evaluator: CS suite / math / medmcqa + retention + F_delta
run_lib.py               shared prompt templates (train == eval), logging, registries
eval_cs.py               commonsense generative eval (used by eval_one_gpu.py)
uio_inprocess.py         F_delta implementation + CS dataset list (imported by eval_one_gpu.py)
math_eval.py             faithful 0-shot GSM8K + Hendrycks MATH eval
bbh_metric_fix.py        lm-eval BBH answer-normalization patch
norm_trace.py            per-step ||dW||_F + loss trainer callback
residual_save.py         rank-2r W0-relative save conversion for residual-init methods
corda_init.py, cordapp_init.py, milora_init.py, data_aware_init.py (PiSSA),
sclora_init.py, lora_null_init.py        method init modules (train_cs.py flags)
fdelta.py                standalone F_delta reference implementation
forgetting_ce.py         CE-drift-to-base metric (WikiText)
ce_batch.py              batch driver for forgetting_ce.py
geo_drift_phase1.py      base-model SVD references (Llama)   } Qwen variants:
geo_drift_phase2.py      per-adapter geometry battery        } *_qwen.py
metamath_prep_395k.py    MetaMathQA-395K training-set builder
math_test_prep.py        Hendrycks MATH test-set builder (alpaca schema)
analyze_full_2026-07-16.py, analyze_adversarial_2026-07-16.py,
analyze_ebatch_2026-07-17.py, flag_diverged.py    frozen-section (sec. 18) recompute scripts
validate_frepro_residual.py, validate_residual_zero_step.py,
validate_cordapp_cpu.py, base_retention_check.py  correctness gates
gpu_pool.py, auto_dispatch.py, make_frepro_jobs.py, build_lean.py
                         how the campaign was run (inert; kept for provenance)
portable_parity_pack/    self-contained copy of the train/eval pipeline + fetch_data.py
scripts/deepseek/        DeepSeek-V4-Flash 284B pipeline (train/eval/CE/geometry)
jobs/                    example job lists (one shell line per adapter); ce_backfill_qwen.txt
                         documents the unfillable Qwen CE cells cited by the paper
results/                 FROZEN data of record: ~1,661 per-run dirs, each with summary.json;
                         forgetting_merged.jsonl (CE drift), geo_drift/ (SVDs + adapter metrics,
                         adapter_metrics_merged.jsonl), quarantine_diverged.txt (71 excluded runs),
                         ds_adapters_evac/ (the 21 surviving DeepSeek-284B adapters + SHA256SUMS)
repro/                   vendored reference repos + training data (CLoRA, CorDA, LLM-Adapters,
                         LoRA-Null, MiLoRA, SC-LoRA); see Data below if empty in your copy
paper/writing/           paper sources, figure/table generators, and data/key_numbers.md
                         (the canonical numbers document; quote only sections 18 and 19)
paper/writing/acl_analysis/   the statistics layer (rq1_stats/, correlations/, adjudication/,
                         observatory/, insights/, verification/) - see stage (f) below
paper/writing/analysis_final/ post-freeze CPU analyses (ladder, operating points, seed stability)
paper/evidence/          validation-gate logs
requirements-freeze.txt  pinned load-bearing package versions (full list: requirements-full.txt)
```

The canonical merged aggregates in `results/` were built by `fleet/evac_merge.py`
(retained from an otherwise ops-only layer that is not part of this release).

## Environment

The venv is built around the **PEFT fork at the repository root** (`../../peft` relative
to this folder, i.e. the `peft/` directory two levels up). This fork is a hard
dependency: it contains the `UIOrthoLoRAConfig` and the modified `layer.py` /
`config.py` used by every training run. Install it editable, then the pins:

```
python -m venv .venv && . .venv/bin/activate
pip install -e ../../peft                 # the fork, NOT pypi peft
pip install -r requirements-freeze.txt    # torch 2.12.0, transformers 5.10.2, etc.
pip install lm-eval                       # retention evals; fleet version unrecorded (known gap)
```

`requirements-freeze.txt` (generated for this release) lists the load-bearing pins;
`requirements-full.txt` is the complete freeze. Python 3.12.

Base models are gated on Hugging Face: `meta-llama/Llama-2-7b-hf` requires accepting
the Llama-2 license and `huggingface-cli login`; `Qwen/Qwen2.5-7B` is ungated.

## Pipeline: one worked example per stage

All commands run from this folder root with the venv active, one GPU per job
(`CUDA_VISIBLE_DEVICES=0 ...`). The worked cell below is a real campaign cell:
LoRA + weight decay, wd = 0.3, lr = 3e-4, Llama-2-7B on Commonsense-170K
(run `frc_lorawd_wd0p3_lr3e4_c256_s42`, taken verbatim from `jobs/frc_reservoir_B.txt`).

### (a) Data prep

```
# math training set (public reconstruction of the campaign file):
cd portable_parity_pack && python fetch_data.py && cd ..
# -> portable_parity_pack/data/metamathqa_395k.json  (script has no flags; prints a
#    SHA256 to compare against the original; see its docstring for the caveat)

# or build it in the layout the root trainer expects:
python metamath_prep_395k.py
# -> repro/LLM-Adapters/ft-training_set/metamathqa_395k.json
```

Honest caveats: (1) `metamath_prep_395k.py` ends with a schema check that reads
`repro/LLM-Adapters/ft-training_set/metamathqa_100k.json`; on a fresh copy without
that file the check crashes after the 395K file is already written (harmless, but
expect the traceback). (2) The commonsense training set
(`repro/LLM-Adapters/ft-training_set/commonsense_170k.json`) and the CS/GSM8K test
sets (`repro/LLM-Adapters/dataset/<task>/test.json`) have **no fetch script**; vendor
them from the public LLM-Adapters repository
(https://github.com/AGI-Edgerunners/LLM-Adapters) into `repro/LLM-Adapters/`.
`python math_test_prep.py` then builds `repro/LLM-Adapters/dataset/MATH/test.json`.

### (b) Train one adapter

```
python train_cs.py --method lora --lora_r 32 --lora_alpha 64 \
  --weight_decay 0.3 --learning_rate 0.0003 --cutoff_len 256 --seed 42 \
  --base_model meta-llama/Llama-2-7b-hf \
  --run_name frc_lorawd_wd0p3_lr3e4_c256_s42 --out_root /scratch/cf_models
```

Training data defaults to `repro/LLM-Adapters/ft-training_set/commonsense_170k.json`
(`--data_path` to override; the math runs pass the MetaMathQA file). If the parent of
`--out_root` does not exist, the script falls back to `./models/` inside this folder.
Output: adapter files (`adapter_model.safetensors`, `run_config.json`, ...) under
`<out_root>/<run_name>/`. Other methods are selected with `--method clora|uiortholora`
or `--method lora` plus one init flag (`--pissa 1`, `--milora 1`, `--corda 1`,
`--cordapp 1`, `--sclora 1`, `--lora_null 1`, `--use_dora 1`).

### (c) Evaluate

```
python eval_one_gpu.py --adapter /scratch/cf_models/frc_lorawd_wd0p3_lr3e4_c256_s42 \
  --run_name frc_lorawd_wd0p3_lr3e4_c256_s42 --base_model meta-llama/Llama-2-7b-hf \
  --adapt_task cs --ret_suite broad --ret_limit 0 --ret_max_gen 512
```

`--ret_limit 0` means the full retention sets (no subsampling); `--ret_suite broad`
adds MMLU/ARC-c/TruthfulQA on top of the core BBH + MMLU-Pro. Math cells use
`--adapt_task gsm8k` (or `math_faithful`). Passing `--adapter none` with a
`--run_name` scores the raw base model (the `base_` ceiling rows). Output:
`results/<run_name>/summary.json`, the per-run source of record.

### (d) CE drift

```
python forgetting_ce.py --runs frc_lorawd_wd0p3_lr3e4_c256_s42 \
  --adapters_root /scratch/cf_models --base_model meta-llama/Llama-2-7b-hf \
  --max_blocks 40
```

Writes `results/<run>/forgetting.json` and appends `results/forgetting.jsonl`.
`--runs` is comma-separated for several adapters; the campaign-scale driver is

```
python ce_batch.py --glob 'frc_lorawd_*' --adapters_root /scratch/cf_models \
  --base_model meta-llama/Llama-2-7b-hf --out results/forgetting.jsonl
```

Note: both scripts default `HF_HUB_OFFLINE=1` and `HF_HOME=/scratch/hf_cache`; on a
fresh machine either pre-download `Salesforce/wikitext` or run once with
`HF_HUB_OFFLINE=0` in the environment.

### (e) Geometry

```
python geo_drift_phase1.py     # one-time: SVD of the 160 Llama target matrices
python geo_drift_phase2.py     # per-adapter battery over every saved adapter
```

Neither script takes CLI flags (thread count via `GEO_THREADS`, default 16/12).
Phase 1 writes `results/geo_drift/base_svd/*.pt`; phase 2 scans the adapter root
(hardcoded to `/scratch/cf_models` in `SCRATCH` at the top of the script; edit that
one line if your adapters live elsewhere) and writes
`results/geo_drift/adapter_metrics.jsonl` plus `results/geo_drift/permatrix/<run>.jsonl`.
For Qwen adapters use `geo_drift_phase1_qwen.py` / `geo_drift_phase2_qwen.py`
(fingerprints are never pooled across model families).

### (f) Analysis layer (statistics on the frozen pool)

The statistics layer lives in `paper/writing/acl_analysis/` and runs on CPU directly
from `results/` (no checkpoints needed):

```
cd paper/writing/acl_analysis/rq1_stats
python 01_head2head_corrected.py     # then 02, 04, 05, 06; adjudication/ and
                                     # correlations/ scripts run the same way
```

Honest caveat: four shared modules currently hardcode the campaign machine's path as
`ROOT = "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"` and need a one-line
edit to point at your checkout root: `adjudication/adjpool.py` (line 34),
`observatory/obs_common.py` (line 24), `observatory/00_build_master.py` (line 27),
`verification/verify_common.py` (line 19). (`rq1_stats/` imports `adjpool`, so the
first edit covers it.) `correlations/corr_common.py` and `insights/` already resolve
the root relative to their own location and need no edit. The same hardcoding exists
in `paper/writing/analysis_a1_a4.py`, `paper/writing/analysis_final/op_points_2026-07-17.py`,
and `paper/writing/make_figs_split_lora_null.py`.

## What is and is not reproducible

Reproducible from this release:
- **The entire analysis and statistics layer.** `results/` is the frozen data of
  record (~1,661 run dirs with `summary.json`; merged CE and geometry aggregates;
  `quarantine_diverged.txt` defining the 71 excluded runs). Every number in
  `paper/writing/data/key_numbers.md` sections 18-19 and every `acl_analysis/`
  exhibit recomputes from these files on CPU.
- **The training/eval pipeline itself**, given the data-prep step and GPU time: every
  adapter in the pool is one job line of the form shown above.

Not reproducible:
- **The 7B adapter checkpoints.** All were destroyed when the training fleet was
  decommissioned. Re-running stages (d)/(e) on a 7B adapter therefore requires
  retraining it first via stage (b). The only surviving checkpoints are the 21
  DeepSeek-V4-Flash 284B adapters in `results/ds_adapters_evac/` (with SHA256SUMS).
- **`paper/writing/tables/table_grand.tex` (full nine-column version).** Its generator
  was lost; the committed file is frozen and its inputs are intact, but regeneration
  needs a rewrite. The compact body version DOES have a committed generator,
  `paper/writing/acl_analysis/rq1_stats/06_make_grand_compact.py`, which copies
  numbers verbatim from the frozen full table (it never recomputes).
- **Base models without a Hugging Face login.** Llama-2-7B is license-gated.
- The exact lm-eval version used on the fleet is unrecorded (see
  `requirements-freeze.txt` note); retention numbers reproduced with a current
  lm-eval may differ marginally.

## Where each paper exhibit comes from

| Exhibit | Source |
|---|---|
| `tables/table_ladder.tex` (nested delta-R2 ladder) | numbers frozen in `paper/writing/data/key_numbers.md` section 19.1; cluster-bootstrap CIs from `acl_analysis/rq1_stats/04_ladder_ci.py` (`ladder_ci.csv/.md`) |
| `tables/table_tost.tex`, `table_mde.tex`, `table_fragility.tex` | `acl_analysis/rq1_stats/05_make_appendix_tables.py` |
| Head-to-head comparisons (Holm-corrected, exact p, CIs) | `acl_analysis/rq1_stats/01_head2head_corrected.py` (`head2head_corrected.csv/.md`), building on `acl_analysis/adjudication/03_head2head.py` |
| Metric league table / commonality decomposition | `acl_analysis/correlations/03_league_table.py` and `04_commonality.py` (the `correlations/` chain 01-06) |
| Operating-point tables and Pareto frontier (`fig_pareto`) | `acl_analysis/adjudication/01_op_points.py` and `02_pareto.py` (outputs under `adjudication/tables/` and `adjudication/figures/`) |
| `tables/table_grand.tex` (full) / compact body version | frozen (generator lost) / `acl_analysis/rq1_stats/06_make_grand_compact.py` |
| Frozen headline numbers (section 18) | `analyze_full_2026-07-16.py`, `analyze_adversarial_2026-07-16.py`, `analyze_ebatch_2026-07-17.py` over `results/*/summary.json` |
