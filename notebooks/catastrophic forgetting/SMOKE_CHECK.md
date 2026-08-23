# SMOKE_CHECK — static verification of every worked-example command in RELEASE_README.md

Method: each command below was checked by READING the script's argparse block and I/O
paths (no training executed). "Flag verified" = the exact flag string appears in an
`add_argument` call in the named file. Line numbers refer to the current files.

Verdict summary: all README flags exist; 7 inconsistencies/caveats flagged (F1-F7 below),
all of which are disclosed in the README text.

---

## (a1) `cd portable_parity_pack && python fetch_data.py`

- CLI: none (verified: no argparse in `portable_parity_pack/fetch_data.py`; bare `main()`).
- Reads: HF dataset `meta-math/MetaMathQA` (network/HF cache).
- Writes: `portable_parity_pack/data/metamathqa_395k.json` (path `data/...` relative to cwd,
  hence the `cd` in the README example; `data/` dir is created by the script).
- **F1**: `fetch_data.py` reconstructs ONLY the MetaMathQA file. The commonsense
  training set `commonsense_170k.json` and the CS/GSM8K test sets under
  `repro/LLM-Adapters/dataset/` have no fetcher anywhere in the release; `repro/` is
  empty on this machine. README discloses this and points at the public LLM-Adapters
  repo. (Matches RELEASE_PLAN gap 5.1.)
- **F2**: `portable_parity_pack/run_all.sh` reads `jobs_parity.txt`, which is NOT in
  the pack (verified by `ls`). The README does not use `run_all.sh`, so no README
  inconsistency, but the pack is not runnable as shipped. Also note CODE_MAP's open
  item: the pack is a 07-14 snapshot vs 07-17 root pipeline files (parity un-diffed).

## (a2) `python metamath_prep_395k.py`

- CLI: none (verified: no argparse; module-level script).
- Reads: HF dataset `meta-math/MetaMathQA`; then
  `repro/LLM-Adapters/ft-training_set/metamathqa_100k.json` (schema check, line 21).
- Writes: `repro/LLM-Adapters/ft-training_set/metamathqa_395k.json` (dir auto-created).
- **F3**: the schema check crashes with FileNotFoundError on a fresh copy (100k file
  absent) AFTER the 395K output is already written. Disclosed in README.

## (a3) `python math_test_prep.py` (mentioned in prose)

- Writes `repro/LLM-Adapters/dataset/MATH/test.json` (line 10, `OUT`). GSM8K
  `test.json` is NOT built by it; it comes with the vendored LLM-Adapters repo
  (covered by F1 disclosure).

## (b) `train_cs.py` worked example

Command flags vs `train_cs.py` argparse (lines 179-250):
- `--method lora` — line 180 (choices lora|uiortholora|clora). OK
- `--lora_r 32` — line 239. OK
- `--lora_alpha 64` — line 240. OK
- `--weight_decay 0.3` — line 188. OK
- `--learning_rate 0.0003` — line 187. OK
- `--cutoff_len 256` — line 185. OK
- `--seed 42` — line 196. OK
- `--base_model meta-llama/Llama-2-7b-hf` — line 183 (also the default). OK
- `--run_name frc_lorawd_wd0p3_lr3e4_c256_s42` — line 181. OK
- `--out_root /scratch/cf_models` — line 182 (also the default). OK

Provenance: the command is the train half of a verbatim job line in
`jobs/frc_reservoir_B.txt` (run `frc_lorawd_wd0p3_lr3e4_c256_s42`), with `--out_root`
made explicit. That run exists in the frozen pool (`results/frc_lorawd_wd0p3_lr3e4_c256_s42/`).
- Reads: `repro/LLM-Adapters/ft-training_set/commonsense_170k.json` (DEFAULT_DATA,
  line 92) — requires the F1 vendoring step; base model from HF (gated).
- Writes: `<out_root>/<run_name>/` adapter dir (`adapter_model.safetensors`,
  `run_config.json` written last as the completion marker); falls back to `./models/`
  if `/scratch` is absent (line ~258) — stated in README.
- Init-flag list in README (`--pissa/--milora/--corda/--cordapp/--sclora/--lora_null/--use_dora`)
  — all verified at lines 211-232.

## (c) `eval_one_gpu.py` worked example

Command flags vs `eval_one_gpu.py` argparse (lines 25-58):
- `--adapter` — line 26 (empty/`none` = base-only mode, requires `--run_name`; README's
  `--adapter none` note matches lines 60-63). OK
- `--run_name` — line 29. OK
- `--base_model` — line 30. OK
- `--adapt_task cs` — line 49 (choices include cs, gsm8k, gsm8k_faithful, math,
  math_faithful, medmcqa). OK
- `--ret_suite broad` — line 46 (choices core|broad). OK
- `--ret_limit 0` — line 33 (int, default 0). OK
- `--ret_max_gen 512` — line 32. OK

Provenance: flags match the eval half of the same `jobs/frc_reservoir_B.txt` line
verbatim. Reads: adapter dir from stage (b); CS test sets
`repro/LLM-Adapters/dataset/<ds>/test.json` (`eval_cs.py` line 19/64 — F1 applies);
retention tasks fetched by lm-eval from HF. Writes: `results/<run_name>/summary.json`
(line 214) and appends `campaign_summary.jsonl` via `run_lib.append_registry` (line 215).
- **F4**: lm-eval is not in `requirements-freeze.txt` (venv gap, noted there and in
  the README); the eval command will not run until it is installed, and the fleet's
  exact version is unrecorded.

## (d1) `forgetting_ce.py` worked example

Command flags vs `forgetting_ce.py` argparse (lines 138-151):
- `--runs` (comma-separated run names) — line 139. OK
- `--adapters_root /scratch/cf_models` — line 141 (also default). OK
- `--base_model` — line 142. OK
- `--max_blocks 40` — line 145 (0 = full test set). OK

Reads: `<adapters_root>/<run>/adapter_model.safetensors` (skips if missing, line 177);
WikiText test set via HF `Salesforce/wikitext` (line 76-78). Writes:
`results/<run>/forgetting.json` + appends `results/forgetting.jsonl` (line 223).
- **F5**: `os.environ.setdefault("HF_HUB_OFFLINE", "1")` + `--hf_home` default
  `/scratch/hf_cache` mean a fresh machine fails to download WikiText unless
  `HF_HUB_OFFLINE=0` is exported or the dataset is pre-cached. Disclosed in README.

## (d2) `ce_batch.py` worked example

Command flags vs `ce_batch.py` argparse (lines 153-174):
- `--glob 'frc_lorawd_*'` — line 156 ("comma-separated fnmatch patterns over
  --adapters_root dir names"). OK
- `--adapters_root`, `--base_model`, `--out results/forgetting.jsonl` — lines 158-162
  (all also defaults). OK
- F5 applies here too (same HF_HUB_OFFLINE/hf_home defaults).

## (e) geometry scripts

- `geo_drift_phase1.py`: no argparse (verified). Reads Llama-2 safetensors from the HF
  cache (`snapshot_download`, MODEL hardcoded line 26); writes
  `results/geo_drift/base_svd/<name>.pt`. Resumable (skips existing .pt). `GEO_THREADS`
  env honored (line 20). Phase-1 outputs already exist in the frozen `results/`.
- `geo_drift_phase2.py`: no argparse (verified). Reads `results/geo_drift/base_svd/*.pt`
  (produced by phase 1 — dependency satisfied) and scans adapter dirs.
- **F6**: phase 2 hardcodes `SCRATCH = "/scratch/cf_models"` (line ~41) with no flag;
  a one-line edit is needed if adapters are elsewhere. Disclosed in README. Writes
  `results/geo_drift/adapter_metrics.jsonl` + `results/geo_drift/permatrix/<run>.jsonl`
  (append-mode, resumable). Qwen variants exist as stated
  (`geo_drift_phase1_qwen.py`, `geo_drift_phase2_qwen.py`).

## (f) analysis layer

- `rq1_stats/01_head2head_corrected.py` (and 02/03/04/05/06): argparse-free; run from
  their own directory as the README shows (each file's docstring carries the same
  invocation). `rq1_common.py` inserts `../adjudication` and `../correlations` on
  sys.path (lines 27-28), so the stage inherits those modules' ROOT constants.
- ROOT constants verified by grep:
  - HARDCODED `/home/guyb/...`: `adjudication/adjpool.py:34`,
    `observatory/obs_common.py:24`, `observatory/00_build_master.py:27`,
    `verification/verify_common.py:19` — README's one-line-edit list matches exactly.
  - Relative (no edit needed): `correlations/corr_common.py:22`,
    `insights/00_build_pool.py:21`, `insights/04_permatrix_layers.py:25`.
  - Also hardcoded outside acl_analysis (listed in README, per RELEASE_PLAN 5.5):
    `paper/writing/analysis_a1_a4.py`, `paper/writing/analysis_final/op_points_2026-07-17.py`,
    `paper/writing/make_figs_split_lora_null.py`.
- Inputs all inside the frozen release tree: `results/*/summary.json`,
  `results/geo_drift/adapter_metrics_merged.jsonl`, `results/forgetting_merged.jsonl`,
  `results/quarantine_diverged.txt` (verified in `corr_common.py` lines 67/104/121 and
  `adjpool.py` lines 35/106). Outputs land inside `acl_analysis/` and
  `paper/writing/tables/` (`05_make_appendix_tables.py` line 8: `../../tables/table_tost.tex`
  etc. — verified).
- Exhibit-provenance table in README verified against script headers/outputs:
  `04_ladder_ci.py` -> `ladder_ci.csv/.md` (CIs only; table numbers frozen from
  key_numbers.md section 19.1, which exists at line 654); `01_head2head_corrected.py`
  -> `head2head_corrected.csv/.md`; `correlations/03_league_table.py` ->
  `league_table.csv/.md`; `correlations/04_commonality.py` -> commonality outputs;
  `adjudication/01_op_points.py` -> `tables/op_points_*.csv` + `op_points.md`;
  `adjudication/02_pareto.py` -> `tables/pareto_frontier.csv`, `tables/pareto_bootstrap.csv`,
  `figures/fig_pareto.{png,pdf}`; `06_make_grand_compact.py` copies verbatim from the
  frozen `tables/table_grand.tex` (docstring confirms it never recomputes).
- **F7**: `tables/table_grand.tex` (full) has NO generator anywhere in the tree
  (grep for `table_grand` in *.py matches only `06_make_grand_compact.py`, a reader).
  Disclosed in README, consistent with RELEASE_PLAN gap 5.3.

## Environment section cross-check

- `requirements-freeze.txt` and `requirements-full.txt` EXIST at the folder root
  (16-line pin list; notes the local editable PEFT fork and the missing lm-eval pin) —
  README references match the actual files.
- PEFT fork location `../../peft` relative to this folder =
  `/home/guyb/UIOrthoLoRA/peft/` — matches CODE_MAP section 1's dependency note.

## Exclusion-list compliance

RELEASE_README.md was grepped against RELEASE_PLAN section 2: it contains no reference
to `archive/`, `handoff/`, `papers/`, `restart_staging/`, `.claude/`,
`paper/.overleaf-git/`, `paper/figs_v2/`, `figures_frozen_backup/`,
`results/fleet_reg/`, `results/evac_logs/`, or `jobs/ce_chunks/`. The single mention of
`fleet/evac_merge.py` is the file RELEASE_PLAN section 2 explicitly keeps.
