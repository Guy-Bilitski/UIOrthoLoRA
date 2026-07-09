# WORKDIR ALIGNMENT — single onboarding doc

**Working dir:** `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/` (note the space — quote it).
**Repo root:** `/home/guy/UIOrthoLoRA`  ·  **Branch:** `ortho_new`  ·  **Written:** 2026-07-06 (repo-hygiene pass).

> This file is the current-state onboarding map. Where older docs disagree, this file + `handoff/20_FAITHFUL_REPRO_SPEC.md`
> win. `README.md` (Phase-1 UIOrthoLoRA go/no-go) and `STATUS.md` (2026-06-29 2×2 campaign) are HISTORICAL —
> see the "Stale docs" note at the bottom and `CLEANUP_MANIFEST.md`.

---

## (a) Goal & thesis

We study **what governs catastrophic forgetting (CF) when a 7B LLM is PEFT-fine-tuned**. The central
claim — **THE MAGNITUDE LAW** — is that *retention is governed by the size of the weight update
‖ΔW‖_F, not by the adapter method*: LoRA, CLoRA, MiLoRA, PiSSA, DoRA and the data-aware inits all fall
on one retention-vs-‖ΔW‖ curve, so the simplest magnitude control (**plain LoRA + weight decay**)
matches or beats elaborate structured/data-aware adapters at equal ‖ΔW‖. The corollary is that reported
single-LR "wins" for fancy adapters over LoRA are **LR/recipe artifacts**: at a fancy method's favored
LR it simply lands at a different ‖ΔW‖, and an LR-swept LoRA+wd frontier dominates it.

The **current centerpiece** is a *faithful CLoRA-recipe reproduction* (`handoff/20`): rebuild CLoRA's
exact commonsense (Table 2) and math (Table 3) settings on LLaMA-2-7B, run full LR sweeps, and put
LoRA / LoRA+wd head-to-head against LoRA-Null, MiLoRA, PiSSA, SC-LoRA, CLoRA (published numbers) and
CorDA++. This *faithful reproduction + LR×wd sweep* is the **primary track**. The older 231/532-cell
"fairness study" (`handoff/19`, eval-matched-calibration re-runs of the data-aware arms) is **valid but
deprioritized** below the faithful repro. The abandoned original goal ("is UIOrthoLoRA an A*-worthy
CLoRA-beater?") is **DEAD** — UIOrthoLoRA only tied CLoRA; it survives only as `uio_inprocess.py`, whose
`fdelta_inprocess`/`CS_DATASETS` helpers `eval_one_gpu.py` still imports.

---

## (b) Experimental settings — CLoRA recipe (ground truth = paper Table 2/3 + `handoff/20`)

**Shared across both tables** (verified faithful, `handoff/20 §1.1`):

| Hyperparameter | Value |
|---|---|
| Base model | `meta-llama/Llama-2-7b-hf` |
| Optimizer | AdamW (`adamw_torch`) |
| LR scheduler | linear |
| Warmup | 100 steps |
| Batch size (effective) | 16 (micro 16 × grad-accum 1) |
| Epochs | 3 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, up_proj, down_proj` (5 × 32 = 160 matrices) |
| A/B init | A ~ Gaussian, B = 0 (PEFT default) |
| Precision | bf16 (paper used fp16; benign B200 change, disclosed) |
| Checkpoint reported | last/final (`save_strategy="no"`) |

**Table 2 — Commonsense (prefix `frc`)**

| Item | Value |
|---|---|
| Train data | Commonsense170K (`commonsense_170k.json`, 170,420 rows) |
| `cutoff_len` | 256 |
| Rank / α | r=32, α=64 |
| In-domain eval | 8 CS datasets (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA), gen MC acc — `eval_cs.run_eval` / `uio_inprocess.CS_DATASETS` |
| Out-domain (retention) | answer-only `bbh_fewshot` (3-shot) + `mmlu_pro` (5-shot CoT) via in-process lm-eval; `broad` adds mmlu/arc_challenge/truthfulqa_mc2 |

**Table 3 — Math (prefix `frm`)**

| Item | Value |
|---|---|
| Train data | MetaMathQA-395K (`metamathqa_395k.json`; built by `metamath_prep_395k.py`) |
| `cutoff_len` | 256 primary (512 as sensitivity pair on lora + lorawd core) |
| Rank / α | r=64, α=128 |
| Eval (adapt) | **faithful 0-shot Alpaca-template** GSM8K (1319, last-number, abs-diff ≤1e-3) + Hendrycks MATH (5000, boxed / "The answer is:" + `is_equiv`) — `math_eval.py`, `--adapt_task math_faithful`. NOT the old lm-eval 5-shot `gsm8k` (train/eval template mismatch, `handoff/20 §4.1`) |
| Retention | same BBH+MMLU-Pro (+broad), `--ret_max_gen 256` (math models don't emit EOS) |

Reference targets (Table 3 GSM8K/MATH): LoRA 60.58/16.88 · PiSSA 58.23/15.84 · MiLoRA 63.53/17.76 ·
CLoRA-k64 64.29/17.52 · CLoRA-k128 64.59/18.38 · CLoRA-k256 63.45/17.58.

LoRA+wd sweep grid (both tables, native r/α): **LR ∈ {1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3} × wd ∈
{0.0, 0.1, 0.2, 0.3, 0.5}** (wd=0 column == plain LoRA; LR3e-4/wd0 == the LoRA baseline, deduped).
Depth-first core = LR{2e-4,3e-4,5e-4} × wd{0.1,0.2,0.3}. Reproduction baselines run at CLoRA's fixed
LR=3e-4; data-aware arms (milora/sclora/lora_null) are LR-swept at the faithful r/α.

---

## (c) Pipeline + LIVE file map

```
  make_frepro_jobs.py  --table {math,cs} --prefix {frm,frc}
        |   (emits per-arm  "train_cs.py ... && eval_one_gpu.py ..."  cells; resumable)
        v
  jobs/frepro_math.txt (55) + jobs/frepro_cs.txt (48)
        |
  build_lean.py   (merges; already-trained cells -> eval-only so they aren't retrained)
        v
  jobs/frepro_lean.txt  (103 cells)   <-- the RUNNING job file
        |
  gpu_pool.py --gpus 8 --tag frepro3 --jobs jobs/frepro_lean.txt   [LIVE, pid 2932862]
        |                         (1 job/GPU; per-job log logs/frepro3_<i>.log)
        +--> train_cs.py   (shared trainer; --method lora|clora + adapter-init flag)
        |        imports run_lib; on demand: corda_init / milora_init / data_aware_init(pissa) /
        |        sclora_init / lora_null_init / residual_save
        +--> eval_one_gpu.py  (in-process CS or math_faithful adapt + BBH/MMLU-Pro retention + F-delta)
                 imports run_lib, eval_cs, uio_inprocess; on demand: math_eval, bbh_metric_fix
        v
  results/<run>/summary.json  +  results/campaign_summary.jsonl  (one line/run)
```

**LIVE files** (running pool + its train/eval import chain — verified by grep, not guessed):

| File | Role |
|---|---|
| `gpu_pool.py` | 8-GPU scheduler (running as tag `frepro3`); 1 job/GPU, per-job logs, sets `OMP/MKL_NUM_THREADS=8` |
| `jobs/frepro_lean.txt` | the 103-cell job list the pool is executing (48 CS + 55 math) |
| `train_cs.py` | shared trainer; every method = `--method lora|clora` + an init flag (`--milora/--pissa/--sclora/--lora_null/--corda`, `--weight_decay`) |
| `eval_one_gpu.py` | in-process evaluator (adapt = CS suite or math_faithful; retention = BBH+MMLU-Pro; F-delta) |
| `run_lib.py` | shared prompt templates (train==eval), logging, registries |
| `eval_cs.py` | commonsense gen eval; `run_eval(model,…)` batching reused in-process |
| `uio_inprocess.py` | UIO-era, but `eval_one_gpu` imports its `fdelta_inprocess` + `CS_DATASETS` → LIVE |
| `math_eval.py` | faithful GSM8K + Hendrycks MATH eval (imported by `eval_one_gpu` for `math_faithful`) |
| `bbh_metric_fix.py` | patches lm-eval BBH answer-only metric normalization (imported by `eval_one_gpu`) |
| `residual_save.py` | rank-2r W0-relative adapter conversion for residual-init methods (scaling-generalized) |
| `corda_init.py` | static CorDA-KPA init (imported by `train_cs --corda`; reused by `cordapp_init`) |
| `milora_init.py` | MiLoRA bottom-r SVD init |
| `data_aware_init.py` | PiSSA top-r SVD init (`pissa_BAR`) |
| `sclora_init.py` | SC-LoRA D+/D− covariance init |
| `lora_null_init.py` | LoRA-Null null-space init |
| `results/` | provenance: `campaign_summary.jsonl`, `train_registry.jsonl`, `eval_registry.jsonl`, per-run dirs |
| `logs/` | per-job logs (pool writes `logs/frepro3_*.log`) |

**ACTIVE-SUPPORT** (part of the current campaign; run out-of-band, not imported at runtime):
`make_frepro_jobs.py`, `build_lean.py`, `jobs/frepro_{math,cs,all}.txt`, `validate_frepro_residual.py`,
`validate_residual_zero_step.py`, `validate_cordapp_cpu.py`, `cordapp_init.py`, `metamath_prep_395k.py`,
`math_test_prep.py`, `base_retention_check.py`, and the paper generator `paper_figs_v2.py` (+ the
`paper/writing/` package).

---

## (d) Adapter roster + implementation status

| Adapter | Flag(s) | Status |
|---|---|---|
| **LoRA** | `--method lora` | done; the anchor; wd=0 column of the sweep |
| **LoRA+wd** (hero) | `--method lora --weight_decay <wd>` | done; **LR×wd sweeping now** |
| **CLoRA** | `--method clora --clora_k <k>` | implemented + faithful (penalty verified); **published numbers used** for the main table (k64/128/256 math also trained: `frm_clora_*` done) |
| **MiLoRA** | `--method lora --milora 1` | implemented + LR-swept (residual method, residual_save) |
| **PiSSA** | `--method lora --pissa 1` | implemented + swept (residual method); wired per `handoff/20 §2.3` |
| **SC-LoRA** | `--method lora --sclora 1 --sclora_beta 0.5` | implemented; **sweeping** (calib = D+ task / D− nq_open, paper default) |
| **LoRA-Null** | `--method lora --lora_null 1` | implemented; **sweeping** (calib = nq_open) |
| **DoRA** | `--method lora --use_dora 1` | implemented; kept as an extra (dropped from headline ~2× train cost) |
| **CorDA (static KPA)** | `--method lora --corda 1` | implemented + faithful; **excluded from tables** (nq_open calib ≠ academic eval confound, `handoff/18`) |
| **CorDA++** | (not yet wired) | **implemented in `cordapp_init.py`, CPU-validated 14/14** (`validate_cordapp_cpu.py`); **wiring into `train_cs.py` PENDING**; candidate-pool `DEFAULT_N` is currently **8 and must be changed to 5** at the next pool restart (paper N unresolved, `handoff/17 §8`) |

Residual-init methods (MiLoRA, PiSSA, SC-LoRA, LoRA-Null, CorDA, CorDA++) **require** the rank-2r
W0-relative conversion at save (`residual_save.py`) or eval explodes (see gotchas).

---

## (e) Current campaign status (parsed 2026-07-06)

- `jobs/frepro_lean.txt` = **103 cells** (48 CS `frc_*` + 55 math `frm_*`); pool `frepro3` started 09:39.
- **Math (`frm`) complete = 7** (all have `summary.json`): `clora_k64/k128/k256`, `lora`, `lorawd_wd0`,
  `milora`, `pissa` (all `lr3e4_c256_s42`, except `lorawd_wd0_lr1e4`). **CS (`frc`) complete = 0** (just started).
- `results/campaign_summary.jsonl` = **430 records / 414 unique run_names** spanning the whole project
  history (not just frepro). Top prefixes: `mtx` 102, `lrsw` 63, `qwsw` 50, `lora` 36, `lrswm` 36,
  `uio` 33, `clora` 18, `scl` 18, `grid` 9, `mtxm` 8, `frm` 7, `dora` 6, `corda` 2. (The `fr_*` rows are
  legacy UIO frontier runs, NOT frepro.) 504 subdirs under `results/`.

---

## (f) Key decisions log

| Date | Decision |
|---|---|
| 2026-06-15 | **Pivot** from "beat CLoRA with UIOrthoLoRA" (DEAD, only ties) to a **controlled CF study**; magnitude-law thesis adopted. |
| 2026-06-13→ | **Eval-protocol / measurement discoveries:** answer-only `bbh_fewshot` (3-shot) reproduces CLoRA base (33.1 vs 34.91), NOT CoT BBH; fast-retention ≈ full + ~0.9pp; UIO must be evaluated **in-process** (reload bugs). |
| ~mid-June | **DoRA discarded** from the headline — ~2× LoRA train cost for no retention edge. |
| 2026-06-29 | 2×2 matrix result: **magnitude law confirmed** (pooled r≈−0.87; on-curve −0.92); LoRA+wd0.3 & MiLoRA r32 win; CorDA/SC-LoRA fall off-curve **but calibration-confounded**. `paper_figs_v2.py` becomes canonical (`paper_assets.py` deprecated). |
| 2026-07-01 | Eval fixes: gen-cap 512/max-len 4096 (`handoff/15`), BBH metric normalization (`handoff/16`). Rows before commits `fe0f9be3`/`2602f57d` are not comparable to post-fix rows. |
| 2026-07-02 | Paper package assembled (`paper/writing/`, `campaign_summary_clean.jsonl`); CorDA scrubbed from deliverables → **6-of-8 law**; CorDA++ plan finalized (`handoff/17`); off-curve language **embargoed** pending eval-matched re-run. |
| 2026-07-05 | **Faithful-repro pivot** (`handoff/20`): rebuild CLoRA's exact CS+math recipe (r32/α64, r64/α128), **faithful math eval** via new `math_eval.py`, **`residual_save.py` generalized to scaling≠1** (α≠r), **PiSSA wired**, MetaMathQA-395K + Hendrycks MATH built. **256-vs-512 math cutoff sensitivity** flagged (256 truncates CoT tail; 512 recommended, disclosed). |
| 2026-07-06 | camp5 drained; `frepro3` pool launched on `jobs/frepro_lean.txt` (this is the live campaign). |

---

## (g) Gotchas (do not relearn the hard way)

- **ONE GPU scheduler at a time.** Two `gpu_pool.py` each grab all 8 GPUs → 2 runs/GPU → OOM (~45h lost
  twice). `frepro3` (pid 2932862) is live — **never launch a second pool**, and **never edit a live `.py`
  the pool executes fresh per job** (it would corrupt every not-yet-started cell). Develop code changes in
  a git worktree/branch and run only after the pool drains.
- **`pgrep -f "<pat>"` self-matches** its own command line. Use the bracket trick `grep '[g]pu_pool'`
  (or kill by explicit PID).
- **Residual rank-2r conversion is mandatory** for residual-init methods (MiLoRA/PiSSA/SC-LoRA/LoRA-Null/
  CorDA/CorDA++). PEFT persists only the adapter, not the mutated base `W_res`; reloading onto the original
  W0 double-counts and eval explodes (‖ΔW‖ blows up, retention→0). `residual_save.py` stacks
  `A''=[A_tr;A_init]`, `B''=[B_tr,−B_init]` (rank-2r, scaling preserved). Gate: `validate_residual_zero_step.py`
  (0-step post-reload ‖ΔW‖ < 1e-4).
- **Eval must be in-process.** PEFT save/reload mangles some adapters (originally UIO rotators; also the
  residual class above) — `eval_one_gpu.py` loads a fresh W0 + adapter and evaluates in one process.
- **Extreme LRs diverge:** LR ≥ 2e-3 → NaN weights → eval crashes (`probability tensor contains inf/nan`)
  and the pool retries forever. Those LRs are out of the grid.
- **Math eval is heavy:** math-tuned models don't emit EOS → use `--ret_max_gen 256` for math retention
  (512 crawls/OOMs).
- **gpu_pool job lines must use the full venv python** (`/home/guy/UIOrthoLoRA/.venv/bin/python`), never
  bare `python` (else rc=127, GPUs silently idle). `make_frepro_jobs.py` already does this.
- **Don't mix pre/post eval-fix rows** (commits `fe0f9be3`/`2602f57d`) in any final table.

---

## Stale docs — read the current ones

Current canonical order: **this file** → `handoff/20_FAITHFUL_REPRO_SPEC.md` (the live plan) →
`paper/writing/FINALIZATION_PLAN.md` + `paper/writing/data/key_numbers.md` (paper single-source-of-truth).
`README.md` (Phase-1 UIO go/no-go) and `handoff/00`–`12` are HISTORICAL; `STATUS.md` (2026-06-29) and
`handoff/17`/`19` describe the pre-pivot camp5 2×2 / fairness campaign and are superseded in priority by
`handoff/20`. See `CLEANUP_MANIFEST.md` for the full stale-file list.
