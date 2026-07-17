# WORKDIR ALIGNMENT — single onboarding doc

**Working dir:** `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/` (note the space — quote it).
**Repo root:** `/home/guy/UIOrthoLoRA`  ·  **Branch:** `ortho_new`  ·  **Written:** 2026-07-06, **updated 2026-07-09** (repo-hygiene pass).

> This file is the current-state onboarding map. Where older docs disagree, this file + `handoff/20_FAITHFUL_REPRO_SPEC.md`
> win. `README.md` (Phase-1 UIOrthoLoRA go/no-go) and `STATUS.md` (2026-06-29 2×2 campaign) are HISTORICAL —
> see the "Stale docs" note at the bottom and `CLEANUP_MANIFEST.md`.

> **⚠ UPDATE 2026-07-17 — campaign FROZEN, fleet EVACUATED.** The fleet was evacuated on 2026-07-17
> (`handoff/41_EVACUATION_2026-07-17.md` is the live state doc); there are **no live pools**, and every
> live-fleet section below (the "LIVE file map" in §c, the campaign status in §e) is **INERT** — kept as
> a historical roster, not a description of a running system. Onboarding now goes:
> `paper/writing/analysis_final/` (story layer, 01–07 + `PAPER_BLUEPRINT.md`) +
> `paper/writing/data/key_numbers.md` **§18 FINAL FREEZE + §19 POST-FREEZE ADDENDUM** (canonical numbers:
> pooled r=−0.847, n=1035; headline wording is "magnitude relation (flat-then-falling with a knee)", not
> "Magnitude Law"). Source of record: `results/*/summary.json` (`campaign_summary.jsonl` / `results_book/`
> are stale). `STATUS.md` and the 07-02 writing suite were archived to `archive/writing_2026-07-17/`.

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

## (c) Pipeline + LIVE file map — **INERT as of 2026-07-17** (fleet evacuated, see `handoff/41`; roster kept for provenance)

```
  make_frepro_jobs.py  --table {math,cs} --prefix {frm,frc}
        |   (emits per-arm  "train_cs.py ... && eval_one_gpu.py ..."  cells; resumable)
        v
  jobs/frepro_math.txt + jobs/frepro_cs.txt   (frepro4 wave: jobs/frepro4_{math,cs}.txt)
        |
  build_lean.py   (merges frepro4_{math,cs} -> frepro4_lean; already-trained cells -> eval-only)
        v
  jobs/frepro4_lean.txt + wave-specific lists (frepro4_main5/reservoir/b4/inject/headline_math2)
        |
  TWO schedulers (single-scheduler-per-GPU discipline):
   gpu_pool.py --gpu_ids <ids> --tag <t> --jobs jobs/<file>.txt   (fixed GPUs, 1 job/GPU)
   auto_dispatch.py --jobs jobs/master_dispatch.txt --gpus 0-7     (self-refilling; absorbs freed GPUs)
        |
        +--> train_cs.py   (shared trainer; --method lora|clora + adapter-init flag)
        |        imports run_lib; on demand: corda_init / cordapp_init / milora_init /
        |        data_aware_init(pissa) / sclora_init / lora_null_init / residual_save
        +--> eval_one_gpu.py  (in-process CS or math_faithful adapt + BBH/MMLU-Pro retention + F-delta)
                 imports run_lib, eval_cs, uio_inprocess; on demand: math_eval, bbh_metric_fix
        v
  results/<run>/summary.json  +  results/campaign_summary.jsonl  (one line/run)
```

**LIVE files** (running pools + dispatcher + their train/eval import chain — verified by grep + `ps`, not
guessed). As of 2026-07-09 the live schedulers were **4 `gpu_pool` pools** (tags `frepro4`, `frepro4b4`,
`frepro4hs`, `frepro4inj`) + **`auto_dispatch`** on `jobs/master_dispatch.txt` — NOT the old `frepro3`
pool. **[INERT 2026-07-17: nothing runs anymore — fleet evacuated, `handoff/41`. "LIVE" below reads
"was live during the campaign".]**

| File | Role |
|---|---|
| `gpu_pool.py` | fixed-GPU scheduler; 1 job/GPU, per-job logs, sets `OMP/MKL_NUM_THREADS=8` |
| `auto_dispatch.py` | self-refilling dispatcher; picks up GPUs after their owning pool exits (never collides) |
| `jobs/master_dispatch.txt` | the master queue `auto_dispatch` drains |
| `jobs/frepro4_{main5,b4,headline_math2,inject}.txt` | the per-pool job lists currently executing |
| `jobs/frepro4_{cs,math,lean,reservoir,qwen}.txt`, `jobs/frepro_{lean,cs,math,all}.txt` | source/prior-wave job lists (kept; `build_lean` reads frepro4_{math,cs}) |
| `train_cs.py` | shared trainer; every method = `--method lora|clora` + an init flag (`--milora/--pissa/--sclora/--lora_null/--corda/--cordapp`, `--weight_decay`) |
| `eval_one_gpu.py` | in-process evaluator (adapt = CS suite or math_faithful; retention = BBH+MMLU-Pro; F-delta) |
| `run_lib.py` | shared prompt templates (train==eval), logging, registries |
| `eval_cs.py` | commonsense gen eval; `run_eval(model,…)` batching reused in-process |
| `uio_inprocess.py` | UIO-era, but `eval_one_gpu` imports its `fdelta_inprocess` + `CS_DATASETS` → LIVE |
| `math_eval.py` | faithful GSM8K + Hendrycks MATH eval (imported by `eval_one_gpu` for `math_faithful`) |
| `bbh_metric_fix.py` | patches lm-eval BBH answer-only metric normalization (imported by `eval_one_gpu`) |
| `residual_save.py` | rank-2r W0-relative adapter conversion for residual-init methods (scaling-generalized) |
| `corda_init.py` | static CorDA-KPA init (imported by `train_cs --corda`; reused by `cordapp_init`) |
| `cordapp_init.py` | CorDA++ init — **now WIRED into `train_cs.py`** (`--cordapp`, 29 refs); no longer pending |
| `milora_init.py` | MiLoRA bottom-r SVD init |
| `data_aware_init.py` | PiSSA top-r SVD init (`pissa_BAR`) |
| `sclora_init.py` | SC-LoRA D+/D− covariance init |
| `lora_null_init.py` | LoRA-Null null-space init |
| `results/` | provenance: `campaign_summary.jsonl`, `train_registry.jsonl`, `eval_registry.jsonl`, per-run dirs |
| `logs/` | per-job logs (pools write `logs/<tag>_*.log`; dispatcher writes `logs/disp_*.log`) |

**ACTIVE-SUPPORT** (part of the current campaign; run out-of-band, not imported at runtime):
`make_frepro_jobs.py`, `build_lean.py`, the `jobs/frepro*` job lists, `validate_frepro_residual.py`,
`validate_residual_zero_step.py`, `validate_cordapp_cpu.py`, `metamath_prep_395k.py`, `math_test_prep.py`,
`base_retention_check.py`, the analysis scripts `geo_drift_phase{1,2}.py`, `forgetting_ce.py`,
`retfix_retention_gate.py`, `retfix_bbh_only_report.py`, and the paper generator `paper_figs_v2.py`
(+ the `paper/writing/` package). Reference PDFs live in `papers/`; vendored method repos + data in `repro/`.

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
| **CorDA++** | `--method lora --cordapp 1` | **WIRED into `train_cs.py`** (2026-07-06 restart; `cordapp_init.py`, CPU-validated 14/14); dynamic-covariance N set for the campaign (`handoff/17 §8`); α=r@2e-5 paper-faithful anchor queued to remove the α=2r confound |

Residual-init methods (MiLoRA, PiSSA, SC-LoRA, LoRA-Null, CorDA, CorDA++) **require** the rank-2r
W0-relative conversion at save (`residual_save.py`) or eval explodes (see gotchas).

---

## (e) Campaign status (parsed 2026-07-09) — **INERT as of 2026-07-17** (superseded by `handoff/41` + key_numbers §18)

- **Two-node 16×B200 fleet** (`handoff/28`): Node A (this host) owns all adapters + analysis; Node B trains
  fresh and syncs summary JSON. Live on A: 4 `gpu_pool` pools (`frepro4`/`frepro4b4`/`frepro4hs`/`frepro4inj`)
  + `auto_dispatch` on `jobs/master_dispatch.txt`.
- **Math (`frm_`) complete ≈ 46/46** (48 `frm_*` result dirs on disk; + method-row/β cells in flight).
- **CS (`frc_`) landing now** — the 65-cell reservoir is the paper's spine (0 done at campaign start).
- `results/campaign_summary.jsonl` ≈ **472 rows** spanning the whole project history (dedup = latest
  `evaluated_at` per `run_name`, see key_numbers §0); ~482 subdirs under `results/`.
- **Full current snapshot: `STATUS.md`** (running pools, progress, analysis-done/queued).

**[2026-07-17]** All of the above is history: the campaign is frozen (final dataset = 1,661 result dirs,
1,500 full evals, n=1035 usable; key_numbers §18), the fleet is evacuated (`handoff/41`), and `STATUS.md`
now lives at `archive/writing_2026-07-17/STATUS.md`. Remaining work is offline analysis + writing.

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
| 2026-07-06 | camp5 drained; `frepro3` pool launched on `jobs/frepro_lean.txt`. Retention decision → **BBH-only** (MMLU-Pro broken for math); PiSSA real-forgetting gate (`handoff/22`); repo-verification gates pass (`handoff/23`). |
| 2026-07-06→ | **Restart to frepro4** (patches in `restart_staging/`): CorDA++ wired into `train_cs.py`; multi-pool + `auto_dispatch` fleet on `frepro4_*` / `master_dispatch.txt`. |
| 2026-07-09 | **9-agent review fleet** (5 paper experts on the PDFs + 4 section validators): all ports faithful; **`fdelta` = CLoRA's F_Δ, not Frobenius** (metrology fix, key_numbers §0); CLoRA Table 4 = external law replication (r=−0.98); geometry-drift verdict (magnitude 1st / rank 2nd / **principal-direction axis REJECTED**, `handoff/27`); numbers switched to canonical key_numbers.md; framing → "flat field governed by ‖ΔW‖, LoRA+wd Pareto-competitive." **Two-node plan** (`handoff/28`). Repo-hygiene pass: superseded scripts/jobs → `archive/`. |

---

## (g) Gotchas (do not relearn the hard way)

- **ONE scheduler per GPU.** Two schedulers claiming the same GPU → 2 runs/GPU → OOM (~45h lost twice).
  The current fleet coexists safely because each `gpu_pool` owns fixed `--gpu_ids` and `auto_dispatch`
  **only picks up GPUs after their owning pool exits** (it reads live `gpu_pool` `--gpu_ids` from `ps`).
  **Never launch a second pool on an already-owned GPU**, and **never edit a live `.py` the pool executes
  fresh per job** (it would corrupt every not-yet-started cell). Develop code changes in a git
  worktree/branch and run only after the relevant pool drains.
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

**Current canonical order (2026-07-17):** `handoff/41_EVACUATION_2026-07-17.md` (state) →
`paper/writing/analysis_final/PAPER_BLUEPRINT.md` + docs 01–07 (story) →
`paper/writing/data/key_numbers.md` **§18–§19** (every quoted number). `README.md` is the front door.

*(Pre-freeze order, kept for provenance: this file → `handoff/20_FAITHFUL_REPRO_SPEC.md` →
`handoff/25`–`28` → key_numbers + `paper/writing/INTERESTING_INSIGHTS.md`.)*
`handoff/00`–`12` are HISTORICAL (UIO era); `handoff/17`/`19` describe the pre-pivot camp5 2×2 /
fairness campaign, superseded in priority by `handoff/20`; `handoff/34`–`40` were archived. Superseded
scripts/jobs live in `archive/` (see `CLEANUP_MANIFEST.md` + `archive/README.md`); `STATUS.md` and the
07-02 writing suite (01–08, CONCLUSIONS_AND_IDEAS, FINALIZATION_PLAN, claims-coverage audit,
registry-refresh, NEXT_EXPERIMENTS, the 07-16/17 assessments, pi_review/author-recommendation notes,
and `handoff/34`–`40`) moved to `archive/writing_2026-07-17/`.
