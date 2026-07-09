# Catastrophic forgetting in PEFT — the magnitude-law study

**Working dir:** `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/` (note the space — quote it).
**Repo root:** `/home/guy/UIOrthoLoRA`  ·  **Branch:** `ortho_new`  ·  **Refreshed:** 2026-07-09.

> **New here? Read `WORKDIR_ALIGNMENT.md` first** (the single onboarding doc: goal, exact settings, live
> file map, adapter roster, gotchas), then `handoff/20_FAITHFUL_REPRO_SPEC.md` (the live plan) and
> `paper/writing/data/key_numbers.md` (the single source of truth for every quoted number).
> This README is the directory front door; the older UIOrthoLoRA "go/no-go" README it replaces is in
> git history.

---

## What this is

We study **what governs catastrophic forgetting when a 7B LLM is PEFT-fine-tuned.** Central claim — **THE
MAGNITUDE LAW:** retention is governed by the size of the weight update **F_Δ** (CLoRA's effective
update-magnitude metric, Eq 3), **not** by the adapter method. LoRA, CLoRA, MiLoRA, PiSSA, DoRA, SC-LoRA,
LoRA-Null and the data-aware inits all fall on one retention-vs-F_Δ curve, so the simplest magnitude control
(**plain LoRA + weight decay**) matches or beats elaborate structured/data-aware adapters at equal F_Δ.
Reported single-LR "wins" for fancy adapters are largely LR/recipe artifacts.

The centerpiece is a **faithful CLoRA-recipe reproduction + LR×wd sweep** on LLaMA-2-7B (Commonsense-170K,
prefix `frc_`; MetaMathQA-395K, prefix `frm_`), replicated on **Qwen2.5-7B**, with F_Δ and ‖ΔW‖ measured on
every run — the column the source papers omit. Every adapter goes through **one shared pipeline** so the
comparison is fair.

Key results (see `paper/writing/data/key_numbers.md` and `paper/writing/INTERESTING_INSIGHTS.md`):
magnitude law r=−0.86 pooled / −0.92 on-curve (CS, n=49), replicated on Qwen; LR is a weaker proxy
(R² 0.32 vs 0.74); CLoRA's own Table 4 is the same line (r=−0.98, slope −14.7 vs our −14.8); geometry is a
**fingerprint/measurement tool**, magnitude is 1st-order and rank a modest 2nd-order lever (the
principal-direction "2nd-order axis" was tested and rejected). LoRA+wd is on the frontier at zero extra cost.

---

## Pipeline & how to reproduce

```
make_frepro_jobs.py --table {math,cs}      -> jobs/frepro_{math,cs}.txt   (per-arm train+eval cells)
build_lean.py                              -> jobs/frepro4_lean.txt        (merges; done cells -> eval-only)
        |
   two schedulers keep the fleet saturated (single-scheduler-per-GPU discipline):
   gpu_pool.py    --gpu_ids <ids> --tag <t> --jobs jobs/<file>.txt   (fixed GPUs, 1 job/GPU)
   auto_dispatch.py --jobs jobs/master_dispatch.txt --gpus 0-7       (absorbs GPUs as pools drain)
        |
   train_cs.py   (shared trainer; every method = --method lora|clora + an init flag:
                  --weight_decay / --milora / --pissa / --sclora --sclora_beta / --lora_null /
                  --corda / --cordapp; residual-init methods use residual_save.py)
   eval_one_gpu.py (in-process: CS suite OR --adapt_task math_faithful; BBH+MMLU-Pro retention; F_Δ)
        v
   results/<run>/summary.json  +  results/campaign_summary.jsonl (one line/run) + train/eval registries
```

- **Venv:** `/home/guy/UIOrthoLoRA/.venv/bin/python` (never bare `python` in job lines → rc=127).
- **Checkpoints:** `/scratch/cf_models/<run>`. **Data:** under `repro/LLM-Adapters/` (commonsense_170k,
  metamathqa_395k, GSM8K/MATH test) — built by `metamath_prep_395k.py` / `math_test_prep.py`.
- **Figures/tables (canonical):** `paper_figs_v2.py` and `paper/writing/make_figs_split_lora_null.py`
  (LoRA-Null split convention). Stats: `paper/writing/analysis_a1_a4.py`.
- **Analysis:** `geo_drift_phase{1,2}.py` (geometry fingerprint), `forgetting_ce.py` (CE-to-base),
  `retfix_retention_gate.py` (retention gates).
- **Reference method repos + PDFs:** `repro/` (CLoRA, CorDA, LLM-Adapters, LoRA-Null, MiLoRA, SC-LoRA) and
  `papers/` (the 5 source PDFs). **Do not modify `results/` or `repro/`.**

**GOTCHAS** (see WORKDIR_ALIGNMENT §g): one scheduler per GPU (two pools → OOM); never edit a `.py` a live
pool runs fresh per job; residual rank-2r conversion is mandatory for residual-init methods; math retention
needs `--ret_max_gen 256` (models don't emit EOS); LR ≥ 2e-3 diverges (NaN → eval crash).

---

## Current status (2026-07-09) — a 2-node campaign

Deadline ~Sun; ~883 GPU-h of demand met by a **two-node fleet** (Node A = this 8×B200 host owns all
adapters + analysis; Node B = a second 8×B200 trains fresh and syncs summary JSON back). Live on A right
now: 4 `gpu_pool` pools (tags `frepro4`, `frepro4b4`, `frepro4hs`, `frepro4inj`) + `auto_dispatch` on
`jobs/master_dispatch.txt`.

- **Faithful MATH (`frm_`): ~46/46 done** (+ method/β cells in flight).
- **Faithful CS (`frc_`): landing now** — the 65-cell reservoir is the paper's spine (0 done at start).
- **Running/queued:** CLoRA k-grid + `frc_lora_l2`, SC-LoRA/CorDA++ boundary cells (inject/b4), 3-seed
  headlines, Qwen block (Node B, `lorawd` math LR-sweep first), CE-to-base full batch (A).
- **Analysis done & validated:** magnitude law (2 models, ceiling-robust stats), geometry-drift verdict,
  efficiency/memory, CE-to-base, CLoRA-Table-4 external replication, fdelta→F_Δ metrology fix.
- **Paper actions pending (no GPU):** fdelta→F_Δ relabel in `paper.tex`/`analyze_matrix.py`/figures;
  saturating-fit law figure; operating-point + efficiency tables; cross-literature overlay.

See `STATUS.md` for the campaign snapshot and `handoff/` for the full decision log.

---

## Handoff-doc index

`handoff/README.md` is the ordered index; **start with `WORKDIR_ALIGNMENT.md`**. Current-era docs (17–28):

| # | Doc | What |
|---|---|---|
| 17 | `17_CORDAPP_IMPL_PLAN` | CorDA++ (dynamic covariance + rank) impl plan → `cordapp_init.py` (now wired) |
| 18 | `18_ADAPTER_AUDIT` | per-adapter faithfulness verdicts; the calibration↔eval confound quarantining CorDA/SC-LoRA |
| 19 | `19_FAIRNESS_STUDY_PLAN` (+`_DRAFT`) | the eval-matched-calibration fairness study (valid, deprioritized) |
| 20 | `20_FAITHFUL_REPRO_SPEC` | **the live plan** — faithful CLoRA CS+math reproduction + LR×wd sweep |
| 21 | `21_CONSORTIUM_SYNTHESIS` | 9-agent verdict: law flat in the competitor blob → BBH-only, α=2r ruling, Tier-A cells |
| 22 | `22_RETENTION_FIX` | BBH-only retention decision + PiSSA real-forgetting gate |
| 23 | `23_REPO_VERIFICATION` | correctness-gate pass (ports faithful, save/reload lossless, base BBH) |
| 24 | `24_PI_STATUS_REPORT` (07-07) | PI status snapshot |
| 25 | `25_SUPERVISION_REPORT` (07-09) | **current supervisor report** — thesis, fair sweep, canonical numbers, boundaries |
| 26 | `26_RESEARCH_PLAN` (07-09) | post-fleet prioritized plan: injected GPU cells, CPU/eval-only work, paper actions |
| 27 | `27_GEOMETRY_DRIFT` (07-09) | geometry-drift verdict (magnitude 1st / rank 2nd / principal-direction rejected; fingerprint tool) |
| 28 | `28_TWO_NODE_PLAN` (07-09) | 16-GPU two-node plan (saves Qwen + 3-seed + full CE) |

Docs `00–16` are historical (UIOrthoLoRA-instrument era + the 2×2 campaign + the 07-01 eval fixes).
