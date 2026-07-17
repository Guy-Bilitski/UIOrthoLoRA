# Catastrophic forgetting in PEFT — the magnitude-relation study ("Magnitude, Not Geometry")

**Working dir:** `/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/` (note the space — quote it).
**Repo root:** `/home/guy/UIOrthoLoRA`  ·  **Branch:** `ortho_new`  ·  **Refreshed:** 2026-07-17.

> **New here? Read `WORKDIR_ALIGNMENT.md` first** (the onboarding doc; its live-fleet sections are now
> INERT — see below), then `paper/writing/analysis_final/PAPER_BLUEPRINT.md` (the story layer, with
> docs 01–07) and `paper/writing/data/key_numbers.md` **§18 FINAL FREEZE (2026-07-17) + §19 POST-FREEZE
> ADDENDUM** (the single source of truth for every quoted number). Current project state:
> `handoff/41_EVACUATION_2026-07-17.md` (fleet evacuated; offline-analysis phase).
> This README is the directory front door; the older UIOrthoLoRA "go/no-go" README it replaces is in
> git history.

---

## What this is

We study **what governs catastrophic forgetting when a 7B LLM is PEFT-fine-tuned.** Central claim — the
**magnitude relation** (flat-then-falling with a knee; title direction: *"Magnitude, Not Geometry"*):
retention is governed by the size of the weight update **F_Δ** (CLoRA's effective
update-magnitude metric, Eq 3), **not** by the adapter method. LoRA, CLoRA, MiLoRA, PiSSA, DoRA, SC-LoRA,
LoRA-Null and the data-aware inits all fall on one retention-vs-F_Δ curve, so the simplest magnitude control
(**plain LoRA + weight decay**) matches or beats elaborate structured/data-aware adapters at equal F_Δ.
Reported single-LR "wins" for fancy adapters are largely LR/recipe artifacts.

The centerpiece is a **faithful CLoRA-recipe reproduction + LR×wd sweep** on LLaMA-2-7B (Commonsense-170K,
prefix `frc_`; MetaMathQA-395K, prefix `frm_`), replicated on **Qwen2.5-7B**, with F_Δ and ‖ΔW‖ measured on
every run — the column the source papers omit. Every adapter goes through **one shared pipeline** so the
comparison is fair.

Key results (canonical: `paper/writing/data/key_numbers.md` §18–19; story: `paper/writing/analysis_final/`):
magnitude relation pooled **r=−0.847 (rank −0.923), n=1035**, across **6 model×task families, 8 methods,
3–5 seeds**; nested ΔR² ladder (§19.1): family FE R²=0.390 → **+0.395 magnitude** (F≈1890) → +0.017 geometry
(F=30) → +0.006 method; LR is a weaker proxy (§18.5); CLoRA's own Table 4 is the same line (r=−0.98, slope
−14.7 vs our −14.8); geometry is a **fingerprint/measurement tool**, magnitude is 1st-order and rank a modest
2nd-order lever (the principal-direction "2nd-order axis" was tested and rejected); SC-LoRA's old −4.15pp
deviation is RESOLVED by E4 eval-matched calibration (+0.92pp above the relation — a calibration artifact,
§18.3). LoRA+wd is on the frontier at zero extra cost.

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

> **Post-freeze note (2026-07-17):** the source of record is `results/*/summary.json`;
> `results/campaign_summary.jsonl` and `results_book/` are **STALE** — do not source numbers from them.

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

## Current status (2026-07-17) — campaign FROZEN, fleet EVACUATED, offline-analysis phase

The data-collection campaign is **over**. The fleet was evacuated on 2026-07-17
(`handoff/41_EVACUATION_2026-07-17.md` is the live state doc) — **no live pools**; the scheduler/fleet
instructions above are kept for provenance and reproduction, not for a running system.

- **Canonical numbers:** `paper/writing/data/key_numbers.md` **§18 FINAL FREEZE + §19 POST-FREEZE
  ADDENDUM** (pooled r=−0.847, n=1035, 6 families, 8 methods, 3–5 seeds).
- **Story layer:** `paper/writing/analysis_final/` (docs 01–07 + `PAPER_BLUEPRINT.md`).
- **Source of record:** `results/*/summary.json` + merged aggregates (`results/forgetting_merged.jsonl`,
  `results/geo_drift/adapter_metrics_merged.jsonl`).
- **DeepSeek 284B:** 21 adapters evacuated + SHA256-verified (`results/ds_adapters_evac/`); 20/21 MedMCQA
  adapt + 21/21 factor-only geometry landed; retention/CE lost (GPU re-eval only). The geometry
  method-fingerprint recurs at 284B (`analysis_final/07`).
- **Qwen CE:** 123 primary-seed cells permanently unfillable (adapters destroyed);
  `jobs/ce_backfill_qwen.txt` kept as disclosure.
- `STATUS.md` and the 07-02 writing suite were archived to `archive/writing_2026-07-17/`.

What remains is pure offline analysis + writing. See `handoff/` for the full decision log.

---

## Handoff-doc index

`handoff/README.md` is the ordered index (now through **41**; docs 34–40 live at
`archive/writing_2026-07-17/handoff/`). **`41_EVACUATION_2026-07-17.md` is the current state doc.**
Campaign-era docs (17–28), kept for provenance:

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
