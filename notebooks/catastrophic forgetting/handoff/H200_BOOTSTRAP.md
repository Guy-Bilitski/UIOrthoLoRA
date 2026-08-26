# H200 bootstrap — Tier A campaign (written 2026-08-26, dev box)

You (Claude) are waking up in a fresh clone of this repo on the H200 machine.
Goal: run the PI-approved Tier A campaign — **Exp 1 intruder slice first**
(abstract-critical, needs maximum lead time), **Exp 2 Qwen rescale ladder
second** (Limitation 1's promise; the ARR reviewer's named best addition,
strengthening the interventional support behind Table 1 / `tab:grand`).
Authoritative spec: `handoff/TIER_A_SPEC_2026-08-23.md`. Submission target
~2026-10-02; the Exp 1 decision-gate readout must exist by ~**Sep 22**.

Everything below was prepared and verified on the dev box on 2026-08-26.
Working dir for all commands: `notebooks/catastrophic forgetting/`.

## What is already done (don't redo)

- `intruder_pass.py` written; `--selftest` PASSES (5/5 synthetic cases, all
  cos thresholds, both base references; warm-started Rayleigh-Ritz matches
  dense SVD to <1e-3). Re-run the selftest once on the host as a sanity gate.
- `gen_tierA_jobs.py` + default `jobs/tierA_exp1_slice.txt` (18 cells,
  coverage-first) and `jobs/tierA_exp2_anchors.txt` (4 cells). **The committed
  files have NO evacuation step and a bare `python`** — regenerate on the host
  (step 3 below) before launching.
- `evacuate_cell.sh` (rsync + checksum verify + `.evacuated` marker).
- Method configs were re-verified against surviving pool job lines + paper
  Appendix A. The spec's prose paragraph ("alpha=32, wd=0.1, MiLoRA r=16") was
  stale; **the generator encodes pool truth**:

  | arm | flags (both families unless noted) |
  |---|---|
  | LoRA+wd | `--method lora --lora_r 32 --lora_alpha 64 --weight_decay 0.3` |
  | MiLoRA | `--method lora --milora 1 --lora_r 32 --lora_alpha 32` |
  | SC-LoRA | `--sclora 1 --sclora_beta 0.5 --sclora_calib_size 256 --calib_source nq_open`; **alpha=64 on Llama-CS, alpha=32 on Qwen-CS** (real pool history) |

  Everything: cutoff 256, 3 epochs, batch 16 (train_cs defaults), **seed 43**
  (42 is the documented SC-LoRA magnitude outlier). SC-LoRA calibration is
  pooled nq_open, NOT eval-matched (pool comparability; settled separately).

## Step 1 — environment

1. venv: `python -m venv .venv && .venv/bin/pip install -r "notebooks/catastrophic forgetting/requirements-freeze.txt"`
   (repo root also has `init.sh` — check it first, it may do this).
2. HF auth: Llama-2 is gated — `huggingface-cli login` with Guy's token
   (ask him; the old campaign token was rotated per RELEASE_PLAN item 0).
   Check disk for the HF cache: Llama-2-7B + Qwen2.5-7B ≈ 30 GB total.
3. Decide/confirm with Guy (open items from the spec):
   - `OUT_ROOT` for adapters (old fleet used `/scratch/cf_models`; adapters
     are small, <5 GB total for the whole campaign).
   - `EVAC_DEST` for per-cell checkpoint evacuation — **spec rule 2, not
     optional**. Any location that survives the machine being reclaimed:
     a NAS path, another host, or worst case this repo's host via ssh. If
     Guy hasn't named one, ask before launch; do NOT launch without it.
4. GPU count: the spec assumed 1 card, strictly serial. If the H200 box has
   more than one GPU, cells are independent — `gpu_pool.py --gpus N` is safe
   and shortens the wall clock; keep each cell's internal chain intact.

## Step 2 — CPU prerequisites (start these first, they run while GPU works)

Base-SVD stores (needed only by `intruder_pass.py`, not by training/eval):

    setsid .venv/bin/python geo_drift_phase1.py > logs/geo1_llama.log 2>&1 &
    setsid .venv/bin/python geo_drift_phase1_qwen.py > logs/geo1_qwen.log 2>&1 &

They stream weights from the HF cache (triggering the downloads) and write
`results/geo_drift/base_svd/` and `base_svd_qwen/` (160 + 196 matrices,
CPU-only, hours). Then the gate:

    .venv/bin/python intruder_pass.py --selftest   # must print PASS

## Step 3 — regenerate job files with real paths, dedupe, smoke

    .venv/bin/python gen_tierA_jobs.py \
        --python-bin $PWD/../../.venv/bin/python \
        --out-root <OUT_ROOT> --evac-dest <EVAC_DEST>

(Generator comments out any cell whose `results/<run>/summary.json` already
exists.) Smoke = the first queue line (cell 2, Llama LoRA+wd 5e-4 — a real
needed cell): run it alone, watch the first ~30 min (loss decreasing, sane
it/s), let its chain finish through eval + CE + evacuation, and verify
`<EVAC_DEST>/<run>/` and the `.evacuated` marker exist. First two cells
calibrate per-cell time; >3.5 h/cell → keep all 18 but re-check the 7-day ask
with Guy.

## Step 4 — launch Exp 1

    mkdir -p logs
    setsid .venv/bin/python gpu_pool.py --gpus 1 --tag tierA1 \
        --jobs jobs/tierA_exp1_slice.txt > logs/tierA1_pool.log 2>&1 &

Queue order is coverage-first (cells 2, 11, 6, 15, 9, 18 — all six
model×method arms at knee-or-above — then the remaining 12), so after ~1 day
every arm has an informative point. Per-cell logs: `logs/tierA1_<i>.log`.

**Divergence rule (cell 12, Qwen LoRA+wd 1e-3):** if F_delta explodes
(pooled precedent: one seed hit F=14.3), retrain ONCE at 5e-4 under run name
`..._lr5e4f_s43`, flag it in the run log, never silently substitute.

## Step 5 — intruder pass as cells land (CPU, incremental)

After each cell's eval finishes (adapter still on disk at OUT_ROOT):

    .venv/bin/python intruder_pass.py --adapter <OUT_ROOT>/<run> \
        --base_model <meta-llama/Llama-2-7b-hf | Qwen/Qwen2.5-7B>

Writes `results/intruder/<run>.json` + a line in
`results/intruder/intruder_registry.jsonl`. ~10 min/adapter on decent CPUs.

Readouts, in order of arrival (spec R1–R3):
- **R1** (checkpoints only): do intruders appear; intruder count vs F_delta
  per arm (collinearity; the pool's spec_max~F_delta r=0.931 predicts yes).
- **R2** (needs evals): within-slice partial correlation of intruder count
  with retention **given log F_delta**; slice runs placed on the frozen
  family curves (Llama-CS, Qwen-CS).
- **R3**: do the three designs differ in intruder formation at matched
  magnitude (links `fig:geometry`'s detection result to the mechanism).

**Decision gate (paper-critical, get PI eyes on it):**
- Intruders ≈ restatement of magnitude → mechanism paragraph in §4.2 +
  appendix exhibit; related-work scoping upgraded; abstract untouched.
- Intruders add retention variance beyond F_delta, persistent across cos
  thresholds AND both models → the abstract's geometry clause is reframed by
  us, pre-submission. This is why Exp 1 runs first.

Report R1 the moment the first 6 cells have it; don't wait for 18.

## Step 6 — Exp 2 (after Exp 1 queue drains, or interleaved if >1 GPU)

    setsid .venv/bin/python gpu_pool.py --gpus 1 --tag tierA2 \
        --jobs jobs/tierA_exp2_anchors.txt > logs/tierA2_pool.log 2>&1 &

When the 4 anchors land:
1. **Ladder (CPU):** `rescale_adapters.py` is hardcoded to the Llama E1
   anchors — write a `rescale_adapters_qwen.py` variant from it (same B-scaling
   math, same control construction): per anchor ~4 F_delta targets spanning
   the frozen family's observed range (targets from the frozen qwsw/qwswm
   curves — `paper/writing/data/key_numbers.md` §18/§19 and
   `paper/writing/analysis_final/` carry the curve fits), → ~16 rescales.
2. **Controls (CPU):** matched-F_delta random-direction updates, 4/setting → 8.
3. **Evals (GPU):** rescales get adaptation+retention; controls retention
   only. `qwswm` retention = BBH only (pool convention; adapt_task gsm8k,
   ret_max_gen 256). Emit eval-only job lines like rescale_adapters.py does.
4. Evacuate everything (evals are the expensive part, keep the adapters too).
5. **Analysis:** existing E1 pipeline (on-curve residuals vs frozen family
   curve, within-set correlation, direction penalty), per setting. Paper
   landing: §4.2 second check becomes two-architecture; Appendix D gains a
   Qwen block; Limitation 1 rewritten from promise to result.

## Optional slot (only if GPU time remains — ask Guy first)

`jobs/qwen_ce_recovery.txt` (124 ready train+CE chains) fills the grand
table's (Table 1) missing Qwen CE `--` cells — MISSING_EXPERIMENTS item 4.
It is NOT part of the approved Tier A spec; treat as a bonus, never at the
expense of Exp 1/2 or the evacuation discipline.

## Standing rules and gotchas

- **Never modify the tested pipeline** (train_cs.py, eval_one_gpu.py,
  run_lib.py, the init modules, the PEFT fork at repo-root `peft/`). New
  analysis goes in NEW files.
- train_cs.py is idempotent: a complete adapter dir → skips training, so
  rerunning a failed chain line is safe.
- Every registry append (train_registry, eval_registry, forgetting,
  intruder_registry) happens under `results/` in the clone — rsync `results/`
  back to the dev box / push regularly; that's the data the paper reads.
- gpu_pool sets `OMP_NUM_THREADS=8` per job; geo/intruder CPU passes respect
  `GEO_THREADS` (default 16) — on a big box you can raise it.
- Do not reorder or thin the queue without telling Guy; coverage-first order
  is a PI decision.
- If anything here contradicts `handoff/TIER_A_SPEC_2026-08-23.md`, the spec
  wins — except the method-config table above, which supersedes the spec's
  stale prose paragraph (verified against pool job lines 2026-08-26).
