# OPERATING STATE — read this first (handoff for any agent / future me)

> ⚡ **LIVE STATE (2026-06-29): read `13_STATE_2026-06-29.md` FIRST** (also mirrored at top-level
> `STATUS.md`). It has the current claim (THE magnitude LAW: retention ~ ‖ΔW‖_F, not method), the
> complete Llama-2 CS LR-sweep findings (incl. the CorDA/SC-LoRA off-curve refinement), the canonical
> figure pipeline (`paper_figs_v2.py`; `paper_assets.py` deprecated), how to resume/salvage, the
> single-seed-first 2×2 campaign, and the hard-won gotchas. Everything below (and 12 and earlier) is
> historical. (Prior 2026-06-17 headline `12_STATE_DUMP.md` is superseded.)

**Project (EVOLVED 2026-06-15):** A **controlled study of WHAT GOVERNS catastrophic forgetting (CF)
in PEFT**, using corrected **UIOrthoLoRA / UILinLoRA as controllable INSTRUMENTS**. The original goal
("is UIOrthoLoRA an A*-worthy CLoRA-beater?") is **DEAD** — even corrected (drop_major) it only TIES
CLoRA and loses the high-CS corner. Setting unchanged: CLoRA's commonsense one-stage setup, LLaMA-2-7B;
in-domain CS-avg vs out-domain retention=mean(answer-only-BBH, MMLU-Pro). **The live science is in
`06_INSIGHTS.md` (findings), `07_RELATED_WORK.md` (lit positioning), `08_FORWARD_PLAN.md` (the plan) —
READ THOSE.** This file = environment + durable findings + what's running.

**Current thesis (2026-06-15) — 3 threads (full detail in 08):**
1. **THE MAGNITUDE (FROBENIUS) LAW.** retention ≈ f(‖ΔW‖_F). Within-architecture corr **−0.96 to −0.98**
   across LoRA / CLoRA / UIOrthoLoRA (points interleave on ONE curve; pooled −0.79 only due to a
   CLoRA-full vs UIO-fast SCALE offset → fast CLoRA re-eval unifies it). Weight-basis DIRECTION is
   irrelevant (μ_E r=−0.09). CLoRA's random-subspace penalty "works" only by shrinking ‖ΔW‖_F (it is
   spectrally neutral, out_top≈0.5). [F-norm > spectral −0.86 > σ-weighted/F∆ −0.46 > weight-dir −0.09.]
2. **THE RANK QUESTION (open / kingmaker).** "Higher rank surprisingly mitigates CF" — is it an
   independent structural effect, or just rank↑⇒‖ΔW‖_F↓? E2b = match task-CS across ranks, compare
   retention. HYP-A folds into the Frobenius law; HYP-B = genuine spectral-tail structural property.
3. **THE DATA-BASIS FRONTIER.** Is data/activation-covariance direction (CorDA/SC-LoRA's claim) a
   SECOND-ORDER correction to the first-order Frobenius tax? (data-basis forensic + CorDA-basis
   instrument E3c.)

**Honesty guardrails (do not relapse):** the CF-mechanism space is CROWDED. "magnitude matters" ≈
CLoRA's F∆; "data basis matters" = CorDA/SC-LoRA; "geometry/direction" = OPLoRA/Subspace-Geometry.
THREE reframes already preempted (see 06 ★). Our defensible contribution = the **controlled/causal
dissociation** + the **cross-architecture Frobenius law** + (if it lands) the **rank result** — NOT
"we discovered magnitude/basis matters." Verify [VERIFY] citation IDs in 07 before any manuscript.

---

## Environment (host = B200 box, user `ubuntu`)
- **venv:** `/home/guy/UIOrthoLoRA/.venv` (python3.12). Run everything with `/home/guy/UIOrthoLoRA/.venv/bin/python`.
  peft 0.19.1 = **editable install of THIS fork** (do NOT pip-install peft). torch 2.12+cu130, transformers 5.10, trl 1.5.1, lm-eval 0.4.12.
- **8× NVIDIA B200** (183 GB each). One UIOrthoLoRA run uses ~112 GB (can't pack 2/GPU). LoRA/CLoRA ~14-50 GB.
- **HF:** token works; weights cached. Use `HF_HUB_DISABLE_XET=1`. Firewall: `*.hf.co`/`*.huggingface.co` open; github HTTPS works.
- **Scratch:** checkpoints go to **`/scratch/cf_models`** (526 GB root vol) — NOT `/home` (44 GB). UIOrthoLoRA in-process runs save nothing.
- **All work lives in** `notebooks/catastrophic forgetting/` (user requirement).
- **Thread caps:** always set `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` (gpu_pool does this) — else 8 concurrent jobs thrash the 128-core CPU (load avg 250+).

## Scripts (all in workdir)
| script | role |
|---|---|
| `train_cs.py` | shared trainer, `--method lora|clora|uiortholora`. CLoRA penalty (`CLoRARegularizer`, GPU-QR P-matrices) lives here. |
| `uio_inprocess.py` | **UIOrthoLoRA: train + eval in ONE process, NO save/reload** (required — see bug below). Logs CS + retention + F∆ + **leakage**. |
| `eval_one_gpu.py` | eval a LoRA/CLoRA adapter fully on 1 GPU in-process (CS + retention + F∆). |
| `eval_cs.py` | commonsense gen eval; `run_eval(model,...)` reusable in-process. |
| `run_retention.py` / `eval_retention.py` | sharded BBH+MMLU-Pro via lm-eval (for adapter-reload methods / base). |
| `run_cs_eval.py` | 8-dataset CS across GPUs. |
| `fdelta.py` | F∆ / ‖ΔW‖ from a reloaded adapter. |
| `leakage.py` | **orthogonality-leakage thermometers** (μ_E, ν_D, Leak11, OffTailF, RelPertF, DriftU/V). Validated. |
| `gpu_pool.py` | scheduler: run a job-list across GPUs (`--gpu_ids`, `--tag`, `--jobs`), 1 job/GPU, per-job logs `logs/<tag>_<i>.log`. |
| `make_report.py` | frontier table (now w/ leakage cols) + repro-check from `results/campaign_summary.jsonl`. |
| `run_pipeline.sh` | auto-pipeline: waits for current wave → launches next. **Launch detached:** `setsid bash run_pipeline.sh >logs/pipeline.log 2>&1 </dev/null &` |
| `jobs/*.txt` | exact gpu_pool job-lists for each wave. |

Results: `results/<run>/summary.json` + `results/campaign_summary.jsonl` (one line/run). Logs: `logs/`.

---

## CRITICAL findings / decisions (do not relearn the hard way)

1. **UIOrthoLoRA checkpoints CANNOT round-trip through PEFT.** Two bugs: (a) SVD basis U/S/Vᵀ were
   dropped (no `uiortholora_` prefix) → **FIXED** in `src/peft/tuners/uiortholora/layer.py` (now frozen
   prefixed Parameters). (b) Orthogonal **rotators still don't reload** (PEFT key-mangling doubles
   `default` in the parametrization keys) → unfixable cleanly. **⇒ Always eval UIOrthoLoRA IN-PROCESS**
   (`uio_inprocess.py`). lm-eval `HFLM(pretrained=<in-mem peft model>, tokenizer=tok)` works — that's the trick.

2. **LR mismatch (the big one).** UIOrthoLoRA's sigma/D/E init at 0.1 → adapter output ~0.01 → grad_norm
   ~10× smaller than LoRA → at LoRA's LR 3e-4 it **under-adapts** (CS 47-62). **Use LR = 1e-2** (loss 0.65 < LoRA 0.76).
   This is itself a publishable finding (per-method LR; spec's RQ1). NEVER compare UIOrthoLoRA at 3e-4.

3. **Eval throughput.** Full MMLU-Pro CoT (12k × 2048-gen) on 1 GPU ≈ 4-16h (UIOrthoLoRA's slow forward).
   ⇒ fast-retention knobs in uio_inprocess/eval_one_gpu: `--ret_limit 64 --ret_max_gen 512` (subsample +
   cap gen). **Calibrate** vs full: LoRA full ret = 21.66, fast ret = 22.52 ⇒ **fast ≈ full + ~0.9** (small, OK).
   Must re-eval CLoRA/LoRA on the SAME fast config to compare to UIOrthoLoRA (= the `calib_*` runs).

4. **Major-term discrepancy (suspected impl bug).** Our ΔW = `E·[U₁·diag(1)·V₁ᵀ + tail]·D` — the
   `U₁·diag(1)·V₁ᵀ` (leading-band, unit singular values) term is **NOT** in the paper's `ΔW=E·Ū_r·Σ'_r·V̄_rᵀ·D`.
   It perturbs the *preserved* subspace (and at use_de=0 it's un-scaled). Leakage thermometers are tail-only
   so they DON'T see it. If clean (use_de=0) configs retain poorly despite ~0 thermometers, this is why →
   test a corrected layer (base = residual, or drop the term). User aware; decided "measurement for now".

5. **use_de gates break orthogonality.** With use_de=1, `diag(E)·M·diag(D)` leaves the tail subspace →
   high leakage / high F∆ (UILinLoRA F∆ 0.75 despite ‖ΔW‖ 9). `use_de=0` is the clean-orthogonality arm.

6. **BBH config:** answer-only `bbh_fewshot` (3-shot) reproduces CLoRA base (33.1 vs 34.91). NOT CoT
   (`bbh`=cot gives 39.5). MMLU-Pro = standard 5-shot CoT `mmlu_pro` (18.96 vs 18.56). Retention = mean of the two.

7. **Param-matching:** rotations cost `2·k_vec²`/module → to match LoRA's 56.1M, **k_vec≈410** is forced;
   k_val ∈ {410,1024,2048,3072,4096} all within ±1%. (`k_val`=# trainable singular values from the bottom;
   `k_vec`=# bottom singular vectors rotated. Tail=adapted bottom; leading=frozen top.)

8. **CLoRA over-constrains at high k** under literal λ=1 sum-reg: k2048 CS collapses to 65 (siqa/hellaswag
   format) vs paper 83.7; k128/k256/k1024 fine (~79). The frontier still has CLoRA's retention trend; k2048
   is the over-constrained corner. Not "fixed" — noted.

9. **CLoRA P-matrices: build via GPU QR** + thread caps (CPU QR × many jobs → load 267, stalls).

10. **gpu_pool job files MUST use the FULL venv python path, NOT bare `python`.** gpu_pool runs each
   job line via `subprocess.call(cmd, shell=True, env=os.environ, executable=/bin/bash)`. It inherits
   the launching shell's PATH. If launched from a shell without the venv on PATH (e.g. a detached
   `setsid bash run_pipeline.sh`, or any non-activated shell), bare `python` → `command not found` →
   **rc=127, all jobs die in 0s, GPUs go silently idle.** This bit Wave 2 (9h idle). Always write
   `/home/guy/UIOrthoLoRA/.venv/bin/python ...` in `jobs/*.txt`. (Wave-1 only worked because its pool
   was launched from a venv-active shell.) Sanity after launching any pool: `grep rc=127 logs/<tag>_pool.log`.

---

## What's running RIGHT NOW (as of 2026-06-15 ~09:45)
- **Phase-2 pool** `logs/phase2_pool.log` (GPUs 1-4,6,7; auto-launched by a watcher when the D1 grid freed):
  24 jobs = (a) **k_val×k_vec grid** `jobs/kval_kvec_grid.txt` (T2 adaptation/Pareto + D3 contrast),
  (b) **λ_E/λ_D sweep** `jobs/d1_lambda.txt` (T1 controlled DIRECTION intervention at ~fixed structure),
  (c) **LoRA+weight_decay** `jobs/mag_control_lora.txt` (T1 D2-killer: does a subspace-free magnitude
  knob reproduce CLoRA's curve?), (d) **CLoRA fast re-eval** `jobs/reeval_fast_baselines.txt`
  (**SCALE UNIFICATION** for the cross-arch Frobenius curve). ~50% through training as of 09:45;
  first results this afternoon. ⚠️ `grid_k1024_v1024` (full rotation, 1024×1024 orthogonal mat) is
  SLOW (~16h) — expected, deprioritize if it blocks.
- **`grid_k410_v410_dE1_lr1e2`** on GPU5: corrected full-rotation Pareto-push (the corrected twin of the
  legacy `uioT_k410` CS72.7@ret25 standout) — in eval phase.
- **data-basis forensic** `jobs/databasis.txt` (T3): queued via watcher (fires when grid_k410 frees GPU5).
  **rank sweep** `jobs/rank_sweep_lora.txt` (T2/E2a, LoRA r∈{4..256}): built, queue when GPUs free.
- **TWO readouts that convert preliminary→discovery-grade (bring to user/Gemini when they land):**
  (1) **scale-unified cross-architecture Frobenius curve** (needs CLoRA fast re-eval); (2) **E2b
  matched-CS rank result** (needs rank sweep + per-rank LR tuning to match CS, then compare retention).
- Analysis: `analyze_magnitude_law.py` (predictor bake-off, join FIXED via _norm), `analyze_d1_d2.py`
  (D1 verdict + D2 overlay), `forensics.py` (weight/spectral), `forensics_databasis.py` (data/CorDA basis).

## HISTORY (chronological, for provenance — superseded by the thesis above)

### Prior state (as of 2026-06-13 ~21:25)
- **Wave 4 DONE (use_de=0 low-LR sweep) — VERDICT HARDENED: clean arm has NO win.** Best clean point `uioW4_k2048_v410_dE0_lr1e3` = CS **57.5**/ret **25.7** (dw_max 3.3) — still dominated on BOTH axes by CLoRA-k2048 (CS 65.4/ret 26.76). Mechanism confirmed across the grid: use_de=0 has no D/E magnitude brake, so anything > LR 1e-3 explodes ΔW and **collapses CS** (lr3e3→CS 32, lr5e3→CS 28; near-random siqa/openbookqa) while only LR 1e-3 gives a sane-but-under-adapted point. Low LR recovers retention purely by bounding magnitude — does NOT recover CS. **Only 1 trailing job left:** `uioW4_k512_v64_dE0_lr1e3` on GPU7 (mid-train, will add another dominated point). 7 GPUs idle; no Wave-5 queued (by design).
- **B1 LEAKAGE MAP DONE (the measurement story) → `handoff/04_LEAKAGE_MAP.md` + `make_leakage_map.py`.** 16 runs across both arms. **Headline: retention tracks a ΔW *magnitude* budget (dw_max), NOT the *directional* leakage (μ_E/ν_D) the thermometers measure.** Clean arm: μ_E≈ν_D≈0.003 for every config, ret swings 4→26, corr(ret,dw_max)=**−0.86**. Leaky arm: μ_E 1.3–1.8 but D/E brake holds dw_max 7–9.5 → ret 22–25. (Caveats baked into the artifact: clean-arm corr(ret,μ_E)=+0.93 is SPURIOUS — μ_E flat at ~0.003; all-runs +0.44 is a use_de confound.) This is the explainability/measurement-tool fallback from `uio-publishability-plan`.
- **FORK RESOLVED (user 2026-06-13): A5 + seeds in parallel.** Wave 5 LAUNCHED — pool `uio_w5` on GPUs 0–6 (`jobs/uio_wave5_a5_seeds.txt`, detached setsid, full venv paths, rc=127-clean, all 7 GPUs 99%). Logs `logs/uio_w5_*.log`, pool `logs/uio_w5_pool.log`. GPU7 still finishing the trailing Wave-4 `k512_lr1e3` on the old w4b pool.
  - **A5 = corrected-major-term layer (IMPLEMENTED + VALIDATED).** New `drop_major` flag (config.py / model.py / layer.py / `uio_inprocess.py --drop_major 1`): sets the frozen major-band singular values to 0 so the adapter delta is confined to the adapted tail (preserved subspace = true identity), fixing finding #4. Validated by `test_a5_drop_major.py` (PASS: major-subspace energy ~1e-8; legacy−A5 == exactly the `E·U1·I·Vt1·D` term to ~1e-7). Smoking gun: in the clean arm the dropped major term has Frobenius ~17× the entire tail at init (Llama 4096-dim: ~45) — it WAS the dominant clean-arm forgetting driver. **6 A5 runs** (5× use_de=0 + 1× use_de=1, drop_major=1) mirror existing legacy twins for a direct retention-delta read. Hypothesis: A5 lifts clean-arm retention toward base (~26) at the CS the clean arm reaches (~55–60). Caveat: A5 does NOT add a tail-magnitude brake, so it likely won't raise the CS ceiling — best case is a "lower-CS / much-higher-retention" corner.
  - **A3 seeds (6 runs, drop_major=0 legacy):** seeds 43+44 on the 3 measurement-paper anchor points — `k1024_v128_dE0_lr1e2` (clean), `k1024_v128_dE1_lr1e2` (leaky), `k2048_v410_dE0_lr1e3` (best-clean-ret). Gives n=3 error bars on the use_de on/off contrast (REQUIRED for credibility).
  - **ETAs:** k512 ~5h, k1024 ~6–8h, k2048 ~11h; 5 jobs queued behind the first 7 → full batch ~16–22h. **Pool is detached and runs on real wall-clock through session dormancy** (per MONITORING REALITY below); review on next reactivation. To check: `grep -E "HEADLINE|rc=" logs/uio_w5_pool.log` + `python make_report.py` + `python make_leakage_map.py`.

### Earlier state (pre-fork, as of 2026-06-12 ~11:05)
- **Wave 3 DONE (adaptation ceiling, use_de=1, CS-only) — CEILING ~74, ROBUST.** LR{2e-2,3e-2,5e-2}×init{0.1,0.5,1.0}: CS clusters 70-74 (best lr5e2→73.7 at all inits; lr3e2+init1.0 destabilized→53). Pushing LR/init only inflates F∆ (1.05-1.07), not CS. ⇒ **UIO under-adapts ~5-6 pts vs LoRA/CLoRA (74 vs 79-80), NOT an LR/init artifact** (Stage-1 adaptation question ANSWERED: real cap).
- **Wave 2 DONE (use_de on/off + leakage) — KEY FINDING: retention ∝ ΔW magnitude (F∆/dw_sv_max), NOT directional leakage (μ_E/ν_D).** use_de=1 (gated): ret 22.6-25.1, μ_E/ν_D 1.3-1.8 (leaky), dw_max 7-9.5 (D/E brake magnitude). use_de=0 (clean): μ_E≈ν_D≈0.003 (pristine direction) BUT ret 3.9-22.2, F∆ 0.93-1.11, dw_max 15-66 (no brake→magnitude explodes→forgets). Monotone: F∆ 0.73→ret25, 1.11→ret3.9. ⇒ thermometers measure direction; preservation needs a magnitude budget. **use_de is measurement, not a win lever (confirmed).**
- **Wave 4 RUNNING (use_de=0 low-LR sweep, GPUs 0-7, ~4h/6.8h as of 14:15 Jun13):** k{256,512,1024,2048} × LR{1e-3,3e-3,5e-3}, FULL ret+leakage. Tests if low LR bounds ΔW magnitude → restores clean-arm retention. Results ~17:00-18:00. Detached pools: `uio_w4a` (gpu0,1,3,4), `uio_w4b` (gpu5,6,7 + k512_lr1e3 queued), k256_lr3e3 (gpu2). Logs `logs/uio_w4*_pool.log`, `logs/uioW4_*.log`. **NO Wave-5 chained — next step is the keep-tuning-vs-measurement-paper fork (user decision).**
- **Wave 1 DONE — VERDICT: NO (dominated).** All 8 uioT points finalized (`results/uioT_*`). On the FAST scale every UIO point sits strictly INSIDE CLoRA's frontier; all have high F∆ 0.73–0.93 (use_de=1 leakage). Best = uioT_k410 CS 72.7/ret 24.98 — dominated on BOTH axes by CLoRA-k1024 (CS 79.9/ret 25.57). No win condition met. See 01_RESULTS for the table. → the case now rests entirely on **Wave 2 (clean use_de=0) + thermometers**, and possibly A5 (corrected major term).
- **Calibration DONE (FAST anchors, validated):** `calib_clora_k1024_fastret` CS 79.86/ret **25.57** (FULL 24.8, +0.77); `calib_clora_k2048_fastret` CS 65.43/ret **26.76** (FULL 25.7, +1.06). Confirms FAST ≈ FULL + ~0.8–1.1. **The bar on the FAST (UIO-comparable) scale: ret 25.57 @ CS 80, or ret 26.76 @ CS 65.** LoRA FAST: CS 78.6/ret 22.5.
- **GPUs 0-7: Wave 2 RUNNING (relaunched 2026-06-12 ~11:05).** `jobs/uio_wave2_plus_calib.txt` = 8 high-retention configs (4× use_de=0 clean + 4× use_de=1, small k_vec). 1 job/GPU, all 8 GPUs 99% util, logs leakage μ_E/ν_D. Logs `logs/uio_w2_{0..7}.log`, pool `logs/uio_w2_pool.log`. ETA ~6-7h train + eval (~18:00-20:00). job0 confirmed `D and E DISABLED` (use_de=0 OK).
- **⚠️ rc=127 INCIDENT (fixed):** first Wave-2 auto-launch (02:02) FAILED — all 8 jobs `python: command not found`. gpu_pool runs job strings via `subprocess.call(shell=True, env=inherited)`; job files used bare `python`, but the relaunched pipeline's shell had NO venv on PATH → rc=127, GPUs idle ~9h (02:02→11:05). **FIX: job files now use the FULL venv python path** `/home/guy/.../.venv/bin/python` (PATH-independent). See finding #10. Old log: `logs/uio_w2_pool_FAILED_rc127.log`.
- **Wave 3 auto-pipeline:** `run_pipeline_w3.sh` (setsid detached, PID ~2495126) waits for Wave-2 "ALL DONE" → launches **Wave 3 = `jobs/uio_wave3_adapt.txt`** (the LR×init adaptation-ceiling sweep, k_val2048/k_vec410, use_de1, `--skip_retention --no_leakage`, full venv paths). Logs `logs/uio_w3_pool.log`, pipeline `logs/pipeline_w3.log`.
- **⚠️ MONITORING REALITY (proven 2026-06-12):** in-session background tasks (`sleep`-based heartbeats, waiters) and ScheduleWakeup are **SUSPENDED while the session is idle** — a `sleep 3300` accrued only 24min in 3h20min wall. **There is NO reliable wall-clock self-wake during dormancy.** Detached `setsid` processes (training pool, run_pipeline*) DO run on real wall-clock. **⇒ Rely on detached auto-pipelines to keep GPUs busy through idle; the session only does intelligent review when reactivated (user message or a task completing during active time).** Continuity does NOT depend on the session waking.

## How to resume / operate
```bash
cd "/home/guy/UIOrthoLoRA/notebooks/catastroph