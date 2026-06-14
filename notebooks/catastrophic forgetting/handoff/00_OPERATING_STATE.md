# OPERATING STATE — read this first (handoff for any agent / future me)

**Project:** Is **UIOrthoLoRA** a real catastrophic-forgetting mitigator worth A*-publishing?
Head-to-head vs **LoRA** (forgetful baseline) and **CLoRA** (SOTA forgetting-mitigator, ACL'25)
on CLoRA's own commonsense one-stage setting, LLaMA-2-7B. Full spec: `../agent_instructions.nd`.

**Decision criterion:** frontier of **in-domain commonsense CS-avg (x)** vs **out-domain
retention = mean(answer-only-BBH, MMLU-Pro) (y)**. GO if UIOrthoLoRA's frontier sits on/above
CLoRA's, esp. dominating CLoRA in some corner. A *tie* is NOT enough for A* (UIOrthoLoRA carries
~4× train cost, ~8× memory, non-round-trippable checkpoints) — needs a clear win or a strong
second story (the **leakage thermometers** + **LR-robustness** are those stories).

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

## What's running RIGHT NOW (update as it changes; as of 2026-06-14 ~14:30)
- **A5 RESULTS IN (6 runs, drop_major=1) — VERDICT SHIFTED from "dominated" to "COMPETITIVE in the high-retention corner."** Dropping the spurious major term improves BOTH axes vs legacy twins (it was actively hurting adaptation, not just leaking into the preserved subspace). A5-vs-legacy:
  - `a5_k2048_v410_dE0_lr1e3`: **CS 64.4 / ret 26.93** (vs legacy 57.5/25.7) — **highest retention of ANY adapter run**, edges CLoRA-k2048 (26.76) and matches base, at ~same CS. No longer dominated.
  - `a5_k1024_v128_dE1_lr1e2`: **CS 69.0 / ret 25.04** (vs legacy 50.5/23.68, **+18.5 CS**) — DOMINATES CLoRA-k512 (67.6/23.06) on both axes. use_de=1+drop_major is a big CS booster.
  - clean-arm twins all improved: k1024_lr1e3 60/19.9→53.6/**26.3**; k2048_lr1e2 46.9/22.2→56.1/24.3; k512_lr1e2 48/17.2→54.8/22.3; k1024_lr1e2 55/18.5→54.8/23.1. dw_max consistently ↓ (major term was big Frobenius mass).
  - **STILL loses the high-CS corner:** no A5 point reaches CLoRA-k1024 (CS 79.9/ret 25.57); A5 tops out ~69 CS. ⇒ competitive/tied in retention corner, not yet a clear A* domination.
- **Wave 5b LAUNCHED (the CS-ceiling test) — pool `uio_w5b` GPUs 1,3,7:** 3× use_de=1+drop_major at higher LR/k to see if the corrected layer can push CS toward 80 (`a5_k1024_v128_dE1_lr2e2`, `a5_k2048_v256_dE1_lr1e2`, `a5_k2048_v410_dE1_lr2e2`). **This decides GO vs competitive-but-tied.** Logs `logs/uio_w5b_*`. ETA k1024 ~7h, k2048 ~11h.
- **Wave 5 seeds (5 still running on `uio_w5` GPUs 0,2,4,5,6):** seeds 43/44 on the 3 anchor points → error bars on the retention-corner edge. seed43_k1024_dE0 done.

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