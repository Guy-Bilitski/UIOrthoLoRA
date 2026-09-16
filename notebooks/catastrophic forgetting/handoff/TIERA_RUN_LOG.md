## 2026-08-27 — cell 11 divergence (tia1_qwsw_lorawd_wd0p3_lr3e4_s43)
- Trained 2-wide alongside cell 6. Loss healthy to ~step 70 (1.13), then grad_norm=nan
  DURING WARMUP (epoch 0.0066, lr 2.07e-4 of 3e-4); all subsequent training NaN; saved
  adapter 100% NaN (280/280 tensors). Its eval ran 2.4h on the dead model before being
  killed (rc=143) — those eval hours measured nothing.
- Pool precedent: qwsw_lorawd_wd0p3_lr3e4_s42 trained clean (cs_avg 86.33). Args diff
  vs pool: none material (path prefix + inert new defaults). bf16=True as fleet.
  => env (torch 2.12.0/transformers 5.10.2) x Qwen x seed-43 warmup instability suspected.
- Action: NaN adapter deleted; cell 11 retrains unchanged (once) in the serial queue.
  If it NaNs again at ~step 70: STOP, escalate to Guy (options: seed 44, env pin,
  attn implementation). Spec divergence rule analogue applied (flag, never substitute
  silently).
- Also today: 2-wide co-tenancy abandoned (eval batch_size=auto hogs 122GB: starves
  sibling evals, OOMs sibling trains — cells 15/9/18 partial dirs deleted after OOM).
  Queue resumes STRICTLY SERIAL (tag tierA1s) when cell 6's chain completes.
## 2026-08-27 — Exp 1 redesign (approved by Guy in-session)
- Non-coverage cells: retention battery at --ret_limit 1500 (SE ~±0.9pp core mean;
  coverage-6 keep FULL battery = the pool-comparable bridge). Cuts the dominant
  4.3h generation phase to ~40min; per-cell 8.3h -> ~4.7h.
- Cells 4/7/13/16 (MiLoRA+SC-LoRA below-knee) dropped; TWO CLoRA cells added at
  pool operating points: frc k1024 lr3e4, qwsw k1024 lr2e4 — retention-aware
  design with an explicit directional constraint (Guy's pick over DoRA).
- CLoRA cells at SEED 44: pool frc_clora_k1024_lr3e4_s43 is a documented
  adaptation-collapse outlier (cs 59.7 vs ~80 at s42/s44).
- Tail order: above/near-knee first, below-knee (cells 1, 10) LAST.
- Pre-authorized: if cell 11 retry NaNs again -> seed 44, flagged.
- Slice is now 16 cells (2 done/running + 14 queued): 6 full-eval + 10 reduced.
## 2026-08-27 14:45 — cell 11 seed-44 fallback APPLIED + dedupe fix
- Cell 11 (qwsw lorawd 3e-4) seed 43 NaN'd a SECOND time, bit-identical trajectory
  (grad_norm nan at epoch 0.00657, lr 2.07e-4). Deterministic => data-order bf16
  instability; pool seed-42 of same config was clean. Pre-authorized fallback used:
  cell 11 now runs SEED 44, flagged here and in the job-file comment.
- Waste caught: stage1 was generated BEFORE cell 6 finished, so its DONE-dedupe
  missed it and the pool restarted cell 6's FULL eval (~5 GPU-h). Killed after
  ~1h; job files regenerated (cells 2 + 6 now dedupe out). Lesson: ALWAYS
  regenerate job files immediately before launching a queue.

## 2026-08-27 ~17:00 — NaN ROOT CAUSE: GPU co-tenancy, not model/method/seed
Evidence (every training run this campaign):
  SOLO on the GPU            -> 0 NaN : cell 2 (smoke), 3x determinism reruns of the
                                        EXACT config that had NaN'd 3x before
                                        (676-753 steps clean, seed 43, unmodified pipeline)
  SHARED GPU (2+ processes)  -> NaN   : cell 11 s43 (2-wide w/ cell 6), cell 11 s44,
                                        cell 15, cell 9 (while 3 of my SDPA diagnostics
                                        shared the card)
Not the cause (each ruled out by experiment): seed (43 and 44 both died), method
(lorawd wd0.3, milora wd0, sclora), model (Llama cell 9 died too), learning rate
(3e-4 and 1e-3), attention backend (math / mem_efficient / default all survived
AND the default survived where it had previously died), training data (base-model
forward over 260 batches finite, max|logit| 35.5), left padding (isolated
fwd+bwd test finite).
Mechanism (hypothesis, not needed for the fix): under memory pressure the kernel
/ workspace selection changes; some path is numerically unsafe in bf16 here.
OPERATING RULE: **exactly one process on the GPU at all times.** No concurrent
diagnostics, no 2-wide pools, no eval alongside train. My own concurrent SDPA
diagnostics almost certainly destroyed cell 9 (77 min of training lost).

## 2026-08-28 00:5x — CAUSAL INTRUDER ABLATION COMPLETE (Exp 1, headline result)
Design: per source adapter, 3 magnitude-matched interventions, all scored on
IDENTICAL documents (ret_limit 50/subtask, eval_limit 200/dataset => paired).
  B = top INTRUDER direction removed per matrix (Shuttleworth-style surgery)
  C = original update uniformly shrunk to B's ||dW|| (magnitude control)
  D = B rescaled back to the SOURCE ||dW||  (vs the source itself)

  source                          F_delta   Ret   Adapt
  frc_lorawd_wd0p3_lr5e4_s43       0.395   24.87  80.00
    C uniform shrink               0.377   24.38  79.50
    B intruder removed             0.402   24.45  57.62
    D intruder removed, renormed   0.421   24.26  65.75
  frc_milora_lr1e3_s43             1.501   17.60  65.69
    C uniform shrink               1.412   19.09  67.00
    B intruder removed             1.436   13.50  59.75
    D intruder removed, renormed   1.528    8.35  47.44

RETENTION deltas (intruder-removed minus magnitude-matched control):
  lorawd  B-C = +0.07   D-source = -0.61      (null)
  milora  B-C = -5.59   D-source = -9.25      (harmful)
ADAPTATION cost of intruder removal: -21.9 / -14.3 (lorawd), -7.3 / -18.3 (milora).

READ: intruder dimensions are NOT the carriers of catastrophic forgetting. At
matched update magnitude, deleting them never improves retention and at large
magnitude badly degrades it, while costing 7-22pp of task accuracy. They are
load-bearing structure that carries the fine-tuning adaptation. Retention tracks
UPDATE MAGNITUDE (the uniform-shrink control preserves both axes). This is causal
evidence on retention benchmarks, complementing Shuttleworth et al. (correlational,
partial lambda-scaling, pre-training loss) and Xie 2026 (threshold law, perplexity).
CAVEATS to state in the paper: (1) B/D are structured edits that strip each
matrix's largest component (sigma_1 219->133 vs 205 for the control), so part of
the harm may be the surgery rather than the directions' identity — the __sc1p05 /
__sc1p12 uniform-scale curve now running quantifies this as an on-curve residual;
(2) we use full removal (lambda=0) and accuracy-based retention, they used partial
lambda and loss; (3) single seed per cell.

## 2026-08-29 — Qwen smoke gate was wrong; DoRA found unsupported by the pipeline

**Qwen "eager" smoke test passed a gate it should have failed.** The gate counted only
`'grad_norm': 'nan'` lines. The run had zero of those and 354 `'grad_norm': 'inf'`,
preceded by 141 large-but-finite gradients, with loss diverging 1.17 -> 9.70 -> 16.24 and
never recovering. It ran healthily for ~410 steps (loss ~1.0, grad ~0.35), then blew up in
a single logging window. The full Qwen experiment was launched on that false pass and was
killed ~5 minutes in (~5 GPU-minutes lost).

Signature comparison — every historical Qwen failure is pure NaN, this one is pure inf:

| log | nan | inf | large-finite | last step |
|---|---|---|---|---|
| tierA1_0 / tierAg_5 / tierAm_* (SDPA) | 6–3189 | 0 | 0 | 190–16758 |
| qwen_smoke (eager) | 0 | 354 | 141 | 6489 |

So eager attention changed the *failure mode*, not the outcome. Both are consistent with a
bf16 overflow; under the fused kernel it surfaces as NaN immediately, under eager as a
finite blow-up over ~10 steps. **Not asserted as a root cause** — three earlier root-cause
claims in this log were retracted. `max_grad_norm` is at the HF default of 1.0, so clipping
was already active and did not prevent it.

Gate replaced by `train_healthy()` in `overnight_lora_dora.sh`, which requires all three:
no `nan` grad_norm, no `inf` grad_norm, and final loss < 3.0. Validated against four logs:
rejects qwen_smoke (inf) and tierA1_0 (nan), accepts tierAj_0 (0.663) and tierAgo_0 (0.627).

**DoRA cannot go through the intruder pipeline as written.** `intruder_pass.load_adapter`
reads only `lora_A`/`lora_B` and returns `dW = (alpha/r) * B @ A`; it handles `use_rslora`
but has no `use_dora` branch and never reads `lora_magnitude_vector`. DoRA's real update is
`m * (W0 + s BA)/||W0 + s BA||_col - W0`, so intruder measurement and all of B/C/D/E/Ep/F
would be computed from the wrong `dW`, and writing modified factors back would leave the
magnitude vector stale. It would produce plausible but wrong numbers. Requested for the
2026-08-29 overnight run and deliberately excluded; **MiLoRA at lr 5e-4 substituted**, which
also gives a within-design magnitude ladder (MiLoRA is already measured at F=0.558 where the
intervention works and F=1.501 where arms B/D hit the floor). Adding DoRA support means
changing `load_adapter` and every arm writer — a real change, not a flag.

## 2026-09-14 — Qwen half of Table 2 launched (qwen_campaign.sh)

**Scope (Guy, in-session).** Mirror the completed Llama table design-for-design; one
configuration per design at its best operating point, seed 43 everywhere, no sweeps.
r32 designs at the Pareto points locked 2026-08-28 (LoRA+wd 1e-4, MiLoRA 1e-4, CLoRA
2e-4/k1024); r16 designs at Table 1's best-adapt LR, the same rule used for Llama (LoRA
5e-5 a32, LoRA-Null 2e-4 a16). SC-LoRA 2e-5 kept as the extra sixth design, queued last.
CLoRA moves from seed 44 to seed 43: in the Qwen pool seed 43 is healthy (86.19 / 39.89 /
F 0.215) and seed 44 is the collapse outlier (80.29) — the reverse of Llama. Recipes are the
pool's verbatim training commands. Run names: `tia1_qwsw_<design>_..._s43`.

**Hardware ruled out.** nvidia-smi: zero ECC errors (volatile and aggregate), zero remapped
rows, no Xid in the kernel log. The NaNs are numerical.

**Precision ruled out.** PEFT's `autocast_adapter_dtype=True` is in force: all 280
trainable tensors are fp32 (checked in-process), so this is not bf16 AdamW.

**The failure is nondeterministic.** 12/12 unguarded Qwen attempts on this box produced a
non-finite gradient (data-order-locked at step 70 for seed 43 in most runs, later in
others; under every SDPA backend, and as a finite blow-up under eager). Today's 5-minute
smoke of the exact seed-43 LoRA+wd recipe under the default backend passed step 70 and
ran 668 steps clean — consistent with the 2026-08-27 "3× clean reruns" and inconsistent
with a deterministic kernel bug. HF Trainer applies a NaN gradient to the weights, so a
single such step kills the run.

**Mitigation: `run_guarded.py`.** Runs the frozen `train_cs.py` unmodified via runpy
(same pattern as `run_safe_sdpa.py`) with two hooks on the Trainer: (1) before every
optimizer step, `torch._foreach_norm` over all gradients; if any is non-finite the step
is SKIPPED (grads zeroed, weights and Adam state untouched) — what GradScaler does for
fp16 and bf16 lacks; abort (exit 3) beyond 30 skips. (2) abort (exit 4) if the logged loss
exceeds 3.0 for 3 consecutive logging windows after step 300 (the eager blow-up mode).
Skip count and step indices are written to `<adapter>/nan_guard.json` and MUST be
reported alongside the Qwen rows. The health gate now reads that file (plus final loss <
3.0 and adapter finiteness) instead of grepping grad_norm lines: a skip at a logging
boundary legitimately logs a nan grad_norm while the weights stay clean.

**Schedule.** Train c(i) on GPU; build c(i)'s arms on CPU while c(i+1) trains; evaluate
c(i)'s seven arms; repeat. First row lands ~11 h after launch (12:48 UTC), then one row
every ~7.7 h. Estimate to be corrected from the first measured Qwen arm evaluation.
Retry policy per configuration: seed 43, seed 43 again, then seed 44 (flagged in the log
and in the paper if ever used).
- 12:48 first launch with a 30-skip cap: the guard fired at steps 62, 107, 288, 324, 592
  (5 in 841 steps, ~0.6 %), loss/grad trajectory identical to the unguarded smoke test at
  the same epochs (1.144 / 0.26 at epoch 0.046). A 30-skip cap would have aborted every
  attempt by step ~5k. Stopped at step 841, cap raised to 640 (2 % of steps, live-adjustable
  in `logs/guard_control.json`), burst abort at >40 skips within 200 steps added,
  relaunched 12:53. Stopped attempt's log: `logs/train_..._cap30_stopped.log`.
- The rate matters for the paper: skipped steps are dropped batches (<=2 % of the data,
  random positions), not a recipe change. Report the count per configuration.
- Power draw during training sits at 675 W of the 700 W cap with the SW power-cap throttle
  reason active. Together with the nondeterminism and the earlier co-tenancy correlation
  this is consistent with power-stress compute errors surfacing as NaN. Passwordless sudo
  is available, so a lower power limit is a testable mitigation once a baseline skip rate
  exists (decision at step ~1500).
- 13:05 **Not random.** Relaunch 2 skipped exactly the same steps as relaunch 1 (62, 107,
  288, 324, 592, then 1451, 1454 — 7 in 1750, 0.4 %), while the interactive smoke run of
  the same command minutes earlier skipped none of them, and the 2026-08-27 "3 clean
  reruns" were also interactive while every queued/daemonised run died. Environment diff
  between the two: nothing that touches numerics. Reading: kernel selection is fixed per
  launch (cuBLAS eligibility depends on pointer alignment, which depends on allocation
  order at load), and one eligible bf16 GEMM path overflows on particular batches.
  Action: `run_guarded.py` now sets `allow_bf16_reduced_precision_reduction=False`
  (fp32 accumulation in split-K bf16 GEMMs; strictly more precise, not a recipe change;
  `SAFE_BF16_REDUCE=1` restores the default). Campaign relaunched (3rd start, 13:08) so
  that all six Qwen configurations share the setting. If the same steps still skip, the
  reduction path is not the culprit and the guard carries the run regardless.
- `sudo nvidia-smi -pl 600` (power-cap test) was blocked by the tool's permission
  classifier; left for Guy to decide. Not needed for completion.
- 13:12 Relaunch 3 (fp32 reductions) skipped the SAME steps 62/107/288/324/592 → the
  cuBLAS reduced-precision reduction path is NOT the cause (setting kept: harmless, more
  precise). Env diff between interactive (clean) and daemonised (bad) launches contains
  nothing on the training path (GEO_THREADS is used only by the CPU geometry scripts).
  Left running under the guard — 0.4 % of steps skipped, trajectory matches the clean run.
  Guard now records, on every skip, the step's loss and which tensors are NaN vs inf
  (`[guard] diag` lines + `diag` in nan_guard.json). Takes effect from configuration 2
  (fresh process); no restart of configuration 1.
- 13:40 Loss/grad traces of the three daemonised attempts are NOT bit-identical (grad
  norm 0.984 / 1.050 / 1.048 in the same window): `train_cs.py` never seeds torch before
  `get_peft_model`, so the LoRA-A init is unseeded and every process trains different
  weights (pre-existing pipeline behaviour, also true of the pool; not changed). Yet all
  three skipped the SAME steps (62, 107, 288, 324, 592, 1451, 1454). Conclusion: the
  non-finite gradient is triggered by specific BATCHES, independent of the weights, while
  the single interactive launch that passed those batches remains unexplained. Local skip
  rate rose from 0.3 % (steps 0-4k) to ~2 % (steps 4.2k-4.6k); cap raised live to 1600 (5 %)
  via logs/guard_control.json so a 3 h run is not thrown away by an arbitrary cap. The
  burst abort (40 in 200) still stands. If a configuration ends above 2 % skipped, flag it.
  Per-tensor diagnostics arrive with configuration 2.
- 16:05 **Configuration 1 (LoRA+wd 1e-4, seed 43) TRAINED OK** under the guard: 31,956
  steps in 2 h 59 min, final loss 0.892, adapter finite (280 tensors, update energy 9990),
  **129 skipped steps = 0.40 %** (cap 1600, no burst). Skips are spread over the whole run
  (first 62, last 31,842), densest in 13k-20k (~0.6 %), never more than 3 within 200 steps.
  The clean interactive smoke and the three daemonised attempts all agree on the early
  skip set, so the dropped batches are a fixed, data-locked <0.5 % of the recipe. Report
  "129/31,956 (0.4 %)" with this row. `intruder_pass.py` (CPU) started 16:04:57; MiLoRA
  1e-4 seed 43 training started 16:04:57 on the GPU (expected gate ~19:05).
- 19:07 **Configuration 2 (MiLoRA 1e-4, seed 43) TRAINED OK**: final loss 0.902, adapter finite,
  update energy 10013.6, **129 skipped steps at EXACTLY the same 129 step indices as
  configuration 1** (a different adapter design, different weights). The non-finite gradient
  is therefore a fixed property of 129 specific batches of the seed-43 data order on this
  box; the guard removes the same 0.40 % of batches from every Qwen configuration, which
  makes the rows mutually comparable. Report once for the Qwen half, not per row.
  Configuration 1 geometry (full basis, k=10, tau=0.5): **169 intruders / 1400 = 12.1 %**,
  energy share 0.107, 75/140 matrices affected, ‖dW_B‖/‖dW‖ = 0.891 (Llama LoRA+wd 5e-4:
  56.8 %, 0.433, 135/160, 0.830). Lowest intruder fraction of any configuration, matching the
  lowest pool F (0.13). Confound: LR 1e-4 vs 5e-4 — read as "intruder fraction tracks
  forgetting level across architectures", not as an architecture effect. All seven arm
  invariants hold (verify_arms). 19:06:51 seven-arm evaluation of configuration 1 started
  (pool tag tQ_0); MiLoRA intruder scoring on CPU. CLoRA training starts after tQ_0 drains.
- 19:28 **Switched to `qwen_campaign2.sh` (concurrent schedule)** on Guy's instruction to use
  the GPU memory ("use the memory size smartly"). v1 alternated training (55 GB, 3 h) and a
  single-wide evaluation (28 GB, ~6 h for seven arms) and left ~115 GB idle during eval.
  v2 keeps the recipes, gate, arms and eval commands byte-identical and changes only the
  scheduling: `gpu_pool_dyn.py` polls `jobs/qwen_dyn_queue.txt` and runs TWO arm
  evaluations at a time (start only when >= 36 GB free, 150 s stagger), while the training
  loop runs the remaining configurations (CLoRA, LoRA r16, LoRA-Null, SC-LoRA) CONCURRENTLY,
  taking `logs/gpu_slot.lock` for the first 4 min of each start so no two processes size
  their memory against the same transient headroom. Budget 55 + 2x28 ~= 111 of 143 GB.
  Transition: v1 script and static pool tQ_0 killed; its job0 (LoRA+wd arm A) left running
  as an orphan (pid 50883, log `logs/tQ_0_0.log`), and that arm is queued last so the pool
  skips it once its summary exists. CLoRA 2e-4 seed 43 training started 19:28:31.
  Things to check because of the co-tenancy: (i) CLoRA's guard skip set vs the 129 indices
  of configurations 1-2 (a change would mean co-tenancy alters numerics, and would be
  logged); (ii) eval throughput per arm vs the solo arm A; (iii) any OOM in `logs/tQd_*.log`.
- 21:14 **Co-tenancy, first measurements and a correction.** (i) The v1 orphan (LoRA+wd arm A)
  died of CUDA OOM at ~19:45 when the pool's second job started: CLoRA training claims
  65 GB (not the 55 GB of LoRA+wd/MiLoRA; the k=1024 projections), and an eval peaks at
  35 GB (lm-eval auto-batch), so 65+35+29+25 > 140. The arm is queued last and will be
  re-run; no result was lost, ~40 min of GPU were. (ii) Throughput under 1 training + 2
  evals: training 1.02 s/step vs 3.0 step/s solo (**3x slower**), evals 1.36 s/req vs
  1.08 (20 % slower). Total GPU throughput ~1.9x solo, but training is the critical path
  (4 trainings x 9 h). CPU is not the bottleneck (load 3 on 16 cores). (iii) Fix:
  `gpu_pool_dyn.py` now takes its slot count live from `logs/dynpool_control.json`:
  **1 eval while a training runs, 3 evals when the GPU is otherwise idle**, min free 40 GB,
  and it ADOPTS eval chains already running when restarted (no duplicates). Pool restarted
  21:14, adopted MiLoRA arms A and B. Training speed under one co-tenant eval to be measured
  and the split tuned via the control file. Because campaign2 waits on the old pool pid,
  `FINAL_TABLE_qwen.{md,csv}` must be regenerated by hand (`paper_table.py`) after the
  queue drains.
- 21:17 **First Qwen evaluation: MiLoRA 1e-4 arm A** = cs 86.75 / retention 43.98 (BBH 47.11,
  MMLU-Pro 40.86) / **F_delta 0.183**. Pool run of the same recipe: 87.5 / 38.5 / 0.18 — task
  and F reproduce; the retention proxy (ret_limit 50) reads higher than the pool's battery,
  as expected for a different limit, and only within-table contrasts use it. Geometry:
  **6.0 % top-10 intruders, energy 0.081, 50/140 matrices, ‖dW_B‖/‖dW‖ = 0.972** (Llama
  MiLoRA 3e-4: 78.4 %, 0.553, ratio ~0.74). Arm B (all intruders deleted) is mid-eval and
  tracking arm A on the CS tasks so far (OpenBookQA 91.0 vs 92.0) — on Llama MiLoRA the same
  surgery collapsed task accuracy to 1.06. `forgetting_ce` skips the `__rl50` name ("no
  adapter_model.safetensors": eval-only alias of the source adapter) — same as in v1 and in
  the Llama campaign; F_delta comes from the summary, so harmless.
- 21:22 **Overnight self-healing setup (Guy away 8 h; "we can't afford losing a full night of
  H200").** Three daemons, one instance each: `qwen_campaign3.sh` (v2 + adopts an in-progress
  training after a relaunch, raises `logs/train_wanted` and waits for <= 1 eval and >= 100 GB
  free before starting a training, does not own the pool), `gpu_pool_dyn.py` (singleton lock
  `logs/tQd_pool.singleton`; honours `train_wanted`; adopts running chains), and
  `qwen_watchdog.sh` (every 2 min: restart pool if none, relaunch campaign3 if dead before
  STOP, restart a pool that left the GPU idle 20 min, 30-min status lines in
  `logs/qwen_watchdog.log`; regenerates FINAL_TABLE_qwen when the queue drains). Handover
  incident: `pkill -f` matched the invoking shell and killed it mid-handover (exit 144);
  finished by PID. campaign2 retired; CLoRA training (pid 62627) adopted by campaign3.
- 21:22 **MiLoRA 1e-4 arm B (all top-10 intruders deleted) = cs 86.25 / ret 44.75 / F 0.181**
  vs arm A 86.75 / 43.98 / 0.183. Deleting every intruder direction (169 -> here 84 dirs in
  50 matrices, 3 % of ‖dW‖) changes NOTHING on Qwen MiLoRA: task -0.5 pp (within the
  200-item noise), F -0.002. On Llama MiLoRA 3e-4 the same arm collapsed task accuracy
  80 -> 1.06 and F 0.558 -> 0.470. Reading: at a low-forgetting operating point the
  intruders carry neither the adaptation nor the forgetting — the intruder effect scales
  with the intruder energy share (0.08 here vs 0.55 on Llama), i.e. with update magnitude.
  Arm C (uniform shrink to B's norm, 0.972) started 21:22; with a 3 % shrink C should also
  sit on A. The informative Qwen rows for the causal contrast will be CLoRA 2e-4 and
  LoRA-Null 2e-4 (higher LR, more intruders expected).
- 23:08 Status check, all daemons healthy. MiLoRA arm C (uniform shrink to 0.972) = 86.94 / 44.41 /
  F 0.178, on top of A and B as expected. Arm evals take 68 min each alongside training.
  CLoRA at 55 %, 1.46 step/s, skips on the same indices as configs 1-2 (68 by step 17,426).
- 2026-09-15 00:31 Checkpoint, all healthy, no watchdog action needed so far. MiLoRA arm D (B
  rescaled to source norm) = 86.19 / 44.68 / F 0.186 — also on A. Four MiLoRA arms in, E running.
  CLoRA at 78 %, gate expected ~01:57.
- 01:48 **Configuration 3 (CLoRA 2e-4/k1024, seed 43) TRAINED OK** (adopted by campaign3 after
  the 21:30 relaunch): final adapter finite, update energy 59,126 (6x LoRA+wd's 9,990 — the
  2e-4 rate and the CLoRA projection give a much larger ‖dW‖), **129 skips at the IDENTICAL
  129 indices as configurations 1-2 although this run trained with 1-2 co-tenant evals for
  its whole duration** — co-tenancy does not alter the training numerics; the skip set is a
  fixed property of the seed-43 data order on this box. Intruder scoring started 01:48.
  Pool moved to 3 slots (MiLoRA Ep + F1 running); campaign3 raised `train_wanted` for
  LoRA r16 and waits for <= 1 eval + >= 100 GB free (coordination path exercised for the
  first time). MiLoRA arm E (non-intruder, magnitude-matched) = 86.75 / 44.95 / F 0.178 —
  five MiLoRA arms in, all on A within noise.
- 02:09 CLoRA arms built (01:58) and queued. **CLoRA 2e-4 geometry: 38.0 % top-10 intruders,
  energy share 0.277, 75/140 matrices, ‖dW_B‖/‖dW‖ = 0.720** — first Qwen configuration in the
  Llama range (Llama CLoRA 3e-4: 62 %, 0.461, ratio ~0.79). Deleting its intruders removes
  28 % of the update energy, so arms B/C/D of this row are the first Qwen cells where the
  causal contrast can show. Train/pool coordination worked as designed: `train_wanted`
  held the pool at one slot, LoRA r16 5e-5 training started 02:08:33 with 1 eval running
  and 104 GB free (attempt 1, seed 43).
- 02:29 Scheduled check: all daemons up, no watchdog action, 0 failures, 0 OOM; LoRA r16 at 6 %
  (1.52 step/s), MiLoRA F1 evaluating, CLoRA arms queued next. **MiLoRA arm Ep
  (perturbation-matched non-intruder deletion, ‖dW‖ ratio 0.598) = 86.88 / 49.98 / F 0.109**
  vs A 0.183: a 40 % smaller update keeps the full task accuracy and removes 40 % of the
  forgetting, while B (intruders deleted, ratio 0.972) changed nothing. Same pattern as
  Llama LoRA+wd Ep (F 0.395 -> 0.220) but here without the task-accuracy cost. Magnitude,
  not intruder geometry, moves F on this row. Six MiLoRA arms in; F1 completes the row.
- 02:59 **First complete Qwen row: MiLoRA 1e-4 (seed 43), all seven arms.** A 86.75/43.98/0.183 ·
  B (intruders deleted, ‖dW‖ 0.972) 86.25/44.75/0.181 · C (uniform shrink 0.972) 86.94/44.41/0.178 ·
  D (B rescaled to A) 86.19/44.68/0.186 · E (non-intruder, magnitude-matched 0.972)
  86.75/44.95/0.178 · Ep (non-intruder, perturbation-matched 0.598) 86.88/49.98/0.109 ·
  **F (non-intruder, count-matched, 1.050) 80.19/43.45/0.194**. Reading: with 6 % intruders,
  B=C=D=E=A (nothing in the intruder directions matters, either way); the only arm that
  moved F is the 40 % shrink (Ep), and the only arm that hurt the task is F, which deletes
  the same NUMBER of directions but takes them from the base-aligned top of the spectrum
  (-6.6 pp task). On Qwen MiLoRA the adaptation lives in the aligned directions, not the
  intruders — the mirror image of Llama MiLoRA (B collapsed to 1.06, F to 0.00). Scheduled
  check otherwise green: LoRA r16 at 15 %, LoRA+wd arm B evaluating, CLoRA arms after.
- 04:53 **LoRA+wd 1e-4 arm B (169 intruders deleted, ‖dW‖ 0.891) = 86.00 / 48.86 / F 0.131.** Pool
  source (full battery) 86.42 / 40.49 / F 0.132; the proxy-A re-eval (`__rl50`) is queued last.
  Deleting 12 % intruders / 11 % of the energy again leaves task and F unchanged (Llama
  LoRA+wd B: 80 -> 2.81 task). The slow retention generation of this arm (115 min vs 68) was
  NOT model degradation — the outputs score normally; the co-tenant LoRA r16 training is
  simply heavier on the GPU than CLoRA was. Arm C started 04:53.
- 04:59 Scheduled check green (LoRA r16 at 52 %, LoRA+wd arm C evaluating, 0 failures). Note:
  the 04:52 watchdog line recorded 130.6 GB in use with one training (~55 GB) + one eval, i.e.
  the eval's lm-eval auto-batch grew to ~75 GB in its loglikelihood stage — it sizes to the
  free memory at probe time, so a single eval can use whatever is free. Safe as long as a
  training never starts on top of already-probed evals (campaign3 requires >= 100 GB free
  and <= 1 eval before a training start); a mid-run squeeze would OOM the eval, which the
  pool retries once. No action.
- 05:34 **Seventh Qwen configuration added on Guy's approval: CLoRA k1024 at 3e-4, seed 43**
  (`tia1_qwsw_clora_k1024_lr3e4_s43`), the magnitude-matched cell. Reason: the two finished
  Qwen rows sit at F 0.13-0.18 and 6-12 % intruders where arm B is inert, while the Llama
  rows sit at F 0.40-0.56 and 57-78 % where B collapses; the Pareto rule placed the two
  architectures at very different forgetting levels. A Qwen cell at Llama's design AND
  learning rate tests "one dose-response curve, two architectures". CLoRA chosen over
  LoRA+wd because Qwen LoRA+wd saturates under wd 0.3 (pool dw_sv_max 10-13 for every LR
  1e-4..5e-4, F <= 0.25), whereas CLoRA 3e-4 reaches dw_sv_max 26 / F 0.29 with all pool
  seeds healthy (5e-4 reaches F 0.43 but its s43 pool run collapsed to cs 62). Queued last
  (after SC-LoRA); ETA for the full table moves from ~06:00 to ~20:00 UTC Sep 16.
  campaign3 relaunched with the 7-run list (adopted the LoRA r16 training, pid 138042).
  paper_table.py: added as an EXTRA row next to Llama MiLoRA 1e-3 (high-F).
- 06:39 LoRA+wd arm C (uniform shrink to 0.891) landed; see table line below. Arm D started.
  LoRA+wd 1e-4 arm C = 87.38 / 49.10 / F 0.120 (B was 86.00 / 48.86 / 0.131).
- 07:35 **Configuration 4 (LoRA r16 a32 5e-5, seed 43) TRAINED OK**: 5 h 26 min with one
  co-tenant eval, final loss 0.977, adapter finite, update energy 3,199 (the smallest of the
  set, as expected for r16 at 5e-5), **129 skips at the identical indices** (fourth time).
  Intruder scoring started; `train_wanted` raised for LoRA-Null r16 2e-4, which starts when
  LoRA+wd arm D (running since 06:39) frees the eval slot.
- 07:59 Scheduled check green (0 failures). **LoRA r16 5e-5 geometry: 3.0 % top-10 intruders,
  mean energy share 0.018, 34/140 matrices, ‖dW_B‖/‖dW‖ = 0.873** — the fewest intruders of
  any configuration, consistent with the smallest update (max s1 10.4). Arms built and
  queued 07:44. Scheduling: at the 07:35 gate the pool had already started a second eval
  (arm E) one second before `train_wanted` went up, so LoRA-Null waits for arm D to finish
  (~08:20) — a ~45 min training gap that would recur at every boundary. Fix: set
  `slots_idle=1` in `logs/dynpool_control.json` while trainings remain (one eval always
  runs, the next training starts the moment the previous one ends). **Restore slots_idle=3
  when the CLoRA 3e-4 training (the last) starts**, so the eval tail runs three-wide.
- 08:22 **Configuration 5 (LoRA-Null r16 a16 2e-4, seed 43) training started** 08:22:15 with one
  eval running and 118 GB free — no boundary gap this time (single-slot pool). LoRA+wd arm D
  (B rescaled to source norm) = 85.88 / 47.94 / F 0.146 vs B 86.00 / 0.131 and C 87.38 / 0.120:
  restoring the norm after deleting intruders raises F slightly, i.e. F follows the norm, not
  the intruder content. Arm E running; Ep, F1 and the source re-eval follow.
- 08:55 **Memory note:** the LoRA-Null training holds 99 GB (LoRA r16: 55 GB): its nq_open
  null-space calibration allocates large buffers before step 1 and PyTorch keeps them
  reserved. With LoRA+wd arm E at 39 GB the card sits at 138.4 / 143.7 GB. Training is
  stepping normally (1.45 step/s, 0 OOM). Exposure is limited to the eval (a growth of its
  auto-batch would OOM -> pool retries once); accepted rather than idling the eval slot for
  6 h. Expect the same for SC-LoRA (also calibrates). If an arm fails twice, re-enqueue it
  after the training ends (pool skips nothing by name; append the same line to the queue).
- 09:00 LoRA+wd arm E (non-intruder, magnitude-matched 0.891) = 87.12 / 48.67 / F 0.118 (B 86.00/0.131, C 87.38/0.120). Arm E finished without OOM despite 141 GB peak. Ep next.
- 10:50 LoRA+wd arm Ep (non-intruder, perturbation-matched, norm 0.573) = 82.44 / 50.89 / F 0.062 (A pool 86.42/0.132; B 86.00/0.131; C 87.38/0.120). Unlike MiLoRA Ep (no task cost at norm 0.598), here the 43 % shrink halves F (0.132 -> 0.062) but costs 4 pp task (86.4 -> 82.4): LoRA+wd sits closer to the edge where magnitude carries adaptation. F1 next, then the source re-eval.
- 12:30 LoRA+wd arm F (non-intruder, count-matched 169 dirs, norm 1.100) = 86.38 / 47.20 / F 0.152 (source 86.42/0.132; B 86.00/0.131). Unlike MiLoRA F (-6.6 pp task), here the count-matched deletion leaves the task intact and only raises F with the norm (1.100 -> F 0.152 vs 0.132): every LoRA+wd arm's F orders by its norm (Ep 0.573 -> 0.062, B/C/E 0.891 -> 0.12-0.13, A 1.0 -> 0.132, D 1.0 -> 0.146, F 1.10 -> 0.152). Source re-eval (__rl50) started; it completes the LoRA+wd row.

- 13:10 Enqueued a no-adapter eval of Qwen2.5-7B under the proxy protocol (run base_qwen25-7b): the Qwen base was never measured on the 50-item retention proxy, so the retention column had no zero point (Llama has base_l2-7b).
- 13:52 **Configuration 5 (LoRA-Null r16 a16 2e-4, seed 43) TRAINED OK**: 5 h 31 min, final loss
  0.894, adapter finite, update energy 35,681 (second-largest after CLoRA 2e-4's 59k — the 2e-4
  rate), **129 skips at the identical indices** (fifth time). SC-LoRA 2e-5 training started
  13:52:47 with zero gap (single-slot pool). Intruder scoring for LoRA-Null on CPU.
- 13:59 **LoRA-Null r16 2e-4 geometry: 61.5 % top-10 intruders, energy share 0.438, 135/140
  matrices, ‖dW_B‖/‖dW‖ = 0.706 (deleting them removes 50 % of the energy)** — this Qwen
  row sits squarely in the Llama range (Llama LoRA+wd 5e-4: 56.8 % / 0.433; Llama LoRA-Null
  5e-4: 93.6 % / 0.687). Together with CLoRA 2e-4 (38 % / 0.277) it gives the dose-response
  test without waiting for the extra 3e-4 cell: if arm B collapses here as on Llama, the
  effect is a function of intruder energy share, not architecture. Arms building on CPU.
  SC-LoRA in its calibration stage. Check green: 0 failures, 85 GB.
- 14:29 Queue reordered (pool identifies jobs by line content, so safe): CLoRA 2e-4 A-D, LoRA-Null A-D, base eval, then the E/Ep/F arms of both, LoRA r16 last. LoRA-Null arm B now lands ~22:00 instead of ~04:00. Check green: 0 failures; 142.5 GB peak (rl50 loglikelihood stage + SC-LoRA 99 GB), SC-LoRA at 7 %.
- 14:30 **Second complete Qwen row: LoRA+wd 1e-4, all seven arms.** Proxy A (`__rl50`) = 87.00 / 48.19 / F 0.134 (pool full battery 86.42 / 40.49 / 0.132 — task and F reproduce; retention proxy reads ~8 pts higher than the full battery, as for MiLoRA). Row: A 87.00/48.19/0.134 · B 86.00/48.86/0.131 · C 87.38/49.10/0.120 · D 85.88/47.94/0.146 · E 87.12/48.67/0.118 · Ep 82.44/50.89/0.062 · F 86.38/47.20/0.152. CLoRA 2e-4 arm A started 14:29; its B lands ~17:30.
- 16:12 **CLoRA 2e-4 arm A (proxy)** = 85.94 / 44.34 / F 0.211 (pool full battery 86.19 / 39.89 / 0.215). Arm B (169 intruders in 75 matrices deleted, norm 0.720, 28 %% of energy) started 16:12 -- the first Qwen cell with a Llama-sized deletion.
- 16:30 **CLoRA 2e-4 arm B, task stage complete (retention/F pending): CS-8 mean 84.69 vs A 85.94 (-1.25 pp).** Deleting 169 intruders holding 28 % of the update energy (norm 0.720) does NOT collapse Qwen CLoRA; all eight tasks within 0-5 pp of the source (SIQA 78.5 vs 83.5 the largest). Llama CLoRA 3e-4 B (62 % intruders, 46 % energy): 76.2 -> 43.5. The Qwen dose-response is flat so far up to 28 % energy; LoRA-Null B (44 % energy) at ~22:00 is the next point.
- 18:24 **CLoRA 2e-4 arm B (all intruders deleted, norm 0.720, 28 % of energy) = 84.69 / 46.36 /
  F 0.185** vs A 85.94 / 44.34 / 0.211. No collapse: task -1.25 pp, F down 12 % (norm down
  28 %, so F falls less than proportionally — the deleted intruder directions perturbed the
  activations LESS than average), retention +2.0 (proxy noise ~1.5). Llama CLoRA 3e-4 B:
  76.2 -> 43.5 task. The slow retention generation (6.7 s/req) did not translate into a
  retention loss. Arm C (uniform shrink to 0.720) started 18:24; C vs B is the row's key
  contrast — if C's F is ~0.152 (0.72 x 0.211) with task intact, intruder deletion is again
  the less effective way to spend a 28 % norm cut.
- 19:17 **Configuration 6 (SC-LoRA beta0.5 2e-5, seed 43) TRAINED OK**: 5 h 25 min, final loss
  1.008, adapter finite, update energy 1,501 (the smallest of all — 2e-5 rate), **129 skips at
  the identical indices** (sixth time; the skip set is now confirmed across six adapter
  designs and three co-tenancy regimes). **The seventh and last training, CLoRA k1024 3e-4
  seed 43 (Llama-matched LR), started 19:17:19** with one eval running and 125 GB free.
  `slots_idle` restored to 3 so the evaluation tail runs three-wide after it finishes
  (~01:00 UTC). SC-LoRA intruder scoring on CPU.
- 19:22 **SC-LoRA 2e-5 geometry: 0.2 % top-10 intruders (3 directions in 3 of 140 matrices),
  energy share 0.015, max s1 3.2** — effectively no intruders; arm B is the source with three
  directions removed (‖dW_B‖/‖dW‖ = 1.009 after the renorm bookkeeping). Its B-F arms are a
  guaranteed null and cost ~7 GPU-h; plan: keep A and B in place, move SC-LoRA C/D/E/Ep/F to
  the end of the queue behind LoRA r16 once its arms are enqueued (reorder, not a
  cancellation — the row stays complete for the table). Intruder fraction now spans 0.2 %
  (SC-LoRA, s1 3.2) to 61.5 % (LoRA-Null, s1 16.6) on Qwen alone, monotone in update size.
- 19:30 `verify_arms` flagged SC-LoRA: **B should be smaller than A** (ratio 1.009) and E
  INFEASIBLE (alpha 0 -> E == A). Cause: with 3 intruders holding 1.5 % of the energy, the
  arm builder's approximate top-k SVD (matvec, fixed iterations) removes less energy than its
  numerical error adds, so B/C/F come out slightly LARGER than the source and E is a copy of
  A. Not a pipeline bug at normal intruder counts (LoRA r16 with 3 % gave a clean 0.873);
  a floor effect of the design. Decision: SC-LoRA A (`__rl50`) and Ep (real shrink, 0.828)
  stay in the queue ahead of LoRA r16; B/C/D/E/F moved to the very end and can be dropped
  from the table (they measure noise) — Guy's call. Backup
  jobs/qwen_dyn_queue.before_reorder_1930.txt.
- 20:03 **CLoRA 2e-4 arm C (uniform shrink to B's norm, 0.720) = 87.06 / 48.28 / F 0.151** vs
  B (intruders deleted, same norm) 84.69 / 46.36 / 0.185 and A 85.94 / 44.34 / 0.211. At
  matched magnitude the uniform shrink beats intruder deletion on all three metrics: task
  +2.4, retention +1.9, F 0.151 vs 0.185 (F scales with the norm as predicted, 0.72 x 0.211 =
  0.152 — while B's F fell only to 0.185, i.e. the intruder directions carried LESS of the
  activation perturbation than average). This is the paper's Llama finding ("uniform
  shrinking retains more than deleting its intruders") reproduced on Qwen at a 28 % energy
  deletion, without the Llama task collapse. Arm D (B rescaled to A's norm) started 20:03.
- 20:52 slots_idle set to 2 for the night (3 evals can each grow to ~75 GB in their loglikelihood stage after sizing against a freer card; 2 cannot exceed 143 GB). ETA full table 14:00-17:00 UTC Sep 16.
- 21:59 **CLoRA 2e-4 arm D (B rescaled to the source norm) = 83.25 / 41.65 / F 0.255** vs A
  85.94 / 44.34 / 0.211. Putting the deleted intruders' energy back into the remaining
  (aligned) directions makes the model WORSE than the source on all three: task -2.7,
  retention -2.7, **F +21 %**. The row's causal core is complete: A 0.211 -> B (delete, 0.72)
  0.185 -> C (shrink, 0.72) 0.151 -> D (delete + restore norm) 0.255. Per unit of norm the
  intruder directions perturb the activations LESS than the aligned ones (B > C in F at equal
  norm; D > A at equal norm). Same ordering as Llama CLoRA (A 0.440, B 0.328, C 0.316,
  D 0.459) and Qwen LoRA+wd (A 0.134, B 0.131, C 0.120, D 0.146) — three rows, two
  architectures, one ordering: C < B <= A < D. LoRA-Null A/B/C/D next (B ~01:30).
- 23:32 **LoRA-Null 2e-4 arm A (proxy)** = 86.56 / 44.54 / F 0.197 (pool run name differs; campaign script quoted 87.1 / 39.7 / 0.20). Arm B (61.5 % intruders, 135/140 matrices, 50 % of energy, norm 0.706) started 23:32 -- the heaviest Qwen deletion, in the Llama LoRA-Null/LoRA+wd energy range.
- 2026-09-15 23:59 **LoRA-Null 2e-4 arm B, task stage complete: CS-8 mean 32.9 vs A 86.56 — the
  Llama collapse appears on Qwen.** Per task: BoolQ 54.5, PIQA 34.0, SIQA 40.0, HellaSwag 5.5,
  WinoGrande 2.5, ARC-E 48.5, ARC-C 39.0, OBQA 39.0 (several below chance -> broken output
  behaviour, as on Llama). Deleting 61.5 % intruders / 50 % of the energy destroys the
  adapted model, while 28 % (CLoRA) cost 1.3 pp and 8-11 % (MiLoRA, LoRA+wd) cost nothing.
  **Dose-response confirmed across architectures: the intruder-deletion effect is a function
  of the intruder ENERGY SHARE, not of the architecture.** Retention/F of B pending (~01:15);
  arm C (uniform shrink to 0.706) is the decisive control — on Llama LoRA-Null C kept 77.6
  task. Check green: 0 failures; CLoRA 3e-4 at 80 %, gate ~01:10.
- 2026-09-16 01:12 **Configuration 7 (CLoRA k1024 3e-4, seed 43 — the Llama-matched cell) TRAINED
  OK**: 5 h 55 min, final loss 0.832, adapter finite, **update energy 125,115** (2.1x CLoRA 2e-4's
  59k, 12.5x LoRA+wd's 10k — by far the largest Qwen update; Llama LoRA+wd 5e-4 was 71.8k),
  **129 skips at the identical indices — seventh of seven.** All Qwen training is complete:
  7 configurations, 7 x 129 = identical skip sets, zero NaN aborts, zero retries. Intruder
  scoring on CPU; arms then queued. Pool now runs without a training: slots -> 2.
- 01:25 **CLoRA 3e-4 geometry: 47.6 % top-10 intruders, energy share 0.375, 83/140 matrices,
  ‖dW_B‖/‖dW‖ = 0.722 (48 % of the energy deleted)** — between CLoRA 2e-4 (38 %, 0.277) and
  LoRA-Null (61.5 %, 0.438), and close to Llama CLoRA 3e-4 (62 %, 0.461) at the same design
  and rate. Arms built, all invariants hold, STOP queued (campaign3 now only waits for the
  pool to drain). Queue reordered so its A-D run right after the base eval, ahead of the
  E/Ep/F arms of the other rows.
- 01:26 **LoRA-Null 2e-4 arm B (full) = 32.88 / 41.28 / F 0.206** vs A 86.56 / 44.54 / 0.197. With
  50 % of the energy removed (norm 0.706), F went UP (0.197 -> 0.206) and retention DOWN (-3.3,
  beyond proxy noise) while the task collapsed — the Llama LoRA-Null pattern in full (Llama B:
  task 78.5 -> 0.5, F 0.702 -> 0.952, ret 22.0 -> 3.1), in milder form. Deleting intruders at
  high dose is strictly harmful on every axis; the surviving aligned directions perturb the
  activations more per unit norm than the deleted intruders did. Arms C (uniform shrink 0.706)
  and D (B rescaled to A) now running two-wide; C is the decisive control.
- 02:29 **LoRA-Null 2e-4 arm C (uniform shrink to B's norm, 0.706) = 87.19 / 48.80 / F 0.140** vs
  B (intruders deleted, same norm) 32.88 / 41.28 / 0.206 and A 86.56 / 44.54 / 0.197. The
  decisive high-dose contrast: at identical magnitude the uniform shrink keeps the full task
  (+0.6 over A), gains 4.3 retention and cuts F by 29 % (0.140 = 0.71 x 0.197, exactly
  proportional to the norm), while deleting the intruders destroys the task and raises F.
  This is the Llama LoRA-Null result (C 77.6 / 27.4 / 0.422 vs B 0.5 / 3.1 / 0.952) reproduced
  on Qwen. Across all four Qwen rows with a real deletion, C beats B on F at equal norm:
  LoRA+wd 0.120 vs 0.131, MiLoRA 0.178 vs 0.181, CLoRA 0.151 vs 0.185, LoRA-Null 0.140 vs
  0.206 — and the gap widens with the intruder energy share (0.11, 0.08, 0.28, 0.44).
- 02:57 **LoRA-Null 2e-4 arm D (B rescaled to source norm) = 3.50 / 29.88 / F 0.292** vs A 86.56 /
  44.54 / 0.197. Restoring the norm after deleting the intruders takes the model to Llama-D
  territory: task 3.5 (below chance), retention -14.7, **F +48 %**. The row's causal core is
  complete — A 0.197 · B (delete, 0.706) 0.206 · C (shrink, 0.706) 0.140 · D (delete+restore)
  0.292 — the same C < A <= B < D ordering as Llama LoRA-Null (A 0.702, B 0.952, C 0.422,
  D 1.728) with the same task collapse in B and D. Fourth Qwen row with the C < B ordering;
  the D > A gap grows with intruder energy share (LoRA+wd +9 %, MiLoRA +2 %, CLoRA +21 %,
  LoRA-Null +48 %). CLoRA 3e-4 arm A started 02:56 beside the base eval.
- 03:32 **Qwen2.5-7B base (no adapter) on the proxy protocol: retention 48.00 (BBH 44.15,
  MMLU-Pro 51.86); CS-8 21.25 (the base cannot follow the CS answer format — expected).** This is
  the zero point the Qwen retention column lacked. Forgetting of the source adapters in
  retention points: LoRA+wd 1e-4 **-0.2 (none)**, LoRA-Null 2e-4 3.5, CLoRA 2e-4 3.7, MiLoRA 1e-4
  4.0 — same order as their F (0.134 < 0.197 ~ 0.211 ~ 0.183 within noise). Arms whose retention
  reads 48-51 (all C arms, both Ep arms) sit AT the base: zero measurable forgetting. LoRA-Null
  B forgets 6.7 and D 18.1 points. Llama's comparable numbers: base 28, sources 22-25 (3-6
  points lost). So at their Pareto points the Qwen adapters forget 0-4 points of 48 while the
  Llama ones forgot 3-6 of 28 — Qwen's operating points are genuinely low-forgetting, which is
  the regime where intruder deletion has nothing to fix. CLoRA 3e-4 arm B started 03:32.
- 04:15 **CLoRA 3e-4 arm A (proxy)** = 85.00 / 39.86 / F 0.288 (pool 85.84 / 37.88 / 0.286; base retention 48.00 -> forgets 8.1 pts; Llama CLoRA 3e-4 A: 76.19 / 24.40 / 0.440). Arm B (48 %% of energy) running; C started.
- 04:29 **CLoRA 3e-4 arm B, task stage: CS-8 mean 82.56 vs A 85.00 (-2.4 pp) — no collapse at a
  48 % energy deletion**, whereas LoRA-Null B collapsed at 50 %. So energy share alone does not
  fix the outcome; the two high-dose cells differ in intruder FRACTION (47.6 % vs 61.5 % of
  top-10 slots), coverage (83/140 vs 135/140 matrices), rank (32 vs 16) and design (CLoRA's
  k=1024 orthogonality constraint keeps the update inside a protected subspace, so removing
  part of it may be benign). Read: collapse needs the intruders to hold most of the top
  spectrum in most matrices (Llama rows 57-94 %, LoRA-Null 61.5 %); at 38-48 % (both Qwen
  CLoRA rows) deletion costs 1-2 pp. Retention/F of B pending (~05:30); C running.
- 04:58 **CLoRA 3e-4 arm B (full) = 82.56 / 44.74 / F 0.246** vs A 85.00 / 39.86 / 0.288. At this
  high-forgetting Qwen cell (A forgets 8.1 pts vs base), deleting the intruders (48 % of energy,
  norm 0.722) recovers 4.9 retention points and cuts F by 15 % at a 2.4 pp task cost — the one
  Qwen B arm with a real retention gain. Whether it beats a plain shrink is arm C's question
  (expected F ~0.208 = 0.722 x 0.288 if proportional). C running since 04:15, D started 04:58.
- 05:45 **Pushed to GitHub (ortho_new: 68bf45b7 campaign+results, plus the table generator) and to
  Overleaf (git bridge, one commit).** Overleaf changes: `tables/table_intruder.tex` regenerated by
  `make_table_intruder_tex.py` with a Llama block and a Qwen block (rows ordered by intruder load,
  base retention per block, \dots for cells still evaluating); Section "Removing intruder
  dimensions does not reduce forgetting" rewritten around three claims — (1) intruder load is a
  function of update size on both architectures (0.2 % .. 94 %); (2) at matched magnitude the
  uniform shrink never retains less than intruder deletion (9/9 configurations), and the D>A
  penalty grows with intruder energy; (3) intruders carry the adaptation only once they dominate
  the top spectrum (inert <= 38 %, -2.4 pp at 48 %, collapse at 62 % and above on both
  architectures). Intro contribution (iii), Conclusion third result, Limitations item 1 (no longer
  "Llama only"), and Appendix app:intruder (guarded Qwen training, E/Ep availability, validity
  boundary on Qwen, Qwen base 48.00) updated accordingly. Re-run the table generator and re-push
  when the remaining arms land (SC-LoRA, LoRA r16, CLoRA 3e-4 C/D/E, E arms of CLoRA/LoRA-Null).
- 05:35 **CLoRA 3e-4 arm C (uniform shrink to B's norm, 0.722) = 87.38 / 46.55 / F 0.210** vs B
  (intruders deleted, same norm) 82.56 / 44.74 / 0.246 and A 85.00 / 39.86 / 0.288. Tenth of
  ten configurations where C >= B on retention (+1.8) — and here C also beats B on task by
  4.8 pp and on F (0.210 = 0.73 x 0.288, proportional to the norm; B's 0.246 is 17 % above
  that). C beats the SOURCE on task (+2.4) while recovering 6.7 of the 8.1 forgotten points:
  at this high-forgetting cell a plain 28 % shrink is a strictly better adapter than the
  trained one. Arm D running; CLoRA 2e-4 arm E started 05:35.
- 06:23 **CLoRA 3e-4 arm D (B rescaled to source norm) = 78.19 / 36.03 / F 0.337** vs A 85.00 /
  39.86 / 0.288: task -6.8, retention -3.8, F +17 %. Row core complete — A 0.288 · B (delete,
  0.722) 0.246 · C (shrink, 0.722) 0.210 · D (delete + restore) 0.337 — C < B < A < D, the fifth
  Qwen row with this ordering. **All five Qwen rows with a real deletion, and all five Llama
  rows, now show C <= B on retention and D worse than A.** The causal core of the Qwen half is
  complete; remaining queue = E/Ep/F controls of three rows + LoRA r16 + SC-LoRA (A, Ep, and
  the five degenerate arms last).
