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
- 06:46 CLoRA 2e-4 arm E (non-intruder removal, magnitude-matched 0.723) = 85.75 / 49.21 / F 0.137 (B 84.69/46.36/0.185, C 87.06/48.28/0.151). Ep running, F1 next.
- 07:36 CLoRA 2e-4 arm Ep (non-intruder removal, perturbation-matched, norm 0.723) = 85.69 / 49.46 / F 0.137 (E 85.75/49.21/0.137 at the same norm; B 84.69/46.36/0.185). F1 running; LoRA-Null E started.
- 08:04 **Session handover.** New agent took over from the handoff doc (QWEN_HANDOFF_2026-09-16.md).
  Health check green: 1 gpu_pool_dyn, 1 qwen_watchdog, 1 qwen_campaign3, 2 eval_one_gpu, 0 failures
  (rc=[1-9]), 0 OOM, disk 952 G, control {slots_training 1, slots_idle 2, min_free_mb 40000}. In
  flight: job29 CLoRA 2e-4 F1 (since 06:46) and job30 LoRA-Null E (since 07:36). Queue lines 33-51
  pending (19 arms) + STOP: LoRA-Null Ep/F1 · CLoRA 3e-4 E/Ep/F1 · SC-LoRA rl50, Ep · LoRA r16
  rl50/B/C/D/E/Ep/F1 · SC-LoRA B/C/D/E/F1 (degenerate, last). Session-bound watchers re-created
  (cron 23,53 * * * * health+results check; Monitor tail -F on the three logs, 30 min, re-armed).
  No daemon action needed.
- 08:29 Scheduled check green: pool/watchdog/campaign3 1 each, 2 evals, 0 trainings (all training
  done, `logs/train_wanted` absent as expected), 73.4/143.8 GB, 0 rc!=0, 0 FAILED, 0 OOM, 952 G
  free. No new arm since Ep: job29 (CLoRA 2e-4 F1, since 06:46) at 86 % of its retention stage,
  job30 (LoRA-Null E, since 07:36) at 49 %. The CLoRA 2e-4 E/Ep summaries were rewritten at 07:58
  (mtime only — values unchanged at 85.75/49.21/0.137 and 85.69/49.46/0.137). No action taken.
- 08:38 **Third complete Qwen row: CLoRA k1024 2e-4, all seven arms. Arm F1 (non-intruder,
  count-matched: the same 532 top-10 slots deleted but taken from the base-ALIGNED directions,
  norm 1.067) = 9.69 / 3.52 / F 0.300** vs A 85.94 / 44.34 / 0.211 and B (532 intruders deleted,
  norm 0.720) 84.69 / 46.36 / 0.185. The count-matched aligned deletion DESTROYS the model
  (task 9.7, retention 3.5 — 44 pts below the base 48.00, i.e. broken output, not a removed
  skill), while deleting the same number of intruder directions cost 1.25 pp. Row:
  A 85.94/44.34/0.211 · B 84.69/46.36/0.185 · C 87.06/48.28/0.151 · D 83.25/41.65/0.255 ·
  E 85.75/49.21/0.137 · Ep 85.69/49.46/0.137 · F1 9.69/3.52/0.300. The F-arm dose-response now
  mirrors the B-arm one, in the opposite direction: deleting 532 aligned directions costs
  0 pp on LoRA+wd (F1 86.38, update energy 10 k), 6.6 pp on MiLoRA (80.19, 24 k) and collapses
  CLoRA 2e-4 (9.69, 59 k) — the larger the update, the more of the adaptation sits in the
  aligned top spectrum. Note E/Ep (non-intruder deletion matched on MAGNITUDE, norm 0.723)
  leave the task intact at 85.7 with the best retention of the row (49.2/49.5, i.e. at the
  base); only the count-matched variant, which removes ~all of the aligned energy and then
  rescales back to 1.067, is destructive. Consistent with the row's verify log
  ([INFEASIBLE] E matched to B, rel diff 0.0095 — non-intruder content is nearly exhausted at
  that norm). LoRA-Null Ep started 08:37 beside LoRA-Null E.
- 08:47 **LoRA-Null 2e-4 arm E (non-intruder removal) = 85.31 / 48.57 / F 0.142** vs A 86.56 /
  44.54 / 0.197, B (intruders deleted, 0.706) 32.88 / 41.28 / 0.206 and C (uniform shrink, 0.706)
  87.19 / 48.80 / 0.140. **Caveat: E is NOT norm-matched on this row** — `verify_arms` reports
  [INFEASIBLE], rel diff 0.2464: removing ALL non-intruder content only takes the norm to 0.7876,
  it cannot reach B's 0.7055 (62 % of the top-10 slots are intruders, so the aligned content is
  exhausted first). Read with that caveat, the cell is the sharpest statement of the row: at
  62 % intruder load, deleting every base-aligned direction leaves the task intact (85.3, -1.3)
  and retention AT the base (48.57 vs 48.00), while deleting the intruders destroys the task
  (32.9) and raises F. So here the adaptation really does live in the intruder directions — and
  deleting them is still the worst available way to shrink the update: C at a SMALLER norm
  (0.706) keeps 87.2 task, 48.80 retention and cuts F by 29 %. E ~ C on F (0.142 at 0.788 vs
  0.140 at 0.706) — F tracks the norm, not which directions were removed.
  **Table provenance flag for Guy:** `make_table_intruder_tex.py`'s NOT_BUILT set sends the
  Llama LoRA r16 / LoRA-Null E cells to "--" for exactly this INFEASIBLE condition, but Qwen
  LoRA-Null is not in the set, so the next regeneration will print 48.57/85.31 in a column whose
  caption asserts norm matching. Options: dagger + stated norm, or "--" for consistency with the
  Llama rows. Not changed unilaterally — Guy's call. LoRA-Null F1 started 08:47.
- 09:46 **LoRA-Null 2e-4 arm Ep = 85.25 / 47.96 / F 0.142** vs E 85.31 / 48.57 / 0.142. Both arms
  carry the SAME norm on this row (0.7876 in `verify_arms`): because the aligned content is
  exhausted at 62 % intruder load, the magnitude-matched (E) and perturbation-matched (Ep)
  constructions coincide, so Ep is effectively a REPLICATE of E, not an independent arm. Useful
  as a precision estimate for the proxy: two independently evaluated adapters with identical
  construction differ by 0.06 pp task, 0.61 retention and 0.000 F — i.e. the retention proxy's
  run-to-run noise is ~0.6 pt, well below the contrasts the row reports (C-B = 7.5 retention,
  54 pp task). Row now complete except F1 (running): A 44.54/86.56/0.197 · B 41.28/32.88/0.206 ·
  C 48.80/87.19/0.140 · D 29.88/3.50/0.292 · E 48.57/85.31/0.142 · Ep 47.96/85.25/0.142.
  CLoRA 3e-4 arm E started 09:46.
- 10:42 **Fourth complete Qwen row: LoRA-Null r16 2e-4, all seven arms. Arm F1 (non-intruder,
  count-matched, norm 1.1261) = 0.00 / 0.00 / F 0.261** vs A 86.56 / 44.54 / 0.197 and E (ALL
  aligned content removed, norm 0.788) 85.31 / 48.57 / 0.142. Total destruction: all eight CS
  tasks exactly 0.0, BBH 0.0, MMLU-Pro 0.0. **Verified this is a real measurement, not a
  pipeline failure**: rc=0, 6898 s, and the loglikelihood-scored tasks in the same run still
  report chance-level numbers (MMLU 25.12 = 4-way chance, ARC-C 30.0, TruthfulQA 43.08,
  retention_broad 19.64) — generation is destroyed, ranking is at chance. Same signature as
  Llama LoRA-Null D (0.00/0.00 in the table). The only log match for error/nan is the standard
  lm-eval tokenizer length warning. Row: A 44.54/86.56/0.197 · B 41.28/32.88/0.206 ·
  C 48.80/87.19/0.140 · D 29.88/3.50/0.292 · E 48.57/85.31/0.142 · Ep 47.96/85.25/0.142 ·
  F1 0.00/0.00/0.261. Reading: E and F1 both delete only NON-intruder directions, and they sit
  at the two extremes of the row — E (everything aligned in the top-10, shrink to 0.788) is
  harmless, F1 (count-matched 868 directions, which reaches far below the top-10 into the
  aligned bulk, then rescales UP to 1.126) destroys the model. So what destroys an adapter is
  how deep the deletion cuts plus the renormalisation that follows, not whether the removed
  directions were labelled intruders. CLoRA 3e-4 Ep started 10:42.
- 11:12 **CLoRA 3e-4 arm E (non-intruder removal, magnitude-matched) = 86.38 / 46.43 / F 0.205**
  vs B (intruders deleted) 82.56 / 44.74 / 0.246 and C (uniform shrink) 87.38 / 46.55 / 0.210,
  **all three at exactly the same norm 0.7217** — `verify_arms` gives [PASS] E matched to B with
  rel diff 0.00000, so this is the campaign's one clean three-way equal-magnitude comparison at
  a HIGH intruder load (47.6 % of slots, 0.375 energy). CLoRA 2e-4's E was [INFEASIBLE]
  (rel diff 0.0095) and LoRA-Null's badly so (0.2464); here the constraint binds on neither side.
  Reading: at identical magnitude, removing the base-ALIGNED directions is indistinguishable
  from a plain uniform shrink (task 86.4 vs 87.4, retention 46.4 vs 46.6, F 0.205 vs 0.210) while
  removing the INTRUDERS is worse on all three axes (82.6 / 44.7 / 0.246). The intruder
  directions are therefore not the harmful component of the update: at matched norm they are the
  worst thing to spend the deletion budget on, and which directions are removed barely matters
  compared with how much norm goes. Row needs only F1 (started 11:12); Ep still running.
- 12:01 **CLoRA 3e-4 arm Ep (non-intruder, perturbation-matched, norm 0.7066) = 85.56 / 46.46 /
  F 0.200** vs E (same construction at 0.7217) 86.38 / 46.43 / 0.205 and C (uniform shrink,
  0.7217) 87.38 / 46.55 / 0.210. This is the only row where Ep and E carry DIFFERENT norms
  (0.7066 vs 0.7217; they coincided on CLoRA 2e-4 and LoRA-Null), so it is a genuine extra
  point — and it lands exactly where the norm predicts: 2 % less norm than E, 2 % less F
  (0.200 vs 0.205), task and retention unchanged within noise. Three non-intruder/shrink arms
  of this row (C, E, Ep) now sit at 46.4-46.6 retention and F 0.200-0.210 while the
  intruder-deletion arm B at the same magnitude sits at 44.74 / 0.246: the F ladder of this row
  is a pure function of the norm except for B, which is 17 % above the line. Row needs only F1
  (running since 11:12). **Queue has moved past the three E/Ep/F1 blocks: SC-LoRA A (`__rl50`)
  started 12:01** — remaining after it: SC-LoRA Ep, LoRA r16 (all seven), then the five
  degenerate SC-LoRA arms.
- 12:34 **Fifth complete Qwen row: CLoRA k1024 3e-4, all seven arms. Arm F1 (non-intruder,
  count-matched, 666 directions, norm 1.0383) = 37.06 / 31.44 / F 0.358** vs A 85.00 / 39.86 /
  0.288 and E (non-intruder, magnitude-matched 0.7217) 86.38 / 46.43 / 0.205. Heavy but PARTIAL
  degradation, unlike the total collapses elsewhere: per-task BoolQ 52.0, PIQA 55.0, HellaSwag
  61.0, OBQA 35.0 but SIQA 7.5 and ARC-E 25.5; MMLU 59.16 and TruthfulQA 47.19 show the model is
  still a working ranker, so this is a damaged CS-format follower, not a destroyed model.
  Row: A 39.86/85.00/0.288 · B 44.74/82.56/0.246 · C 46.55/87.38/0.210 · D 36.03/78.19/0.337 ·
  E 46.43/86.38/0.205 · Ep 46.46/85.56/0.200 · F1 31.44/37.06/0.358.
  **CORRECTION to the 08:38 entry.** That entry read the F-arm damage as ordering with update
  size (LoRA+wd 10 k energy -> 0 pp, MiLoRA 24 k -> -6.6 pp, CLoRA 2e-4 59 k -> collapse). This
  row breaks it: CLoRA 3e-4 has a 2.1x larger update (125 k) and deletes MORE directions (666 vs
  532), yet its F1 is far LESS damaged (37.06 vs 9.69). Full F1 ladder by task accuracy:
  LoRA+wd 86.38 (norm 1.100) · MiLoRA 80.19 (1.050) · CLoRA 3e-4 37.06 (1.038) ·
  CLoRA 2e-4 9.69 (1.067) · LoRA-Null 0.00 (1.126). Neither update energy nor deletion count
  orders this; what survives of the earlier reading is only the weak claim that count-matched
  ALIGNED deletion is damaging on four of five rows while intruder deletion at the same count is
  not (B: -1.3 pp on CLoRA 2e-4, -2.4 on CLoRA 3e-4). The mechanism behind the F1 ordering is
  not established by this campaign and should not be asserted in the paper. SC-LoRA Ep started
  12:34; SC-LoRA A still running.
- 13:20 **SC-LoRA 2e-5 arm A (proxy source) = 86.50 / 47.72 / F 0.115** (BBH 50.00, MMLU-Pro
  45.43). It anchors the bottom of the Qwen intruder axis: 0.2 % of top-10 slots, energy share
  0.015, update energy 1,501 (smallest of the seven), and now also the lowest F (0.115) and the
  smallest forgetting of any Qwen source — 0.28 points below the base 48.00, i.e. none
  measurable, at full task accuracy 86.50. Qwen sources ordered by intruder load:
  SC-LoRA 0.2 % 86.50/47.72/0.115 (forgets 0.3) · MiLoRA 6 % 86.75/43.98/0.183 (4.0) ·
  LoRA+wd 12 % 87.00/48.19/0.134 (-0.2) · CLoRA 2e-4 38 % 85.94/44.34/0.211 (3.7) ·
  CLoRA 3e-4 48 % 85.00/39.86/0.288 (8.1) · LoRA-Null 62 % 86.56/44.54/0.197 (3.5). F rises
  with intruder load across the set (0.115 -> 0.288) with LoRA+wd and LoRA-Null off the line,
  which is the same "intruder load tracks update size" reading already in the paper — not an
  independent result, since both quantities are functions of the update magnitude.
  **LoRA r16 A started 13:20**: its seven arms are the last informative block (ETA ~16:30),
  then the five degenerate SC-LoRA arms. SC-LoRA Ep still running.
- 14:00 **SC-LoRA 2e-5 arm Ep (non-intruder, perturbation-matched, norm 0.8278) = 87.19 / 48.34 /
  F 0.098** vs A 86.50 / 47.72 / 0.115. This is the ONLY non-source SC-LoRA arm the geometry
  supports (B/C/D/E/F all come out at ratio 1.000-1.011 — see the 19:30 entry of 2026-09-15), and
  it behaves exactly like the C arms of every other row: a 17 % shrink of an already tiny update
  buys +0.7 task, +0.6 retention and 15 % less F. Retention 48.34 sits marginally ABOVE the base
  48.00, i.e. no measurable forgetting at either end of this row. The Qwen block's F range is now
  0.098 (SC-LoRA Ep) to 0.358 (CLoRA 3e-4 F1). **LoRA r16 arm B started 14:00** — the last
  informative block is underway; LoRA r16 A still running (72 % of generation at 13:59).
- 14:19 **LoRA r16 5e-5 arm A (proxy source) = 85.44 / 48.18 / F 0.122** — retention 0.18 points
  ABOVE the base 48.00, i.e. no measurable forgetting, at 85.44 task. Second-lowest F of the Qwen
  sources after SC-LoRA (0.115), consistent with its 3 % intruder slots / 0.018 energy share and
  the second-smallest update (energy 3,199). Both ends of the Qwen intruder axis (SC-LoRA 0.2 %,
  LoRA r16 3 %) therefore sit at zero forgetting, which is what makes their B/C/D arms
  uninformative for the causal contrast — there is nothing to remove. Run took 3,520 s (59 min)
  versus the ~78 min of recent arms, the smallest adapter evaluating fastest; if the rest of the
  row keeps this pace the ETAs I gave Guy at 13:23 move ~15-20 min earlier. Arm C started 14:18
  (B running since 14:00).
- 15:06 **LoRA r16 5e-5 arm B (42 intruders deleted, norm 0.8732) = 85.81 / 48.42 / F 0.106** vs
  A 85.44 / 48.18 / 0.122. Inert, as the 3 % intruder load predicts: task +0.4, retention +0.2
  (both within the ~0.6 proxy noise measured at 09:46), and F falls 13 % for a 13 % norm cut —
  exactly proportional, so the deleted directions perturbed the activations at the average rate.
  Sixth Qwen row to show B inert or harmful rather than helpful; with MiLoRA (6 %) and LoRA+wd
  (12 %) this fixes the bottom of the dose-response curve at three configurations where removing
  every intruder changes nothing. Arm C (uniform shrink to the same 0.8732) is running since
  14:18 and is the comparison that matters — at this load both should sit on A. Arm D started
  15:06.
- 15:18 **LoRA r16 5e-5 arm C (uniform shrink to B's norm 0.8732) = 86.38 / 48.82 / F 0.107** vs
  B (42 intruders deleted, same norm) 85.81 / 48.42 / 0.106 and A 85.44 / 48.18 / 0.122.
  **Eleventh of eleven configurations in which the uniform shrink retains at least as much as
  intruder deletion at equal norm** (+0.4 retention here, +0.6 task; F identical at 0.106/0.107
  because at 3 % intruder load the two constructions remove almost the same subspace). The
  headline count in the manuscript ("9/9" at the 05:45 Overleaf push, then 10/10) becomes 11/11
  once this row and CLoRA 3e-4 are included — worth updating in the text when the table is
  regenerated. As expected at this load the C-B gap is the smallest of the campaign: the gap
  grows with intruder energy share (LoRA r16 0.02 -> +0.4 retention; MiLoRA 0.08 -> -0.3;
  LoRA+wd 0.11 -> +0.2; CLoRA 2e-4 0.28 -> +1.9; CLoRA 3e-4 0.38 -> +1.8; LoRA-Null 0.44 ->
  +7.5). Arm E started 15:17; D running since 15:06.
- 16:11 **LoRA r16 5e-5 arm D (B rescaled to the source norm) = 81.50 / 46.81 / F 0.121** vs
  A 85.44 / 48.18 / 0.122, B 85.81 / 48.42 / 0.106 and C 86.38 / 48.82 / 0.107. **Twelfth of
  twelve configurations where D is worse than A** (task -3.9, retention -1.4) — the pattern is
  now unbroken across both architectures and the full 0.2-94 % intruder range.
  **But a caveat on HOW it is worse.** The F version of the claim holds and tightens: the D-vs-A
  F penalty tracks intruder energy share almost monotonically — 0.018 -> -1 % (here),
  0.08 -> +2 %, 0.11 -> +9 %, 0.28 -> +21 %, 0.375 -> +17 %, 0.44 -> +48 %. The TASK penalty does
  not: this row has the smallest energy share of the campaign (0.018) yet loses 3.9 pp, more than
  CLoRA 2e-4 at 0.28 (-2.7 pp) and far more than MiLoRA at 0.08 (-0.6 pp). Likely mechanism:
  restoring the norm after deleting 42 directions multiplies everything that remains by
  1/0.873 = 1.145, and a 14.5 % amplification of the whole update is damaging regardless of how
  little energy the deleted directions held. So the manuscript should keep the D>A claim on F
  (where it is ordered) and state the task cost as present-but-unordered rather than
  energy-scaled. Arm Ep started 16:11; E running since 15:17, F1 last.
- 16:19 **LoRA r16 5e-5 arm E (non-intruder removal, magnitude-matched 0.8732, [PASS] rel diff
  0.00000) = 86.62 / 49.14 / F 0.105** vs B (intruders deleted, same norm) 85.81 / 48.42 / 0.106
  and C (uniform shrink, same norm) 86.38 / 48.82 / 0.107. Three arms at one norm again, and the
  ordering is E >= C >= B on task and retention with F identical to three decimals — at 3 %
  intruder load the three constructions are interchangeable, which is the expected null at the
  bottom of the dose-response curve and the counterpart to CLoRA 3e-4, where the same three-way
  comparison at 48 % separates them. E's 49.14 is the highest retention of the row, 1.1 points
  ABOVE the base 48.00.
  **MILESTONE: every cell Table~\ref{tab:intruder} needs is now measured except SC-LoRA B/C/D**
  (queue lines 47-49, the degenerate arms). Both LoRA r16 and SC-LoRA dW/A cells are in, the
  LoRA r16 A-E block is complete, and the Qwen block has no other gaps. The table can be
  regenerated and pushed as soon as Guy rules on the SC-LoRA arms: keep them (last cell ~20:25)
  or print "--" and regenerate now. F1 started 16:18 and is the row's last arm, but F is not a
  table column.
- 17:28 **Sixth and last informative Qwen row complete: LoRA r16 5e-5, all seven arms. Arm F1
  (non-intruder, count-matched, 42 directions, norm 1.0892) = 85.88 / 47.36 / F 0.128** vs
  A 85.44 / 48.18 / 0.122. Inert: task +0.4, retention -0.8, F +5 % for a +9 % norm — the same
  null as LoRA+wd's F1 and the expected outcome when the count-match removes only 42 of 1400 top
  slots. Row: A 48.18/85.44/0.122 · B 48.42/85.81/0.106 · C 48.82/86.38/0.107 · D 46.81/81.50/
  0.121 · E 49.14/86.62/0.105 · Ep (running) · F1 47.36/85.88/0.128.
  Full F1 ladder by deleted count, which fits better than the energy reading corrected at 12:34
  but still has two inversions: 42 dirs (3 %) 85.88 · 169 (12 %) 86.38 · 84 (6 %) 80.19 ·
  532 (38 %) 9.69 · 666 (48 %) 37.06 · 868 (62 %) 0.00. MiLoRA (84 dirs, worse than LoRA+wd's
  169) and CLoRA 3e-4 (666 dirs, better than CLoRA 2e-4's 532) both break monotonicity, so the
  campaign supports only the qualitative statement: count-matched ALIGNED deletion is harmless
  below ~12 % of slots and damaging above ~38 %, with the severity in between unordered.
  **All six informative Qwen rows are now complete** (MiLoRA, LoRA+wd, CLoRA 2e-4, CLoRA 3e-4,
  LoRA-Null, LoRA r16) plus SC-LoRA A/Ep. Only the five degenerate SC-LoRA arms remain queued;
  Ep of LoRA r16 is still running (76 % at 17:24, slowed to 4.15 s/it).
- 17:36 **LoRA r16 5e-5 arm Ep (non-intruder, perturbation-matched, norm 0.5752) = 78.38 / 47.53 /
  F 0.055** vs A 85.44 / 48.18 / 0.122. The deepest shrink of the campaign (42 % of the norm
  removed) and its lowest F: 0.055, less than half the source's, with retention holding at 47.53
  (base 48.00) but task down 7.1 pp. Ep arms ordered by norm: LoRA+wd 0.573 -> 82.44 (-4.0 pp),
  LoRA r16 0.575 -> 78.38 (-7.1), MiLoRA 0.598 -> 86.88 (0.0), CLoRA 3e-4 0.707 -> 85.56 (+0.6),
  CLoRA 2e-4 0.723 -> 85.69 (-0.3), LoRA-Null 0.788 -> 85.25 (-1.3), SC-LoRA 0.828 -> 87.19
  (+0.7). Below ~0.6 the shrink starts costing task accuracy on two of three rows; above ~0.7 it
  is free everywhere. MiLoRA at 0.598 is the exception that stops this being a clean threshold.
  **THE ROW AND ALL INFORMATIVE EVALUATION ARE COMPLETE.** Seven arms x six rows (MiLoRA,
  LoRA+wd, CLoRA 2e-4, CLoRA 3e-4, LoRA-Null, LoRA r16) + SC-LoRA A/Ep + the Qwen base = every
  cell the manuscript needs. Everything still queued (SC-LoRA B/C/D/E/F1) is the degenerate block
  that measures the arm builder's numerical floor. SC-LoRA B started 17:29, C 17:36.
- 18:51 **SC-LoRA 2e-5 arm B (3 intruders "deleted", norm ratio 1.0086) = 86.75 / 48.78 / F 0.115**
  vs A 86.50 / 47.72 / 0.115. **F is identical to three decimals** and task differs by 0.25 pp:
  the degeneracy diagnosed at 19:30 on 2026-09-15 is now confirmed empirically, not just from the
  verify log. Removing 3 directions holding 1.5 % of the energy — from an adapter whose norm the
  builder then leaves 0.9 % LARGER than the source — is arithmetically a no-op, and the result
  behaves like a re-evaluation of A (retention +1.06, at the edge of the 0.6 pt proxy noise;
  both sit at the base 48.00). This settles the open question with evidence: B is the arm of this
  row that would show the largest effect if any existed, and it shows none, so C/D/E/F1 are
  predictable nulls. Recommend printing "--" for SC-LoRA B/C/D in the table (consistent with the
  Llama LoRA/LoRA-Null E cells, which are "--" for the same INFEASIBLE reason) and dropping queue
  lines 49-51. **Not acted on — Guy's call.** Arm D started 18:51; C running since 17:36.
- 18:54 **SC-LoRA 2e-5 arm C (uniform shrink to B's 1.0086) = 86.56 / 47.50 / F 0.116.** Row so
  far: A 47.72/86.50/0.115 · B 48.78/86.75/0.115 · C 47.50/86.56/0.116 · Ep 48.34/87.19/0.098.
  A, B and C are three independent evaluations of what is arithmetically the same adapter (the
  builder's "deletion" and "shrink" both land within 0.9 % of the source norm), so they give the
  campaign's best NOISE ESTIMATE: task spread 0.25 pp, F spread 0.001, **retention spread 1.28
  points** (47.50 to 48.78).
  **This is larger than the 0.6 pt figure taken from the LoRA-Null E/Ep replicate at 09:46, and
  it matters for the write-up.** With retention noise up to ~1.3 pt, the C-vs-B retention margins
  on the low-load rows are inside the noise band: LoRA r16 +0.4, MiLoRA -0.3, LoRA+wd +0.2. Only
  CLoRA 2e-4 (+1.9), CLoRA 3e-4 (+1.8) and LoRA-Null (+7.5) clear it. The "11/11 C >= B" count
  remains true as measured and can stand as a descriptive statement, but the per-row margins
  below ~1.3 must not be interpreted as effects, and the claim's real weight comes from the three
  high-load rows. Task accuracy is the sharper axis (0.25 pt noise): there the C-vs-B gaps of
  +2.4 (CLoRA 2e-4), +4.8 (CLoRA 3e-4) and +54 (LoRA-Null) are far outside noise. Arm E started
  18:54; D running since 18:51, F1 last.
