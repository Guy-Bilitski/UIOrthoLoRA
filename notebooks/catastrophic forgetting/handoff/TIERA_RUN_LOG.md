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
