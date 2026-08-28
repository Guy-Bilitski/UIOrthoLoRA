# Intruder-dimension experiment — plan and current results

Updated 2026-08-28 14:40 UTC. Repo: `notebooks/catastrophic forgetting/`, branch `ortho_new`.

## 1. Question

Do intruder dimensions cause catastrophic forgetting, or is forgetting caused by the
size of the weight update? (Shuttleworth et al. arXiv 2410.21228 report intruders and
link them to forgetting, but never control for the update size that creates them.)

## 2. Plan

**Configurations.** One per adapter design = its best PARETO operating point on the
frozen pool's measured (adaptation, retention) frontier. No learning-rate sweep.

| # | model | method | lr | status |
|---|---|---|---|---|
| 1 | Llama-2-7B | LoRA+wd (r32, a64, wd0.3) | 5e-4 | DONE |
| 2 | Llama-2-7B | MiLoRA (r32, a32) | 3e-4 | DONE (trained+measured) |
| 3 | Llama-2-7B | CLoRA (r32, a64, k=1024) | 3e-4 | training now |
| 4 | Qwen2.5-7B | LoRA+wd | 1e-4 | queued |
| 5 | Qwen2.5-7B | MiLoRA | 1e-4 | queued |
| 6 | Qwen2.5-7B | SC-LoRA (beta 0.5, nq_open calib) | 2e-5 | queued |
| 7 | Qwen2.5-7B | CLoRA (k=1024) | 2e-4 | queued |

Llama SC-LoRA is DROPPED: its only pool run (lr 1e-3) collapsed to 0.64 retention, so
no trustworthy operating point exists.

**Training** (frozen pool recipe, `train_cs.py` unmodified): commonsense_170k,
170,420 examples, 3 epochs, batch 16, cutoff 256 tokens, bf16, seed 43 (44 for CLoRA),
= 31,956 steps, ~3.1 h/config.

**Evaluation** (`eval_one_gpu.py` unmodified): task = CS-8 mean accuracy, 200 items per
dataset. Retention = mean(BBH, MMLU-Pro), 50 documents per subtask. Update size =
`fdelta_token_weighted`. Same documents for every adapter, so all comparisons are paired.

**Intruder measurement** (`intruder_pass.py`, CPU): per adapted matrix take the top-10
left singular vectors of `W0+dW`; a direction is an INTRUDER if its maximum |cosine|
against ALL 4096 pretrained left singular vectors is < 0.5. 160 matrices per Llama model
(q,k,v,up,down x 32 layers). Note: this measures "not aligned with any single pretrained
direction", i.e. delocalised — 100% of random directions qualify.

**Causal intervention** — three modified copies per config, all evaluated identically:

| arm | what it is |
|---|---|
| **B** | delete EVERY intruder in the top-10 window from every matrix (this also shrinks the update) |
| **C** | take the ORIGINAL update and shrink it uniformly to exactly B's magnitude (size control, no geometry) |
| **D** | arm B rescaled back up to the SOURCE magnitude |

Two magnitude-matched contrasts: **B vs C** and **D vs source**. If deleting intruders
helps retention beyond the size change, geometry is causal; if not, forgetting is a
size effect.

## 3. Results so far

### 3a. How many intruders each design creates

| config | update size F | retention | task | intruders (top-10) | intruder energy share |
|---|---|---|---|---|---|
| Llama LoRA+wd 5e-4 | 0.395 | 24.87 | 80.00 | 908/1600 = **56.8 %** | 0.433 |
| Llama MiLoRA 3e-4 | 0.558 | 24.31 | 80.00 | 1255/1600 = **78.4 %** | 0.553 |
| (extra) Llama MiLoRA 1e-3 | 1.501 | 17.60 | 65.69 | 1595/1600 = 99.7 % | 0.924 |

At the same task accuracy (80.00) MiLoRA carries more intruders than LoRA+wd. Only 2 of
4 designs so far — not yet a claim.

### 3b. Causal test — CONFIG 1 ONLY (Llama LoRA+wd)

Deleting all 908 intruders removes 31.2 % of the update energy (||dW||^2 71,788.9 ->
49,411.5, norm ratio 0.8296). Arm C removes the identical amount uniformly.

| arm | F | retention | task |
|---|---|---|---|
| source | 0.395 | 24.87 | 80.00 |
| C: uniform shrink x0.8296 (size control) | 0.327 | **25.70** | 61.69 |
| B: all intruders deleted | 0.367 | **22.98** | **2.81** |
| D: deleted, rescaled x1.2054 | 0.442 | **20.24** | **8.81** |

- **B - C** (matched size): retention **-2.72 pp**, task **-58.88 pp**
- **D - source** (matched size): retention **-4.63 pp**, task **-71.19 pp**

**Reading:** removing the same amount of update *generically* IMPROVES retention
(25.70 vs 24.87, as the magnitude law predicts) and keeps most of the task. Removing the
same amount *targeted at intruder directions* destroys the task (80 -> 2.8) and makes
retention slightly WORSE. Intruder dimensions carry the fine-tuning, not the forgetting.

### 3c. What is NOT done

- Causal test for configs 2-7 (arms for MiLoRA 1e-3 are built and queued; others follow
  automatically as each config finishes).
- All four Qwen configs (no Qwen cell has completed yet — see §4).
- Design comparison needs >= 3 designs; currently 2.

## 4. Known issue: Qwen NaN

Qwen training NaNs at batch 70 (the first batch containing a 256-token sample),
NONDETERMINISTICALLY — the same config and seed sometimes NaNs and sometimes trains past
step 700. Ruled out by controlled test: GPU co-tenancy, `OMP_NUM_THREADS`/`MKL_NUM_THREADS`,
seed, method, learning rate, attention backend (sdpa/eager/math), left padding, and the
training data itself (base-model forward over 260 batches is finite, max|logit| 35.5).

Mitigation in place (not a diagnosis): every training job waits for a free GPU, then
retries up to 6 times, each attempt verified by `adapter_health.py` (a NaN adapter is
deleted and the attempt repeated). A failed attempt is detected in ~2.5 minutes.

## 5. Timing

~3.8 h per config (3.1 h train + 0.7 h eval), plus ~2 h for its three intervention arms.
Queue `jobs/tierA_go.txt`, tag `tierAgo`, one process on the GPU at a time.

| | ETA |
|---|---|
| config 3 (Llama CLoRA) | ~17:00 Aug 28 |
| configs 4-7 (Qwen) | ~Aug 29 evening |
| all intervention arms | ~Aug 30 |

## 6. Key files

- `handoff/INTRUDER_RESULTS.md` — full detail, methods, caveats
- `intruder_pass.py` — intruder measurement (`--selftest` passes 6/6)
- `intruder_ablate.py` — builds arms B/C/D
- `scale_adapter.py` — uniform-scaled copies (magnitude curve)
- `intruder_report.py` — prints the tables above
- `results/intruder/<run>.json` — per-adapter geometry
- `results/<run>/summary.json` — per-adapter retention/task/F
