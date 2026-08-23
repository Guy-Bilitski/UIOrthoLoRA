# Tier A experiment spec — intruder slice + Qwen rescale ladder (2026-08-23)

Hardware: **one H100** (single card, jobs strictly serial). Submission in ~40
days; two weeks of PI vacation in the middle. Everything below is designed to
run unattended: chained per-cell jobs, checkpoint evacuation after every cell,
no step requires a human until the analysis readouts.

Standing rules (both experiments):
1. **Per-cell chaining (PI decision 2026-08-23):** train -> eval -> evacuate
   checkpoint -> next cell. Never train-all-then-eval-all. Every finished cell
   is a complete, pool-comparable data point.
2. **Every adapter checkpoint is kept and rsynced off-node the moment its cell
   finishes.** The 2026-07 fleet evacuation lost all 7B checkpoints; that
   failure mode is designed out, not remembered.
3. All cells use the pool's training recipe (Appendix A.3 / `app:traincfg`):
   3 epochs, method's own recipe, per-method configuration of Appendix A.1.
4. **Seed 43 everywhere.** Not 42: SC-LoRA's seed-42 runs sit at ~2x the
   update magnitude of seeds 43-45 (documented pool anomaly); a single-seed
   slice must not stand on the known-outlier seed.

---

## Experiment 1 — intruder dimensions slice (3 methods x 3 rates x 2 models)

**Question.** Do intruder dimensions (new leading singular directions of
W0+dW, near-orthogonal to the pretrained top subspace; Shuttleworth et al.)
appear in our runs; is their count/strength just a restatement of update
magnitude; and does it predict retention beyond F_delta?

**Why these cells.** Three methods span the design space: LoRA+wd (magnitude
control, no geometry constraint), MiLoRA (minor-subspace initialization),
SC-LoRA (data-aware init, strongest geometry signature). Three rates per
method per model, chosen from the frozen pool's rate->magnitude mapping so
the resulting F_delta spans below-knee / near-knee / well-above-knee
(family knees: Llama-CS sweep ~1.0 F_delta, Qwen-CS ~0.21; pooled Llama-CS
curve knee ~0.4).

### Cells (18 total; expected F_delta = pool median at that rate)

| # | model | method | lr | expected F_delta | zone |
|---|-------|--------|-----|------|------|
| 1 | Llama-2-7B | LoRA+wd | 1e-4 | 0.23 | below |
| 2 | Llama-2-7B | LoRA+wd | 5e-4 | 0.40 | knee (pooled) |
| 3 | Llama-2-7B | LoRA+wd | 1e-3 | 0.46 | above (wd caps it) |
| 4 | Llama-2-7B | MiLoRA | 1e-4 | 0.25 | below |
| 5 | Llama-2-7B | MiLoRA | 3e-4 | 0.56 | near |
| 6 | Llama-2-7B | MiLoRA | 1e-3 | 1.51 | far above |
| 7 | Llama-2-7B | SC-LoRA | 2e-5 | 0.17 | below |
| 8 | Llama-2-7B | SC-LoRA | 2e-4 | 0.60 | near |
| 9 | Llama-2-7B | SC-LoRA | 1e-3 | 1.74 | far above |
| 10 | Qwen2.5-7B | LoRA+wd | 2e-5 | 0.10 | below |
| 11 | Qwen2.5-7B | LoRA+wd | 3e-4 | 0.21 | knee |
| 12 | Qwen2.5-7B | LoRA+wd | 1e-3 | 0.28 | above (divergence risk: one pooled 1e-3 seed hit F=14.3; fallback 5e-4) |
| 13 | Qwen2.5-7B | MiLoRA | 5e-5 | 0.13 | below |
| 14 | Qwen2.5-7B | MiLoRA | 2e-4 | 0.29 | above |
| 15 | Qwen2.5-7B | MiLoRA | 1e-3 | 0.96 | far above |
| 16 | Qwen2.5-7B | SC-LoRA | 2e-5 | 0.12 | below |
| 17 | Qwen2.5-7B | SC-LoRA | 2e-4 | 0.43 | above |
| 18 | Qwen2.5-7B | SC-LoRA | 1e-3 | 1.09 | far above |

Configurations: each method exactly as in the pool (Appendix A.1): LoRA+wd
r=32, alpha=32, adapter weight decay 0.1; MiLoRA r=16 minor-init; SC-LoRA
r=32 with its **pooled nq_open calibration** (NOT eval-matched — the slice
must be comparable to the pool arms; the calibration question is settled
separately and stays out of this experiment).

**Queue order (coverage-first):** 2, 11, 6, 15, 9, 18 first (all six
model x method arms, knee-or-above where the signal lives), then the
remaining 12. After ~6 cells (~1 day) every arm has an informative point.

### Per-cell chain
1. Train (`train_cs.py` path, standard recipe, seed 43).
2. In-process evals, exactly the pool battery:
   - **Adaptation:** CS-8 mean accuracy (LLM-Adapters eval suite).
   - **Retention:** lm-evaluation-harness, FULL sets, no subsampling:
     BBH, MMLU-Pro (core retention = their mean), MMLU, ARC-Challenge
     (scored, excluded from headline per pool convention), TruthfulQA.
   - **CE/KL drift:** `forgetting_ce.py` on the held-out WikiText text
     (Metric 4), so the slice also lands in the RQ3 stores.
   - **F_delta:** `fdelta.py` (CLoRA F diagnostic, the dw axis).
   - **Geometry:** existing dW-vs-base-spectrum pass (geo.json fields).
3. Evacuate: rsync adapter dir (factors + config + all json) to the storage
   target, verify size+checksum, only then mark the cell done.

### CPU analysis (no GPU; runs incrementally as cells land)
New script `intruder_pass.py` (analysis-side, new file — does not touch
tested training code):
- Per adapted matrix: top-k (k=64) singular directions of W0+dW via subspace
  iteration with matvecs (W0 x + B(A x)); never forms the dense sum.
- Intruder = new leading direction whose max |cosine| against the pretrained
  top-k left singular vectors is below threshold; report at cos thresholds
  {0.5, 0.7, 0.9} (threshold sweep, not one magic number).
- Per run: intruder count, total intruder energy share, per-matrix margin
  sigma_1(dW)/spike-threshold.
- **Validation before any real adapter exists:** synthetic W0 (use the cached
  base SVD matrices) + constructed low-rank updates that provably do/do not
  cross the spiked-deformation threshold; detector must score 100% on these.

Readouts, in order of arrival:
- R1 (needs checkpoints only): do intruders appear at all; count vs F_delta
  per arm (the collinearity question; spec_max~F_delta r=0.931 predicts yes).
- R2 (needs evals): within-slice, partial correlation of intruder count with
  retention given log F_delta; and slice runs placed on the frozen family
  curves.
- R3 (secondary): does the constrained-design trio differ in intruder
  formation at matched magnitude (links fig:geometry's detection result to
  the mechanism).

**Decision gate for the paper:**
- Intruders track magnitude, add ~nothing beyond it -> mechanism paragraph in
  4.2 + appendix exhibit; related-work scoping upgraded to a positive
  statement; abstract untouched.
- Intruders add retention variance beyond F_delta (persistent across cos
  thresholds and both models) -> the abstract's geometry clause is reframed
  BY US, pre-submission. This readout must exist >=10 days before the
  deadline; it is why Exp 1 runs first on the machine.

**Fallbacks:** per-cell time > 3.5h -> keep all 18 (they're chained; the
queue just runs longer) but re-check the 7-day ask. A diverged cell (F_delta
explodes, e.g. Qwen lorawd 1e-3) -> retrain once at the fallback rate (5e-4),
flag in the run log, do not silently substitute.

**GPU budget:** 18 cells x 2-3.5h = **1.5-2.5 days**. First two cells
calibrate the estimate.

---

## Experiment 2 — Qwen rescale ladder (E1 cross-architecture)

**Question.** Does the interventional result (rescaling a trained adapter
moves retention along the family curve; same-size random directions land
below it) hold on Qwen, both settings? Fulfills Limitation 1's promise
("the most valuable missing experiment"); mirrors the Llama E1 battery
(15 rescales +1.29+-2.07 on-curve; 9 random-direction controls -1.76+-1.32;
direction penalty 3.05pp).

### Design
- **Anchors (GPU trains, chained like Exp 1):** per setting (Qwen-CS,
  Qwen-math): plain LoRA r=32 @ 3e-4 and LoRA+wd @ 5e-4, seed 43 ->
  **4 anchor cells**, full eval battery (math setting: retention = BBH only,
  pool convention).
- **Rescale ladder (no GPU):** `rescale_adapters.py` on each anchor to ~4
  target F_delta values per anchor spanning the family's observed range
  (exact targets computed from the frozen qwsw/qwswm curves when anchors
  land) -> **~16 rescaled adapters**.
- **Random-direction controls (no GPU to build):** matched-F_delta random
  updates, ~4 per setting -> **8 controls**.
- **Evals (GPU):** the 24 derived adapters get the retention battery
  (+ adaptation for the rescales; controls need retention only).
- Same evacuation rule; rescales/controls are cheap to rebuild but their
  evals are not — keep everything.

**Analysis:** existing E1 pipeline (on-curve residuals against the frozen
family curve, within-set correlation, direction penalty), rerun per setting.

**Paper landing:** 4.2 "Not an artifact of the recipe" second check becomes
two-architecture; Appendix D rescaling paragraph gains a Qwen block;
Limitation 1 rewritten from promise to result.

**GPU budget:** 4 trains (~0.5 day) + ~24-28 evals (~1-1.5 days) =
**1.5-2 days**.

---

## Combined schedule on the one H100

| Slot | What | Machine time |
|---|---|---|
| 1 | Exp 1, coverage-first 6 cells | ~1 day |
| 2 | Exp 1, remaining 12 cells | ~1-1.5 days |
| 3 | Exp 2 anchors (4 cells) | ~0.5 day |
| 4 | Exp 2 ladder + control evals | ~1-1.5 days |

Total: **~4-5.5 GPU-days; ask for 7.** CPU analysis (intruder pass, ladder
stats) runs off-machine, incrementally, including during vacation.

## Prerequisites (Phase 0 — before the machine arrives)

- [ ] Base weights available where the CPU analysis runs: Llama-2-7B +
      Qwen2.5-7B are NOT in the local HF cache (checked 2026-08-23);
      either download here (~30 GB) or run base-SVD rebuild + intruder pass
      on the H100 box's CPUs. DECIDE.
- [ ] Rebuild `results/geo_drift/base_svd/` for both models
      (`geo_drift_phase1.py`, `geo_drift_phase1_qwen.py`; currently empty).
- [ ] `intruder_pass.py` written + synthetic validation green.
- [ ] Job files written in chained per-cell form: `jobs/tierA_exp1_slice.txt`
      (18 cells, coverage-first order), `jobs/tierA_exp2_anchors.txt`
      (4 cells); Exp 2 eval queue generated after anchors land. Dedupe
      against `results/` before launch.
- [ ] Storage target for evacuated checkpoints confirmed (<5 GB total;
      local disk default unless PI names a NAS/bucket).
- [ ] H100 box environment: `requirements-freeze.txt` install + one smoke
      cell (any Exp 1 low-rate cell) before committing the queue.

## Open items (PI)
- Vacation dates (fixes when the Exp 1 decision-gate readout gets PI eyes).
- Checkpoint storage target.
- Confirm the 7-day machine ask.
