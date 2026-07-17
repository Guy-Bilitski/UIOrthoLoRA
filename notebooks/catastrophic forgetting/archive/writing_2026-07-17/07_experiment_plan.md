# 07 — Experiment Plan for Blockers B4 & B5 (NEW runs; nothing fabricated)

Author date: 2026-07-02. Scope: the two blockers that require NEW compute — **B4 (calibration
fairness)** and **B5 (seeds + param-match)**. B1/B2/B3/B6 are solvable from existing data and are
NOT in this doc. No results are invented here; this is a costed run/analysis spec.

Hardware: single 8×B200 box, **one GPU scheduler at a time** (gotcha in handoff/13 §7 — two pools
each grab all 8 GPUs → OOM). Cost model below is derived from measured cell times in the handoff
docs: **Llama-2 cell ≈ 1.5–2.5 h** (train+broad eval, gen_cap-dependent), **Qwen cell ≈ 6–10 h**
(handoff/16, 03 §b). "Cell" = one (arm × LR × seed × model × domain) train+eval unit. 8 GPUs run
8 cells in parallel, so **wall-clock ≈ ceil(cells/8) × per-cell-hours**. All costs below are
Llama-2-CS unless stated (the mature domain where B4/B5 must land first); Qwen/math extensions are
called out separately and are the expensive tail.

Machinery recap (from make_campaign_jobs.py): arms live in the `ARMS` dict; LR grid is the 7-entry
`LRS`; seeds in `SEEDS=["42"]`. A new arm = one dict entry (run-name key → train flags). Seeds =
edit `SEEDS`. The generator is resumable (skips any cell with an existing `results/<run>/summary.json`),
so new arms/seeds are added by editing those and re-running `make_campaign_jobs.py`, then launching
the single pool via `run_all_experiments.sh` (`SKIP_VALIDATION=1 nohup bash run_all_experiments.sh …`).

---

## Shared prerequisite (blocks B4 only): the `--calib_source` flag + eval-matched aux set

Today all three calibration-using arms **hardcode nq_open** as the calibration source
(train_cs.py lines 218, 241, 260 — CorDA cprompts, SC-LoRA dminus, LoRA-Null kprompts). B4 needs
the calibration source to be selectable, and needs an eval-distribution-matched set that is
**DISJOINT from the retention test** (retention eval = BBH + MMLU-Pro core, plus MMLU/ARC-c/
TruthfulQA broad).

### P0 code change (a few hours, no GPU)
1. Add `ap.add_argument("--calib_source", choices=["nq_open","eval_matched"], default="nq_open")`
   to train_cs.py. `nq_open` = today's behavior exactly (keeps every existing cell reproducible and
   the flag backward-compatible — do NOT change the default).
2. Add a helper `load_calib_prompts(source, n)` and replace the three inlined `nq_open` blocks
   (CorDA/SC-LoRA-D-/LoRA-Null) with a call to it, so a single source definition feeds all three
   arms (handoff/14 Fix 1: "shared across all calibration-using arms").
3. `eval_matched` = **MMLU auxiliary_train + ARC-Challenge train split**, question strings only,
   pooled and truncated to 256. Both are *train/aux* splits, provably disjoint from the
   test/validation splits lm-eval scores on (verify disjointness with an id/text hash check — this
   is validation-checklist item "Disjointness" in handoff/14 §6). Concretely: `cais/mmlu`
   config `auxiliary_train` (question field) + `allenai/ai2_arc` config `ARC-Challenge` split
   `train` (question.stem). Fetch is offline-blocked here → hand the two HF pulls to the user (same
   as the nq_open prefetch that's already cached).
4. Emit `calib_source` into `summary.json` (so figures can group by it) and stamp it in the log
   line instead of the hardcoded "(nq_open calib)" strings.

Validation before trusting ANY B4 cell (handoff/14 §6, non-negotiable — this is the discipline that
caught the wikitext bug):
- Init-output invariance **after save→reload** (not in-memory): CorDA/SC-LoRA/LoRA-Null init model
  ≈ base within tol. In-memory has passed on a reloaded-corrupt model before (residual_save bug class).
- Disjointness: `calib(eval_matched) ∩ retention-test = ∅`.

---

## B4 — Calibration fairness (the load-bearing kill-shot; C2 in 03/roadmap)

**Question it settles:** is "SC-LoRA (−3.3pp) / CorDA (−3.0pp) forget MORE than their ‖ΔW‖ budget
predicts" (ANCOVA p<0.001) a REAL geometric second-order effect, or an artifact of calibrating on
nq_open (factoid QA) while evaluating on academic/reasoning tasks? KPM/null-space only protect the
directions the covariance exercises; a mismatched calib set protects the wrong subspace → the
off-curve deviation could be pure measurement bias. Until this runs, ALL "data-aware inits forget
more / LoRA beats CorDA" language is embargoed (03 §c C2).

### B4 arms (add to `ARMS` in make_campaign_jobs.py)
Three new arm keys, each = the existing arm flags **plus `--calib_source eval_matched`**:

```
"corda_r16_em":     "--method lora --corda 1 --lora_r 16 --lora_alpha 16 --calib_source eval_matched",
"sclora_r32_em":    "--method lora --sclora 1 --sclora_beta 0.5 --lora_r 32 --lora_alpha 32 --calib_source eval_matched",
"lora_null_r16_em": "--method lora --lora_null 1 --lora_r 16 --lora_alpha 16 --calib_source eval_matched",
```

The existing nq_open runs (`corda_r16`, `sclora_r32`, `lora_null_r16`) ARE the sensitivity
counterpart — the nq_open-vs-eval-matched comparison is just `_em` arms vs their existing twins at
matched (arm, LR, seed). No extra "sensitivity arm" cells are needed beyond these three `_em` arms;
the sensitivity signal = paired delta against data we already have.

### B4 cells & cost (Llama-2-CS first, the mature domain)
- 3 arms × 7 LRs × seed 42 = **21 cells**. At ≤2.5 h/cell, ceil(21/8)=3 waves → **≈ 7.5 h wall-clock
  (< 1 GPU-day). CHEAP.**
- The nq_open twins already exist (CorDA re-running clean, SC-LoRA/LoRA-Null done), so no re-run of
  the baseline is required — B4 is purely the 21 `_em` cells.

### B4 extensions (only if the CS result is ambiguous or a referee demands breadth)
- Add seed 43/44 to the 3 `_em` arms at the 2–3 near-frontier LRs (overlaps B5 — see below).
- Replicate on Llama-2-math and Qwen once those domains' baselines land. Qwen is expensive
  (6–10 h/cell): 21 Qwen-CS `_em` cells = ceil(21/8)=3 waves × ~10 h ≈ **30 h ≈ 1.25 GPU-days**.
  Defer until the L2-CS verdict is in.

### B4 analysis (what settles it)
Refit the pooled `retention ~ β·log‖ΔW‖_F` law on the calibration-free arms (LoRA, LoRA+wd, MiLoRA,
DoRA, CLoRA), then compute each data-aware arm's **signed residual from that pooled fit** under BOTH
calibrations, paired by (arm, LR, seed):
1. **If eval-matched moves CorDA & SC-LoRA residuals to ≈0 (inside the calibration-free arms'
   residual band):** the off-curve finding was a calibration artifact → honest headline = "the law
   is method-free once calibration is fair" (a CLEANER result; 03 §c C2 RISK note).
2. **If the `_em` residuals stay significantly negative:** it's a real second-order geometric
   effect that survives fair calibration → we keep a nuanced "data-aware inits forget slightly more
   even fairly calibrated" claim, now defensible.
   Test: ANCOVA with calib_source as a factor + paired t/Wilcoxon on the per-cell residual delta
   (nq_open − eval_matched), report CI on the mean shift. n=7 (×3 seeds if extended) per arm.

**B4 is the single experiment that most de-risks the strong claim** — see "De-risking" at the end.

---

## B5 — Seeds + param-matched LoRA+wd control (C3 + C4)

**Question it settles:** (i) is "LoRA+wd surpasses fancy adapters" attributable to MAGNITUDE (the wd
knob shrinking ‖ΔW‖) rather than CAPACITY? The arms mix ranks — LoRA/DoRA/CorDA **r16**;
MiLoRA/SC-LoRA/LoRA+wd **r32**; CLoRA k1024 — and ONLY LoRA carries a wd knob. (ii) Is the Pareto
win real or single-seed noise? The 3-seed mtx_ matrix already exposed collapse basins (seed 44
fragile).

### B5a — param-matched capacity controls (isolates magnitude vs capacity)
Current `lorawd_wd0p3` is r32+wd0.3. To attribute the win to magnitude not capacity we need the 2×2
of {r16, r32} × {wd0, wd0.3}, so rank and the wd knob vary independently:

```
"lora_r32":         "--method lora --lora_r 32 --lora_alpha 64",                       # r32, wd0  (capacity control, no wd)
"lorawd_r16_wd0p3": "--method lora --lora_r 16 --lora_alpha 32 --weight_decay 0.3",    # r16, wd0.3 (wd at matched-to-r16-arms rank)
# existing: lora_r16 (r16,wd0)  and  lorawd_wd0p3 (r32,wd0.3) complete the 2×2
```

This gives the clean 2×2 {r16,r32}×{wd0,wd0.3}. If retention tracks wd (i.e. ‖ΔW‖) and is flat
across rank at fixed wd, the win is magnitude, not capacity — exactly the attribution the referee
demands (03 §c C3).

Cells: 2 new arms × 7 LRs × seed 42 = **14 cells** ≈ ceil(14/8)=2 waves × 2.5 h ≈ **5 h. CHEAP.**

### B5b — optional "wd helps everyone" fairness arm (stronger version of C3)
Give the wd knob to the 2–3 arms nearest the frontier (MiLoRA, DoRA) at wd0.3, to show "wd helps
every method and geometry adds nothing on top." Add e.g. `milora_r32_wd0p3`, `dora_r16_wd0p3` by
appending `--weight_decay 0.3` to those arms.
Cells: 2 arms × 7 LRs = **14 cells ≈ 5 h.** Optional; do only if reviewers push on C3.

### B5c — seeds 43/44 on headline cells (error bars; C4)
Do NOT re-seed the whole grid. Add 43/44 ONLY to the cells that appear in a headline table/figure.
Headline set = per method, its best-adapt LR cell + one near-frontier neighbor (the points drawn in
the Pareto/gotcha exhibit). Estimate ≈ 8 arms × ~1.5 headline LRs = ~12 cells × 2 seeds = **24 cells**
≈ ceil(24/8)=3 waves × 2.5 h ≈ **7.5 h. CHEAP-MODERATE.** Set the exact LR list AFTER the Pareto/
gotcha figure (B3) identifies which points are load-bearing — otherwise you re-seed cells you never
plot. Implement by setting `SEEDS=["43","44"]` and running the generator against a headline-only
arm/LR subset (temporarily trim `LRS`/`ARMS`, or add a `--only` filter), so the resumable generator
emits just those cells.

### B5 analysis (what settles it)
- **Capacity vs magnitude (B5a):** on the {r16,r32}×{wd0,wd0.3} 2×2, regress retention on
  log‖ΔW‖_F and test whether the rank main-effect and rank×wd interaction are ≈0 once ‖ΔW‖ is in
  the model. Win-is-magnitude ⇔ rank coefficient ~0, wd acts only through ‖ΔW‖.
- **Seeds (B5c):** for each headline method, report the per-method **signed residual from the pooled
  ‖ΔW‖ fit with a 3-seed CI** (mean ± t·SE, n=3). "LoRA+wd on/above the frontier" survives iff its
  residual CI does not overlap zero the wrong way vs the fancy arms. This is the same residual test
  as B4 — one pooled-fit analysis serves both blockers.

---

## Prioritized order (single scheduler, L2-CS first)

| Pri | Item | New cells (L2-CS) | Wall-clock | Cheap? | Gates / unblocks |
|-----|------|-------------------|-----------|--------|------------------|
| **0** | `--calib_source` flag + eval_matched aux set + reload-invariance/disjointness validation | 0 (code) | few h, **no GPU** | free | prerequisite for B4; user must prefetch MMLU-aux + ARC-train |
| **1** | **B4** — 3 `_em` arms × 7 LR × s42 | **21** | **≈ 7.5 h** | **CHEAP** | LOAD-BEARING; unembargoes all off-curve language; refutes/confirms the kill-shot |
| **2** | **B5a** — param-match 2×2 (`lora_r32`, `lorawd_r16_wd0p3`) × 7 LR | **14** | **≈ 5 h** | **CHEAP** | makes "LoRA+wd wins" a capacity-fair claim |
| **3** | **B5c** — seeds 43/44 on ~12 headline cells | **24** | **≈ 7.5 h** | CHEAP-MOD | error bars; kills n=1 desk-reject; run AFTER B3 fig picks the cells |
| 4 | B5b — wd knob on MiLoRA/DoRA (optional) | 14 | ≈ 5 h | CHEAP | "wd helps everyone" stronger C3 |
| 5 | B4 seeds 43/44 on `_em` near-frontier LRs | ~18 | ≈ 6 h | CHEAP | error bars on the fairness verdict |
| 6 | B4/B5 replication on **Qwen-CS** | ~35 | **≈ 45 h ≈ 2 GPU-days** | **EXPENSIVE** | second-model generality; defer until L2 verdicts land + Qwen 2×2 drained |
| 7 | B4/B5 replication on L2-math + Qwen-math | ~50+ | **several GPU-days** | **EXPENSIVE** | dual-domain generality of the fairness/param-match result |

**Total for the cheap, load-bearing L2-CS core (Pri 0–3): ≈ 59 new cells, ≈ 20 h wall-clock
(< 1 GPU-day of the 8×B200 box).** Everything that de-risks the STRONG claim on the mature domain is
under a day of compute. The expensive tail (Pri 6–7) is second-model/second-domain breadth, not
correctness of the claim, and should wait behind the live combined pool (03 §d item 1, ~6–8 d).

Sequencing note: Pri 1–5 contend for the single scheduler and must run AFTER the live combined 2×2
pool drains (or during a deliberate pause) — never a second concurrent pool. Order fairness(B4) →
controls(B5a) → seeds(B5c) because B4 can retroactively invalidate off-curve claims that B5 would
otherwise be built to explain (03 §d sequencing note).

---

## Which single experiment most de-risks the strong claim

**B4 (Pri 1): the eval-matched calibration re-run of CorDA / SC-LoRA / LoRA-Null (21 cells, ≈ 7.5 h).**

Rationale: it is simultaneously the **cheapest** (< 1 GPU-day) and the **highest-leverage** item.
The whole thesis is "geometry is causally inert; forgetting is governed by ‖ΔW‖." The only current
evidence AGAINST method-freeness is the SC-LoRA/CorDA off-curve deviation — and that deviation is
confounded by an unfair (nq_open vs academic-eval) calibration. B4 resolves it either way:
- if the `_em` arms snap onto the curve → the law becomes *cleanly* method-free (strengthens the
  thesis and removes the top referee kill-shot, 03 §c C2 / §"attack surfaces" #2);
- if they stay off → we have a real, now-defensible second-order effect and stop overclaiming.

Neither B5a (capacity) nor B5c (seeds) can be interpreted until B4 settles what the residuals even
mean, because the pooled-fit residual is the shared analysis object for all three. B4 first.
