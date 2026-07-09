# THE RESEARCH PLAN — post-fleet synthesis (2026-07-09, research-planner agent)

Inputs: 9-agent review fleet (5 paper experts on the PI-supplied PDFs + 4 section validators), queue
ground-truth, key_numbers.md, FINAL_TABLE_PLAN.md. Full findings: scratchpad fleet_findings.md +
tasks/<id>.output transcripts. Deadline: GPUs returned ~Jul 12-13.

## Queue reality (validated)
- Faithful MATH frm_ = 46/46 DONE (+ method rows/β cells in flight). **Faithful CS frc_ = 0 done — the
  65-cell reservoir + frepro4_cs queue IS the paper's spine.** Priority = LoRA+wd grid → CLoRA k-grid →
  MiLoRA/a1r/SC-LoRA/LoRA-Null → CorDA++ → baselines (incl. frc_lora_l2, the PI's LoRA-L2 side-by-side).
- 3 of the fleet's asks were ALREADY QUEUED (SC-LoRA β0.8/0.9 sweep; matched-α LoRA/MiLoRA cells =
  analysis-only once frc lands; CS seed pairs). 390/402 saved adapters retain weights → all CPU
  analyses unblocked.

## (A) NEW GPU cells — INJECTED (jobs/frepro4_inject.txt, ~25 GPU-h, funded ahead of Qwen tail)
1. frc_lorawd wd0/wd0.2/wd0.3 @2e-5 — answers CorDA++'s "LoRA catastrophically forgets at 2e-5"
   (frc_cordapp_lr2e5 already queued → complete comparison).
2. frc_cordapp_a1r @2e-5 (α=r) — the one paper-faithful CorDA++ anchor; removes the α=2r confound.
3. frc_sclora_b0p9_em_r128 @5e-5 — fully paper-faithful SC-LoRA (recommended β + eval-matched calib +
   their rank); tests whether the −4.15pp deviation survives their own recipe.
Dispatch: to first freed GPUs (headline pool exits tonight), before reservoir chunks. Qwen = last
(effectively sacrificial; PI-approved ordering unchanged).

## (B) CPU/eval-only THIS WEEK (parallel, no GPU contention)
- Geometry drift (PI): principal angles of ΔW vs base-W subspaces, per layer/model/adapter, over the
  390 saved adapters. ~1 day.
- Efficiency (PI): taxonomy audited (no-precompute: LoRA/LoRA+wd/DoRA; weight-SVD: MiLoRA/PiSSA;
  SVD+calib: LoRA-Null/SC-LoRA/CorDA; CLoRA per-step reg cost) + wall-clock scraped from logs. ~0.5 day.
- Matched-α LoRA-vs-MiLoRA decomposition (uses queued frc cells; their published design confounds
  α=2r-LoRA vs α=r-MiLoRA). ~2h once cells land.
- CE-to-base (WikiText-103, MiLoRA Table-8 metric) over saved adapters → plots our F_Δ on their axis.
- Cross-literature law figure: CLoRA Table 4 (r=−0.98, slope −14.7 vs ours −14.8!) + MiLoRA T7/8 +
  LoRA-Null T4b + CorDA++ TVII overlay. Numbers already extracted verbatim in fleet briefs.
- DEFER: CLoRA harness-attribution rescore (footnote suffices if time runs out).

## (C) PAPER actions next week (no GPU)
1. Relabel fdelta → F_Δ EVERYWHERE (key_numbers done; paper.tex/analyze_matrix.py/figures pending) —
   a CLoRA-reading reviewer catches the Frobenius mislabel; F_Δ comparability is a strength.
2. Law figure: saturating fit + ceiling line (asymptote 26.8) + knee (0.36 ≈ LoRA+wd op point) + below-
   ceiling-slope inset (−21 vs pooled −14.8). Stats: Spearman −0.896, quad p=0.004, partial-r −0.868,
   permutation p<5e-5, per-method slope CIs (SC-LoRA sole outlier −26 [−33,−19]).
3. Operating-point table: + usable-band (cs≥75 AND ret≥24: LoRA+wd 3/7, CLoRA/LoRA-Null 0/7),
   mean-retention, efficiency columns; Pareto plot.
4. Borrowed evidence + citations: MiLoRA Table 8 + LoRA-Null Table 4b; Kalajdzievski 2024; verbatim
   single-shared-LR quotes (MiLoRA/CLoRA/LoRA-Null/CorDA++; say "MOST prior comparisons" — SC-LoRA
   tuned baselines in 2/3 experiments).
5. Honest boundaries: high-k CLoRA (pending faithful k-grid verdict), SC-LoRA scoping (updated by A3),
   Qwen-math pending, repro shortfall, MATH scorer offset, single-seed.

## (D) Deferred / dropped
- SC-LoRA init-erosion probe (needs unsaved per-step checkpoints → post-submission).
- Qwen tail (sacrificial; CS replicates r=−0.86 core, math anti-replicates — present as in-progress).
- DoRA/PiSSA full LR sweeps (single 3e-4 points suffice for the Table-2 mirror).
- c512 sensitivity on non-LoRA math anchors (run only if a GPU idles).

## Key fleet verdicts recorded
All 5 ports FAITHFUL (CorDA++ cells cleared; KPM = correct mode). fdelta = CLoRA's F_Δ (NOT Frobenius)
— fixed. CLoRA "MMLU" column = MMLU-Pro (comparison commensurable). LoRA-L2 = L2 on params, coeff 1e-5,
norm-not-direction penalty (mechanism loss-vs-optimizer unspecified). All 26 published numbers in the
supervision report verified against the PDF. Ceiling package computed & integrated (report).
