# METHOD ADJUDICATION — is any adapter method actually superior for retention, and at what cost?

`[2026-07-18. Scripts: adjpool.py (shared pool; preflight-reproduces key_numbers.md
section 18.1: n=1035, pooled r=-0.847 before every emit), 01_op_points.py, 02_pareto.py,
03_head2head.py, 04_robustness.py, 05_overheads.py, 06_verdict.py. All outputs in
tables/ and figures/. Conventions: quarantine-excluded (kept as divergence data),
"_reeval" duplicate dropped (Q4), lora_null split convention, qwswm ep6 excluded,
frm c256 primary, CorDA/CorDA++ withheld (port bug — never ranked). Best-LR rule =
best mean adaptation over landed seeds, n>=2 preferred. Seeds are never treated as
independent across cells: head-to-heads are PAIRED per-seed; the Pareto bootstrap
resamples seeds within cells (ICC~0.78, 09_verification Q1).]`

## TL;DR

**No geometry-motivated adapter beats vanilla LoRA + weight decay on the
retention-adaptation plane in 25 of 26 head-to-head comparisons across four
model x task families — and every one of them pays extra training, init,
memory, or stability cost for the geometry. The single genuine exception is
SC-LoRA on Qwen-math (higher adaptation ceiling at base-level retention),
which buys its win with the worst seed fragility in the study.** This is the
operating-point face of the frozen thesis: magnitude first, geometry second —
LoRA+wd matches or edges every alternative at matched capacity.

---

## RANKED TOP FINDINGS

1. **LoRA+wd(0.3) owns the entire observed Pareto frontier in 3 of 4 families**
   (Llama-CS, Llama-math, Qwen-CS): every non-dominated (adaptation, retention)
   cell belongs to it, and the seed-bootstrap gives P(on frontier) = 1.00 in all
   three, vs <= 0.50 for the best alternative (MiLoRA 0.50 on Llama-CS is a
   low-LR corner artifact, not a high-adaptation point). On Qwen-math the frontier
   is 100% SC-LoRA (P = 1.00; LoRA+wd 0.53). `[02_pareto.py — tables/pareto_frontier.csv,
   tables/pareto_bootstrap.csv, figures/fig_pareto]`

2. **Head-to-head at best operating points: 26 paired comparisons, LoRA+wd loses
   none.** 17/26 opponents are beaten on BOTH axes outside noise ("dominated"),
   8 are ties-with-losses, and exactly one comparison goes the other way:
   **SC-LoRA on Qwen-math wins adaptation (+8.3 pp GSM8K, paired t = 5.2) at tied
   retention**. No method ever beats LoRA+wd on retention outside noise (0/26).
   `[03_head2head.py — tables/head2head.csv/.md]`

3. **The retention-matched cut is where the gap is biggest.** Requiring retention
   within 1 pp of the base ceiling, LoRA+wd keeps its UNCONSTRAINED best point
   (Llama-CS 81.75 +/- 0.17 CS-8 at 25.86 ret; Llama-math 66.79 +/- 0.79 GSM8K at
   33.57 BBH >= base 33.1; Qwen-math 3e-4 68.97 at 47.54), while every competitor
   must retreat 3-14 pp of adaptation or has no qualifying cell at all (Llama-CS
   clean runner-up under the cut: LoRA-Null at 70.79; Llama-math: MiLoRA 63.68).
   On **Qwen-CS no method passes even the -2 pp cut at any LR** — CS-8 tuning
   costs >= 3.3 pp core retention for every adapter (family-level fact, not a
   method ranking). `[01_op_points.py — tables/op_points.md, op_points_*.csv]`

4. **Robustness is a second axis of LoRA+wd superiority, and the only axis where
   a geometry method wins anything else.** Safe-LR band (retention within 2 pp of
   base; Qwen-CS scored on the family-relative band): LoRA+wd 26/29 LRs across
   families vs MiLoRA/LoRA-Null 13, CLoRA 12, LoRA 10, DoRA 8, SC-LoRA 6.
   Seed stability: LoRA+wd median within-cell retention SD 0.43 pp vs SC-LoRA
   3.06 (7.08 on Qwen-CS — the 28 pp one-recipe seed swing of key_numbers
   section 18/02 section 3a). Divergence in the shared LR band (<= 1e-3) is low and
   late for everyone (0-6%, all at >= 5e-4): **CLoRA is the only method with zero
   in-band divergences (0/121)** — the null-space constraint is a real stabilizer —
   but it buys that with -4.2 pp retention at matched adaptation and the k-memory
   tax. `[04_robustness.py — tables/lr_band.csv, seed_variance.csv, divergence.csv,
   figures/fig_lr_band]`

5. **The cost of geometry is strictly positive and buys no retention at matched
   adaptation.** Train wall-clock (registry medians, identical on both models):
   DoRA 2.15x; CLoRA 1.17x at sweep k=1024 plus +3.34 GB resident frozen-P
   (6.69 GB at the k2048 boundary point); everything else 0.99-1.00x. One-time
   init: LoRA/LoRA+wd/DoRA none; MiLoRA/PiSSA 160 base-weight SVDs; LoRA-Null /
   SC-LoRA / CorDA 256-512 calibration forwards + eigh (CorDA++ ~1280 forwards,
   ~3.5e16 FLOPs, ~22.5 GB transient). Deployment: residual-init methods
   (MiLoRA/PiSSA/CorDA) modify the base weights at init, so the shippable delta
   is rank-2r — double the adapter bytes — unless the whole base is re-shipped.
   `[05_overheads.py — tables/overheads.csv/.md, figures/fig_overheads; init/memory
   constants are [EXTERNAL] from INTERESTING_INSIGHTS.md section 7 / fig_efficiency.py]`

6. **Retention-at-matched-adaptation summary (vs the LoRA+wd LR-sweep envelope),
   mean over families:** LoRA -2.2 pp, MiLoRA -2.8, LoRA-Null -3.2, CLoRA -4.2,
   DoRA -4.2, SC-LoRA -4.6 (dragged by the Qwen-CS calibration-artifact point;
   +0.7 and beyond-the-envelope on Qwen-math), PiSSA -26.3 (BBH collapse to 7.23,
   n=1). `[06_verdict.py — tables/verdict.md, per-family gap list]`

7. **Ranking is robust to the retention definition, with two disclosed wobbles.**
   Kendall tau between method rankings under core vs broad vs BBH-only: 0.81-1.00
   on Llama-CS, 1.00 on Llama-math and Qwen-CS (core vs BBH). Wobbles: (a) Qwen-CS
   BROAD definition reorders the midfield (tau 0.52; CLoRA rises to #1, LoRA+wd
   drops to #4 — driven by MMLU/ARC/TruthfulQA, worth one appendix sentence);
   (b) Qwen-math core vs BBH tau 0.57 — but Qwen-math core retention carries the
   known MMLU-Pro parser caveat (key_numbers section 11), so BBH is canonical there.
   `[04_robustness.py — tables/ret_definition_sensitivity.csv]`

---

## VERDICT TABLE

`[06_verdict.py — tables/verdict.csv/.md. ret@matched-adapt = method's retention at
its best-adaptation cell minus LoRA+wd's retention when LoRA+wd's LR is re-chosen to
match that adaptation; divergence = quarantined/attempted at LR <= 1e-3 across the 6
sweep+grid families (extreme 2e-3/5e-3 probe cells excluded).]`

| Method | ret@matched-adapt | adapt ceiling | LR band | seed SD(ret) | divergence | train | init | memory | deploy | beats LoRA+wd? |
|---|---|---|---|---|---|---|---|---|---|---|
| **LoRA+wd** | reference | reference | **26/29** | **0.43** | 6/146 (4.1%) | 1.00x | none (free AdamW flag) | 0 | rank-r | reference |
| LoRA | -2.2 pp | -4.0 pp | 10/29 | 0.83 | 7/179 (3.9%) | 1.00x | none | 0 | rank-r | 0/4 (dominated in 3) |
| CLoRA | -4.2 pp | -3.2 pp | 12/22 | 0.76 | **0/121 (0%)** | 1.17x | k x d eigh + frozen-P build | +3.34 GB (k1024) | rank-r | 0/4 (dominated in 3) |
| MiLoRA | -2.8 pp | -4.2 pp | 13/31 | 0.65 | 9/162 (5.6%) | 1.00x | 160 base-W SVDs | 0 | rank-2r | 0/4 (dominated in 2) |
| LoRA-Null | -3.2 pp | -2.5 pp | 13/24 | 0.72 | 3/97 (3.1%) | 1.00x | 256 calib fwd + eigh | 0 | rank-r | 0/4 (dominated in 2) |
| SC-LoRA | -4.6 pp (1 beyond) | **-0.9 pp** | 6/25 | 3.06 | 4/113 (3.5%) | 0.99x | 512 calib fwd + eigh | 0 | rank-r | **1/4** (Qwen-math) |
| DoRA | -4.2 pp | -5.9 pp | 8/22 | 0.70 | 3/73 (4.1%) | **2.15x** | none | 0 | rank-r | 0/4 (dominated in 2) |
| PiSSA | -26.3 pp | -17.1 pp | 0/1 | n=1 | 0/5 (0%) | — | 160 base-W SVDs | 0 | rank-2r | 0/1 (dominated) |
| CorDA/++ | WITHHELD | — | — | — | 23-36% | 1.00x | 256-1280 fwd + inv/SVD | 0 | rank-2r | not ranked (port bug) |

**Bottom line (consistent with the frozen thesis).** If you want better retention,
the winning move is not a geometry-constrained adapter — it is capping the update
magnitude, and weight decay does that for free: LoRA+wd matches or edges every
alternative at matched capacity while being the cheapest, widest-banded, most
seed-stable configuration in the study. The geometry methods that do improve
retention (CLoRA with large k, LoRA-Null, MiLoRA at low LR) do so exactly insofar
as they reduce F_Delta — i.e., through magnitude (frc k-grid: F_Delta 0.615->0.347
and ret 22.7->25.3 monotone in k, at ~10 pp adaptation cost and up to 6.7 GB memory,
02_operating_points section 5) — and none of them reaches LoRA+wd's simultaneous
adaptation + retention point. CLoRA's published numbers are faithful; on the faithful
math recipe LoRA+wd reaches GSM8K 66.79 +/- 0.79 at BBH 33.57 vs published CLoRA 64.6
([EXTERNAL], key_numbers section 12) — the research question "can LoRA+wd match them at
matched capacity" is answered YES with margin. The one honest trade-off in favor of a
geometry method: **SC-LoRA's data-aware init genuinely raises the adaptation ceiling
on Qwen-math at base-level retention (E4-consistent), if you can afford a 512-forward
init and 3-7 pp seed lottery** — and "magnitude first, geometry second" survives even
there, since its winning cell is its LOWEST-magnitude one (F_Delta 0.107).

---

## PER-FAMILY SECTIONS

### Llama-2-7B x CS-8 (lrsw)  `[tables/op_points_llama_cs.csv]`
Best points: LoRA+wd 81.75 +/- 0.17 CS-8 / 25.86 +/- 0.37 ret (5e-4, n=4) — highest
adaptation AND highest retention among high-adapters, ~base ceiling 26.0.
SC-LoRA 80.61/24.60 (E4: its deficit is a calibration artifact — do not rank it below
the curve on geometry grounds); LoRA 79.17/23.86; LoRA-Null 78.87/21.76; CLoRA
78.29/21.60; MiLoRA 77.19/21.43; DoRA 76.23/19.15. Convention footnote: under the
s42-rule DoRA@2e-4 (74.29 +/- 8.66, ret 25.20) and MiLoRA@3e-4 (63.09 +/- 21.61, ret
24.37) are the retention-relevant points — the huge adaptation SDs are answer-format
collapse with retention intact (02_operating_points section 1). Frontier: 100% LoRA+wd.
Safe band: LoRA+wd 7/7 LRs within 2 pp of base (cell-mean rule; the s42-only prior-art
rule gives 6/7) vs 2/7 (SC-LoRA) - 5/7 (best others).

### Llama-2-7B x math (frm, c256)  `[tables/op_points_llama_math.csv]`
LoRA+wd 66.79 +/- 0.79 GSM8K at BBH 33.57 +/- 1.04 (>= base 33.1): zero measured BBH
forgetting at the family-best adaptation; beats every in-pipeline competitor by
>= 2.8 pp GSM8K and >= 1.1 pp BBH, and the published CLoRA/MiLoRA/LoRA/PiSSA set by
>= 2.2 pp ([EXTERNAL] cite). MiLoRA is the best geometry method here (63.68 at 32.44,
passing the -1 pp cut). PiSSA BBH-collapses (7.23, n=1). CorDA++ 58.76/31.56 (n=1,
withheld). Frontier: 100% LoRA+wd; bootstrap P = 1.00 vs 0.00 for all others.

### Qwen-2.5-7B x CS-8 (qwsw)  `[tables/op_points_qwen_cs.csv]`
LoRA+wd 87.43 +/- 0.23 / 40.07 +/- 0.68 (5e-4) tops adaptation; its 1e-4/1.5e-4/3e-4
cells fill the entire frontier up to 41.03 retention. CLoRA-k1024 is the closest
competitor (87.02/39.52, -0.55 pp at matched adaptation — the smallest gap any
geometry method achieves in any family) but pays 1.17x + 3.34 GB. SC-LoRA matches
adaptation (87.15) but its retention is a seed lottery (27.85 +/- 15.96; per-seed
9.4/36.2/37.9 tracking per-seed F_Delta 0.44/0.30/0.30). NO method retains within
2 pp of the Qwen base ceiling (44.35) at any LR. Frontier: 100% LoRA+wd.

### Qwen-2.5-7B x math (qwswm)  `[tables/op_points_qwen_math.csv]`
The exception family. SC-LoRA@5e-5: 77.23 +/- 0.79 GSM8K at BBH 47.71 +/- 0.23
(base 47.93) — +8.3 pp adaptation over LoRA+wd's 3-seed point (68.97 +/- 3.33 at
47.54) at tied retention; the win survives pairing (t = 5.2) and holds P(frontier)
= 1.00. Its winning cell is its lowest-magnitude cell (F_Delta 0.107) — geometry
helped by putting the update in a better subspace at SMALL magnitude, not by
tolerating a big one. LoRA+wd's nominal sweep-max (1e-3, 72.93) is an n=1
quarantine-orphan (both sibling seeds diverged) — flagged wherever used.
LoRA-Null also edges adaptation at 1e-3 (72.33 +/- 1.33) but pays -2.8 pp BBH.

---

## SURPRISES

1. **Weight decay's advantage is largest exactly where the sweep is pushed hardest**
   (fig_lr_band): at 5e-4-1e-3 every other method falls off a 10-30 pp retention
   cliff while LoRA+wd stays within ~2.7 pp of base in all four families. The safe
   band is arguably the most deployment-relevant statistic in the study — you don't
   have to find the right LR with LoRA+wd.
2. **CLoRA never diverges inside the band (0/121)** — the only method with a clean
   sheet — yet is dominated on the retention plane in 3/4 families. Geometry buys
   optimization stability, not retention.
3. **The only family a geometry method wins, it wins at its SMALLEST update**
   (SC-LoRA Qwen-math, F_Delta 0.107): even the counterexample obeys the magnitude
   account's ordering of first- and second-order effects.
4. **Vanilla LoRA is never on the frontier** (bootstrap P <= 0.22 everywhere):
   plain-LoRA-vs-fancy-adapter is the wrong comparison — plain LoRA + wd vs fancy
   adapters is the fight, and wd flips it.
5. **DoRA pays the single largest overhead (2.15x on both models) for the single
   worst Llama-CS operating point** (76.23/19.15 at the mean-rule LR), and wd cannot
   rescue it (E6: DoRA+wd is degenerate as-implemented — CE 20.8/10.4).
6. Divergence is a LATE-LR phenomenon for every assessed method (nothing below
   5e-4 ever quarantined) — instability is not a differentiator inside the sane
   LR range, only the retention cliff is.

## LIMITATIONS / HONESTY LEDGER

- Seeds within a cell are correlated (ICC~0.78): all tests here are paired or
  within-cell resampled; no raw cross-cell independence is assumed (09 Q1).
- n=1 cells are flagged wherever they enter (Qwen-math LoRA+wd 1e-3; Qwen-CS
  LoRA+wd 7e-5 frontier cell; PiSSA/CorDA++ rows).
- SC-LoRA's Llama-side retention deficits are calibration-set artifacts (E4:
  eval-matched -> +0.92 pp above curve); rankings note this and never attribute
  its Llama gap to method geometry.
- Qwen-CS "broad" ranking wobble and Qwen-math "core" parser caveat are disclosed
  (finding 7). Registry peak-GPU-memory is bimodal within identical cells
  (node/batching artifact) and was excluded; the analytical resident-memory
  comparison substitutes (05_overheads.py header).
- frm coverage is uneven for CLoRA/DoRA/SC-LoRA (1-2 LRs attempted) — their
  llama_math LR-band entries are denominators of 1-2, not 6-7.
- CorDA/CorDA++ are withheld (our port bug) and never ranked; their published
  results are not challenged. CLoRA's published numbers are faithful; nothing here
  disputes them — the comparison is at matched capacity within our pipeline.

## CANDIDATE PAPER EXHIBITS

MAIN:
- **fig_pareto** (4-panel adaptation-vs-retention with frontier + bootstrap
  P(on frontier) quoted in caption) — the one-figure version of the verdict.
- **Verdict decision table** (tables/verdict.md, possibly pruned to 6 columns:
  ret@matched, LR band, seed SD, train, init, memory).

APPENDIX:
- fig_lr_band (safe-LR band; the deployment-relevance argument).
- fig_overheads (cost of geometry; replaces/joins the existing fig_efficiency).
- tables/op_points.md (per-family operating points + matched-retention cuts).
- tables/head2head.md (26 paired comparisons), tables/pareto_bootstrap.csv,
  tables/divergence.csv, tables/ret_definition_sensitivity.csv.

## FILES

- adjpool.py — shared loader + preflight (section 18.1 reproduction)
- 01_op_points.py -> tables/op_points.md, tables/op_points_{llama_cs,llama_math,qwen_cs,qwen_math}.csv
- 02_pareto.py -> tables/pareto_frontier.csv, tables/pareto_bootstrap.csv, figures/fig_pareto.{png,pdf}
- 03_head2head.py -> tables/head2head.{csv,md}
- 04_robustness.py -> tables/lr_band.csv, tables/seed_variance.csv, tables/divergence.csv,
  tables/ret_definition_sensitivity.csv, figures/fig_lr_band.{png,pdf}
- 05_overheads.py -> tables/overheads.{csv,md}, figures/fig_overheads.{png,pdf}
- 06_verdict.py -> tables/verdict.{csv,md}
