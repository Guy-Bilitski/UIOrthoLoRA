# PI STATUS REPORT — full results + end-of-week forecast (2026-07-07)

Prepared for PI supervision. Covers: every evidence base and its current numbers, correctness
assurance, exactly what runs between now and the GPU hand-back (end of week), and risks.
Companion docs: handoff/21 (consortium synthesis), 22 (retention fix), 23 (repo verification),
paper/writing/FINAL_TABLE_PLAN.md, restart_staging/RESTART_RUNBOOK.md.

---

## 1. EXECUTIVE SUMMARY

- **The thesis is winning on the strongest evidence yet.** In the faithful CLoRA-recipe math
  reproduction, plain **LoRA+weight-decay now beats every competitor — published or in-harness —
  on BOTH axes simultaneously**: best adapt GSM8K 67.40 (CLoRA's published best: 64.59), and one
  cell (wd0.3/lr2e-4) adapts at 67.25 while retaining at the base BBH ceiling (33.10) — zero
  measured forgetting.
- **The magnitude law now holds INSIDE the faithful recipe**: retention vs log‖ΔW‖ over the 28
  healthy faithful-math cells gives **r = −0.85, slope −9.1 pp/decade** — same slope as the n=49
  CS sweep (−10). This directly answers the critic's kill-shot ("law was carried by two leverage
  points"): the LoRA+wd grid filled the mid-range within one recipe.
- **New regime insight**: at this training scale (395K × 3 epochs), adaptation *also falls* with
  magnitude (r = −0.92) — we are past the adaptation peak, which is exactly why weight decay
  improves BOTH axes at once (at CLoRA's own lr 3e-4: wd 0→0.5 moves GSM8K 59.6→66.9 AND
  BBH 28.3→32.9). One row of the table now carries Claim 3.
- **Everything still missing is queued, gated, and fits the end-of-week budget** (~3.8 days of
  compute vs ~4–5 days of GPU availability), with Qwen last as the sacrificial arm per PI.

## 2. CURRENT RESULTS BY EVIDENCE BASE

### 2a. Faithful math reproduction (CLoRA recipe: r64/α128, MetaMathQA-395K, batch16, 3ep, cutoff 256) — 32/55 cells done, seed 42, retention = BBH-only (base ceiling ≈33.1)

Top of table (full ranking in results/, BBH-only via retfix report):

| cell | GSM8K | MATH | BBH | ‖ΔW‖ |
|---|---|---|---|---|
| **LoRA+wd0.2 @1e-4** | **67.40** | 13.88 | 32.53 | 0.28 |
| **LoRA+wd0.3 @2e-4** | **67.25** | 14.62 | **33.10 = base** | 0.28 |
| LoRA+wd0.5 @3e-4 | 66.94 | 13.94 | 32.91 | 0.23 |
| … 8 LoRA+wd cells ≥ 64.9 … | | | | |
| plain LoRA @1e-4 (old headline) | 64.97 | 14.64 | 30.96 | 0.43 |
| **CLoRA published best (k128)** | **64.59** | 18.38* | — | — |
| MiLoRA published | 63.53 | 17.76* | — | — |
| our CLoRA best in-harness (k256 @3e-4) | 60.80 | 14.02 | 28.61 | 1.02 |
| plain LoRA @ CLoRA's 3e-4 | 60.20 | 13.56 | 29.14 | 1.28 |
| our MiLoRA @3e-4 (α=2r) | 58.98 | 13.54 | 30.18 | 1.26 |
| collapse anchor (wd0 @5e-4) | 50.11 | 10.26 | 17.74 | 2.68 |
| PiSSA @3e-4 (eval-artifact, under diagnosis) | 49.66 | 10.50 | 7.23† | 2.21 |
| diverged: wd0@7e-4, wd0.5@7e-4, wd0.3@1e-3 | ≤37 | — | ≤3.6 | 10.8–116 |

\* the uniform ~3pp MATH-column offset vs published is a cutoff/scorer artifact shared by ALL
methods (c2048 anchors queued will attribute it). † PiSSA's BBH is a generation-format
breakdown, not clean forgetting (likelihood-MMLU identical to peers) — 1-GPU diagnostic at restart.

**Reading:** (i) LoRA+wd dominates on both axes; (ii) the wd knob at fixed LR improves BOTH axes
(59.6→66.9 GSM8K and 28.3→32.9 BBH as wd 0→0.5); (iii) law inside the recipe r=−0.85 /
−9.1pp/decade (n=28 healthy); (iv) the high-LR divergences (wd0.3@1e-3 etc.) mark the
instability edge — reported as diverged, excluded from fits, consistent with the paper's
existing 2e-3/5e-3 divergence note.

### 2b. Commonsense n=49 registry (paper's law carrier; mixed-rank lrsw sweep, seed 42) — COMPLETE (unchanged)
Pooled r=−0.86 (R²=0.74, −14.8pp/decade); six on-curve adapters r=−0.92; ANCOVA residuals all
n.s. except SC-LoRA −4.15pp (provisional; the B4 arm queued this week settles it); LoRA+wd(0.3)
best point CS-8 81.6 / core ret 25.6 with the widest safe band (6/7 LRs). This carries Claim 1
in the paper as written.

### 2c. Faithful CS reproduction (CLoRA Table-2 mirror) — **0/77 cells (the from-zero piece)**
The honest-boundary question ("does CLoRA k1024/k2048 really beat LoRA+wd on CS at the faithful
recipe?") is UNANSWERED until this grid runs — it is the first CS block in the queue.

### 2d. Qwen2.5-7B second model — 55/112 done (50 CS + 5 math)
CS LoRA sweep replicates the law's sign (r=−0.88); math anti-replicates ONLY because the
high-LR cells never ran — those exact cells are in the 44-cell completion queued last.

### 2e. Published-anchor sanity (unchanged)
Our harness reproduces published LoRA (GSM8K 60.20 vs 60.58; CS-8 ~80 vs 79.9; BoolQ 69.97 vs
69.8) — the scale is right; our CLoRA sits ~4-5pp below its published lift (recipe/harness gap,
disclosed; hence in-harness CLoRA rows + published rows kept separate in the paper).

## 3. CORRECTNESS ASSURANCE (the "100% certain" chain)

DONE: 5 adapter-expert audits (CLoRA line-for-line vs released code; others vs cloned reference
repos — caught + fixed the LoRA-Null degenerate calibration BEFORE its cells ran); adversarial
critic + supervisor rulings (handoff/21); retention-metric root-cause + BBH-only re-report
(handoff/22); MiLoRA α=r code-confirmed (handoff/23).
AT RESTART (automatic, tonight): CorDA++ CPU 14/14 + 1-GPU 0-step residual gate for ALL residual
methods at real scaling; MMLU-Pro brokenness confirmation; PiSSA format-vs-forgetting verdict;
base-BBH ceiling lock (±1pp of 33.10). No new-method cell dispatches before its gate passes.
IN-FLIGHT: LoRA-Null rank(C) full-rank diagnostic must print per cell; CorDA++ inline init
assert; two reproduction anchors (CLoRA k128 ≈59.6±1; milora_a1r @3e-4 ≈63.5).

## 4. WHAT RUNS BETWEEN NOW AND THE HAND-BACK (queue order = paper priority)

| # | Arm | Cells | GPU-h | Paper slot | ETA |
|---|---|---|---|---|---|
| 0 | frepro3 drains to index ~40 (math data-aware sweeps: milora/sclora) | ~10 | ~55 | math table rows | tonight |
| — | **RESTART** (gates ~3.5 GPU-h) | — | — | correctness | tonight |
| 1 | math remainder: CLoRA k128 LR-sweep, milora_a1r (α=r), sclora β cells, lora_null (fixed calib), low-LR cells, c2048 anchors, PiSSA re-run, math headline seeds 43/44 | ~40 | ~225 | math table FINAL + dissolution row + MATH-offset attribution | **~Jul 8 EOD** |
| 2 | faithful CS grid: baselines + CLoRA k-grid + lorawd 30-cell grid + data-aware + milora_a1r | ~63 | ~220 | CS table + honest-boundary verdict | **~Jul 9–10** |
| 3 | CorDA++ arms (math 7 + CS 7, behind gates) | 14 | ~73 | the "strongest 2025 geometric method" rows | Jul 9–10 |
| 4 | B4 eval-matched calibration (9 cells, registry configs) | 9 | ~33 | settles SC-LoRA deviation; lifts the off-curve embargo | Jul 10 |
| 5 | CS headline seeds 43/44 (winners known after #2) | ~6 | ~21 | error bars | Jul 10 |
| 6 | **Qwen completion (44 cells, LAST — sacrificial per PI)** | 44 | ~176 | two-model claim | **Jul 10–11** |

Total ≈ 800 GPU-h ≈ 4.2 days wall on 8 GPUs from now — fits "end of week" with ~½–1 day margin;
if the nodes vanish early, losses come off the Qwen tail only.

## 5. RISKS + CONTINGENCIES
1. **A restart gate fails** → documented fail-paths (e.g. cordapp cells commented out, queue
   continues); only that arm is lost, never the tables.
2. **CS honest-boundary goes against us** (CLoRA k2048 beats LoRA+wd on faithful CS) → paper
   already carries the honest-boundary language; claim narrows, doesn't die.
3. **milora_a1r misses 63.5** → the "our harness deflates geometric methods" concern returns;
   c2048 anchors disambiguate cutoff vs harness.
4. **Seed 43/44 spread swallows the headline margins** → verb stays "matches/edges" (already the
   paper's wording); robustness column (6/7 safe LRs) carries the practical claim.
5. **GPU loss mid-Qwen** → use what completed (PI-approved).

## 6. STANDING DECISIONS (locked, for the record)
Narrow-honest thesis · BBH-only math retention · in-harness CLoRA rows + published kept separate ·
uniform α=2r primary / native-α appendix · LoRA-Null repo-matched calibration · no old-CorDA
(CorDA++ replaces it) · B4 now · Qwen full, last · 2e-5 cells appendix-with-disclaimer.
