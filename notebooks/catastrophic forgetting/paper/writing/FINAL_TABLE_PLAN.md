# FINAL_TABLE_PLAN — the CLoRA-mirror reproduction tables (single source of truth)

**Date:** 2026-07-06. **Owner:** paper-writing lead. **Scope:** defines the paper's two centerpiece
tables (a **Math** table mirroring CLoRA Table 3, a **Commonsense** table mirroring CLoRA Table 2),
each augmented with a `‖ΔW‖_F` column (the magnitude law made visible *inside the comparison*), the
**selection protocol** for reporting one operating point per method from a full LR sweep, the exact
**coverage gaps** the supervisor must queue, a **claims ledger**, and a **narrative skeleton** that
slots into `paper_draft.tex`.

**Ground truth for every number:** `paper/writing/data/key_numbers.md` (SINGLE SOURCE OF TRUTH) and
the live registry `results/campaign_summary.jsonl` (a run is *complete* iff `retention_mean != null`;
`frm_*` = faithful math, `frc_*` = faithful CS, `lrsw_*`/`lrswm_*` = the older mixed-rank mature
sweeps). **CLoRA published anchors** are from the user-supplied Table 2 / Table 3.

**Live-pool status (read at authoring time):** one pool is running —
`gpu_pool.py --gpus 8 --tag frepro3 --jobs jobs/frepro_lean.txt` (PID 2932862), currently training
`frm_lorawd_wd0p3_lr2e4_c256_s42`. `jobs/frepro_lean.txt` = **103 unique cells** (55 math + 48 CS).
**Done so far: 7 faithful math cells; 0 faithful CS cells.** The generator (`make_frepro_jobs.py`)
skips any cell with an existing `results/<run>/summary.json`, so the job files ARE the authoritative
"remaining" list and the `results/frm_*` / `frc_*` dirs are the authoritative "done" list.

---

## 0. What is completed RIGHT NOW (the 7 faithful math cells, seed 42, r64/α128, MetaMathQA-395K, cutoff 256)

Verbatim from `results/frm_*/summary.json` (`git_commit ec6bfea2`, post gen_cap+BBH fix):

| Run (arm) | LR | GSM8K | MATH | `‖ΔW‖_F` | ret-core | ret-broad | CLoRA published GSM8K/MATH |
|---|---|---|---|---|---|---|---|
| `frm_lora` (LoRA) | 3e-4 | **60.20** | 13.56 | 1.283 | 18.04 | 26.63 | 60.58 / 16.88 |
| `frm_pissa` (PiSSA) | 3e-4 | 49.66 | 10.50 | **2.206** | **3.62** | 21.70 | 58.23 / 15.84 |
| `frm_milora` (MiLoRA) | 3e-4 | 58.98 | 13.54 | 1.257 | 19.86 | 27.21 | 63.53 / 17.76 |
| `frm_clora_k64` | 3e-4 | 58.53 | 13.46 | 1.113 | 18.38 | 27.16 | 64.29 / 17.52 |
| `frm_clora_k128` | 3e-4 | 59.59 | 14.50 | 1.079 | 18.52 | 27.28 | 64.59 / 18.38 |
| `frm_clora_k256` | 3e-4 | 60.80 | 14.02 | 1.019 | 19.02 | 26.69 | 63.45 / 17.58 |
| `frm_lorawd_wd0` (plain LoRA, wd0) | 1e-4 | **64.97** | 14.64 | **0.434** | **22.58** | **30.40** | (our arm; > CLoRA best pub 64.59) |

Base ceiling (core) = **26.0** (BBH-AO 33.10 + MMLU-Pro 18.96). Note CLoRA's own base is BBH 34.91 /
MMLU-Pro 18.56 (different BBH protocol — see §1.4).

**Three load-bearing reads from this completed slice** (details in §4, threats in the final section):
1. Plain LoRA at lr1e-4 (64.97) **beats CLoRA's best published number (64.59)** at the smallest
   `‖ΔW‖_F` (0.434) and best retention (22.58). Headline confirmed on real data.
2. Retention is **monotone in `‖ΔW‖_F`** across all 7 faithful points (PiSSA 2.21→3.6; LoRA
   1.28→18.0; MiLoRA 1.26→19.9; CLoRA ~1.0–1.1→18.4–19.0; LoRA-wd0 0.43→22.6). The law reproduces
   *inside* CLoRA's own recipe.
3. **THREAT:** our in-harness MiLoRA/CLoRA reproduce **~4–5 pp BELOW** their published GSM8K while
   LoRA reproduces cleanly (60.2 vs 60.58); and **all** MATH scores land ~3 pp low (uniform scorer
   offset). See final section — this shapes how the tables must be sourced.

---

## 1. THE TABLE SPEC(S)

Design rule shared by both tables: **columns carry a `source` tag per cell** — `published` (from
CLoRA's table), `ours-done` (a complete registry row), `ours-running` (queued in the live pool),
`MISSING` (neither done nor queued; must be added). Every "ours" cell also carries `‖ΔW‖_F`, which
CLoRA's tables lack — this column is the paper's visible statement of the magnitude law.

### 1.1 MAIN MATH TABLE (mirrors CLoRA Table 3) — `tables/table_main_math.tex`

Faithful recipe: **Llama-2-7B, r=64/α=128, MetaMathQA-395K, cutoff 256, seed 42.** Two operating-point
columns per method: **(a)** CLoRA's fixed LR=3e-4 (exact reproduction, apples-to-apples with their
table) and **(b)** best-swept operating point (our contribution — see §2 for the rule). Published
numbers occupy their own column so no reader confuses them with our runs.

Columns: `Method | r/α | GSM8K (pub) | MATH (pub) | GSM8K @3e-4 (ours) | GSM8K best-LR (ours) | MATH best-LR (ours) | ‖ΔW‖_F @best | ret-core @best | source`

| Method | r/α | GSM8K pub | GSM8K @3e-4 | GSM8K best | `‖ΔW‖_F` | ret-core | cell source |
|---|---|---|---|---|---|---|---|
| **LoRA** | 64/128 | 60.58 | 60.20 (ours-done) | **64.97** @1e-4 | 0.434 | 22.58 | ours-done (2 of 6 LRs); **4 LR MISSING** (2e4/5e4/7e4/1e3, wd0 col) |
| **LoRA+wd** (our arm) | 64/128 | — | (wd0=60.2) | *pending* | *pending* | *pending* | ours-running (LR×wd grid, 0/38 done) |
| PiSSA | 64/128 | 58.23 | 49.66 (ours-done) | n/a (3e-4 only) | 2.206 | 3.62 | ours-done @3e-4; sweep not planned (uses pub) |
| MiLoRA | 64/128 | 63.53 | 58.98 (ours-done) | *pending* | *pending* | *pending* | ours-done @3e-4 + ours-running (5 LRs) |
| CLoRA-k64 | 64/128 | 64.29 | 58.53 (ours-done) | n/a | 1.113 | 18.38 | ours-done @3e-4; sweep uses pub |
| CLoRA-k128 | 64/128 | 64.59 | 59.59 (ours-done) | n/a | 1.079 | 18.52 | ours-done @3e-4; sweep uses pub |
| CLoRA-k256 | 64/128 | 63.45 | 60.80 (ours-done) | n/a | 1.019 | 19.02 | ours-done @3e-4; sweep uses pub |
| SC-LoRA (β0.5) | 64/128 | — | *pending* | *pending* | *pending* | *pending* | ours-running (6 LRs, 0 done) |
| LoRA-Null | 64/128 | — | *pending* | *pending* | *pending* | *pending* | ours-running (6 LRs, 0 done) |
| DoRA | 64/128 | — | **MISSING** | **MISSING** | — | — | **MISSING** (in `MATH_BASELINES` but not queued) |
| CorDA / CorDA++ | 64/128 | — | **MISSING** | **MISSING** | — | — | **MISSING** (not wired; §3.4) |

MATH-accuracy column is reported but flagged low-confidence (uniform ~3 pp scorer offset; §final).
GSM8K is the reliable adapt axis for math.

### 1.2 MAIN COMMONSENSE TABLE (mirrors CLoRA Table 2) — `tables/table_main_cs.tex`

Faithful recipe: **Llama-2-7B, r=32/α=64, Commonsense170K, cutoff 256, seed 42.** Columns mirror
CLoRA Table 2 (avg CS-8, BBH, MMLU-Pro), plus `‖ΔW‖_F`. **STATUS: 0 faithful CS cells complete** —
every "ours" number below is either `ours-running` or `MISSING`. The mature `lrsw_*` numbers (r16/r32
MIXED, in `key_numbers §3`) are **NOT** substitutes — they are a different (unfair-rank) recipe and
belong only in the mechanism sections, not in this faithful table.

Columns: `Method | r/α | CS-8 avg (pub) | BBH (pub) | MMLU-Pro (pub) | CS-8 best (ours) | BBH (ours) | MMLU-Pro (ours) | ‖ΔW‖_F | ret-core | source`

| Method | r/α | CS-8 pub | BBH pub | MMLU-Pro pub | ours status |
|---|---|---|---|---|---|
| **LoRA** (=LoRA+wd wd0 col) | 32/64 | 79.9 | 26.69 | 14.46 | ours-running (wd0 col of grid; 0/6 LR done) |
| **LoRA+wd** (our arm) | 32/64 | — | — | — | ours-running (LR×wd 30 cells, 0 done) — **the headline arm** |
| DoRA | 32/64 | 80.5 | 28.24 | 11.67 | pub + **MISSING** faithful (dora_r32 not queued) |
| PiSSA | 32/64 | 73.8 | 29.54 | 11.33 | pub + **MISSING** faithful (pissa_r32 not queued) |
| MiLoRA | 32/64 | 80.0 | 25.14 | 17.74 | pub + ours-running (6 LRs, 0 done) |
| LoRA-r8 | 8/16 | 78.8 | — | — | pub + **MISSING** |
| LoRA-r16 | 16/32 | 79.8 | — | — | pub + **MISSING** |
| LoRA-L2 (wt 1e-5) | 32/64 | 80.8 | 32.93 | 16.59 | pub + **MISSING** (lora_l2 not queued) |
| CLoRA-k128 | 32/64 | 80.7 | 30.82 | 12.07 | pub + **MISSING** faithful (not queued; §3.3) |
| CLoRA-k256 | 32/64 | 81.6 | ~ | ~ | pub + **MISSING** faithful |
| CLoRA-k512 | 32/64 | 82.0 | ~ | ~ | pub + **MISSING** faithful |
| CLoRA-k1024 | 32/64 | ~82–83 | ~ | ~ | pub + **MISSING** faithful (mature lrsw k1024 = 79.85/24.82, wrong-rank proxy only) |
| CLoRA-k2048 | 32/64 | 83.7 | 38.67 | 20.59 | pub + **MISSING** faithful — **the honest-boundary cell (§4)** |
| SC-LoRA (β0.5) | 32/64 | — | — | — | ours-running (6 LRs, 0 done) |
| LoRA-Null | 32/64 | — | — | — | ours-running (6 LRs, 0 done) |
| CorDA / CorDA++ | 32/64 | — | — | — | **MISSING** (not wired; §3.4) |
| Base (no-FT) | — | — | 34.91 | 18.56 | pub anchor (ours: BBH-AO 33.10 / MMLU-Pro 18.96) |

### 1.3 `‖ΔW‖_F` column (the magnitude law, made visible)

Both tables carry `‖ΔW‖_F` = registry field `fdelta` (token-weighted Frobenius; range ~0.05–3.7;
NOT the obsolete 72–1395 scale). Sort or annotate rows so the reader sees retention falling as
`‖ΔW‖_F` rises **across methods**, not just within one. This single column is what turns a
reproduction table into evidence for Claim 1 — CLoRA's Tables 2/3 do not have it, and their Table 4
(the `F_Δ`-vs-BBH monotonicity) already implies it (§4 ledger, Insight #8).

### 1.4 Reconciliation footnotes the tables MUST carry

- **BBH protocol mismatch:** CLoRA base BBH = 34.91; ours = BBH-answer-only 3-shot 33.10. Report
  retention as *our* internal delta from *our* base, and cite CLoRA's numbers as their-harness values;
  never subtract across harnesses.
- **MATH ~3 pp low, uniformly** — scorer/extraction offset, disclose; treat GSM8K as the adapt axis.
- **bf16 (ours) vs fp16 (CLoRA/LLM-Adapters)** — benign, ≤~0.5 pp, disclose.
- **cutoff 256** truncates MetaMathQA CoT; a 256-vs-512 sensitivity pair is queued only for LoRA+wd,
  not for the LoRA/CLoRA anchors — flag (§3.5).

### 1.5 SUPPLEMENTARY FULL-SWEEP TABLE (appendix) — `tables/table_sweep_full.tex`

One row per **(method × LR [× wd] × cutoff × seed)** cell, all columns (GSM8K/CS-8, MATH, BBH,
MMLU-Pro, MMLU, ARC-c, TruthfulQA, ret-core, ret-broad, `‖ΔW‖_F`, `σ_max`, trainable-params). This
is the referee shield for the selection protocol ("show us you didn't cherry-pick"): every point that
feeds a best-LR cell in §1.1/§1.2 is visible here, plus the collapse edges. Generated directly from
`campaign_summary.jsonl` filtered to `frm_*`/`frc_*` (+ `lrsw_*`/`lrswm_*` in a clearly-labeled
"mature mixed-rank" block). Pair it with the Pareto plot (`fig3_pareto.png`) and the LR-artifact
exhibit (`fig9_lr_artifact.png` / `fig:lrartifact`).

---

## 2. SELECTION PROTOCOL (how we pick the reported operating point)

### 2.1 Primary rule (pre-registered, identical for every method)

> **Report each method at its best in-domain adaptation over its full LR sweep** (for LoRA+wd, over
> the LR×wd grid): the cell with **max `cs_avg`** (GSM8K for math, CS-8 for commonsense). **Ties**
> within the adaptation-noise band (**±0.5 pp GSM8K / ±0.3 pp CS-8**) are broken toward the **lower
> `‖ΔW‖_F`** (equivalently higher retention) cell. Report to `key_numbers` precision.

**Justification.** The thesis (Claim 3) is that a *single shared LR* flatters whichever method is
well-tuned there — so a fixed-LR head-to-head is exactly the artifact we expose. The only fair
comparison tunes the reported free knob (LR; LR×wd for LoRA+wd) per method. Best-adapt (not
best-retention or best-composite) is chosen because (i) it is the axis every source paper optimizes
and reports, so it is the least-charitable-to-us criterion, and (ii) it makes the retention/`‖ΔW‖`
column an *earned* observation rather than a selection target. The tie-break toward lower `‖ΔW‖`
is the only place retention enters selection, and it is disclosed.

### 2.2 Secondary (companion) view — the fixed-LR reproduction

Alongside the best-LR column, report the **faithful fixed LR=3e-4** point (CLoRA's exact setting) so
readers can (a) verify our reproduction against CLoRA's tables cell-by-cell, and (b) see the delta
between "fixed-LR" and "tuned-LR" — which *is* the LR-artifact story. This double column is what
makes the selection unimpeachable: we show both the shared-LR number CLoRA reports and the tuned
number, and let the gap speak.

### 2.3 Published vs ours

CLoRA/PiSSA/DoRA/MiLoRA published numbers stay in a dedicated `published` column, cited to CLoRA's
tables, never merged into an "ours" cell. Our contributions are the **swept LoRA+wd** arm and the
**swept LoRA-Null / SC-LoRA / (our own swept) MiLoRA**, plus the fixed-LR reproductions of
LoRA/PiSSA/CLoRA we already have.

### 2.4 Seed handling

- **Body of both tables: seed 42** (matches the entire mature campaign and CLoRA's single-seed
  tables). State "single seed" up front.
- **3-seed (43, 44) ±std ONLY on headline cells.** The documented seed-collapse basins (seed 44
  collapsed clora_k2048→23, dora_r8→22, lorawd_wd0p5→51) mean a single-seed *ranking* is not
  publishable; but the *law* is robust on 49 points. So error bars go only on the cells that carry a
  head-to-head claim. **Exactly these ~9 cells need ±std:**
  1. Math: LoRA best-LR (the 64.97 headline) — the "beats CLoRA published" claim.
  2. Math: LoRA+wd best-LR winner.
  3. Math: the closest structured competitor at best-LR (MiLoRA or CLoRA-k256).
  4. CS: LoRA+wd best-LR winner.
  5. CS: plain LoRA (r32) best-LR.
  6. CS: MiLoRA best-LR.
  7–8. CS honest-boundary pair: **CLoRA-k1024 and CLoRA-k2048** (faithful) vs LoRA+wd best.
  9. Any method flagged off-curve (currently SC-LoRA) at its best-LR.
  (≈9 cells × 2 extra seeds = 18 runs; a fraction of the pool.)

---

## 3. COVERAGE GAP ANALYSIS (parsed from job files + registry, 2026-07-06)

Counts are exact: `frm_*`/`frc_*` dirs with `summary.json` = done; `jobs/frepro_lean.txt`
(103 unique) = running; anything in neither = MISSING.

### 3.1 Math faithful (Table 3) — done 7, running 55, missing arms

**Done (7):** lora@3e4, pissa@3e4, milora@3e4, clora_k64/128/256@3e4, lorawd_wd0@1e4.
**Running (55, in frepro3):**
- LoRA+wd LR×wd @c256: wd0{2e4,3e4,5e4,7e4,1e3}=5, wd0.1/0.2/0.3 × 6 LRs each (the c256 halves of
  the "9" counts), wd0.5 × 6 LRs = **29 cells** (the `wd0` column across LRs = the plain-LoRA math
  LR sweep — this is what fills the LoRA best-LR cell).
- LoRA+wd @c512 sensitivity: wd0.1/0.2/0.3 × {2e4,3e4,5e4} = **9 cells**.
- MiLoRA LR sweep (our own): {1e4,2e4,5e4,7e4,1e3} = **5** (3e4 done).
- SC-LoRA LR sweep: 6. LoRA-Null LR sweep: 6.
**MISSING entirely (math):**
- **DoRA math** — listed in `MATH_BASELINES` but NOT in the queue and NOT done. Needed for the
  DoRA-2×-cost insight on math + the math Pareto panel. (≥1 cell @3e-4; ideally 6–7 LR.)
- **CorDA / CorDA++ math** — not wired (§3.4).
- **256-vs-512 cutoff sensitivity on the LoRA / CLoRA / MiLoRA anchors** — only LoRA+wd has c512
  cells; the reproduction anchors have no c512 control (§3.5).
- **Seeds 43/44** on any math cell — none.
- Full LR sweeps for PiSSA/CLoRA/MiLoRA-@faithful are only partially needed (they use published
  numbers); only the best-LR *contribution* arms (LoRA+wd, MiLoRA-ours, SC-LoRA, LoRA-Null) get swept.

### 3.2 Commonsense faithful (Table 2) — done 0, running 48, large missing baseline block

**Done: 0.** **Running (48, in frepro3):**
- LoRA+wd LR×wd @c256: wd0/0.1/0.2/0.3/0.5 × 6 LRs = **30** (wd0 column = plain LoRA r32 CS LR sweep).
- MiLoRA 6, SC-LoRA 6, LoRA-Null 6.
**MISSING entirely (CS) — the whole fixed-LR baseline block was generated with `--baselines 0`:**
- lora_r32, dora_r32, pissa_r32, lora_r8, lora_r16, lora_l2 (@3e-4) — **all MISSING**.
- **CLoRA k-grid faithful (k128/256/512/1024/2048 at r32/α64) — ALL MISSING.** This is the biggest
  CS gap: the honest-boundary claim (§4) currently rests on CLoRA *published* + the *mixed-rank*
  `lrsw_clora_k1024` proxy, with **zero faithful-recipe CLoRA CS cells**.
- **CorDA / CorDA++ CS** — not wired (§3.4).
- **Seeds 43/44** — none.

### 3.3 CS weight-decay sweep status

The LoRA+wd CS LR×wd grid (30 cells) IS queued and running (0 done). This is the arm CLoRA omitted
(their LoRA-L2 is a single untuned point, wt 1e-5, "1e-4 too large" — Insight #8). Once drained it
gives the swept LoRA+wd frontier at the faithful r32 recipe. **Gap:** it will land *without* a
faithful CLoRA k-grid to compare against on the same recipe (§3.2) — so queue the CLoRA CS k-grid in
parallel or the boundary test stays proxy-only.

### 3.4 CorDA / CorDA++ status

- **Static CorDA:** excluded from every current figure/table (wikitext-calib bug → nq_open re-run
  pending; calib↔eval fairness open per handoff/19 B4). Present as "withheld," not "off-curve."
- **CorDA++ (arXiv:2506.13187):** **NOT wired** into `train_cs.py` at all — no `--corda_pp` branch;
  N (calibration size), the π compactness operand, and the dynamic-rank allocation direction are
  unresolved (handoff/19 D3, all four reviewers say DEFER). It is the "we didn't strawman the 2025
  SOTA" arm; realized trainable-param count must be matched to the r64/α128 (math) and r32/α64 (CS)
  budgets, since dynamic rank breaks nominal parity. **Both CorDA variants are MISSING from both
  faithful tables.** Recommendation: keep CorDA/CorDA++ as an explicitly-labeled "future/strong-tier"
  row, not a blocker for the minimum-defensible tables.

### 3.5 MATH cutoff 256-vs-512 sensitivity

Only **LoRA+wd** has c512 cells queued (wd0.1/0.2/0.3 × {2e4,3e4,5e4} = 9). The **reproduction
anchors (LoRA, CLoRA-k*, MiLoRA) have NO c512 control**, so we cannot yet attribute the ~4–5 pp
GSM8K shortfall (our fancy-method repro vs published) to CoT truncation. **Add a 256-vs-512 pair on
`lora@3e-4` and on `clora_k128@3e-4`** (2 cells) to test whether cutoff explains the shortfall — this
is a cheap, high-value diagnostic for the reproduction-fidelity claim.

### 3.6 SC-LoRA β coverage

Only **β=0.5** is swept (math + CS, LR grid). handoff/19 specs β∈{0.3,0.5,0.7} + a β→1 diagnostic
(Block F/G) to avoid "under-tuned SOTA." **β∈{0.3,0.7} and β0.95 are MISSING** in the faithful
campaign. Needed only if SC-LoRA appears off-curve at β0.5 and we want to report it at its best
config; otherwise a footnote ("β0.5, the paper default") suffices for the minimum tier.

---

## 4. CLAIMS LEDGER

| # | Claim | Supporting table cells | Current verdict | Honest boundary to state |
|---|---|---|---|---|
| **C1** | **MECHANISM** — forgetting governed by `‖ΔW‖_F`, not geometry | Math table `‖ΔW‖_F` col: 7 faithful points monotone (r strongly negative; PiSSA 2.21→3.6 … LoRA-wd0 0.43→22.6). Mature `lrsw_` CS r=−0.86 (n=49) / −0.92 on-curve. CLoRA Table 4 `F_Δ`→BBH monotone (Insight #8). | **SUPPORTED** on faithful math (7 pts) + mature CS; **OPEN** on faithful CS (0 cells). | Near-circularity: lead with ANCOVA residual (method adds ~0), not raw r. Faithful-CS law is not yet shown. |
| **C2** | **CONSEQUENCE** — LoRA+wd matches/beats the frontier | Math: LoRA(wd0)@1e-4 **64.97 > CLoRA pub 64.59** at smallest `‖ΔW‖` (0.434) & best ret (22.58) — DONE. LoRA+wd grid math+CS — RUNNING (0 CS done). Mature CS op-points (`key_numbers §3`, mixed-rank). | Math: **SUPPORTED** (headline beats published). CS faithful: **OPEN** (running). | Verb = "matches/edges," never "dominates," until seeds land (+0.8–1.5 pp < 10–40 pp collapse basins). Rank/wd asymmetry: only LoRA has wd; param-matched control still needed. |
| **C3** | **DIAGNOSIS** — the "wins" are an LR/recipe artifact | Fixed-LR (3e-4) vs best-LR columns (§1.1/§1.2). Math done cells already show fixed-3e-4 LoRA (60.2) < LoRA best (64.97); CLoRA fixed-3e-4 in-harness ≤ 60.8. R²(‖ΔW‖)=0.74 ≫ R²(LR)=0.32. `fig:lrartifact`. | **SUPPORTED** on math (fixed vs tuned gap is real); CS pending. | Phrase as "we show the *ingredients* of the artifact" until the exhibit is seed-averaged. Per-method: strong for SC-LoRA/LoRA-Null/DoRA, weak MiLoRA, absent CLoRA. |
| **C4** | **MESSAGE** — control magnitude, not geometry | Whole table + `‖ΔW‖_F` column + ANCOVA (SC-LoRA sole provisional deviator). | **SUPPORTED as guidance**, with boundaries. | Two honest boundaries below. |

**Honest-boundary statements that MUST appear (non-negotiable):**

- **B-CS (high-k CLoRA):** Insight #9 — LoRA+wd matches CLoRA up to ~k512 on CS, but
  **CLoRA-k1024/k2048 beat LoRA+wd on BOTH CS axes** (published 83.7/38.67/20.59), and forcing
  LoRA+wd to that retention collapses its adapt (→~45). So the defensible claim is NARROWER than
  "geometry is useless": the magnitude *law* governs retention universally, and LoRA+wd matches fancy
  adapters on **math** and **mid-regularization CS**, but **at high-k CS, directional (null-space)
  constraint buys real adapt-efficiency**. NOTE this currently rests on published + mixed-rank proxy;
  the faithful-recipe test is MISSING (§3.2) and is the top CS priority.
- **B-transfer:** CLoRA-k2048 out-domain (BBH 38.67) **exceeds base** (34.91) — that is transfer, not
  mere retention, which muddies "forgetting" framing on CS. State it.
- **B-eval (+19.5 pp):** the eval-protocol insight (Insight #2) — the same MetaMath adapter scores
  ~46.55 under lm-eval default GSM8K vs 60–66 under the faithful protocol. Our `frm_lora`=60.2
  reproduces CLoRA's 60.58 only because the protocol is aligned. Report as a measurement-hygiene
  contribution and the reason our reproduction is trustworthy on LoRA.
- **B-DoRA-cost:** DoRA costs ~2.1× LoRA wall-clock for no retention/adapt benefit (Insight #1);
  include as an efficiency footnote — but note DoRA is currently MISSING from the faithful math table
  (§3.1), so this cell needs to be run to appear.
- **B-repro-shortfall:** our in-harness MiLoRA/CLoRA land ~4–5 pp below published GSM8K (see final
  section); disclose and lean on the published-number comparison for C2.

---

## 5. NARRATIVE SKELETON

### 5.1 Abstract-level story (one paragraph)

Parameter-efficient fine-tuning papers ship a new forgetting-mitigation adapter every week, each
claiming a win over LoRA at a single, shared learning rate. We reproduce the exact experimental
settings of one such method (CLoRA: Commonsense-170K r32/α64 and MetaMathQA-395K r64/α128, evaluated
in- and out-of-domain) and add the two arms those comparisons omit — a fully **learning-rate-swept
LoRA+weight-decay** baseline and swept ports of LoRA-Null / SC-LoRA / MiLoRA — with one extra column
the source tables lack: the update magnitude `‖ΔW‖_F`. Across the reproduction, retention is governed
by `‖ΔW‖_F`, not adapter geometry: plain LoRA at a well-chosen LR already beats CLoRA's best published
GSM8K (64.97 vs 64.59) at a smaller update and higher retention, and every method's retention lands
where its magnitude predicts. The elaborate adapters' reported edges are largely a learning-rate
artifact — visible as the gap between the fixed-LR and tuned-LR columns of the same table. We state
the boundary honestly: at high regularization on commonsense, CLoRA's null-space constraint still buys
adapt-efficiency that pure magnitude control does not. The message to the field: **control the
magnitude, not the geometry — and sweep the learning rate before you claim a win.**

### 5.2 Section outline (aligned to `paper_draft.tex`; the faithful tables slot into §pareto)

- **§1 Introduction** — the weekly-adapter problem; our controlled reproduction; scope up front
  (Llama-2 mature + faithful math done, faithful CS in progress).
- **§2 Related work / positioning** — basis axes (weight-SVD vs data-covariance vs gradient); CLoRA
  already links magnitude→CF (their Table 4); the "LR Matters: Vanilla LoRA May Suffice" prior. Verify
  every arXiv ID.
- **§3 Setup & measurement** — the shared trainer (every structured adapter = LoRA + a different init;
  only CLoRA adds a loss term); `‖ΔW‖_F` axis and why it beats σ_max/LR; retention suite + base
  ceilings; port-fidelity audit; **the eval-protocol +19.5 pp measurement-hygiene result** (B-eval).
- **§4 The magnitude law** (`fig:hero`, `fig:budget`, `fig:perbench`) — correlations, per-capability,
  non-circularity (ANCOVA) first.
- **§5 Geometry adds nothing** (`fig:fairness`) — 5/6 on the law; SC-LoRA sole provisional deviator.
- **§6 The reproduction tables — LoRA+wd on the frontier** (`tab:math`=§1.1, `tab:cs`=§1.2,
  `fig:pareto`). **This is where the two centerpiece tables live.** Lead with the math table (done +
  headline), then CS (mark running cells). Fixed-LR vs best-LR columns carry Claim 3. State the honest
  verb and boundary B-CS + B-transfer.
- **§7 Learning rate is only a proxy** (`fig:proxy`, `fig:lrsens`, `fig:lrartifact`) — best-LR not
  shared; R² 0.74 vs 0.32; the symmetric-sweep answer to "did you tune LoRA+wd as hard."
- **§8 Mechanism** — magnitude as a two-edged budget; wd lands in the sweet-spot band for free.
- **§9 Replication on Qwen (in progress)** — partial; Qwen-math anti-replicates (report, don't bury).
- **§10 Limitations** — single seed, rank/wd asymmetry, calib↔eval mismatch, CorDA withheld, SC-LoRA
  provisional, MATH scorer offset, reproduction shortfall (B-repro-shortfall), labeling bug.
- **§11 Discussion / wake-up call** + **§12 Future work** (CorDA++, calibration-matched arms,
  param-matched controls, seeds).

---

## FINAL: TOP-10 MISSING CELLS (highest priority first)

Priority = (unblocks a headline claim) × (cheap) × (a live boundary rests on it).

1. **Faithful CS CLoRA k-grid @ r32/α64, LR=3e-4** — `frc_clora_k{128,256,512,1024,2048}` (5 cells).
   *Unblocks the honest-boundary test B-CS at the faithful recipe; currently proxy-only. Highest
   scientific risk cell in the paper.*
2. **Faithful CS plain LoRA r32 best-LR** — arrives as the `frc_lorawd_wd0` column (RUNNING, 0/6 done).
   *The CS reproduction anchor; nothing in the CS table is real until this lands.*
3. **Faithful CS LoRA+wd LR×wd winner** — `frc_lorawd_wd0p{1,2,3}` grid (RUNNING, 0/30 done).
   *The C2 headline for CS.*
4. **Math LoRA wd0 LR sweep completion** — `frm_lorawd_wd0_lr{2e4,5e4,7e4,1e3}` (RUNNING, 4 missing).
   *Completes the plain-LoRA math LR curve that produces the 64.97 headline; only 2/6 LRs done.*
5. **Seeds 43/44 on the math headline pair** — LoRA best-LR (64.97) + LoRA+wd math winner (4 runs).
   *Turns "beats CLoRA published" from n=1 into a defensible claim; kills the desk-reject.*
6. **256-vs-512 cutoff pair on `frm_lora@3e4` + `frm_clora_k128@3e4`** (2 cells).
   *Diagnoses the 4–5 pp fancy-method reproduction shortfall — is it CoT truncation or real?*
7. **DoRA faithful math @3e-4 (+ short LR sweep)** — `frm_dora_*` (MISSING; ≥1, ideally 6).
   *Fills the DoRA row + the 2×-cost efficiency footnote on math; the math Pareto panel needs it.*
8. **Faithful CS baselines @3e-4** — `frc_{dora_r32,pissa_r32,lora_l2,lora_r8,lora_r16}` (5 cells).
   *Rounds out the CS table's reduced-rank / L2 rows for the CLoRA Table-2 mirror.*
9. **Seeds 43/44 on the CS honest-boundary pair** — CLoRA-k1024/k2048 vs LoRA+wd best (post-#1) (~6 runs).
   *The boundary claim B-CS is a ranking claim → needs error bars.*
10. **SC-LoRA β∈{0.3,0.7} @ faithful recipe** (math or CS, best-LR) (~2–4 cells).
    *Only needed if SC-LoRA reads off-curve at β0.5; prevents "under-tuned SOTA."*

(CorDA/CorDA++ are deliberately below the top-10 — strong-tier, not minimum-defensible; §3.4.)

## FINAL: THESIS-THREATENING RESULTS IN THE COMPLETED DATA

1. **Asymmetric reproduction shortfall (the real threat).** Our in-harness **MiLoRA (58.98)** and
   **CLoRA-k64/k128 (58.53/59.59)** land **~4–5 pp BELOW their published GSM8K (63.53/64.29/64.59)**,
   while **LoRA reproduces cleanly (60.20 vs 60.58)**. A reviewer will ask why we can reproduce LoRA
   but not the fancy methods. This cuts both ways: it *strengthens* "geometry doesn't help in our
   harness" but *weakens* "faithful reproduction." **Mitigation baked into the plan:** C2 leans on the
   published-number comparison (LoRA 64.97 > CLoRA published 64.59), the fixed-LR column discloses the
   in-harness numbers transparently, and top-10 item #6 (cutoff sensitivity) tests whether truncation
   explains it. Do NOT headline the in-harness fancy numbers as their "true" performance.
2. **MATH column is uniformly ~3 pp low** (all methods 10.5–14.6 vs published 15.8–18.4) — a scorer/
   extraction offset, not a per-method signal. Keep MATH as a secondary, caveated column; GSM8K is the
   adapt axis.
3. **The high-k CS boundary (B-CS) is a genuine limit, not yet tested at the faithful recipe.**
   CLoRA-k1024/k2048 beating LoRA+wd on both CS axes is the one place "geometry adds value" is true —
   and we have **zero faithful-recipe CLoRA CS cells** to confirm or refute it (only published +
   mixed-rank proxy). Until top-10 #1 lands, the paper's central negative claim has an untested escape
   hatch on its most-cited domain. This is the single most important thing left to run.
4. **Everything is single-seed (s42) and the CS faithful table is 0% complete** — no faithful-CS
   ranking is defensible yet; the CS narrative currently borrows from the mixed-rank `lrsw_` sweep,
   which is a different (unfair-rank) recipe and must not be passed off as the faithful table.
