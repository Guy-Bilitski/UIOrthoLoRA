# handoff/33 — QUEUE ALIGNMENT AUDIT before the Sunday-morning data freeze (2026-07-11 19:42, research-planner)

Every count below was recomputed live this pass from `results/*/summary.json`, `/scratch/cf_models`,
`jobs/master_dispatch.txt`, `jobs/frc_reservoir_B.txt`, and `jobs/ce_chunks/*`. Ground-truth docs
reconciled: handoff/30 (seeds), /31 (A* roadmap), /32 (plan reassess + KILL/dedup + B-offload runbook),
`FINAL_TABLE_PLAN.md`, `paper/writing/data/key_numbers.md`, `artifact_status_report.html`.

## 0. Clock & capacity (the numbers everything hangs on)
- **Now: Sat 2026-07-11 19:42.** Freeze: **Sun 07-12 08:00 = 12.3 h away.** Bonus: Sun 08:00 → Mon
  ~08:00 = **~24 h.**
- Measured pace: **A ≈ 42 cells/day (1.75/h), B ≈ 27 cells/day (1.13/h).**
- **By freeze: Node A ≈ 22 cells, Node B ≈ 14 cells.**
- **By Monday 08:00: Node A ≈ 64 cells (≈ its whole 65-cell queue minus the ce_chunk8 tail), Node B ≈
  all 36 reservoir cells with margin.**
- **The freeze is decided by the FIRST ~22 Node-A cells. The order of those 22 lines is the entire
  game.** Everything after position ~22 is a *bonus-hours* item, not a *frozen-paper* item.

## 1. LIVE STATE CORRECTIONS vs the launch snapshot (two things changed under us)
1. **The B-offload dedup (handoff/32 runbook Step 3) has ALREADY been applied** — `master_dispatch.txt`
   was rewritten at **19:35 today**; the 36 reservoir cells are **gone from A's queue** (verified: 0
   overlap now). A's pending dropped 103 → **65** (1 lrsw seed + 7 frm + 5 b4 + **45 frc spine** + 7
   ce). **No A/B double-run risk remains.** (The launch brief's "82 frc / ~103 pending" was the
   pre-19:35 snapshot.)
2. **Qwen math is effectively DONE locally, and the high-LR discriminating cells LANDED.** 44 `qwswm_`
   summaries present (by LR: 2e5×7, 5e5×7, 1e4×7, 2e4×7, **5e4×6, 1e3×7**, 3e4×3). The 5e-4/1e-3
   cells that handoff/31/32 called "in flight / pending" are **in**. **Consequence: the Qwen-math arm
   can be recomputed NOW (BBH-only, ceiling-aware, explosion-exclusion) and reclassified from
   "pending" to a stated result — a zero-GPU upgrade.** The "~6 left tonight" on B are the mid-LR
   (3e4/5e4) fill, not the decisive points.

## 2. MISSING-CELLS VERDICT (every planned table/figure cell: DONE / QUEUED / true MISSING)

| Paper deliverable | Cells | Status |
|---|---|---|
| Magnitude-law figs (Llama CS n=49; Qwen CS n=49) | lrsw_ / qwsw_ | **DONE** |
| Math law (frm_ n=49) | 56 frm_ done | **DONE** |
| Math table §1.1 — LoRA/LoRA+wd/PiSSA/MiLoRA/CLoRA/DoRA/CorDA++ rows | frm_ | **DONE** (incl. `frm_dora_lr3e4` — the old "DoRA MISSING" gap is closed; 64.97 headline s42+s43 done, **s44 QUEUED**, running) |
| Math table §1.1 — SC-LoRA + LoRA-Null rows | frm_sclora_lr1e4/2e4, frm_lora_null_lr1e4/2e4 | **QUEUED** (A, pos 44–48) |
| CS table §1.2 — LoRA+wd winner col + plain-LoRA anchor | frc_lorawd_wd0*/wd0 grid | **QUEUED** (A spine; 3/33 done) — AT RISK on order |
| CS table §1.2 — CLoRA k-grid (the B-CS honest boundary) | frc_clora_k128/256/512/1024/2048 | **QUEUED** (A spine, pos 32–36) — **AT RISK: buried behind 27 lorawd cells** |
| CS table §1.2 — DoRA/PiSSA/LoRA-L2/LoRA-r8/r16 baseline rows | frc_dora_r32, frc_pissa_r32, frc_lora_l2, frc_lora_r8/r16, frc_lora_r32 | **QUEUED** (A spine, pos 37–42) — AT RISK on order |
| CS table §1.2 — MiLoRA/SC-LoRA/LoRA-Null competitor rows | frc_milora/sclora/lora_null | **QUEUED** (**B reservoir**, front-loaded) |
| CS table §1.2 — CorDA++ (Tier-B, labeled) | frc_cordapp | **QUEUED** (B reservoir tail) |
| §3 seed error bars (7 CS op-points ×3 seeds) | lrsw_ s43/s44 | **DONE 13/14**; last cell `lrsw_dora_r16_lr2e4_s44` running (pos 1) |
| Qwen 2nd-model panel (CS + math) | qwsw_ / qwswm_ | **DONE** (see §1.2 correction) |
| Cross-literature overlay / geometry figure | existing data | **DONE** (analysis/writing only) |
| Calibration-confound control (b4) — SC-LoRA arm | b4_sclora ×4 | **DONE** (the thesis-completing arm is in hand) |
| Calibration-confound control (b4) — LoRA-Null/CorDA++ extension | b4_lora_null ×3, b4_cordapp ×2 | **QUEUED** (A, pos 61–65, dead last) |
| Cutoff-sensitivity diagnostic | frm_lora_lr3e4_c2048, frm_clora_k128_lr3e4_c2048 | **QUEUED** (A, pos 43,46) |
| CE forgetting table §6 (frm_ + mature sweeps) | ce_chunk1 done; chunk2–7 | **QUEUED** (A) — see §5 |
| **Efficiency table — instrumented peak-memory column** | (LoRA, DoRA, CLoRA-k1024, one SVD-init) | **TRUE MISSING — queued NOWHERE** |

**VERDICT: exactly ONE true gap — instrumented peak-memory runs for the "Compute & memory per
adapter" table.** Everything else is DONE or QUEUED. But three QUEUED families that the paper's
centerpiece CS table depends on (CLoRA boundary, CS baselines, math SC-LoRA/LoRA-Null rows) sit
**behind the freeze line** in the current order → they are "queued but won't land frozen" unless
reordered. That is a *sequencing* failure, not a coverage failure, and it is fixable for free.

## 3. ORDER-CHECK & RECOMMENDED REORDER (Node A — the decisive action)

**Current pending order (verified):** [1] dora seed, [2] 64.97 math seed, **[3–29] the entire 27-cell
frc_lorawd grid**, [30–31] clora k2048 s43/s44 (premature — s42 not landed), **[32–36] CLoRA k-grid
s42**, [37–42] CS baselines, [43–48] frm math completions, [49–55] ce chunks, [56–60] lorawd lr2e5
tail + clora k1024 seeds, [61–65] b4 confound.

**At the current order, the frozen ~22 = 2 seeds + ~20 lorawd grid cells.** It NEVER reaches the CLoRA
honest-boundary (the paper's #1 untested claim, FINAL_TABLE_PLAN threat #3 / top-10 #1), the CS
baseline rows, or the math SC-LoRA/LoRA-Null rows. The mature `lrsw_` n=49 + 3 done frc_lorawd already
carry "LoRA+wd on the frontier," so grinding the full faithful wd grid before the boundary spends the
scarce frozen slots on the *least* marginal cells.

**RECOMMENDED REORDER — move these 13 lines to the top of the pending block (right after the two seeds
at lines 1–2), in this order. I will not apply it; apply and do the single consolidated dispatcher
restart (skip-done makes it idempotent; in-flight locks are respected).**

```
# --- FROZEN-SET PRIORITY (promote above the un-started lorawd grid) ---
frc_clora_k1024_lr3e4_c256_s42     # B-CS honest boundary — THE #1 cell
frc_clora_k2048_lr3e4_c256_s42     # B-CS honest boundary (transfer-above-base cell)
frc_clora_k512_lr3e4_c256_s42
frc_clora_k256_lr3e4_c256_s42
frc_clora_k128_lr3e4_c256_s42
frc_lorawd_wd0p3_lr5e4_c256_s42    # LoRA+wd CS winner (expected op-point) + full wd column @5e4 already running
frc_lorawd_wd0_lr5e4_c256_s42      # plain-LoRA r32 anchor at the winner LR
frc_lora_r32_lr3e4_c256_s42        # CS-table plain-LoRA r32 fixed-LR anchor
frc_dora_r32_lr3e4_c256_s42        # CS-table DoRA row
frc_pissa_r32_lr3e4_c256_s42       # CS-table PiSSA row
frc_lora_l2_lr3e4_c256_s42         # CS-table LoRA-L2 row (CLoRA's own baseline)
frm_sclora_lr1e4_c256_s42          # math-table SC-LoRA row
frm_lora_null_lr1e4_c256_s42       # math-table LoRA-Null row
```
(Leave the currently-running frc_lorawd wd*@lr1e4/lr2e4 wave in flight — it keeps producing the wd
column for free. Everything below the 13 promoted lines keeps its order.)

**Frozen-22 after reorder:** 2 seeds + 5 CLoRA boundary + 2 winner/anchor + 4 CS baselines + 2 math
rows + ~7 lorawd cells finishing from the in-flight wave = **every load-bearing table cell lands in
the frozen paper.** This is the whole payoff of the audit.

**Node B reservoir order: already correct** — MiLoRA (1–6), SC-LoRA (7–12), LoRA-Null (13–18) are
front-loaded ahead of the a1r/em variants and CorDA++, so the three CS competitor rows are the first
to land. Optional micro-tweak: hoist `frc_milora_lr3e4`, `frc_sclora_lr1e4`, `frc_lora_null_lr5e4`
(the mature best-LR per method) to positions 1–3 so each competitor has one best-LR cell in the frozen
set even if only ~14 land. Caveat to log: the faithful `frc_sclora` sweep floors at lr1e4, but
SC-LoRA's mature best-adapt LR is 5e-5 — its faithful best-LR row may understate it; the low-LR point
lives in the b4 eval-matched arm instead.

## 4. BONUS-HOUR MENU (Sun 08:00 → Mon 08:00; ~42 more A-cells + ~27 B-cells) — ranked by marginal paper impact

1. **CLoRA-boundary error bars — `frc_clora_k1024`+`k2048 @lr3e4 s43,s44` (4 cells, ~20 GPU-h, A).**
   The B-CS boundary is a *ranking* claim (CLoRA-k1024/k2048 beats LoRA+wd) → needs seeds. Highest
   marginal value once the s42 boundary lands frozen.
2. **Instrumented peak-memory runs — LoRA, DoRA, CLoRA-k1024, one SVD-init (MiLoRA or SC-LoRA); 4–5
   short profiled train+eval capturing `torch.cuda.max_memory_allocated` (~8 GPU-h, A).** Closes the
   ONLY true-missing deliverable (efficiency table currently analytical/"not instrumented", review
   FIX 5). Needs a ~5-line logging patch first (PI/ops); pairs with the init-time column (LoRA+wd = no
   SVD/calibration precompute — the PI efficiency workstream).
3. **b4 confound extension — `b4_lora_null_r16_lr{2e5,1e4,3e4}` + `b4_cordapp_r32_lr{1e4,3e4}` (5
   cells, ~25 GPU-h, A).** Extends the "the one deviation is a calibration confound" control beyond
   SC-LoRA to LoRA-Null/CorDA++ — robustness on the thesis-completing result.
4. **Node-B reservoir remainder — the ~22 competitor cells not landed by freeze (MiLoRA/SC-LoRA/
   LoRA-Null tails + CorDA++ labeled row) (~110 GPU-h, B).** Fills the CS competitor rows to full
   sweeps; B clears these by Monday on otherwise-idle capacity.
5. **Faithful CS completion on A — remaining `frc_lorawd` wd-intervention chains (full wd column at
   lr2e4 & lr5e4) + `frc_lora_r8`/`frc_lora_r16` reduced-rank rows (~14 cells, ~70 GPU-h).** Rounds
   out the faithful wd→F_Δ→retention intervention figure and the CLoRA-Table-2 reduced-rank rows.

Below the line (run only if idle): CE `chunk8` catch-all + CE on the 28 newest adapters (§5), the
`frm_ c2048` cutoff diagnostics, extra `frm_sclora/lora_null` LRs.

## 5. CE-PLAN CROSS-CHECK (do chunks 2–8 union to all adapters, no gaps/overlaps?)

**Design:** chunk1 = `--glob 'frm_*'` (60 faithful-math adapters); chunks 2–7 = explicit
`--runs_file` lists (non-frm Llama adapters); chunk8 = `--glob '*'` runtime catch-all. `ce_batch.py`
is per-adapter skip-done (done-set refreshed before each adapter from `results/forgetting*.jsonl`),
skips non-Llama (Qwen) and untrained adapters.

**Verified against the 468 adapters present on `/scratch/cf_models`:**
- **Overlaps: NONE** across chunks 2–7 (286 listed lines, 286 unique, 0 duplicates). Dead list
  entries (name with no adapter on disk): **0**.
- **Coverage:** 60 frm_ (chunk1) + 286 (chunks 2–7) = 346 explicit. 94 Qwen are correctly excluded
  (CE is Llama-only). That leaves **28 Llama adapters present but in NO explicit chunk** — they are
  covered ONLY by the chunk8 `*` catch-all.
- **The 28 chunk8-only adapters are exactly the cells that landed AFTER the chunk lists were built
  (2026-07-10 00:48):** the 14 `lrsw_ s43/s44` seed headline cells, the done `frc_lorawd` spine cells,
  and the `b4_*` confound cells (+ `smoke_qwcs`, ignorable).
- **Verdict: no permanent gaps — the union is complete IFF chunk8 runs.** But chunk8 is dead-last and
  a single huge job (glob over ~314 Llama adapters), so at freeze the newest 28 (incl. the §3 seed
  replicates) likely have NO CE score. That only affects the *secondary* CE-error-bars-on-seeds and
  the b4-CE rows; the §6 forgetting table's primary op-points (all `_s42`) are in chunks 2–4.
- **Concrete fix (cheap):** add a small `chunk_new.txt` listing the 14 lrsw seed cells + the 3 done
  frc_lorawd + b4 cells and slot one `ce_batch.py --runs_file jobs/ce_chunks/chunk_new.txt` line
  *before* ce_chunk8 — gives CE on the headline replicates without waiting on the catch-all. Otherwise
  leave chunk8 as the bonus-hours sweep.

## 6. RESIDUAL RISK (unchanged headline from handoff/31 C.4 / 32 D, now sequencing-bounded)
The faithful-CS head-to-head is the paper's only direct evidence for claim 3 on CLoRA's flagship
domain, and its highest-value cells (CLoRA k1024/k2048 boundary) are behind the freeze line **only
because of queue order**. The reorder in §3 converts this from a data risk into a solved item for
free. Freeze-day fallback stays pre-committed: any frc_ row that hasn't landed → CLoRA **published** +
labeled mixed-rank proxy, never presented as the faithful recipe.

### changed since handoff/32
- **Reservoir dedup already applied (19:35)** — A is 65 pending, no double-run; reservoir lives only
  on B now. (handoff/32 listed this as a to-do; it's done.)
- **Qwen math high-LR cells LANDED** (44 qwswm_ local incl. 5e4/1e3) → math arm recomputable now,
  reclassify "pending" → stated result (BBH-only, ceiling-aware). Zero GPU.
- **New sequencing finding:** at current order the frozen ~22 A-cells are 2 seeds + lorawd grid and
  MISS the CLoRA boundary / CS baselines / math SC-LoRA+LoRA-Null rows. 13-line reorder (§3) fixes it.
- **One true-missing deliverable identified:** instrumented peak-memory for the efficiency table
  (queued nowhere) — now the #2 bonus-hour item with a concrete recipe.
- **CE union verified gap-free-modulo-chunk8**; 28 newest adapters (incl. §3 seed replicates) fall to
  the catch-all only → optional `chunk_new` fix.
