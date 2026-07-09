# handoff/31 — Qwen status, additional-data plan, and the A* roadmap (2026-07-10)

Research-planner deliverable answering the PI's three questions verbatim: *"Where are the Qwen
runs? Are there any additional data that we need? What is the roadmap to make this a basis for a
real A* publication?"*

Ground truth reconciled this pass: `paper/writing/data/key_numbers.md` (authoritative numbers, but
§11 Qwen block is STALE — see below), handoff/25 (supervision), handoff/27 (geometry), handoff/28
(two-node plan), handoff/29 (session state), handoff/30 (seed3 error bars), `FINAL_TABLE_PLAN.md`,
`artifact_review_round_final.md`, `author_recommendations_2026-07-09.md`,
`pi_review_figures_title_2026-07-09.md`, live queue state (`jobs/*`, `results/` counts, both nodes),
and an independent recompute of the Qwen law from raw `summary.json` (Node A local, all 49 CS cells).

**Deadline context:** ~2 days of 16-GPU compute left (Sun EOD); nodes returned end of week. Node A
(d001) and Node B (d002) both saturated (8/8 GPUs, verified 2026-07-10). The 2nd node already SAVED
Qwen from sacrificial status (handoff/28).

---

## PART A — QWEN STATUS (the honest current picture)

### A.0 Headline correction to key_numbers §11 (dated 07-02, now STALE)
key_numbers §11, handoff/25, the artifact, and both review docs all describe Qwen as **"LoRA-only,
~13/112 cells, CS r=−0.88."** That was true on 07-02. **It is no longer true.** Between 07-02 and
today the Qwen **commonsense** arm completed as a **full multi-adapter sweep**, and the Qwen **math**
multi-adapter sweep is now draining on Node B. The Qwen evidence base is materially stronger than
every current writing artifact claims. This is the most important single update in this document.

### A.1 What Qwen data exists RIGHT NOW (counts by arm, verified from summary.json)

**Commonsense (CS) arm — `qwsw_*`, Node A, COMPLETE (49 assessed cells + 1 excluded):**

| adapter | LRs present | cells | note |
|---|---|---|---|
| lora_r16 | 7 (2e-5…1e-3) | 7 | the published "r=−0.88" arm |
| lorawd_wd0p3 | 7 | 7 | |
| lora_null_r16 | 7 | 7 | |
| dora_r16 | 7 | 7 | |
| milora_r32 | 7 | 7 | |
| sclora_r32 | 7 | 7 | |
| clora_k1024 | 7 | 7 | |
| corda_r16 | 1 | 1 | **EXCLUDED** (policy §8; calibration mismatch) |

= **7 adapters × 7 LRs = 49 assessed CS cells**, mirroring the Llama-2 n=49 structure exactly.
Every cell has real retention (`headline.retention_mean` = mean(BBH, MMLU-Pro); e.g. LoRA@3e-4 =
33.54, CS-avg 86.7, BBH 41.44, MMLU-Pro 25.65 — Qwen's higher base ceiling).

**Math arm — `qwswm_*`, split A+B, IN PROGRESS (only n=10 assessed today):**
- Assessed union = **10 cells**: LoRA (5 LRs 2e-5/5e-5/1e-4/2e-4/3e-4 — **all low/mid, none above
  3e-4**) + LoRA+wd0.3 (5 LRs, incl. one 5e-4). F_Δ spans only **0.038–0.159 (~0.6 decade)** — a
  collapsed range, no high-magnitude points. (The 5 `qwswm_lorawd` rows on Node B are byte-identical
  to A's already-synced copies — no new info yet; +1 SMOKE discarded.)
- Node B (d002): running `jobs/frepro4_qwen_B_keep.txt` = **44 math cells** (7 LRs each ×
  clora_k1024/dora/lora_null/lorawd/milora/sclora + 2 lora top-ups). **5 done, 39 genuinely-new
  remaining** (2026-07-10) — these carry the 5e-4/1e-3 high-F_Δ points. Dispatcher live
  (`--tag qwenB`), 8/8 GPUs busy. ~4–5 GPU-h/cell → ~25 h wall → **expected to finish within the
  window** (watch the tail).

### A.2 What each arm's evidence supports

- **CS arm (DONE) → full 2nd-model law replication across 7 adapters.** Independent recompute
  (n=49, CorDA excluded, F_Δ = `fdelta_token_weighted`, retention = `retention_mean`):
  - **pooled core: r = −0.857, R² = 0.735, slope −31.98 pp/dec**
  - pooled broad: r = −0.937, R² = 0.878, slope −26.10
  - LoRA-only core: r = −0.883 (reproduces the published −0.88 exactly)
  - within-method core: DoRA −0.905, LoRA −0.883, MiLoRA −0.882, SC-LoRA −0.838, CLoRA −0.767,
    LoRA-Null −0.730, **LoRA+wd −0.165** (flat by construction — wd caps F_Δ so its 7 points do
    not span the axis; same shallow-slope signature LoRA+wd shows on Llama).
  - The Qwen pooled core r (−0.857) is **numerically indistinguishable from Llama's −0.858** — the
    law transfers across architecture with the same strength. (Steeper slope, −32 vs −15 pp/dec, is
    the Qwen retention scale, not a different law; report r/direction as the headline, slope as
    model-specific. **Independent `data-verifier` cross-node recompute confirms all of the above to
    2 decimals** (pooled core −0.86, broad −0.94, LoRA-only −0.88; within-method as listed); no
    collapse/explosion cells; F_Δ = `headline.fdelta` == `fdelta_token_weighted`.)

- **Math arm (IN PROGRESS) → the anti-replication resolution.** Today the math sweep is
  **low-LR-only** (F_Δ 0.038–0.159), which is why it fails to replicate. **Honest metric statement
  (verifier-confirmed):** on the **campaign-correct BBH-only** metric the Qwen math law is **FLAT**
  — pooled r=−0.05 (ns), LoRA-only r=−0.24 (ns) — nowhere near Llama-2 math's −0.97. The
  frequently-quoted **"+0.67"** (and pooled +0.60) is on **core retention, which includes the
  known-broken MMLU-Pro math parser** — it must NOT be reported as a positive math law; the correct
  statement is "flat on BBH-only; a spurious positive appears only when the broken MMLU-Pro column is
  mixed in." The B_keep queue adds the **5e-4/1e-3 cells across 6 adapters** — the high-F_Δ points
  needed to trace the curve. Until those land, the math arm **does NOT replicate** and is reported as
  pending, on BBH-only.

### A.3 Honest current verdict (state exactly this in the paper)
> On a second architecture (Qwen-2.5-7B), the magnitude law **replicates in full on commonsense**:
> across seven adapters × seven learning rates (n=49), retention falls on a single F_Δ curve with
> pooled r = −0.86 (identical to Llama-2's −0.86), and every adapter individually is negative. On
> **math**, replication is **not yet established** — the current Qwen-math sweep is low-learning-
> rate-only (F_Δ 0.038–0.159), and on the campaign-correct BBH-only metric the fit is flat
> (r = −0.05, n.s.); the high-learning-rate cells that would resolve it are in flight and are
> reported as pending, not as a result.

Do not upgrade the math claim until the Node-B cells land and are recomputed. Do not merge Qwen and
Llama numbers into one fit (different base ceilings). Math retention stays **BBH-only** (MMLU-Pro
parser broken for math) — the "+0.67" core figure is a parser artifact, not a result.

---

## PART B — ADDITIONAL DATA NEEDED (prioritized, costed, accept/reject)

Costs are GPU-h (≈5 GPU-h/cell: ~2 h train + ~3 h broad eval; DoRA ~7 h). "Node/slot" = where it
runs and its position vs the current priority stack. Compute budget remaining ≈ **2 days × 16 GPUs**.

| # | Item | Cost | Node/slot | Decision | One-line justification |
|---|------|------|-----------|----------|------------------------|
| 1 | **Finish Qwen math multi-adapter sweep (39 cells)** | ~195 GPU-h (running) | B, live #1 | **ACCEPT (already running)** | Converts the math anti-replication into a 2nd-model 2nd-task result; the single biggest evidence upgrade left and it costs nothing new. |
| 2 | **14 seed-3 (s43/s44) lrsw_ headline cells** | ~75 GPU-h (queued) | A, promote to #2 | **ACCEPT** | Error bars on the 7 CS §3 operating points — the adversarial review's #1 gap. Queued in `master_dispatch` (handoff/30); needs the consolidated dispatcher restart to go live. |
| 3 | 3rd seed for the **math** headline pair (LoRA best-LR 64.97 + LoRA+wd math winner), s43/s44 | ~20 GPU-h (4 cells) | A, #3 | **ACCEPT (small)** | The "LoRA beats CLoRA published 64.97 vs 64.59" claim is currently n=1; the s43 sibling is 65.88 (still edges 64.59). 4 cells makes the headline defensible; cheap. |
| 4 | Qwen **multi-adapter math** already covered by #1; no extra Qwen adapters needed for CS (CS is done) | 0 | — | **REJECT (already have it)** | The feared "Qwen is LoRA-only" gap is already closed — CS is a full 7-adapter sweep; adding more Qwen CS adapters buys nothing. |
| 5 | Qwen **3-seed** error bars (elevate Qwen to co-headline) | ~70 GPU-h (14 cells) | B tail, post-#1 | **REJECT for this deadline / DEFER** | Qwen is a replication arm, not a headline; single-seed is acceptable for "the law's sign/monotonicity transfers." Revisit only if a reviewer demands it (post-deadline). |
| 6 | **3rd model (Mistral-7B)** minimal LoRA + LoRA+wd LR sweep (2 adapters × 7 LRs) | ~70 GPU-h min (14 cells); ~250 GPU-h for a credible 5-adapter arm | none free pre-deadline | **REJECT pre-deadline / ACCEPT post-deadline** | A* reviewers love 3 models, but there is no free node before Sunday, and a thin 2-adapter arm invites the same "LoRA-only" critique Qwen just escaped. This is the top **camera-ready / rebuttal** add, not a pre-freeze item. |
| 7 | **Faithful CS table spine (`frc_`, ~51 cells) — currently 0 done** | ~255 GPU-h | A, live #1 (CS spine) | **ACCEPT but TRIAGE (at risk)** | The CLoRA Table-2 mirror head-to-head. 0 cells done and it is the queue bottleneck; likely will NOT fully land in 2 days on one node. Prioritize the **LoRA+wd winner column + plain-LoRA anchor + faithful CLoRA k-grid** (the B-CS boundary cells) over the full baseline block; fall back to published + mature-proxy with honest labels for the rest. |
| 8 | **Instrumented peak-memory** to replace analytical §5 estimates | ~8 GPU-h (short profiled runs, 1 per adapter family) | A, GPU-free window or 1 slot | **ACCEPT (cheap, closes a review flag)** | Review FIX 5: §5 memory numbers are analytical/"not instrumented" but presented as measured. A handful of `torch.cuda.max_memory_allocated` runs (LoRA, DoRA, CLoRA-k1024, an SVD-init method) turns an attackable estimate into a measurement. High credibility per GPU-h. |
| 9 | **Full CE-to-base batch over all ~445 adapters** (only 6 done today) | ~30–45 GPU-h eval-only (reads local `/scratch`) | A ONLY, GPU-free window | **ACCEPT (opportunistic)** | Independent-metric confirmation of the law + the "MiLoRA≈LoRA at matched magnitude" point at scale. Eval-only, adapters are already on A's disk; run whenever GPUs briefly free. Not on the critical path but high value-per-hour. |
| 10 | **Cutoff 256-vs-512 pair** on `frm_lora@3e4` + `frm_clora_k128@3e4` | ~10 GPU-h (2 cells) | A, #4 | **ACCEPT (diagnostic)** | Tests whether the ~4–5 pp fancy-method reproduction shortfall is CoT truncation vs real — protects the reproduction-fidelity claim (FINAL_TABLE_PLAN #6). |
| 11 | Registry FREEZE + MMLU-Pro parser | 0 GPU (CPU/writing) | A, immediate | **ACCEPT (do first)** | Review: `campaign_summary.jsonl` is a moving target (473→457 rows vs the 359→343 key_numbers was built on). Freeze a dated snapshot for all paper numbers. MMLU-Pro math parser stays broken → math retention **BBH-only** (already the rule; document, do not attempt a fix under deadline). |
| 12 | CorDA / CorDA++ arms | ~40+ GPU-h | none | **REJECT / DEFER** | Old CorDA excluded (calibration bug); CorDA++ not wired into `train_cs.py`. Keep as an explicit "future/strong-tier" row, not a blocker (FINAL_TABLE_PLAN §3.4). Never present old-CorDA numbers as CorDA++. |
| 13 | Geometry drift — extend to Qwen | ~CPU-only (needs Qwen adapters, streamed-deleted on B) | — | **REJECT** | Off-thesis (geometry is a fingerprint, not a lever); both review docs say skip a Qwen geometry row. Llama geometry (320 adapters) is sufficient. |

**Net GPU asks that are genuinely new and small:** #3 (20) + #8 (8) + #9 (~40, eval-only) + #10
(10) ≈ **~80 GPU-h**, all fitting the GPU-free windows and single freed slots. Everything else large
(#1, #2, #7) is already queued/running. #6 (3rd model) is deliberately post-deadline.

---

## PART C — THE A* ROADMAP (today → submission)

### C.1 Where the paper stands (paper.tex is far along)
`paper/writing/paper.tex` (1251 lines) already has the full skeleton: Intro, Related Work, Setup &
Measurement, **The Magnitude Law**, **Geometry Is a Second-Order Fingerprint**, LR Is Only a Proxy,
**Weight Decay Bounds F_Δ for Free** (the practical corollary), Limitations, Discussion, Supporting
Figures, **CLoRA published cross-check** (appendix), Reproducibility. Writing is *not* the
bottleneck — **data freeze, seeds, and the faithful-CS table are.**

### C.2 Milestone plan (ordered)

**M1 — Registry freeze + Qwen-CS integration (TODAY, no GPU).**
- Freeze a dated `campaign_summary_frozen_2026-07-10.jsonl`; point key_numbers + all figures at it.
- **Rewrite key_numbers §11 and every "Qwen LoRA-only" mention** to the full multi-adapter result
  (n=49, pooled core r=−0.86, broad −0.94, within-method table above). Specific verifier-flagged
  stale prose to fix: §11 L215 "Other 5 adapters NOT yet run" is **false** (all 7 CS adapters done);
  L217 "~13 of ~112 cells" → **~59/112** (49 CS + 10 math); L216 restate math on **BBH-only**
  (flat r=−0.05, not "+0.67" which is the broken-parser core). Add the pooled Qwen CS law to §1's
  table. This is a pure win sitting unclaimed in the data. Also update handoff/25, the artifact
  §M3/M4, and both review docs' now-false "Qwen is LoRA-only" premise.
- Fix the review BLOCKERS that are writing-only: dek overclaim (FIX 1), seed disclosure (FIX 2),
  drop "pending"/"not yet" language (FIX 3), title change (drop "Not the Geometry" → magnitude-law-
  positive, first-order-qualified title).

**M2 — Stats hardening (as seeds land, days 1–2).**
- Promote the 14 lrsw_ s43/s44 cells (Part B#2) to run right after the frc_ spine head; put
  3-seed retention error bars on the 7 CS §3 cells + the 2 math headline cells (Part B#3).
- Ceiling-aware statistics (PI workstream 3, CPU-only, DO NOW): the retention curve saturates at the
  base ceiling, so report **Spearman** (already −0.90 CS) and a **spline/censored fit** alongside
  linear R², and the **below-ceiling slope**. This is a writing/analysis task on existing data and
  pre-empts the "your R² is inflated by the ceiling" critique.
- Keep published vs in-pipeline numbers in separate columns; headline cells ≥3 seeds; math BBH-only.

**M3 — Data freeze (Sun EOD / node return).**
- Freeze whatever the frc_ CS spine and Qwen math have reached. Decide the CS table's final form
  (full faithful vs winner-column + honest proxy fallback) based on frc_ completion — see C.4 risk.
- Final CE full-batch (Part B#9) and instrumented memory (Part B#8) opportunistically before freeze.

**M4 — Figure set finalization (CPU/writing, overlaps M2–M3).**
- BUILD the **cross-literature overlay** (our F_Δ→BBH + CLoRA Table-4 F_Δ→BBH, **BBH↔BBH**, slope
  as a range not a point-match; "parallel law, not one line") — both review docs call this the
  single highest-value figure. Add the **Qwen 2nd-model panel** (now a real 7-adapter scatter, not a
  LoRA line).
- One **consolidated geometry figure**: ANCOVA/drop-outlier null as the centerpiece (panel B), the
  fingerprint heatmap, and the SC-LoRA per-layer erosion inset. Compact subsection, not a co-equal
  section (author rec 3b). CorDA labeled "fingerprint only, excluded from all law claims."
- **Efficiency table** with instrumented memory + init times (LoRA+wd = zero SVD/calibration
  precompute — the PI efficiency workstream; correct SC-LoRA to 512 calib forwards, FIX 5).

**M5 — Internal red-team round (before submission).**
- Re-run `adversarial-critic` + `data-verifier` on the frozen paper.tex + frozen registry (the
  round in `artifact_review_round_final.md` was on the artifact; repeat on the paper).
- Verify every arXiv ID and every "other paper" claim (guardrail: no unverified claims about other
  papers; LoRA-L2 = "same KIND of knob, not identical," per handoff/25).

### C.3 Honest venue-tier assessment (what this evidence supports)

**Current base (after M1–M2, before a 3rd model):** a strong empirical + measurement paper.
- The magnitude law is multiply-corroborated: **two models** (Llama r=−0.858, Qwen r=−0.857, each
  n=49 × 7 adapters), an **independent published table** (CLoRA Table-4, r=−0.98), and an
  **independent metric** (CE-to-base, MiLoRA≈LoRA at matched magnitude). The geometry null (ΔR²≈0,
  6/7 on-curve) and the geometry-as-fingerprint measurement tool are genuine, careful contributions.
- Honest tier: **top empirical/measurement venue or strong workshop today; a credible ML-conference
  main-track submission after M1–M2** (2-model replication + seed error bars + ceiling-aware stats).
  What still caps it below a *confident* A* accept: single-seed adaptation rankings, the faithful-CS
  head-to-head table not yet complete, and only 2 architectures.

**What moves it to A* main-track (NeurIPS/ICML), cheapest strongest lever first:**
1. **CHEAPEST STRONGEST — reframe the weight-decay sweep as a controlled causal intervention, and
   ship the 2-model replication (both nearly free).** The paper currently states the law
   *observationally*. But wd is a **knob that monotonically shrinks F_Δ**, and we already swept it
   (the frm_/lrsw_ wd grids) — so "turning the wd knob moves F_Δ, which moves retention along the
   *same* curve as the observational fit" is a **controlled manipulation, not a correlation**, at
   **zero new compute** (data exists; CPU/writing only). Combined with the just-completed Qwen
   2-model × 7-adapter replication, this converts the story from "a correlation in one model" to "a
   law with a controlled knob, replicated across two architectures." That is the depth a 3rd model
   cannot buy, and it is essentially free. **This is our single highest-leverage move.**
2. **Land the seed error bars (M2) and the faithful-CS winner column (C.4).** Removes the two
   concrete desk-reject triggers (no error bars; no faithful head-to-head on the most-cited domain).
3. **3rd model (Mistral-7B), POST-deadline for rebuttal/camera-ready.** Genuinely raises the ceiling
   ("law holds across 3 architectures") but is infeasible pre-freeze and, done thin, re-opens the
   "LoRA-only" critique. Schedule it for the rebuttal window, not now.

Why (1) over (3) as the *cheapest strongest*: a 3rd model adds breadth (generality) but costs ~250
GPU-h we do not have before Sunday and adds no new *mechanism*; the causal reframing adds the thing
A* reviewers reward most — a controlled test that rules out reverse causation — for free, and the
2nd-model replication (the usual reason to want model #3) is **already in hand for CS and landing for
math**. Breadth we now have (2 models × 2 tasks × 7 adapters); the missing A* ingredient is
**causal/mechanistic depth**, which is our cheapest lever.

### C.4 The single biggest DATA risk (call it out to the PI)
**`frc_` faithful-CS table = 0 cells done, and it is the queue bottleneck on one node.** The
magnitude *law* does NOT depend on it (the mature `lrsw_` n=49 sweep carries the law, and Qwen now
seconds it), and the faithful *math* table is DONE (50 `frm_` cells, incl. the "LoRA 64.97 > CLoRA
64.59" headline). But the faithful *CS* head-to-head (CLoRA Table-2 mirror) may not fully land in 2
days. **Mitigation:** triage the frc_ queue to the **LoRA+wd winner column + plain-LoRA anchor +
faithful CLoRA k-grid** (the B-CS high-k boundary — FINAL_TABLE_PLAN #1, the paper's one untested
escape hatch); for any baseline rows that don't finish, fall back to CLoRA **published** + the mature
mixed-rank proxy, each **explicitly labeled** (never passed off as the faithful recipe). The paper is
defensible without a complete frc_ table; it is NOT defensible if the frc_ proxy is mislabeled.

---

## STANDING PLAN SNAPSHOT (numbered, costed — the live TODO)

1. **[NOW, 0 GPU]** Registry freeze + rewrite all "Qwen LoRA-only" → full n=49 multi-adapter
   (r=−0.857 core). *WHY:* unclaimed 2nd-model result strengthens the central claim. *STATUS:* ready.
2. **[NOW, 0 GPU]** Writing-only review BLOCKERS: dek, seed disclosure, drop "pending", title.
   *WHY:* removes the 2 FAIL items + guardrail violations. *STATUS:* ready.
3. **[NOW, CPU]** Ceiling-aware stats (Spearman/spline/censored, below-ceiling slope) on existing
   data. *WHY:* PI workstream 3; pre-empts inflated-R² critique. *STATUS:* ready.
4. **[running, B]** Finish Qwen math (39 cells, ~195 GPU-h). *WHY:* resolves math anti-replication.
   *STATUS:* live, ~25 h wall; watch the tail.
5. **[queued→promote, A]** 14 lrsw_ s43/s44 + 4 math-headline seeds (~95 GPU-h). *WHY:* error bars =
   review gap #1; headline ≥3 seeds. *STATUS:* in master_dispatch; needs consolidated restart.
6. **[running→triage, A]** frc_ CS spine — prioritize winner column + CLoRA k-grid (B-CS boundary).
   *WHY:* faithful CS head-to-head; the one untested escape hatch. *STATUS:* 0/~51 done — AT RISK.
7. **[opportunistic, A]** CE full batch (~445 adapters, eval-only) + instrumented memory + cutoff
   256/512 pair (~60 GPU-h total). *WHY:* independent metric at scale; closes efficiency-measurement
   flag; reproduction diagnostic. *STATUS:* GPU-free-window work.
8. **[after data, CPU]** Cross-literature overlay (BBH↔BBH) + Qwen 2nd-model panel + consolidated
   geometry figure + efficiency table. *WHY:* highest-value figures; PI efficiency + geometry-drift
   workstreams. *STATUS:* pending freeze.
9. **[causal reframe, CPU]** Recast the wd sweep as a controlled F_Δ intervention. *WHY:* cheapest
   strongest A* lever (correlation → controlled manipulation, zero compute). *STATUS:* ready.
10. **[pre-submit]** Red-team round (adversarial-critic + data-verifier on frozen paper.tex).
    *WHY:* the artifact was reviewed; the paper has not been. *STATUS:* after M4.
11. **[POST-deadline]** 3rd model (Mistral) LoRA+LoRA+wd sweep; Qwen 3-seed; CorDA++ wiring.
    *WHY:* camera-ready/rebuttal breadth. *STATUS:* deferred, no free node.

### changed since last version (vs handoff/28/30)
- **Qwen CS re-classified from "LoRA-only, in progress" to "COMPLETE full 7-adapter × 7-LR
  replication, pooled core r=−0.857 ≈ Llama −0.858"** (independent recompute; key_numbers §11 is
  stale and must be rewritten). This is the headline change.
- Qwen math quantified as 5 done + **39 in flight on Node B** (high-LR resolution cells); still
  reported as not-yet-replicating until they land.
- Surfaced the **frc_ faithful-CS spine = 0 done** as the single biggest data risk, with a triage +
  labeled-fallback mitigation.
- Named the **cheapest strongest A* lever = wd-as-causal-intervention + the now-in-hand 2-model
  replication** (both ~free), ahead of a 3rd model (deferred post-deadline).
- Folded the adversarial review's BLOCKERS (seed disclosure, dek, "pending" language, title) into
  M1 as zero-GPU writing tasks; added registry freeze and instrumented-memory as accepted items.
