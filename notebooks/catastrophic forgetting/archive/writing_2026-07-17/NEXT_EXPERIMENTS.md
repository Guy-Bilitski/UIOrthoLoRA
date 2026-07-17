# Next Experiments — pre-submission roadmap (October 2026 ARR)

**Date:** 2026-07-14. **Owner:** planning agent, for lead sign-off.
**Ground truth precedence:** `claims_coverage_audit_sat.md` (2026-07-11) > `data/key_numbers.md`
(07-02 base + 07-10 §4/§11 updates) > `FINALIZATION_PLAN.md` / `03_gaps_and_roadmap.md` (07-02,
partly stale — Qwen CS, frm_ math, seeds 43/44, and the first B4 cells have ALL landed since).
**Cost basis:** `07_experiment_plan.md` — Llama-2 cell ≈ 1.5–2.5 h, Qwen cell ≈ 6–10 h, 8×B200 box,
ONE scheduler at a time, wall-clock ≈ ceil(cells/8) × per-cell-hours. "GPU-day" below = one day of
the full 8-GPU box.

---

## STATUS UPDATE 2026-07-14 (post registry refresh)

Ground truth: `data/key_numbers.md` **§16** (supersedes §11 and the audit framing this plan was
built on); full verification in `registry_refresh_2026-07-14.md`. Registry after the lead-approved
qwswm ingestion: **622 rows, 606 unique**. Superseded items below are marked in place with
"→ CLOSED 07-14" rather than deleted.

**Closed by landed data:**

- **1.1 param-match (W2/B5a): CLOSED, with result.** `frc_lorawdr16_wd0p3_lr5e4` (r16, 28.0M
  params, plain-LoRA-matched): **CS 81.04 / ret 26.27 / F_Δ 0.334** — reproduces the r32 operating
  point at half the parameters. Plain-LoRA rank ladder r8→r16→r32 @ lr3e-4: F_Δ 0.518→0.603→0.739
  with retention 24.0→23.0→22.2 — the rank effect vanishes once F_Δ is controlled. Disclosed
  anomaly: the lr3e-4 r16 sibling is a deterministic answer-format-collapse cell (CS 13.53
  below-chance per-dataset, retention intact 26.84, healthy training, reeval-reproduced); its
  seed replicate is queued.
- **1.3 base ceilings (C5): CLOSED for Llama-2** (`base_llama2_noft`, 07-12): broad ceiling
  **35.26** (MMLU 40.88, ARC-C 44.80, TQA 38.85); core 25.89 confirms the canonical 26.0
  (external snapshot stays canonical; decision 2026-07-14). TruthfulQA immunity is real, not a
  floor artifact (base 38.85; fine-tuned 31.4–39.5, mean 35.7, 1/49 above base; slope −0.46 ns —
  a constant ≈−3 pp magnitude-independent offset). "Ret-broad uncalibrated" hedges are retired
  in the paper. **Qwen base ceilings still open.**
- **1.4 Qwen-math high-LR: LANDED, and the 51-cell ingestion is done** (lead-approved
  2026-07-14). s42 BBH fit: **n=56, r=−0.70, slope −14.4 pp/decade** over F_Δ 0.029–0.90
  (~1.5 decades); two diverged cells excluded with disclosure; slope inside the cross-domain band
  (−12.7…−18.9); all six non-wd adapters individually negative; LoRA+wd flat by construction
  (+0.09) — same signature as CS. Qualitative/directional replication (single-seed, BBH-only;
  MMLU-Pro excluded by convention — the "+0.67" framing is superseded and must not be quoted).
  **Nothing remains to run for math s42.**
- **1.0 backlog: largely drained.** frm math seeds landed (six 3-seed configs; headline
  66.79±0.79 verified; SC-LoRA math on the frm law n=48 r=−0.90, residual −1.8 pp); DoRA s44
  landed → all 7 CS operating points are 3-seed (unblocks 2.2); much of frc landed (wd grid +
  rank series + r16 control); b4 partially landed (`b4_lora_null_r16_lr2e5` in registry;
  lora_null lr1e4/3e4 + cordapp full evals still pending).

**Still open — honest remaining GPU-day total ≈ 1.0 core (≈ 1.5 with Tier-2 leftovers, +0.7 if
the reservoir is promoted):**

| item | cost (GPU-days) |
|---|---|
| 1.2 CorDA clean nq_open CS re-run (zero post-07-11 corda cells — unchanged) | ~0.1 |
| 1.5 Qwen CS seeds 43/44, headline cells | ~0.7 |
| 1.3-Qwen base ceilings (no-FT, eval-only) | ~0.05 |
| b4 completions (lora_null lr1e4/3e4 + cordapp full evals) | ~0.05 |
| r16-collapse seed replicate (the 1.1 anomaly cell) | ~0.05 |
| Tier-2 leftovers (2.1 basin probe 0.1; 2.6 B4 `_em` seeds 0.25; 2.7 B5b wd-for-everyone 0.2) | ~0.55 |
| optional: promote `frc_reservoir_B.txt` | ~0.7 |

GPU-free and now unblocked: **2.2 seed-averaged ANCOVA recompute** (DoRA s44 landed).

---

## 1. Executive summary

1. **Solid:** the Llama-2 CS magnitude law + full statistical battery reproduces to the decimal (audit PASS table); Qwen CS is a full 49-cell replication (r=−0.86); the `frm_` math headline is 3-seed (67.3±0.8); CS headline seeds s43×7 / s44×6 and the first B4 calibration cells have landed and resolve SC-LoRA onto the curve. *→ 07-14: s44 is now ×7 (DoRA s44 landed) and Qwen math replicates qualitatively (n=56, r=−0.70) — the lrsw pooled law re-verified this pass (r=−0.858, −14.78, n=49).*
2. **Exposed:** four gaps are queued NOWHERE — the param-matched LoRA+wd control (W2), the CorDA clean nq_open re-run, the C5 base-ceiling no-FT evals, and the Qwen seeds + Qwen-math high-LR completion; plus the "+0.67 Qwen-math anti-replication" needs its parser-artifact resolution written down and its high-LR cells run. *→ 07-14: three of the four are CLOSED by landed data (W2, C5-Llama, Qwen-math high-LR incl. the ingestion); the "+0.67" is superseded (MMLU-Pro excluded by convention). Remaining: CorDA, Qwen CS seeds, Qwen base ceilings — see STATUS UPDATE.*
3. **Cheap:** all Tier-1 Llama work is ≈0.5 GPU-days; the expensive tail is entirely Qwen (≈2.5 GPU-days). *→ 07-14: the Qwen tail shrank to ≈0.75 (seeds + base ceilings only).*
4. **Totals:** queue drain ≈1.0 GPU-day + Tier 1 new ≈3.0 + Tier 2 ≈0.6–1.3 → **≈5–5.5 GPU-days wall-clock to "all Tier 1 + Tier 2 landed"**; Tier 3 adds ≈2–6 more (CorDA++ dominates). *→ 07-14 recompute: **≈1.0 core + ≈0.55 Tier-2 leftovers ≈ 1.5 GPU-days** (+0.7 optional reservoir) — see STATUS UPDATE table.*
5. **Deadline math:** at ~6 GPU-days compute + ~2 weeks analysis/writing integration, compute must return by **~Sep 1** for a comfortable mid-Oct ARR submission; the crash-only-Tier-1 floor is T0 ≈ Sep 22.

---

## 2. Tier 1 — reviewer-critical (run first when compute returns)

> Sequencing rule (07_plan): single scheduler, never two pools. Order below = dispatch order.
> Item 1.0 is not "new" but must drain first — it carries DoRA s44, the frm_ math seeds, and the
> b4 expansion that Tier 2 analyses depend on.

### 1.0 Drain the existing `master_dispatch.txt` backlog (58 cells) — prerequisite
- **What:** the 58 genuinely-pending cells: `frc` 45 (CS wd-ablation c256 + faithful-math structured methods), `frm` 7 (math seeds 43/44), `b4` 5 (calibration expansion), `lrsw` 1 (DoRA s44).
- **Defends:** the wd dose-response for Claim 3 (frc), 3-seed math headline (frm), B4 fairness verdict breadth (b4), and the last missing CS seed sibling (DoRA s44 → unlocks the W5 seed-averaged ANCOVA).
- **Reviewer if missing:** "your math headline and ANCOVA are still single-/two-seed and your wd story has no dose-response."
- **Cost:** 58 Llama cells ≈ 8 waves × 2.5 h ≈ 20 h ≈ **1.0 GPU-day**.
- **Status:** QUEUED (active master dispatch). `frc_reservoir_B.txt` (40 more cells) stays behind it as optional Tier-2 breadth. **→ LARGELY CLOSED 07-14:** frm seeds, DoRA s44, and most of frc landed; b4 partially (lora_null lr1e4/3e4 + cordapp full evals still pending, ~0.05 GPU-day).

### 1.1 (a) Param-matched LoRA+wd control — B5a 2×2 completion (W2)
- **What:** `lorawd_r16_wd0p3` (LoRA+wd at r16, matched to plain LoRA/DoRA) + `lora_r32` (plain LoRA at r32, matched to MiLoRA/SC-LoRA/CLoRA/LoRA+wd), each × 7 LRs × s42. Completes the {r16,r32}×{wd0,wd0.3} 2×2 (existing `lora_r16` and `lorawd_wd0p3`@r32 are the other two corners). Analysis: regress retention on log F_Δ, test rank main-effect and rank×wd interaction ≈ 0 once F_Δ is in the model.
- **Defends:** Claim 3 (LoRA+wd on the Pareto frontier) against the capacity confound; upgrades "wins by magnitude" from asserted to shown.
- **Reviewer if missing (O4, near-certain):** "LoRA+wd wins because you gave it 2× the rank of plain LoRA/DoRA plus a regularizer knob no one else got." The audit (W2) flags this as real vs plain LoRA — the queued `frc_lorawd` wd-ablation is r32/r64 only, so it does NOT close the r16 hole.
- **Cost:** 14 cells ≈ 2 waves × 2.5 h ≈ 5 h ≈ **0.2 GPU-day**.
- **Status:** **QUEUED NOWHERE** (audit queue-coverage gap #1). Highest-priority new dispatch. **→ CLOSED 07-14 (with result):** landed via the frc c256 recipe rather than this lrsw 2×2 spec — LoRA+wd@r16 = CS 81.04 / ret 26.27 / F_Δ 0.334 (r32 op-point at half the params); rank ladder r8→r16→r32 slides down-curve (F_Δ 0.518→0.603→0.739, ret 24.0→23.0→22.2). Only leftover: the lr3e-4 r16 collapse-cell seed replicate (~0.05 GPU-day).

### 1.2 (b) CorDA clean nq_open CS re-run
- **What:** 7 `lrsw`-protocol CorDA CS cells (r16, 7 LRs, s42) with the fixed nq_open calibration, gated by the non-negotiable validation from 07_plan: 0-step ΔW→0 self-check + init-output invariance **after save→reload** (the residual-save bug class ate the last set: current-latest rows are the contaminated wikitext re-eval incl. one F_Δ=515 explosion). Optional twin: `corda_r16_em` × 7 (eval-matched calibration) to fold CorDA into the B4 verdict.
- **Defends:** the coverage claim (O1 — "the one adapter that could falsify 'geometry is inert' is missing"). Good case: CorDA lands on the curve → coverage upgrades toward 8/8 and the fingerprint section gains its `ein_bot`=0.49 outlier as an on-law geometry-is-inert exhibit. Bad case: known now, reframed honestly before a reviewer forces it.
- **Reviewer if missing:** "You excluded the flagship data-aware method from every headline exhibit; 'method-free' is untested exactly where it's most likely to fail."
- **Cost:** 7 cells ≈ 1 wave × 2.5 h ≈ **0.1 GPU-day** (0.2 with the `_em` twin). Plus the no-GPU validation pass.
- **Status:** **QUEUED NOWHERE** for CS (only `frc_cordapp` *math* is queued — that does not re-enter CorDA into the CS law). Audit gap #2.

### 1.3 (c) Base-ceiling no-FT evals — C5
- **What:** eval-only, no training: base Llama-2-7B scored on MMLU, ARC-Challenge, TruthfulQA (BBH 33.10 and MMLU-Pro 18.96 already exist). Recommended add-on while we're here: the same 3 no-FT evals on base Qwen-2.5-7B, since the paper now quotes Qwen broad retention (r=−0.94).
- **Defends:** interpretability of broad retention and the per-benchmark slope table (§7 of key_numbers) — in particular whether "TruthfulQA is immune (−0.5 pp/dec)" is real or a floor artifact.
- **Reviewer if missing:** "Retention percentages without a no-FT ceiling are uninterpretable; your 'TruthfulQA immune' headline could be a floor effect." (Currently disclosed as uncalibrated — honest, but a self-inflicted asterisk on a whole figure.)
- **Cost:** 5 Llama eval runs ≈ **0.2 GPU-day** (FINALIZATION estimates ~0.5 d for the full eval-only pass incl. Qwen; no training pool needed — can run during any scheduler pause).
- **Status:** **QUEUED NOWHERE** (audit gap #3; "no explicit no-FT eval cells found queued"). **→ CLOSED for Llama-2 07-14** (`base_llama2_noft`, evaluated 07-12: broad ceiling 35.26 — MMLU 40.88, ARC-C 44.80, TQA 38.85; core 25.89 confirms canonical 26.0; TQA immunity real). **Qwen base ceilings STILL OPEN** (~0.05 GPU-day, eval-only).

### 1.4 (d-i) Qwen-math high-LR completion + the "+0.67" resolution
- **What:** the high-LR resolution cells (lr 5e-4 / 1e-3 × 6 adapters; the full completion file is `frepro4_qwen.txt`, ~39 cells per key_numbers §11) — currently **not in the active master dispatch** (audit: only 5 low-LR `qwswm_` cells landed; key_numbers 07-10 counted 10 — audit wins, recount at dispatch time).
- **Resolving the anti-replication — two parts, one already done-by-analysis:**
  1. **The r=+0.67 is NOT a real anti-replication and must never be quoted.** Per key_numbers §11 (07-10, data-verifier-confirmed): +0.67 is computed on CORE retention, which includes the **known-broken MMLU-Pro math parser** — a parser artifact. On the campaign-correct **BBH-only** metric the landed low-LR cells give a FLAT fit (pooled r=−0.05 ns, LoRA-only −0.24 ns). So yes: it is an artifact of (a) the broken parser and (b) only low-LR cells (F_Δ spans 0.038–0.159, ~0.6 decade — no forgetting-regime points). REBUTTAL_PREP O8 still carries the old "+0.67, report openly" framing — supersede it: report "flat, low-LR-only, pending" not "wrong sign."
  2. **The experiment:** the high-LR cells put points in the forgetting regime and settle whether Qwen-math joins the law. If the slope goes negative, the 2-model × 2-domain claim completes; if it stays flat, math is honestly scoped to Llama.
- **Defends:** Claim 1 breadth (O8: "one of your four model×domain cells contradicts the law").
- **Cost:** ~39 Qwen cells ≈ 5 waves × 8 h ≈ 40 h ≈ **1.7 GPU-days** (Qwen cells 6–10 h). If budget bites, the minimal decisive slice is 2 LRs × 6 adapters = 12 cells ≈ **0.7 GPU-day**.
- **Status:** **NOT in active dispatch** (sits in `frepro4_qwen.txt`). Audit gap #4a. **→ LANDED + CLOSED 07-14:** the high-LR cells were run and the 51-cell qwswm ingestion is done (lead-approved). s42 BBH fit: n=56, r=−0.70, slope −14.4 pp/dec over ~1.5 decades; all six non-wd adapters negative; lorawd flat by construction (+0.09); two diverged cells excluded with disclosure. The "+0.67" resolution is written down (superseded — MMLU-Pro excluded by convention, unreliable on early Qwen-math rows). **Nothing remains to run for math s42.**

### 1.5 (d-ii) Qwen seeds 43/44 — headline cells only
- **What:** do NOT re-seed the 49-cell grid (≈4+ GPU-days of Qwen time — unaffordable and unnecessary; the audit calls single-seed "acceptable for a replication of the law … but disclose"). Seed only the cells a figure/table prints: 7 adapters × best-adapt LR × s43/s44 = 14 cells.
- **Defends:** pre-empts "your second-model replication is one seed" (O2's Qwen extension) and lets the Qwen op-point table carry error bars like the Llama one now does (N2).
- **Reviewer if missing:** tolerable if disclosed — this is the *lowest*-stakes Tier-1 item; it is Tier 1 only because Qwen wall-clock is the schedule bottleneck, so it must be dispatched early or not at all.
- **Cost:** 14 Qwen cells ≈ 2 waves × 8 h ≈ 16 h ≈ **0.7 GPU-day**.
- **Status:** **QUEUED NOWHERE** (audit gap #4b).

**Tier 1 subtotal (new dispatch): ≈ 3.0 GPU-days (0.5 Llama + 2.5 Qwen) + 1.0 GPU-day backlog drain.**

---

## 3. Tier 2 — robustness upgrades

| # | Item | What / why | Cost | Status |
|---|------|------------|------|--------|
| 2.1 | **MiLoRA collapse-basin characterization (W1)** | Table 1's MiLoRA CS-8=79.9 is a lucky seed (3-seed 57.7±22.7: 79.9/58.7/34.5). Text fix (print 3-seed mean±SD) is free and mandatory. The *experiment*: seeds 43/44 at the two neighboring LRs (2e-4, 5e-4) to establish whether the basin is LR-local (one fragile op-point) or method-wide — turns a liability into a "seed-fragility is the adaptation axis, retention stays law-bound (24.2±0.5)" finding. | 4 Llama cells ≈ **0.1 GPU-day** | queued nowhere (new) |
| 2.2 | **Seed-averaged ANCOVA recompute (W5)** | The entire geometry-inertness battery (F(6,41)=7.05, F(6,35)=9.32, SC-LoRA −4.15, LOMO 9.05) is s42-only and driven by the one cell that regresses to the curve on s43/s44. Recompute intercept/slope ANCOVA + LOMO on seed-averaged retention. Expected outcome per B5 audit finding: SC-LoRA residual shrinks toward 0 → "6/7 → 7/7 on curve," strengthening the thesis. | **0 GPU** (analysis; needs DoRA s44 from the backlog, item 1.0) | blocked on queue drain **→ UNBLOCKED 07-14** (DoRA s44 landed; run now) |
| 2.3 | **DoRA s44** | The one missing CS seed sibling; unblocks 2.2. | 1 cell (inside item 1.0) | **QUEUED** (lrsw, master dispatch) **→ CLOSED 07-14** (landed 07-11; all 7 CS op-points 3-seed) |
| 2.4 | **frm_ math seeds 43/44** | 3-seed error bars on the 67.3 math headline and its comparators. | 7 cells (inside item 1.0) | **QUEUED** (frm, master dispatch) **→ CLOSED 07-14** (six 3-seed frm configs; headline 66.79±0.79 verified; SC-LoRA math on-law, resid −1.8 pp) |
| 2.5 | **B4 expansion** (calibration cells beyond the 4 landed `b4_sclora_*`) | Widens the eval-matched-calibration verdict that already lifted SC-LoRA to 26.5–27.0 (at/above the 26.0 ceiling); feeds O3. | 5 cells (inside item 1.0) | **QUEUED** (b4, master dispatch) **→ PARTIAL 07-14** (`b4_lora_null_r16_lr2e5` landed, ret 26.75 @ F_Δ=0.149 ≈ ceiling; lora_null lr1e4/3e4 + cordapp full evals pending) |
| 2.6 | **B4 `_em` seeds 43/44 at near-frontier LRs** (07_plan Pri 5) | Error bars on the fairness verdict itself — O3's residual risk is "your resolution of the confound is itself n=1." | ~18 Llama cells ≈ **0.25 GPU-day** | queued nowhere (new; only if 2.5's verdict is tight) |
| 2.7 | **B5b — wd knob on MiLoRA + DoRA** (REBUTTAL_PREP O4/O11 upgrade) | `milora_r32_wd0p3`, `dora_r16_wd0p3` × 7 LR: upgrades Claim 3 to "wd helps everyone; geometry adds nothing on top" — the strongest anti-O4 and anti-Biderman(O11) form. Optional; run if 1.1's verdict is anything but crushing. | 14 cells ≈ **0.2 GPU-day** | queued nowhere (new, optional) |
| 2.8 | **`frc_reservoir_B.txt`** (40 cells: CorDA++/LoRA-Null/MiLoRA/SC-LoRA faithful-math + α=1r) | Structured-method math breadth (N1 already gives n=35; this widens it) + the CorDA++ math arm. | ≈ **0.7 GPU-day** | **QUEUED** (reservoir, behind master) |

**Tier 2 subtotal (new dispatch): ≈ 0.35–0.55 GPU-day (+0.7 if the reservoir is promoted).**

REBUTTAL_PREP items needing **no** data (do during the compute wait): O5/O9 re-analyses are DONE; O11 (Biderman) is an in-text reconciliation, done; O12 (why re-run) is framing, done; the W4 rank-partial respec and the B3/B4-audit number fixes are edits/recomputes on existing data.

---

## 4. Tier 3 — nice-to-have / camera-ready

| # | Item | Verdict | Cost |
|---|------|---------|------|
| 3.1 | **Third model family** (e.g. Mistral-7B CS, 7 adapters × 7 LR × s42) | Not demanded by any doc; O6/O8 are answered by Qwen + math. Only if the October cycle slips or reviews explicitly ask for it. | ~49 cells, ≈ **1.5–3 GPU-days** (speed unknown; assume Llama-like) |
| 3.2 | **Rank ladder for the W4 rank-partial spec** | W4's primary fix is analysis-only: the rank cells (`lora_r4…r256`, `mtx_lora_r8…r128`) ARE in the registry (the paper's "not in registry" footnote is false); recompute with a stated spec (−0.69…−0.74 raw; sign-flips under log-rank — so soften the claim). A *clean* dedicated ladder (r∈{4…256} × 1 fixed LR × 3 seeds ≈ 21 cells) is only needed if we want to keep a hard rank-partial number in the paper. | 0 GPU (respec) / **0.3 GPU-day** (clean ladder) |
| 3.3 | **Instrumented peak-memory for the efficiency table (W3)** | Text fix now ("~15% overhead; 6.7 GB is a k2048 *projection*, swept max k1024 ≈ 3.3 GiB, analytical"). Camera-ready: short profiling runs (few steps, `torch.cuda.max_memory_allocated`) for LoRA / DoRA / CLoRA k-ladder. | ≈ **0.05 GPU-day** |
| 3.4 | **DoRA true-ΔW recompute (REBUTTAL_PREP O13)** | Re-run the 7 DoRA CS cells retaining checkpoints, recompute ΔW with the magnitude vector. Correction direction can only push DoRA further above the curve — disclosed as a lower bound, so explicitly deferred to camera-ready. | 7 cells ≈ **0.1 GPU-day** + recompute |
| 3.5 | **CorDA++ advanced arm (C6)** | "You strawmanned SOTA" insurance; strong-tier only. Note `frc_reservoir_B` already carries CorDA++ *math* cells — a CS arm is the remaining gap. Requires realized-param-count matching (dynamic rank breaks nominal parity). | ≈ **3–4 GPU-days** |

---

## 5. Execution schedule (T0 = day compute returns)

Constraints: one 8-GPU scheduler, never two pools; C5 evals need no training pool (fit into any
pause); analysis items are GPU-free and run continuously.

| Window | GPU track (single scheduler) | Parallel no-GPU track |
|--------|------------------------------|----------------------|
| **T0 → T0+1.5d** | Drain `master_dispatch.txt` backlog (58 cells, ≈1 GPU-day) — lands DoRA s44, frm seeds, b4 expansion, frc wd-ablation. Slot the 5 C5 eval-only runs into gaps. | Add Tier-1/Tier-2 lines to dispatch (§6); CorDA 0-step/reload validation prep; W4 respec; W5 ANCOVA script ready to fire on DoRA-s44 arrival; audit B1–B5 text edits (paper is 9 days stale). |
| **T0+1.5d → T0+2.5d** | **Tier-1 Llama block** (≈0.5 GPU-day): B5a 14 cells → CorDA CS 7 (+7 `_em` optional) → Tier-2 MiLoRA basin 4 cells. Order B5a→CorDA because B5a is claim-critical, CorDA is coverage. | Run W5 seed-averaged ANCOVA (2.2); regenerate fig2/ANCOVA exhibits; frm-seed math table with 3-seed bars. |
| **T0+2.5d → T0+5.5d** | **Qwen block** (≈2.5 GPU-days): qwswm high-LR ~39 cells (≈1.7d) → qwsw headline seeds 14 cells (≈0.7d). This is the wall-clock tail — start it as early as the Llama block allows. | B5a 2×2 analysis (rank main-effect test); CorDA verdict → coverage-sentence rewrite (O1); B4 expanded verdict (O3). |
| **T0+5.5d → T0+6.5d** | **Tier-2 contingent cells** (≈0.5 GPU-day): B4 `_em` seeds (2.6) and/or B5b wd-on-MiLoRA/DoRA (2.7), only where the Tier-1 verdicts were tight. Promote `frc_reservoir_B` (0.7d) if schedule allows. | Qwen-math law verdict → O8 rewrite; Qwen op-point error bars. |
| **T0+6.5d → T0+9d** | (idle / reservoir / Tier-3 profiling 3.3) | Full figure/table regeneration from live registry; all audit blockers B1–B5 + W1–W5 closed in text. |

**Wall-clock to "all Tier 1 + Tier 2 landed": ≈ 6–7 calendar days of compute + ≈2–3 days analysis
overhang ⇒ T0+9 days to an internally consistent, fully-updated draft.**

### Back-plan against ARR October (submission ≈ Oct 13–15)

| Milestone | Date | Slack logic |
|---|---|---|
| Submission (ARR October) | **~Oct 13–15** | fixed |
| Buffer (nothing scheduled) | Oct 6 → Oct 13 | 1 week — protects against a bad B5a/CorDA verdict forcing a reframe |
| **Freeze** (numbers/figures locked, key_numbers final recompute) | **Oct 2** | after internal review integration |
| Internal review round (supervisor + one hostile read vs REBUTTAL_PREP) | Sep 15 → Sep 26 | 2 weeks incl. response edits |
| Full draft with all Tier-1/2 data integrated | **Sep 14** | = T0 + 9d + ~4d writing polish |
| ⇒ **Latest comfortable T0** | **~Sep 1** | 9d compute+analysis + 4d polish → Sep 14 draft |
| ⇒ Crash floor (Tier 1 only, no 2.6/2.7, internal review compressed to 1 wk, buffer 3d) | **T0 ≈ Sep 22** | do not plan for this |
| **If compute returns before ~Aug 20** | — | run Tier 3.1/3.5 (third model / CorDA++ CS) in the slack; else defer to camera-ready |

Everything in §3 marked "no-GPU" plus the audit's B1–B5 text blockers should be finished **before
T0** — the compute wait is writing time, not dead time.

---

## 6. Queue actions — concrete additions

Mechanics (07_plan): new arms = entries in the `ARMS` dict of `make_campaign_jobs.py`; seeds = edit
`SEEDS`; the generator is resumable (skips cells with an existing `summary.json`) and emits the
dispatch lines. Run names follow `<prefix>_<arm>_<lr>_<seed>` as seen in the audit. LR grid = the
7-entry `LRS` (2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3).

**A. New ARMS entries (from 07_plan §B5a/§B4, verbatim specs):**
```
"lora_r32":         "--method lora --lora_r 32 --lora_alpha 64",
"lorawd_r16_wd0p3": "--method lora --lora_r 16 --lora_alpha 32 --weight_decay 0.3",
"corda_r16_nq":     "--method lora --corda 1 --lora_r 16 --lora_alpha 16",                          # clean nq_open re-run
"corda_r16_em":     "--method lora --corda 1 --lora_r 16 --lora_alpha 16 --calib_source eval_matched",  # optional B4 twin
# Tier 2 optional (B5b):
"milora_r32_wd0p3": "<milora flags> --weight_decay 0.3",
"dora_r16_wd0p3":   "<dora flags> --weight_decay 0.3",
```

**B. Lines to add to `master_dispatch.txt`** (7 LRs each unless noted; s42 unless noted):

```
# Tier 1.1 — B5a param-match 2x2 completion (14 cells, QUEUED NOWHERE, W2)
#   → CLOSED 07-14: landed via the frc c256 recipe (frc_lorawdr16_* + frc_lora_r{8,16,32}_lr3e4).
#   Do NOT dispatch these lines; only the r16-collapse seed replicate remains:
#   frc_lorawdr16_wd0p3_lr3e4_c256_s43 (anomaly replicate)
lrsw_lorawd_r16_wd0p3_lr{2e-5,5e-5,1e-4,2e-4,3e-4,5e-4,1e-3}_s42
lrsw_lora_r32_lr{2e-5,5e-5,1e-4,2e-4,3e-4,5e-4,1e-3}_s42

# Tier 1.2 — CorDA clean nq_open CS re-run (7 cells; GATE: 0-step dW=0 + post-reload init-invariance)
#   fresh arm name (corda_r16_nq) so contaminated lrsw_corda_* rows are never silently superseded
lrsw_corda_r16_nq_lr{2e-5,5e-5,1e-4,2e-4,3e-4,5e-4,1e-3}_s42
# optional +7: lrsw_corda_r16_em_lr{...}_s42   (folds CorDA into the B4 verdict)

# Tier 1.3 — C5 base ceilings (eval-only, no training; schedule in any pool pause)
#   → Llama-2 line CLOSED 07-14 (base_llama2_noft landed 07-12, one row covers all 5 benchmarks).
#   The Qwen line is the only one still to dispatch.
b4_base_noft_llama2_{mmlu,arcc,tqa}_eval          # + bbh/mmlupro already known (33.10 / 18.96)  → CLOSED 07-14
b4_base_noft_qwen25_{mmlu,arcc,tqa,bbh,mmlupro}_eval   # recommended, since Qwen broad retention is now quoted — STILL OPEN

# Tier 1.4 — Qwen math high-LR completion: IMPORT frepro4_qwen.txt into the active dispatch
#   (do not retype; ~39 cells, lr5e-4/1e-3 x 6 adapters resolution block). Minimal slice if trimmed:
#   → CLOSED 07-14: run + ingested (51 qwswm cells appended to the registry, lead-approved);
#   nothing remains for math s42. Do NOT dispatch.
qwswm_{lora,lorawd_wd0p3,dora,milora,sclora,clora_k1024}_lr{5e-4,1e-3}_s42

# Tier 1.5 — Qwen CS seeds, headline (best-adapt) cells only (14 cells)
qwsw_{lora,lorawd_wd0p3,lora_null,dora,milora,sclora,clora_k1024}_lr<best-adapt>_s{43,44}

# Tier 2.1 — MiLoRA collapse-basin probe (4 cells)
lrsw_milora_r32_lr{2e-4,5e-4}_s{43,44}

# Tier 2.6 — B4 _em seeds at near-frontier LRs (~18 cells; contingent on 2.5 verdict)
b4_{sclora,lora_null,corda}_em_lr<2-3 near-frontier LRs>_s{43,44}

# Tier 2.7 — B5b wd-for-everyone (14 cells; contingent on 1.1 verdict)
lrsw_milora_r32_wd0p3_lr{...}_s42
lrsw_dora_r16_wd0p3_lr{...}_s42
```

**C. Already queued — verify present, do NOT duplicate:** `lrsw_dora_*_s44` (1), `frm_*_s{43,44}`
(7), `b4_*` expansion (5), `frc_lorawd_wd0…wd0.5` c256 ablation + faithful-math structured (45);
`frc_reservoir_B.txt` (40) stays as the promotable reservoir.
*→ 07-14: DoRA s44, the frm seeds, and the frc block have LANDED; of the b4 expansion only
`b4_lora_null_r16_lr2e5` is in the registry (lora_null lr1e4/3e4 + cordapp full evals remain).*

**D. Registry hygiene at dispatch time:** keep the latest-`evaluated_at` dedup rule; exclude all old
`lrsw_corda_*` rows from any fit regardless of timestamp (contaminated wikitext re-eval, one
F_Δ=515 explosion); never quote Qwen-math core retention until the MMLU-Pro math parser is fixed
(BBH-only, key_numbers §11). *→ 07-14 framing update (§16): keep the BBH-only convention, but
describe MMLU-Pro as "excluded by convention (unreliable on early Qwen-math rows)", not "broken" —
the new r32 rows' MMLU-Pro values pass sanity.*
