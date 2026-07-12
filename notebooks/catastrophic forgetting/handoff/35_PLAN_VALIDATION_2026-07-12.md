# handoff/35 — PLAN VALIDATION & PRIORITIZED TODO (research-planner, 2026-07-12 ~12:30 Sun)

Validates the Qwen 3-seed design + both-node queue priorities against the **artifact**
(`paper/writing/artifact_status_report.html`, the PRIMARY deliverable; paper.tex is PAUSED by PI —
no paper work planned here beyond noting it's parked). Every number below was re-verified live this
pass against `results/*/summary.json` and `results/forgetting*.jsonl` (non-finite CE cells filtered).
Node B state confirmed read-only via ssh. **I did not launch/kill/modify anything.**

## 0. CLOCK, CAPACITY, LIVE STATE (verified this pass)
- **Now: Sun 2026-07-12 ~12:30.** Nodes returned end of week; **the live threat is a hard Monday-morning cutoff.**
- **Node A** (`master_dispatch.txt`): **48 pending** / 44 done. Top pending (verified): `frc_dora_r32_lr3e4`,
  `frm_sclora_lr1e4`, `frm_lora_null_lr1e4`, `frc_lorawdr16_wd0p3_lr3e4`, then the `frc_lorawd` wd×lr grid.
- **Node B** (`frcB.txt`): **25 pending**, 8 GPUs busy (util 81–94%), frcB dispatcher alive.
  `chain_qwen3seed.sh` **running and correctly armed** (last poll `pend=25 disp=alive procs=8`); qw3s
  dispatcher **not yet launched** (correct — waiting on frcB drain). `frcB.txt` **exists on Node B**
  (the chain's drain-detector works; the Node-A copy is named `frc_reservoir_B.txt`, 26 pending — a naming
  difference only, not a fault).
- **qwen3seed_B.txt**: 28 cells, **0 done**. No qwen s43/s44 cell exists anywhere yet.
- Measured pace ≈ **1 cell/h/node** (7 GPU-h per 7B cell, 8-wide). frcB drains ~**Mon midday**; qw3s
  (28 cells ≈ 28 wall-h) then finishes ~**Tue evening**.

---

## 1. VALIDATION VERDICTS (tasks A / B / C)

### A. Qwen 3-seed design — VALID, with two gaps
- **The 14 operating points are the RIGHT cells.** All 14 (`qwsw_*` ×7 CS + `qwswm_*` ×7 math) match
  the artifact's per-adapter best-LR points **exactly** (verified to 2 dp against s42 summaries; e.g. CS
  LoRA+wd 87.48/41.01/0.137, SC-LoRA 87.07/**9.44**/0.441; math LoRA+wd 66.64/47.57/0.104, SC-LoRA
  76.88/45.85/0.183). CorDA correctly excluded (1-LR, off-law). Job lines are verbatim s42 clones — recipe-identical. Good.
- **Ranking is sound.** Pairs both seeds of each point consecutively (yields a completed n=3 at a point
  rather than two half-done points), and front-loads the three highest-value points: math LoRA+wd headline
  → CS LoRA+wd → **CS SC-LoRA 9.4 outlier**. Under a hard cutoff this is the correct order.
- **"2 extra seeds at operating points" beats "3-seeding the law-fit cells" — correct choice.** The law
  (r=−0.86, n=49) is a *within-single-seed cross-cell correlation*; tripling its points adds little to the
  coefficient. What is currently single-seed and *ranking-bearing* is the **operating-point tables** — so
  seeding those is the right target, and it de-risks the one shock number (SC-LoRA CS 9.4).
- **GAP A1 (missing, high-value, cheap): Qwen base no-FT eval.** The artifact infers the Qwen retention
  reference ("no direct no-finetuning eval was run … lowest-update runs bound the base reference at ≥ 41";
  math "base BBH ≈ 48"). The entire "LoRA+wd pins retention at the base ceiling" story (41.0 CS / 47.6 BBH)
  rests on an **inferred** ceiling. Not in any queue. See TODO #4.
- **GAP A2 (thin evidence): the Qwen SC-LoRA CS 9.4 is a single seed** displayed as a hard "bad" number,
  while the artifact itself calls SC-LoRA's F_Δ "seed-unstable" on Llama. Until s43/s44 land it should carry
  a one-line single-seed caveat (TODO #1) — cheap insurance against a reviewer distrusting a 4× outlier.

### B. Queue-priority audit — top-5 highest-value cells queued NOWHERE
Ranked by value ÷ cost (all verified absent from `master_dispatch`, `frc_reservoir_B`, `qwen3seed_B`):
1. **Qwen base no-FT broad eval** (BBH/MMLU-Pro/MMLU/ARC/TQA) — calibrates the whole Qwen retention axis. ~1–2 GPU-h + patch.
2. **Llama base-ceiling no-FT eval** for MMLU/ARC/TruthfulQA (`key_numbers §12`: "uncalibrated"; artifact broad battery uncalibrated for 3 of 5). ~1–2 GPU-h, **same patch/session as #1**.
3. **Instrumented peak-memory** (efficiency §5 — the one *true-missing* deliverable; §5 hedges "peak GPU memory not instrumented", CLoRA +0.42–6.7 GB ladder is analytical). 4–5 short profiled runs ~8 GPU-h + ~5-line patch.
4. **CLoRA CS boundary error bars** `frc_clora_k1024/k2048 @lr3e4 s43,s44` — §3 boundary box makes a *ranking* claim (k2048 above base; 68.5 "seed-fragile") from one seed. 4 cells ~28 GPU-h.
5. **Killed Qwen-math CLoRA tail** `qwswm_clora_k1024_lr5e4_s42` — **LOW value, do NOT hand-requeue.** Verified: 6 CLoRA Qwen-math LRs remain (2e5,5e5,1e4,2e4,3e4,1e3) spanning the range; the lost point is interior. Qwen-math law is already robust at **n=47** without it.

### C. Artifact claims thinner than their wording / would strengthen with Monday-night data
- **Qwen operating-point rankings** ("LoRA+wd again pairs best accuracy with best retention on both arms")
  are single-seed. Artifact *frames* Qwen as "replication breadth, not error bars" (§3) so it is **currently
  guardrail-compliant** — but the flagship upgrade to mean±SD is exactly what qw3s delivers, and it likely
  **won't land** (RISK-1). Keep the breadth framing until it lands.
- **Qwen math law is mildly stale: artifact says n=42; live recompute gives n=47** (r=−0.70, slope −15.0 —
  **unchanged**; 5 cells landed since the artifact fit). Zero-GPU freshness refresh (TODO #2).
- **CE stats verified**: math ρ=0.976 (n=49 frm_ + 1 CorDA++ = 50 ✓), CS-arm ρ=0.944 (~claimed 0.946).
  **9 non-finite CE cells must be filtered** (lr1e3 divergences + 4 MiLoRA) — they are; keep filtering.
- **Efficiency §5** peak-memory is honestly disclosed as un-instrumented — instrumenting it (TODO #7) closes the last true gap.

---

## 2. THE PRIORITIZED TODO PLAN

Order = value × (probability-it-lands-before-cutoff) ÷ cost. PI's fixed decisions (wait-then-launch;
paper parked) are respected, not relitigated — risks are flagged, not overridden.

**1. Add single-seed caveats to the Qwen tables (SC-LoRA 9.4 + math bold points).**
   WHAT: one line under the Qwen CS table — "Qwen points are single-seed (s42); SC-LoRA's F_Δ is seed-unstable
   (see §3), so read its 9.4 as one draw pending s43/s44." WHY: protects the §3-Qwen ranking claims and the
   9.4 outlier against a distrustful reviewer (guardrail: bold claims need 3 seeds). COST: **writing, 0 GPU.**
   WHERE: artifact. TRIGGER: now. STATUS: **do first.** ORDER: zero-cost credibility insurance that is
   correct regardless of whether qw3s lands.

**2. Refresh the Qwen-math law n (42 → 47) in the artifact.**
   WHAT: recompute pooled Qwen-math BBH-vs-log-F_Δ over the 47 live non-degenerate cells; update the §1
   "n = 42" to n = 47 (r=−0.70, slope −15.0 — I verified these; exclusions = 1 SMOKE + 1 collapsed
   `qwswm_lorawd_wd0p3_lr1e3` F_Δ 15.8/acc 0). WHY: §1 "second architecture, math — in full" claim; keeps the
   artifact matching live data. COST: **CPU-only recompute + writing, 0 GPU.** WHERE: artifact. TRIGGER: now.
   STATUS: ready. ORDER: cheap, removes a staleness a numeric auditor would catch.
   MISALIGNMENT: artifact n=42 predates 5 landed cells; r/slope unaffected, so low-severity.

**3. Keep Node A draining in current order — it is correctly prioritized.**
   WHAT: let `master_dispatch` run: `frc_dora_r32` (CS-table DoRA row) → `frm_sclora_lr1e4` +
   `frm_lora_null_lr1e4` (the missing §6 CE math rows) → `frc_lorawdr16_wd0p3_lr3e4/5e4` (the **W2
   param-matched r16 LoRA+wd control** — one already landed per git status). WHY: §6 CE table completion +
   closes the W2 "wins on capacity + an extra knob" referee objection. COST: queued, ~4 cells ×7 = ~28 GPU-h,
   Node A. TRIGGER: running. STATUS: **on track, no action.** ORDER: highest-value *queued* Llama cells; they
   fold straight into §6 and close an open referee objection.

**4. Base no-FT eval, both models (Qwen + Llama) — the top unqueued win.**
   WHAT: patch `eval_one_gpu.py` (currently `--adapter required=True`, no base mode) to accept a no-adapter
   pass; run base broad-retention on **Qwen2.5-7B** and **Llama-2-7B** (BBH/MMLU-Pro/MMLU/ARC/TQA). WHY:
   converts the Qwen ceiling from inferred "≥41 / ≈48" to measured (props the LoRA+wd-at-ceiling headline on
   both Qwen arms) and calibrates the Llama broad battery's 3 uncalibrated benchmarks (`key_numbers §12`).
   COST: **~2–4 GPU-h total** + ~10-line patch (CPU/writing). WHERE: either node's spare capacity (evals are
   short). TRIGGER: needs the patch first (small; assign to author/ops — NOT a planner action). STATUS:
   **highest value-per-GPU-h not in any queue.**
   MISALIGNMENT: ~20 marginal `frc_lorawd` wd-grid cells sit queued on Node A *ahead* of this un-queued base
   eval that calibrates the entire retention axis — the base eval is worth more than the wd-grid tail.

**5. WAIT-THEN-LAUNCH Qwen 3-seed (qw3s) — the PI flagship; ranking validated, DO NOT reorder.**
   WHAT: chain auto-launches the 28-cell qw3s dispatcher when frcB drains (~Mon midday). WHY: upgrades the
   Qwen operating-point tables from single-seed to mean±SD — the PI's stated #1 priority today. COST: 28 cells
   ×7 ≈ 196 GPU-h, Node B. TRIGGER: `chain_qwen3seed.sh` (armed, verified). STATUS: **armed, correct.** ORDER:
   fixed by PI; ranking already front-loads the 3 highest-value points.
   **RISK-1 (SEVERE, PI-accepted): under a hard Monday-morning cutoff, ~ZERO qwen seed cells land** — frcB
   won't have drained, so qw3s never starts. The PI's #1 priority is scheduled to land *last* and may not land
   at all. PI was told and kept the order. **Contingency IF a cutoff looks imminent** (supervisor's call, not
   mine): the ranking's top 2 points (math LoRA+wd s43/s44, CS SC-LoRA 9.4 s43/s44 = 4 cells ≈ 28 GPU-h) are
   the only ones worth an emergency hoist onto spare Node-A capacity; everything below position 6 is bonus.

**6. CE `chunk_new` before `chunk8`; then chunks 2–8.**
   WHAT: ensure `ce_batch.py --runs_file jobs/ce_chunks/chunk_new.txt` runs ahead of the `chunk8` catch-all
   (the 28 newest Llama adapters — incl. §3 seed replicates + b4 — otherwise only chunk8 covers them). WHY:
   CE error-bars on the §3/§6 headline replicates. COST: queued, Node A. TRIGGER: as A drains. STATUS: queued.
   ORDER: cheap completeness; secondary (primary §6 op-points are already scored, ρ=0.976 verified).

**7. Instrumented peak-memory (efficiency §5) — closes the last true-missing deliverable.**
   WHAT: ~5-line patch logging `torch.cuda.max_memory_allocated`; 4–5 short profiled train+eval runs (LoRA,
   DoRA, CLoRA-k1024, one SVD-init, one calib method). WHY: §5 currently hedges "peak GPU memory not
   instrumented" and the +0.42–6.7 GB CLoRA ladder is analytical (audit W3). Turns an analytical claim into a
   measurement. COST: ~8 GPU-h + patch. WHERE: Node A/B bonus capacity. TRIGGER: PI bonus window
   (handoff/33 §c item 2). STATUS: bonus. ORDER: only true coverage gap, but needs a patch and only matters if
   PI wants the efficiency table instrumented before freeze.

**8. CLoRA CS boundary error bars — `frc_clora_k1024/k2048 @lr3e4 s43,s44`.**
   WHAT: 4 seed cells on the §3 boundary. WHY: the §3 boundary box's ranking/"seed-fragile 68.5" claims are
   single-seed. COST: ~28 GPU-h, Node A. TRIGGER: PI bonus window (handoff/33 §c item 1). STATUS: bonus.
   ORDER: below the base evals — the boundary is already honestly hedged as "boundary not refutation," so error
   bars are a nice-to-have, not load-bearing.

**9. STANDING PI workstreams (CPU-only, no GPU, do anytime).**
   (a) **Geometry-drift principal angles**: the §2 battery already does energy-projection (e_top/ein_top etc.)
   from saved ΔW; add per-layer/model/adapter **principal angles of ΔW vs base-W subspaces** as a
   complementary read (no training). COST: CPU-only. WHY: PI geometry-drift workstream; corroborates "geometry
   acts through size/allocation." (b) **Ceiling-aware stats**: §1 already carries Spearman ρ=−0.90 + spline
   knee + below-ceiling (Tobit) slope −21 — PI's ceiling-aware workstream is **largely satisfied**; optional
   add = an explicit censored (Tobit) fit line in the artifact text. COST: CPU-only. ORDER: opportunistic;
   they strengthen §1/§2 without touching the GPU budget or the cutoff risk.

**10. DO NOT DO (explicitly de-prioritized).**
   - Hand-requeue the killed `qwswm_clora_k1024_lr5e4` (verified low-value; law robust at n=47). SKIP.
   - CorDA clean nq_open CS re-run (queued nowhere; PI keeps CorDA→CorDA++ math-only; big lift, off-message). SKIP.
   - Any paper.tex work — PAUSED by PI; ASK before touching (audit B1–B5 remain, but parked).
   - 3-seeding the law-fit cells instead of operating points (lower marginal value — see verdict A).

---

## 3. MISALIGNMENTS & RISKS (collected)
- **MISALIGNMENT-1:** Base no-FT calibration (Qwen ceiling + Llama MMLU/ARC/TQA) is un-queued while ~20
  marginal `frc_lorawd` wd-grid cells sit queued ahead of it on Node A. The base eval calibrates the entire
  retention axis at ~2–4 GPU-h; it out-values the wd-grid tail. (TODO #4)
- **MISALIGNMENT-2 (low-severity):** Artifact Qwen-math law says n=42; live data is n=47 (r/slope unchanged). (TODO #2)
- **RISK-1 (SEVERE, PI-accepted):** wait-then-launch means the PI's #1 priority (Qwen 3-seed) likely lands
  ZERO cells under a Monday-morning cutoff. Contingency = hoist the top 4 cells only, if a cutoff looks
  imminent (supervisor's call). (TODO #5)
- **RISK-2 (thin evidence):** Qwen SC-LoRA CS 9.4 is a single-seed 4× outlier shown as a hard number; caveat
  it until s43/s44 land. (TODO #1)
- **RISK-3 (data hygiene):** any CE correlation MUST filter the 9 non-finite cells (lr1e3 divergences + 4
  MiLoRA); verified done in the live stats — keep this in every future CE recompute.

## 4. CHANGED SINCE handoff/33 (last planner version)
- **Qwen 3-seed is now the live top priority** (new PI order today); 28-cell `qwen3seed_B.txt` built + chain
  armed. handoff/33 predated this entirely. Design validated here (14 points exact, ranking sound).
- **handoff/33's "one true-missing deliverable = instrumented peak-memory" still stands** (now TODO #7) — but
  a **second, higher-value unqueued gap surfaced: base no-FT calibration on both models** (now TODO #4/#1 of the top-5).
- **W2 param-matched r16 LoRA+wd control is no longer a gap** — `frc_lorawdr16_*` is queued/landing on Node A.
- **CLoRA boundary + CS baselines + math SC-LoRA/LoRA-Null rows** that handoff/33 flagged as "behind the freeze
  line" have largely **landed** (frm CE rows are next on A; boundary k-grid is in the artifact §3).
- **New: Qwen-math law freshness (n 42→47)** and the **killed CLoRA tail confirmed low-value** (don't requeue).
- Verified live this pass: 48 A-pending / 25 B-pending / 0 qwen-seed-done; Node B chain armed & 8 GPUs busy;
  math 3-seed headline 66.79±0.79 / 33.57±1.04 exact; CE ρ=0.976 (math) / 0.944 (CS) with NaN-filtering.
