# DATA-COMPLETENESS AUDIT (adversarial, pre-freeze) — 2026-07-12

(Authored by the adversarial-critic agent; persisted by the supervisor — the agent's
sandbox blocked the file write. Content verbatim from its final report.)

**Method.** Every load-bearing statistic re-derived from `results/*/summary.json`,
`results/geo_drift/master_labeled.jsonl`, and `results/forgetting*.jsonl` with
`/home/guy/UIOrthoLoRA/.venv/bin/python`. Goes beyond handoff/35 (which validated the
queue plan). Read-only; no jobs/queues touched. ALREADY-QUEUED items (Qwen 3-seed, base
no-FT evals, peak-mem probes, frc_clora_k2048 s43/s44, CE chunks) are not re-flagged.

## 0. Reconciliation — supervisor's CS-CE count of 20 is CORRECT; prior "13" retracted
Re-run on the 3-file union deduped by `run_name`, non-finite filtered, **arm assigned by
each run's `summary.json` `adapt_task`** (not by name-prefix): union=78, 9 non-finite,
**CS-arm=20**, math-arm=49. CS Spearman **0.9459** (→§6 ρ=0.95 ✓), CE range
**1.984→2.806** (→"1.98→2.81" ✓), F_Δ **0.254→0.988** (→"0.25–0.99" ✓), **7 families** ✓.
The error: classification by prefix (only `lrsw_`→CS) dumped the 4 `b4_sclora_*` and 3
`frc_lorawd_*` CS cells into math. **§6's 20-cell CS-CE claim is fully backed — NOT a
gap.** (Future rule: classify CE arm by `adapt_task`, never prefix.)

## 1. BLOCKERS (claim↔data mismatch; must fix before freeze; all zero-GPU)

**BLOCKER-1 — §4 "retention at or above the 33.1 base reference in all three" is FALSE.**
The 3 seeds of `frm_lorawd_wd0p3_lr2e4_c256_{s42,s43,s44}` are BBH **33.10 / 34.77 /
32.85**. The **s44 seed (32.85) is 0.25 pp below the 33.1 base** — only 2 of 3 are ≥
base. Fix: *"BBH 33.57 ± 1.04 (33.10/34.77/32.85) — every GSM8K seed above 64.6, and the
3-seed mean at or above the 33.1 base (two seeds above, the third within 0.3 pp)."*

**BLOCKER-2 — §1 "4-task capability composite (r=−0.945, slope −13.7)" is actually the
5-task composite INCLUDING TruthfulQA.** The honest 4-task capability composite
mean(BBH,MMLU-Pro,MMLU,ARC-C), excluding the TQA control, is **r=−0.937, slope −16.96**.
The −0.945/−13.66 reproduces exactly from `retention_broad` = the **5-task mean that
includes TruthfulQA** — contradicting the doc's own rule (stated two paragraphs up): TQA
*"never in a retention composite."* Fix (preferred, keeps the rule): *"the 4-task
capability composite … r=−0.937, slope −17.0 pp/decade"* (law survives — |r| up vs the
−0.86 headline pair, slope steeper). Also fix the same mislabel in handoff/34's ledger
and key_numbers.md so it doesn't re-propagate.

**BLOCKER-3 — §6 "covers every assessed method family" but SC-LoRA & LoRA-Null math CE
rows don't exist.** `frm_sclora_*` and `frm_lora_null_*` result dirs are absent; the §6
math table has 7 rows and omits SC-LoRA and LoRA-Null on the math arm. Fix: land the
queued cells (GAP-1) or soften to *"covers the families whose faithful-math cells have
landed (SC-LoRA/LoRA-Null pending)."*

## 2. GAP-FILLABLE (fits Node-A ~40 GPU-h; exact specs)

**GAP-1 (closes BLOCKER-3)** — `frm_sclora_lr1e4_c256_s42` + `frm_lora_null_lr1e4_c256_s42`
(already on `master_dispatch`) **then CE-score them**. ≈15–18 GPU-h. Ops action: confirm
BOTH the train cells and their CE scoring land before freeze; else apply the BLOCKER-3
wording.

**GAP-2** — queued list covers only `frc_clora_k2048_lr3e4 s43/s44`, but the §3 boundary
box makes a ranking claim on BOTH boundary points. Add **`frc_clora_k1024_lr3e4_c256_{s43,s44}`**
(2 cells ≈14 GPU-h) so the k1024 "still below LoRA+wd" point is also error-barred. Low
priority (box is hedged), but it's the one boundary cell the queue misses.

## 3. WORDING (zero-GPU)
- **W1 — Llama math law: unify n + lead with honest within-support r.** §1 caption says
  n=43, §4 says n=49 for the same sweep (live: 52 with c512 / 43 c256-only). BBH vs log
  F_Δ: full **r≈−0.92/−13.8**; excl 2 diverged collapse cells (F_Δ>50, acc 0) **r≈−0.83**;
  within trained regime **F_Δ<5 → r≈−0.76**. The −0.92 is inflated by 2 collapse leverage
  points. Pick one cell set (recommend c256-canonical n=43), state it in both §1 and §4,
  keep the "diverged excluded" pattern **plus** the within-support number.
- **W2 — Disclose the math arm is LoRA+wd-dominated:** ≈39 of ~52 faithful-math cells are
  LoRA+wd; competitors contribute ~11–13 cells mostly at a single LR (3e-4). It's a
  within-LoRA+wd magnitude sweep with sparse competitor anchors, not a full 7-adapter
  sweep like CS. (Full cross-method math sweep = 42 cells ≈210 GPU-h — does NOT fit;
  disclosure fix only.)
- **W3** — §4 BBH bar shows 33.1 (s42) labeled "3-seed mean (33.57±1.04)"; and "sits
  exactly at base 33.1" uses s42, not the mean (33.57 is above base). Reconcile.
- **W4** — Header tiles use s42 peaks (CS 81.6/25.6; math 67.3) while §3/§4 use 3-seed
  means (81.8/25.9; 66.8). Claims survive on the means; align tiles or mark "s42 point."
- **W5** — §6 "50 scored math cells" is 49 live (Spearman identical, 0.976). Refresh 50→49.
- **W6** — Qwen §3 tables/rankings stay single-seed (`qw*_s43/s44`=0 live; qw3s likely
  lands zero before cutoff per RISK-1 — NOTE: superseded, PI extended nodes to Tue
  morning, qw3s expected to land). Keep the "breadth, not error bars" framing until
  seed cells actually land; ensure no downstream text states Qwen *rankings* as
  error-barred.
- **W7** — Document the §2 geometry fingerprint pooling: displayed values aren't
  reproducible by a naive per-cell mean (SC-LoRA ein_top 0.50 naive vs 0.41 shown;
  stable-rank ~2× lower — LoRA 4.4 vs 8.8). Qualitative signatures all hold; state the
  pooling (energy-weighted, ‖ΔW‖²) and over which cell set, for auditability.

## 4. VERIFIED (reproduced to displayed precision)
- **CS law n=49:** pooled r=−0.858, slope −14.78, R²=0.736, Spearman −0.896;
  within-method −0.864…−0.972; LR R² 0.32 vs F_Δ 0.74; battery BBH −0.79(−14.3)/
  MMLU-Pro −0.89(−15.2)/MMLU −0.93(−23.4)/ARC-C −0.93(−14.9)/TQA −0.10(−0.5);
  partial-r(adapter) −0.868; LR-demeaned −0.817/−20.6; partial-r(LR,ret|F_Δ) +0.461;
  permutation p≈5×10⁻⁶; knee≈0.37→past-knee slope −20.5, resid SD 0.60/3.46. All ✓.
- **Qwen:** CS core −0.857/−31.98 (n=49); CS broad −0.937/−26.1; math n=47 r=−0.70
  slope −14.98 (n refresh correct). ✓
- **3-seed CS operating points (all 7):** exact ±SD match. **3-seed math headline**
  66.79±0.79 / 33.57±1.04 (means ✓; see BLOCKER-1 for the "all three ≥ base" wording).
- **CE §6:** all 8 math rows exact; headline seed noise ±0.005; MiLoRA↔LoRA 3.66↔3.57;
  ext-repro LoRA 3.57/PiSSA 6.31, ratio 1.77; CS 20 cells (0.9459) + math 49 cells
  (0.9759) back their claims; 9 non-finite correctly filtered.
- **CLoRA Table 4:** all-10 fit r=−0.9805, slope −14.65; baselines −12.66, k-series
  −18.86 — matches extraction and overlay exactly. Overlay "ours" fit (−14.34, r −0.79,
  n=49) reproduces.
- **Geometry:** battery = exactly 320 rows; every design signature reproduces (MiLoRA
  minor, SC-LoRA input-principal, CorDA input-minor, LoRA-Null e_top); SC-LoRA erosion
  0.703→0.211 exact; no §2 claim cites `frc_` cells.
- **b4 calibration-confound block:** data complete for the 3 cited LRs; F_Δ halves
  (0.559→0.260, 0.813→0.365, 1.412→0.689) at unchanged adaptation (CS 80.14↔80.14) with
  retention recovery — mechanism holds.
- **F_Δ α=r vs α=2r scale:** disclosed in §6, no undisclosed cross-scale mixing.
  **Error bars:** every existing triplet is shown; no hidden "free" error bars.

## 5. The 3 changes that most increase credibility
1. **Kill the two hard numeric self-contradictions** (BLOCKER-1 s44 BBH 32.85<33.1;
   BLOCKER-2 the −0.945/−13.7 is the TQA-inclusive 5-task number, true 4-task=−0.937/−17.0).
   Both zero-GPU, both the first things a numeric auditor checks.
2. **Make the Llama math law honest** (W1+W2): one n across §1/§4, lead with
   within-support r (−0.76…−0.83) beside the leverage-inflated −0.92, and disclose the
   ~75% LoRA+wd composition — preempts the strongest "second-arm" attack.
3. **Close or caveat §6 "every assessed method family"** (BLOCKER-3/GAP-1) and add
   `frc_clora_k1024 s43/s44` (GAP-2) so both boundary points carry error bars.

**Files:** artifact `paper/writing/artifact_status_report.html`; ledgers
`paper/writing/data/key_numbers.md` and `handoff/34_SESSION_STATE_2026-07-12.md` (both
carry the −0.945/−13.7 mislabel); data roots `results/*/summary.json`,
`results/geo_drift/master_labeled.jsonl`, `results/forgetting{,_chunk1,_chunk_new}.jsonl`.
