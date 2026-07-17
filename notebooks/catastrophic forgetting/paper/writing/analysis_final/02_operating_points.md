# 02 — Operating-point tables (FINAL dataset) `[2026-07-17, post-fleet-kill freeze]`

**Source:** `results/*/summary.json` (1,500 evaluated runs; 71 excluded via
`results/quarantine_diverged.txt` → **1,429 usable**). Metrics from `summary.json → headline`:
adaptation = `cs_avg` (holds GSM8K on math arms, `adapt_task`), retention = `retention_mean`
(CS arms) / `bbh` (math arms, Llama base BBH ref 33.1), F_Δ = `fdelta` (token-weighted).
LR/seed/method parsed from run names. Cross-checked against `paper/writing/data/key_numbers.md`
§16/§18 and `paper/writing/analysis_final/*.txt`. Analysis script:
`paper/writing/analysis_final/op_points_2026-07-17.py` (pure stdlib; run from repo root),
raw output: `paper/writing/analysis_final/op_points_output_2026-07-17.txt`.

**Best-LR convention.** Tables below pick each adapter's **best-mean-adaptation LR across all
landed seeds** (the mandated rule). The 07-14 artifact picked best LR **on seed 42** and then
seed-averaged; where the two conventions disagree (DoRA, MiLoRA on Llama CS; LoRA+wd on Qwen CS)
both rows are shown and the disagreement is explained — in every case it is caused by
answer-format-collapse seeds (intact retention, collapsed accuracy) dragging cell means, not by
retention changes.

---

## 1. Llama-2 CS — per-adapter best operating point (`lrsw_`, 7-LR grid 2e-5…1e-3)

| Method | best LR | CS-8 (mean±SD) | Ret-core (mean±SD) | F_Δ range | n (seeds) | safe band s42 (ret≥24, /7) |
|---|---|---|---|---|---|---|
| **LoRA+wd(0.3)** | 5e-4 | **81.75 ± 0.17** | **25.86 ± 0.37** | 0.384–0.411 | 4 (42–45) | **6/7** |
| SC-LoRA | 5e-5 | 80.61 ± 0.41 | 24.60 ± 1.85 | 0.284–0.559 | 3 (42–44) | 1/7 |
| LoRA | 3e-4 | 79.17 ± 0.20 | 23.86 ± 0.48 | 0.603–0.626 | 4 (42–45) | 5/7 |
| LoRA-Null | 5e-4 | 78.86 ± 0.17 | 21.76 ± 1.32 | 0.696–0.706 | 4 (42–45) | 5/7 |
| CLoRA-k1024 | 5e-4 | 78.29 ± 0.25 | 21.60 ± 0.39 | 0.631–0.661 | 4 (42–45) | 5/7 |
| DoRA † | 5e-4 | 76.23 ± 1.65 | 19.15 ± 1.39 | 1.214–1.246 | 3 (42–44) | 4/7 |
| MiLoRA † | 5e-4 | 77.19 ± 0.42 | 21.43 ± 0.87 | 0.840–0.866 | 4 (42–45) | 5/7 |

Base core ceiling 26.0 (canonical, §16). LoRA+wd is simultaneously highest-adaptation, highest
retention among high-adapters (≈ ceiling), lowest F_Δ, and widest safe band — unchanged verdict.

**† Convention note (DoRA / MiLoRA).** Under the 07-14 convention (best LR on s42, then
seed-average) the rows are:

| Method | LR (s42-best) | CS-8 | Ret-core | why the mean-rule flips |
|---|---|---|---|---|
| DoRA | 2e-4 | 74.29 ± 8.65 | 25.20 ± 0.33 | s44 format-collapse (acc 64.36, ret intact 25.27); healthy-seed mean 79.26 |
| MiLoRA | 3e-4 | 57.69 ± 22.67 (with s45: 63.09 ± 21.61) | 24.20 ± 0.48 (with s45: 24.37 ± 0.50) | s43/s44 format-collapse (58.67 / 34.54, ret intact 23.8–24.1); healthy-seed mean 79.58 |

Both 07-14 rows **reproduce exactly** from the final data (74.29±8.65 / 25.20±0.33 and
57.69±22.67 / 24.20±0.48). The huge SDs are answer-format collapse, not forgetting — retention
SD stays ≤0.5 pp in those cells. Recommended paper treatment: quote the 2e-4 (DoRA) / 3e-4
(MiLoRA) points as the *retention-relevant* operating points with a collapse-seed disclosure,
since the mean-rule 5e-4 picks are simply the highest LR whose seeds all avoided the collapse
basin (and pay 3–5 pp retention for it). Either way the frontier ranking vs LoRA+wd is unchanged.

Safe-band detail (s42 retention over 2e-5…1e-3):
LoRA+wd 26.74/27.80/26.94/27.66/26.18/25.55/**23.20** → 6/7 (only 1e-3 fails);
SC-LoRA 25.29/22.47/16.38/10.22/9.38/3.60/3.19 → 1/7;
LoRA fails at 5e-4/1e-3; LoRA-Null and CLoRA fail at 5e-4* (23.64/21.88) and 1e-3;
DoRA fails 3e-4/5e-4/1e-3 (1e-3 s42 quarantined, counted unsafe); MiLoRA fails 5e-4/1e-3.
(*LoRA-Null 5e-4 s42 = 23.64, just under threshold.)

Extended-LR cells (2e-3/5e-3, outside the 7-LR band) exist for some adapters and are mostly
quarantined; they never win best-LR and are excluded from the band by definition.

---

## 2. Llama-2 math — per-adapter best operating point

### 2a. Faithful math recipe (`frm_`, c256) — THE math headline

| Method | best LR | GSM8K (mean±SD) | BBH (mean±SD) | F_Δ range | n |
|---|---|---|---|---|---|
| **LoRA+wd(0.3)** | 2e-4 | **66.79 ± 0.79** | **33.57 ± 1.04** | 0.278–0.282 | 3 |
| LoRA (wd0) | 1e-4 | 63.99 ± 0.87 | 31.29 ± 0.35 | 0.434–0.448 | 3 |
| MiLoRA | 1e-4 | 63.68 ± 0.80 | 32.44 ± 0.10 | 0.448–0.452 | 3 |
| LoRA-Null | 1e-4 | 63.76 (n=1, s42) | 32.15 | 0.474 | 1 |
| CLoRA-k256 | 3e-4 | 60.65 ± 0.40 | 28.63 ± 0.11 | 1.004–1.019 | 3 |
| SC-LoRA | 1e-4 | 60.47 ± 0.53 | 27.94 ± 0.52 | 0.853–0.861 | 3 |
| LoRA r32 | 3e-4 | 59.59 ± 1.53 | 28.15 ± 0.87 | 1.283–1.297 | 3 |
| DoRA | 3e-4 | 59.19 ± 0.50 | 28.09 ± 1.07 | 2.840–2.879 | 3 |
| CorDA++ | 1e-4 | 58.76 (n=1) | 31.56 | 0.632 | 1 |
| PiSSA | 3e-4 | 49.66 (n=1) | **7.23** (collapse) | 2.206 | 1 |

- **Headline verified:** LoRA+wd 3-seed GSM8K **66.79 ± 0.79** (67.25/65.88/67.25) at BBH
  **33.57 ± 1.04** — i.e., at/above the base BBH ref 33.1. This **beats every published number
  in the CLoRA comparison set: CLoRA 64.6, MiLoRA 63.5, LoRA 60.6, PiSSA 58.2** (published
  numbers are [EXTERNAL], key_numbers §12 — cite, don't recompute), by +2.2 pp over the best
  of them, while paying zero measured BBH forgetting.
- In-pipeline anchors at the shared lr3e-4 (07-14 quoted s42-only): LoRA r32 59.59 ± 1.53
  (s42 60.20), CLoRA-k128 60.00 ± 0.52 (s42 59.59), MiLoRA 59.06 ± 0.20 (s42 58.98), PiSSA
  BBH-collapses to 7.23 (n=1) — all consistent with the 07-14 values, now 3-seed except PiSSA.
- c512-context robustness cells (frm `_c512`) reproduce the wd ordering; best single cells:
  wd0.3 lr2e-4 c512 = 69.52/68.08/67.85 GSM8K at BBH 33.57/34.28/33.16.

### 2b. Older math sweep (`lrswm_`) — superseded for headline use (key_numbers §4)

Best-LR rows (GSM8K / BBH, 3 seeds each unless noted): LoRA+wd 5e-4 50.67±1.33 / 31.70±0.64;
SC-LoRA 1e-4 59.09±0.37 / 31.09±1.08 (n=2); CLoRA-k1024 3e-4 48.62±1.09 / 31.49±1.33;
LoRA 3e-4 47.82±1.22 / 31.18±0.40; MiLoRA 3e-4 47.61±0.72 / 31.33±1.22; DoRA 3e-4
46.45±1.04 / 31.43±0.21. Quote `frm_` (2a) for the math story; `lrswm_` only feeds the pooled
magnitude-relation fits.

---

## 3. Qwen-2.5 (base core ceiling **44.35**, base BBH 47.93 / MMLU-Pro 40.77 — `base_qwen25_noft`, landed 07-16)

### 3a. Qwen CS (`qwsw_`)

| Method | best LR | CS-8 (mean±SD) | Ret-core (mean±SD) | F_Δ range | n (seeds) |
|---|---|---|---|---|---|
| **LoRA+wd(0.3)** ‡ | 5e-4 | **87.43 ± 0.23** | 40.07 ± 0.68 | 0.245–0.246 | 3 (42–44) |
| SC-LoRA § | 1e-4 | 87.15 ± 0.15 | 27.85 ± 15.96 | 0.299–0.441 | 3 (42–44) |
| LoRA | 5e-5 | 86.43 ± 0.41 | 37.95 ± 0.88 | 0.122–0.123 | 3 (42–44) |
| LoRA-Null | 2e-4 | 86.23 ± 1.60 | 38.95 ± 0.68 | 0.197–0.217 | 3 (42–44) |
| CLoRA-k1024 | 1e-4 | 87.02 ± 0.19 | 39.52 ± 1.15 | 0.127–0.130 | 4 (42–45) |
| DoRA | 1e-4 | 86.80 (n=1, s42) | 38.50 | 0.159 | 1 (3-seed alt: 2e-4 86.44±0.76 / 38.05±0.98) |
| MiLoRA | 1.5e-4 | 87.33 (n=1, s42, gap-fill LR) | 37.64 | 0.234 | 1 (4-seed alt: 2e-4 86.39±0.97 / 36.68±0.12) |

‡ **LoRA+wd convention note:** the 07-14 quote (86.9±0.6 / 40.7±0.3 at lr1e-4) **verifies
exactly** from final data (86.85±0.56 / 40.70±0.27); the mean-rule best has moved to 5e-4
(87.43±0.23 / 40.07±0.68) because all three 5e-4 seeds landed post-07-14. Both are frontier
points; lr1e-4 keeps +0.6 pp retention, lr5e-4 +0.6 pp adaptation. LoRA+wd is the only adapter
with s42 retention ≥ 24-equivalent at all 7 LRs (never drops below 38.2 mean until 1e-3 range).

§ **SC-LoRA seed story — VERIFIED:** at its best-adapt LR (1e-4) retention is
**9.44 / 36.22 / 37.88** (s42/s43/s44), tracking per-seed F_Δ **0.441 / 0.299 / 0.302** —
seed-level magnitude fluctuations produce a 28 pp retention swing at identical recipe
(the within-cell micro-test, key_numbers §17.5/§18.6, in one row). Same signature at 2e-4
(F_Δ 0.63 s42 → ret 7.4 vs 0.43 s43/s44 → ret ~30) and 3e-4.

### 3b. Qwen math (`qwswm_`; retention = BBH, base 47.93; ep6/SMOKE variants excluded)

| Method | best robust LR | GSM8K (mean±SD) | BBH (mean±SD) | F_Δ range | n |
|---|---|---|---|---|---|
| SC-LoRA | 5e-5 | **77.23 ± 0.79** | 47.71 ± 0.23 | 0.107 | 3 |
| LoRA+wd(0.3) ¶ | 3e-4 | 68.97 ± 3.33 | 47.54 ± 0.43 | 0.099–0.104 | 3 |
| LoRA-Null | 1e-3 | 72.33 ± 1.33 | 44.76 ± 0.64 | 0.370–0.394 | 3 |
| CLoRA-k1024 | 1e-3 | 70.46 ± 0.96 | 39.98 ± 0.12 | 0.432–0.442 | 3 |
| LoRA r32 | 5e-4 | 70.44 ± 0.86 (n=2; s42 quarantined) | 41.77 ± 2.31 | 0.385–0.386 | 2 |
| MiLoRA | 2e-4 | 65.35 ± 4.03 | 46.16 ± 1.28 | 0.136–0.150 | 3 |
| LoRA r16 | 3e-4 | 61.97 ± 4.80 | 47.13 ± 0.43 | 0.157–0.168 | 3 |
| DoRA | 3e-4 | 70.96 (n=1, s42) | 30.10 | 0.157 | 1 (DoRA coverage mostly single-seed) |

¶ The literal mean-rule argmax for LoRA+wd is lr1e-3 (72.93, n=1, s44) — but its s42/s43
siblings are quarantined (diverged), so 3e-4 is the defensible 3-seed operating point.
Notable: on Qwen math, **SC-LoRA at 5e-5 is the standout point** (77.2 GSM8K at base-level
BBH, F_Δ 0.107) — low-magnitude SC-LoRA sits ON the frontier, consistent with the E4 finding
that SC-LoRA's Llama deficit is a calibration-set artifact (§18.3). LoRA+wd remains
frontier-adjacent but is not the top adapter on this arm; do not over-claim it here.

---

## 4. LoRA+wd corollary controls (all `frc_`, c256, lr as stated)

### 4a. Plain-LoRA rank ladder @ lr3e-4 — capacity buys magnitude, not retention

| rank | CS-8 s42 (07-14 quote) | CS-8 all-seed | Ret-core all-seed | F_Δ all-seed | n |
|---|---|---|---|---|---|
| r8 | 78.99 (**79.0** ✓) | 74.37 ± 8.80 (s45 collapse 61.18) | 24.70 ± 0.50 | 0.516 ± 0.007 (0.518 ✓) | 4 |
| r16 | 79.56 (**79.6** ✓) | 79.37 ± 0.66 | 23.56 ± 0.71 | 0.603 ± 0.001 (0.603 ✓) | 3 |
| r32 | 73.46 (**73.5** ✓) | 77.05 ± 2.14 | 22.13 ± 0.41 | 0.747 ± 0.015 (0.739 ✓) | 5 |

The 07-14 numbers were s42-only and verify exactly. **Multi-seed softens the r32 accuracy-drop
claim** (77.05±2.14; s42 was the low seed) — quote the seed-robust part: F_Δ rises monotonically
0.516→0.603→0.747 and retention falls monotonically 24.70→23.56→22.13 with rank; no retention
benefit from capacity once F_Δ is (not) controlled.

### 4b. r16 param-matched LoRA+wd(0.3) — capacity confound dead; collapse twin now seed-resolved

- `frc_lorawdr16_wd0p3_lr5e4`: s42 **81.04 / 26.27 / F_Δ 0.334** — verifies the 07-14 quote
  exactly. Now 5 seeds: 81.04/77.79/81.19/51.49/81.39 (s45 = format collapse, ret intact 26.78);
  healthy-seed mean 80.35, retention 26.27–26.86 in ALL five seeds. Reproduces the r32 flagship
  (81.75/25.86) at half the parameters (28.0M).
- lr3e-4 twin: s42 collapse **13.53** verifies (re-eval 13.54, deterministic), retention intact
  26.84. **NEW — the queued seed replicates landed: the basin is seed-specific, not
  LR-deterministic**: s43 71.95, s44 61.33 (partial), s45 80.67, retention 26.9–27.3 everywhere.
  Update the §16 wording from "deterministic … seed replicate queued" to "seed-42-specific
  (partially s44), deterministic under re-eval, retention unaffected".

### 4c. wd×LR grid @ lr3e-4 — monotone F_Δ cap and retention recovery

s42 (the 07-14 claim) verifies: wd 0→0.5 ⇒ **F_Δ 0.747→0.255, ret 21.10→27.47** (0.75→0.26 /
21.1→27.5 ✓), stepping 21.10/24.08/26.05/27.28/27.47. All-seed means: F_Δ
0.744→0.553→0.430→0.350→0.257 (strictly monotone, SD ≤0.014); retention
22.27→24.46→25.91→26.35→25.56 (monotone through wd0.3; the wd0.5 dip is one s45 seed at 23.8).
Accuracy at high wd is seed-fragile (collapse basins: e.g. wd0.3 acc SD 22.5) — the *retention*
monotonicity is the claim, and it is seed-robust.

### 4d. E6 wd-generalization (from `analysis_final/e5_e6_salvage.txt` + §18.3) — split verdict

- **MiLoRA+wd0.3 — wd transfers:** both cells above the family curve (**+1.75 / +2.36 pp**
  residual, §18.3). Raw: lr2e-4 ret 27.99 @ F_Δ 0.248 (adapt 59.98 — low-adapt cell);
  lr5e-4 **ret 26.66 @ F_Δ 0.296 with adapt 80.22** (vs plain MiLoRA lr5e-4: 20.76 ret @ 0.840)
  — wd caps MiLoRA's magnitude exactly as it does LoRA's.
- **DoRA+wd0.3 — degenerate:** benchmark evals lost, but CE salvage shows forgetting-CE
  **20.83 (lr2e-4) / 10.37 (lr5e-4)** vs DoRA twins 2.13/2.57, spec_max up to 1183 — naive
  AdamW wd on DoRA (including its magnitude vector) breaks training. Boundary statement: wd is
  not a universally free knob; transfers to MiLoRA, breaks DoRA-as-implemented.

---

## 5. High-k CLoRA boundary (`frc_clora_k*_lr3e4_c256`, seeds 42–46)

| k | F_Δ (mean±SD) | Ret-core (mean±SD) | CS-8 (mean±SD) | n | per-seed CS-8 |
|---|---|---|---|---|---|
| 128 | 0.615 ± 0.014 | 22.68 ± 0.29 | 76.83 ± 4.77 | 4 | 79.53 / 78.91 / 79.21 / **69.69** |
| 256 | 0.586 ± 0.018 | 23.31 ± 0.78 | 78.91 ± 0.43 | 3 | 79.41 / 78.61 / — / 78.72 |
| 512 | 0.521 ± 0.019 | 23.31 ± 0.77 | 79.39 ± 0.36 | 5 | 78.86 / 79.20 / 79.74 / 79.49 / 79.64 |
| 1024 | 0.449 ± 0.010 | 24.13 ± 0.52 | 74.88 ± 10.10 | 4 | 80.14 / **59.72** / 79.93 / 79.71 |
| 2048 | 0.347 ± 0.007 | 25.26 ± 0.17 | 69.38 ± 4.26 | 5 | **68.53** / 75.63 / **65.73** / 71.52 / **65.51** |

- **Monotone boundary verified:** F_Δ falls 0.615→0.347 and retention rises 22.68→25.26
  monotonically in k (k256/k512 retention tie at 23.31) — a bigger null space caps magnitude,
  exactly the magnitude account (CLoRA works *through* F_Δ).
- **Accuracy fragility at high k verified and now stronger:** k1024 3-seed 80.1/59.7/79.9 ✓
  (s45 adds 79.71 → one collapse in four); k2048 68.5/75.6/65.7 ✓ (s45/s46 add 71.52/65.51 →
  now **4 of 5 seeds below 72**, mean 69.38±4.26). Constraining 2048 of 4096 directions reliably
  costs ~10 pp adaptation — the retention gain has a steep adaptation price, while LoRA+wd
  reaches lower F_Δ (0.35–0.41) at 81.8 CS-8. (Guardrail: CLoRA's published numbers are faithful;
  this is a capacity/adaptation trade-off at matched recipe, not a challenge to their results.)
- Math k-grid (`frm_clora_k64/128/256`) is flat by comparison (GSM8K 59.4→60.7, BBH 28.1→28.6,
  F_Δ 1.14→1.01): small k barely moves magnitude on the math recipe.

---

## 6. Efficiency (carried over — source: `paper/writing/INTERESTING_INSIGHTS.md` §7, corroborated by `paper/writing/fleet_findings.md`; not recomputable from summary.json)

- **LoRA+wd:** zero init cost, wall-clock identical to LoRA (17,126 vs 17,138 s); wd is a free
  AdamW flag.
- **DoRA:** 2.13× wall-clock (r16 2.22×) for no retention/adaptation benefit on our axes.
- **CLoRA:** frozen-P resident memory k×1,753,088 floats = **0.42 / 0.84 / 1.67 / 3.34 / 6.7 GB**
  for k128…k2048 (+9.5–14% wall-clock, k-scaling) — the §5 boundary points pay this on top of
  the adaptation cost.
- Data-aware init taxes (MiLoRA/PiSSA SVDs; SC-LoRA/LoRA-Null/CorDA calib forwards) per §7.

---

## 7. Changes vs the 07-14 artifact (every number that moved)

1. **Llama CS LoRA+wd:** 81.80±0.16 / 25.93±0.42 → **81.75±0.17 / 25.86±0.37** (s45 landed; n=3→4). F_Δ 0.38–0.41 and safe band 6/7 unchanged.
2. **Llama CS LoRA:** 79.08±0.11 / 23.81±0.58 → **79.17±0.20 / 23.86±0.48** (s45).
3. **Llama CS LoRA-Null:** 78.93±0.12 / 22.14±1.32 → **78.86±0.17 / 21.76±1.32** (s45).
4. **Llama CS CLoRA:** 78.41±0.11 / 21.59±0.48 → **78.29±0.25 / 21.60±0.39** (s45).
5. **Llama CS SC-LoRA:** unchanged (80.61±0.41 / 24.60±1.85; no s45 at 5e-5).
6. **Llama CS DoRA:** 07-14 row (74.29±8.65 / 25.20±0.33 @2e-4) reproduces exactly; under the best-**mean**-LR rule the operating point moves to 5e-4 (76.23±1.65 / 19.15±1.39) — s44@2e-4 format collapse; see §1 note.
7. **Llama CS MiLoRA:** 07-14 row (57.69±22.67 / 24.20±0.48 @3e-4) reproduces exactly; s45 landed (3e-4 now 63.09±21.61 / 24.37±0.50); mean-rule point moves to 5e-4 (77.19±0.42 / 21.43±0.87).
8. **Math headline:** LoRA+wd GSM8K 66.79±0.79 / BBH 33.57±1.04 — **unchanged, verified**. Gap to published CLoRA 64.6 = +2.2 pp.
9. **Math in-pipeline anchors:** 07-14 quoted s42-only; now 3-seed: LoRA r32 60.2→59.59±1.53, CLoRA-k128 59.6→60.00±0.52, MiLoRA 59.0→59.06±0.20. PiSSA BBH 7.23 unchanged (still n=1).
10. **Qwen CS LoRA+wd:** 07-14 point (86.9±0.6 / 40.7±0.3 @1e-4) verifies exactly; mean-rule best now 5e-4 (**87.43±0.23 / 40.07±0.68**, 5e-4 seeds landed post-07-14).
11. **Qwen SC-LoRA seed story:** verified bit-for-bit — ret 9.44/36.22/37.88 tracking F_Δ 0.441/0.299/0.302 @1e-4.
12. **Qwen math:** now a full 3-seed multi-adapter table (was "replicates qualitatively, s42-heavy"); SC-LoRA@5e-5 (77.23±0.79 GSM8K @ base-level BBH) is the standout point, LoRA+wd robust point 3e-4 = 68.97±3.33 / 47.54±0.43.
13. **Rank ladder:** s42 values verify (79.0/79.6/73.5, F_Δ 0.518/0.603/0.739); multi-seed keeps F_Δ/retention monotonicity but softens the r32 accuracy drop (77.05±2.14) and adds an r8 s45 collapse — reframe per §4a.
14. **r16 param-matched:** 81.04/26.27 verifies; 4 more seeds landed (healthy mean 80.35, one s45 collapse). **lr3e-4 "deterministic collapse twin" is now seed-42-specific** (s43 71.95 / s45 80.67 healthy) — §16's "seed replicate queued" is resolved; soften "deterministic" to "deterministic under re-eval, seed-specific".
15. **wd×lr grid:** s42 claim verifies (F_Δ 0.75→0.26, ret 21.1→27.5); all-seed means monotone through wd0.3 (22.27→26.35), wd0.5 dips to 25.56 on one seed.
16. **CLoRA k-grid:** k1024 (80.1/59.7/79.9) and k2048 (68.5/75.6/65.7) verify; s45/s46 additions make the k2048 adaptation cost *more* robust (4/5 seeds ≤71.5).
17. **Dataset totals:** 622-row registry → 1,500 evaluated dirs / 1,429 usable after 71-run quarantine (`campaign_summary.jsonl` and `results_book/` are stale — do not source).

---

## 8. Figure-ready statements

1. "At its best operating point, LoRA+wd(0.3) reaches **81.8 CS-8 with 25.9 core retention**
   (base ceiling 26.0) at **F_Δ ≈ 0.39–0.41** — simultaneously the highest adaptation, highest
   retention among high-adapters, smallest update, and the widest safe LR band (6/7 LRs keep
   retention ≥ 24; every alternative ≤ 5/7)." (4 seeds)
2. "On the faithful math recipe, LoRA+wd hits **GSM8K 66.8 ± 0.8 at BBH 33.6 ± 1.0** — above
   base BBH (33.1) — exceeding published CLoRA (64.6), MiLoRA (63.5), LoRA (60.6) and PiSSA
   (58.2), and every in-pipeline competitor by ≥ 2.8 pp GSM8K and ≥ 1.1 pp BBH." (3 seeds)
3. "The pattern replicates on Qwen-2.5: LoRA+wd 87.4 ± 0.2 CS at 40.1 ± 0.7 retention
   (base 44.35), with retention ≥ 38 at every LR in the sweep."
4. "One recipe, three seeds, 28 pp of retention: Qwen SC-LoRA @1e-4 retains 9.4 / 36.2 / 37.9
   as its per-seed F_Δ lands at 0.44 / 0.30 / 0.30 — seed-level magnitude fluctuations alone
   traverse the entire relation."
5. "Weight decay caps magnitude monotonically (F_Δ 0.74 → 0.26 as wd 0 → 0.5) and buys back
   ~4–6 pp retention; it transfers to MiLoRA (+1.8/+2.4 pp above-curve) but breaks DoRA
   as-implemented (forgetting-CE 20.8/10.4) — a knob, not a universal free lunch."
6. "CLoRA's null-space width k is a magnitude dial: F_Δ 0.615 → 0.347 and retention
   22.7 → 25.3 monotonically from k128 → k2048 — but at k ≥ 1024 adaptation becomes seed-fragile
   (k2048: 69.4 ± 4.3 CS-8), while LoRA+wd reaches lower F_Δ at 81.8. CLoRA works through
   magnitude; wd gets there without the up-to-6.7 GB k-memory tax."
7. "More rank, more forgetting: at fixed LR, r8→r32 raises F_Δ 0.52→0.75 and lowers retention
   24.7→22.1 (monotone in all seeds); a param-matched r16 LoRA+wd reproduces the r32 flagship
   point (81.0–81.4 CS, 26.3–26.9 retention) at half the parameters."

**Retired/updated framings:** DoRA "74.3±8.7" and MiLoRA "57.7±22.7" rows should not be printed
as retention evidence without the format-collapse footnote (retention was intact in every
collapsed seed); the r32 "73.5 accuracy" and the "deterministic 13.5 collapse twin" claims are
s42-specific and must be quoted as such.
