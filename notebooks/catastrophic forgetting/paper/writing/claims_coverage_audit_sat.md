# Claims-coverage audit — pre-freeze (Saturday)

**Auditor role:** adversarial reviewer / claims-vs-data map.
**Date:** 2026-07-11. **Python:** `/home/guy/UIOrthoLoRA/.venv/bin/python`.
**Live registry:** `results/campaign_summary.jsonl` (499 raw rows → 483 unique by latest `evaluated_at`).
**Paper snapshot:** `paper/writing/paper.tex` (frozen against the 07-02 320-row `campaign_summary_clean.jsonl`).
**Artifact (authoritative current claims):** `paper/writing/artifact_status_report.html` (updated 07-11 16:55).

> **Headline finding:** the paper's *core* is bulletproof — every load-bearing Llama-2
> commonsense law statistic reproduces to the decimal against live data. The problem is
> the opposite of overclaiming: **the paper is ~9 days stale and UNDER-claims what the
> data now supports** (full Qwen replication, 3-seed error bars, the B4 calibration
> control, a 5-adapter math sweep), while a handful of *secondary* numbers
> (cross-literature slope, the ΔR²=0.0002 geometry null, the rank partial) are stated
> with a precision the released data does not reproduce. Both must be fixed before freeze.

---

## SEVERITY-RANKED GAP LIST

### BLOCKER (fix before freeze — factual/reproducibility errors or paper↔artifact contradictions)

**B1. Qwen replication: paper says "partial / LoRA-only / does not replicate quantitatively"; the data has a FULL 49-cell replication.**
- Paper (abstract L96, intro L176, §law L452-461, claims table): "~13 of ~112 cells, mostly LoRA … r=−0.88, n=7 … replicates sign/monotonicity, not the quantitative law."
- Live data (`qwsw_*_s42`): **7 adapters × 7 LRs = 49 cells, pooled core r=−0.857, R²=0.735, slope −31.98, p=3.7e-15; broad r=−0.937, n=49.** Numerically indistinguishable from Llama's −0.858. The artifact already claims this ("replicates in full … n=49 … r=−0.86").
- Why blocker: the paper and the supervisor artifact now directly contradict each other on a headline claim; a reviewer/supervisor comparing them loses trust. Also a large missed strength (a second-architecture replication is worth far more than "sign only").
- Fix: rewrite the Qwen paragraph to "full 49-cell commonsense replication, pooled r=−0.86, R²=0.74 (single seed, s42); steeper slope (−32) reflects Qwen's retention scale, not a different law; do not merge fits (different base ceilings). LoRA+wd flat by construction (wd caps F_Δ). Qwen math remains in-progress/pending." Keep single-seed disclosed.

**B2. Math table (Table `tab:math`) is on the stale/weak `lrswm_` recipe (GSM8K 50.6) and contradicts the authoritative `frm_` math headline (67.3) used by key_numbers §4 and the artifact.**
- Paper math table: LoRA+wd 50.6 / LoRA 46.5 / DoRA 33.3.
- `key_numbers.md` §4 (07-10): *"Quote the frm_ numbers, not the 50.6 row."* `frm_lorawd_wd0p3_lr2e4_c256`: GSM8K **67.25** (3-seed 66.8±0.8: s42 67.25 / s43 65.88 / s44 67.25); plain LoRA(wd0) `frm_lorawd_wd0_lr1e4` 64.97; MiLoRA 62.85; CLoRA-k256 60.80; PiSSA 49.66/ret3.62.
- DoRA table row is also stale: DoRA now has a full `lrswm_` sweep, best-adapt GSM8K **46.9 @ lr3e-4**, not 33.3 @ lr2e-5.
- Why blocker: internal inconsistency; the paper's own single-source-of-truth says the table is superseded, and the artifact headline (67.3) will not match the paper (50.6).
- Fix: replace the math table with the `frm_` recipe (3-seed for the LoRA+wd headline), OR keep `lrswm_` but explicitly state it is a weaker cutoff-limited recipe and delete the artifact's 67.3 claim. Recommended: adopt `frm_`, and use the **within-harness** comparison (our LoRA+wd 67.3 vs our CLoRA-k256 60.8, +6.5 pp on the SAME pipeline) rather than "67.3 vs CLoRA-published 64.6" (cross-harness; our own CLoRA reproduction is only 59-61 → apples-to-oranges).

**B3. Subspace-alignment "geometry buys ΔR²=0.0002" is metric-cherry-picked and reverses on the on-curve subset.**
- Claim (intro L169-170, §geometry L536/L604-605, discussion L1030): "subspace alignment buys ΔR²≈0.0002 (p=0.53) … no second geometric axis survives."
- Independent recompute (geo_drift join, n=308/219): ΔR²≈0.0002 reproduces **only for `ein_top`** (all-cs ΔR²=0.00018, but **p=0.60 not 0.53**). **Every other alignment metric adds a significant increment** — `e_top` ΔR²=0.007-0.010 (p<0.001), `amp_top` 0.002-0.003 (p<0.05), `e_bot`/`ein_bot` 0.003-0.004 (p<0.05). On the **on-curve-six** set **all five metrics are significant** (ΔR²=0.003-0.019, p<0.03).
- Why blocker: this is billed as "the result that separates our claim from everything prior." A reviewer who recomputes will flip it. The absolute effect is tiny, so the honest claim survives — but the *number* does not.
- Fix: state the exact metric and set ("the strongest single candidate axis, input-principal alignment `ein_top`, adds ΔR²=0.0002, n.s."), and report the **range across all five alignment metrics** ("no alignment term adds more than ΔR²≈0.02, and increments, though occasionally significant, are practically negligible"). Do not lead with the minimum. Also fix p=0.53→0.60.

**B4. Cross-literature "r=−0.98, slope −14.7 across ten CLoRA rows" is not reproducible from the printed table and the slope is wrong.**
- Claim (§law L487-489, fig `crosslit` L502-507, app L1148-1149): CLoRA Table 4 "across its ten adapter rows" gives r=−0.98, slope −14.7, "matching our −14.8."
- Table `tab:clora` prints only **6 rows (4 usable: LoRA, LoRA-L2, k1024, k2048)**. From those 4: **r=−0.99 (≈ ok) but slope = −16.2, not −14.7**; including Base (F=2.42) collapses r to −0.45. The full 10-row Table 4 and the "−12.7 baseline / −18.9 k-series" decomposition are **not in the workdir** → not independently checkable.
- Why blocker: a checkable factual claim ("ten rows," "−14.7 matching −14.8") that the shown evidence does not support. Corroboration, not core — but stated with false precision.
- Fix: either print CLoRA's full 10-row Table 4 so the fit is verifiable, or retag r/slope as `[EXTERNAL — CLoRA Table 4]` and correct the slope (the reproducible excerpt gives ≈−16). State "Base excluded as a reference row" explicitly (the result is filter-dependent on it).

**B5. Stale limitations/TODOs claim experiments were "not run" that HAVE run — SC-LoRA's provisional deviation is now resolvable by existing data.**
- Paper says (Limits L964-974, L945-954; claims table): B4 eval-matched calibration "not run … off-curve language embargoed"; seeds 43/44 "not collected."
- Data: **B4 exists** — `b4_sclora_*` (4 cells, 07-09/10): eval-matched calibration lifts SC-LoRA retention from 22.5 → **26.5-27.0 (at/above the 26.0 ceiling)**. **Seeds exist** — `lrsw_*_s43` (all 7 headline cells), `_s44` (6 of 7). At SC-LoRA's op-point, retention 22.5 (s42) → **25.6 / 25.8 (s43/s44)**, i.e. on the curve; 3-seed mean 24.6±1.8.
- Why blocker: (a) the paper asserts experiments weren't run when they were — a factual error a supervisor will catch; (b) two independent lines (seeds + B4) now show SC-LoRA's −4.15 pp deviation is a seed/calibration artifact, which *strengthens* the thesis (moving toward "7/7 on curve"). The entire "SC-LoRA is the one significant deviator + `ein_top`-erosion mechanism" narrative (§fingerprint L653-659, Fig geometry panel D) currently rests on a single unlucky seed.
- Fix: remove the "not run"/"not collected" TODOs; add "seeds 43/44 and the B4 eval-matched calibration both place SC-LoRA back on the curve — the s42 −4.15 pp residual is a seed/calibration artifact, not a geometric penalty." Reframe SC-LoRA from "the one deviator" to "resolved confound," and demote the erosion-mechanism paragraph to a fingerprint aside. Note: this changes the ANCOVA (the F(6,41)=7.05 / F(6,35)=9.32 full-pool significance is *entirely* SC-LoRA-driven and s42-only).

### WEAKNESS (an adversarial reviewer will press; fix or pre-empt)

**W1. MiLoRA's Table 1 / Table `tab:oppoints` operating point (CS-8 = 79.9) is a lucky seed.** 3-seed: **57.7 ± 22.7** (s42 79.9 / s43 58.7 / s44 34.5) — a collapse basin. Retention is stable (24.2±0.5). The paper ranks MiLoRA #2-3 on adaptation off a non-reproducible cell. Framing ("accuracy is the seed-sensitive axis, tied within band") partly covers it, but the hard number 79.9 sits un-caveated next to LoRA+wd's rock-solid 81.8±0.2. Fix: footnote/annotate MiLoRA's CS-8 with its 3-seed SD, or report the 3-seed mean; do not print 79.9 as a bare point estimate.

**W2. Param-matched LoRA+wd control (B5a, strict r16) is missing AND queued nowhere.** LoRA+wd is r32 (56.1M params) — capacity-matched to MiLoRA/SC-LoRA/CLoRA (all r32) but **2× plain LoRA and DoRA (r16, 28.0M)**. The queue has an r32 wd-ablation (`frc_lorawd_wd0…wd0.5`, c256) but **no r16 LoRA+wd cell anywhere** (all queued wd runs are `lora_r 32`/`64`). The referee objection "LoRA+wd wins on capacity + an extra knob no one else got" is real vs plain LoRA. Fix: either queue LoRA+wd @ r16 (matched to plain LoRA) + plain LoRA @ r32, or state precisely "LoRA+wd is capacity-matched to the three r32 elaborate adapters it is compared against; the r16 control is deferred." (The precise version is a partial *strength* the paper currently omits.)

**W3. Efficiency claims mis-stated.** (`app:repro` L1226-1240.) CLoRA "~10% per-step overhead" — **measured 14-17%** (1.166× at k1024) from `train_registry.jsonl`; understated. CLoRA "6.7 GB" is the **k2048 config, which was NOT swept** (max swept k1024 ≈ 3.3 GiB), and the GB ladder is **analytical, never instrumented** — stated as fact without hedge. (DoRA 2.1× PASS = 2.14×, verified from training wall-clock.) Fix: "~15% overhead"; "up to 6.7 GB (k2048 projection; swept max k1024 ≈ 3.3 GiB), analytical resident size, peak GPU memory not instrumented."

**W4. Rank partial correlation −0.56 does not reproduce, and its footnote is false.** Footnote L697-699 says the rank sweep is "not in the released summary registry" — but `lora_r4…r256` and `mtx_lora_r8…r128` all carry `retention_mean`+`fdelta`. Recomputed partial(retention, rank | log F_Δ) = **−0.69 to −0.74** (raw rank), −0.41 (method-controlled), and **sign-flips to +0.7/+0.99 under log-rank**. The "modest negative residual rank effect" direction survives, but −0.56 does not. Fix: remove the false "not in registry" hedge, recompute with a stated spec, and report the value that spec yields (or soften to "a residual negative rank effect, collinear with stable rank").

**W5. The whole geometry/ANCOVA battery is single-seed (s42).** F(6,41)=7.05, F(6,35)=9.32, the −4.15 pp SC-LoRA residual, LOMO RMSE 9.05 — all s42-only, all driven by the one cell that regresses to the curve on s43/s44 (see B5). The law itself is seed-robust (retention SD ≤0.5 for on-law cells; median 0.43 across the 34-config `mtx_` 3-seed matrix), but the *geometry-inertness* battery is not. Fix: at minimum, add SC-LoRA's 3-seed retention to the residual discussion; ideally recompute the ANCOVA residual for SC-LoRA seed-averaged.

### NICE-TO-HAVE (missed strengths + small provenance fixes)

**N1. Math law now spans 5 adapters, not 3.** Paper: "sparse … only three adapters (n=14)." Live `lrswm_ s42`: LoRA/LoRA+wd/DoRA/CLoRA/MiLoRA all n=7 → pooled **n=35, r=−0.947, R²=0.896, slope −8.85** (3-adapter subset now n=21, r=−0.968, slope −9.26; paper's "n=14, −10.1" is stale). Missed strength — broaden the math law.

**N2. 3-seed error bars now exist for 6-7 CS operating points** (the artifact already uses them). LoRA+wd 81.8±0.2 / 25.9±0.4 is *tighter and slightly better* than the s42 point (81.6/25.6). This lets the paper upgrade Claim 3 from "matches/edges, within seed noise" to "edges on both axes outside 3-seed noise vs plain LoRA (81.8±0.2 vs 79.1±0.1; 25.9±0.4 vs 23.8±0.6)." Currently the paper is over-hedged.

**N3. Scale-validation BoolQ mis-attributed.** Paper L320: LoRA@3e-4 "BoolQ 69.97." The run that gives the paired CS-8=79.1 (`lrsw_lora_r16_lr3e4_s42`) has **BoolQ 71.13** (3-seed 70.65); no run gives 69.97 with a matching CS-8. Fix: cite the actual run's 71.1 (or 3-seed 70.7). Still within scale of the canonical 69.8; just use the real number.

**N4. CE comparison provenance.** CE numbers PASS (LoRA 3.57, PiSSA 6.31, MiLoRA 3.66; ordering PiSSA>LoRA; Spearman ρ=0.94 at n=6 / 0.976 at n=49). Two clarifications: (a) all CE is on the **`frm_` math recipe** (r64/α128), not the CS sweep — say so, since it's embedded in the CS geometry section; the "matched rank" claim is correct (both r64; MiLoRA's `adapter_r`=128 is the 2r residual save). (b) Always attach n to the Spearman (0.94 = 6 rows; 0.976 = 49 cells) — never "0.94 across 49 cells."

**N5. F_Δ labeling.** Ensure no residual "Frobenius / ‖ΔW‖_F" labels leak into captions/figures; `key_numbers.md` §0 fixed the definition to CLoRA Eq 3 (`fdelta`). The paper body already relabels correctly (`\dw = F_Δ`).

---

## CLAIM-BY-CLAIM COVERAGE MAP (verified against live data)

### PASS — reproduces to the decimal (safe to freeze)

| Claim | Paper value | Recomputed (live) | Cells / seeds |
|---|---|---|---|
| Pooled law (CS) | r=−0.86, R²=0.74, slope −14.8, n=49, p=3.4e-15 | r=−0.858, R²=0.736, −14.78, n=49, p=3.4e-15 | lrsw s42, 7 adapters excl CorDA |
| On-curve law | r=−0.92, R²=0.84, −10.0, n=42 | r=−0.915, R²=0.838, −9.96 | excl SC-LoRA |
| Within-method r | −0.86…−0.97 (7 values) | −0.864…−0.972, all match | n=7 each |
| Spearman | ρ=−0.90, p=3.5e-18 | −0.896, 3.5e-18 | n=49 |
| Hockey-stick | knee 0.36, asymptote 26.8, beats line+quad on AIC | knee 0.366, plateau 26.8, AIC 99.6<107.3<114.0 | n=49 |
| Below-ceiling slope | −21 (Tobit −22) | −20.8 (ret<25, n=23) | n=49 |
| Partial corr \| method | −0.87 | −0.868 | n=49 |
| Axis choice | F_Δ 0.74 vs sv_mean 0.36 vs sv_max 0.33 | 0.736 / 0.363 / 0.328 | n=49 |
| LR proxy | R² 0.32 (LR) vs 0.74 (F_Δ) | 0.321 / 0.736 | n=49 |
| ANCOVA intercepts | ΔR²=0.13, F(6,41)=7.1 | ΔR²=0.134, F=7.05, p=3.3e-5 | s42 |
| On-curve intercepts | F(5,35)=1.79, p=0.14 | 1.79, 0.141 | s42 |
| Slopes (all/on-curve) | F(6,35)=9.3 / F(5,30)=0.28, p=0.92; SC-LoRA −26.0 | 9.32 / 0.28, 0.923; −26.0 | s42 |
| LOMO | on-curve 2.5; SC-LoRA 9.1 / −7.4 | 2.46; 9.05 / −7.40 | s42 |
| Per-benchmark slopes | MMLU −23.4, MMLU-Pro −15.2, ARC −14.9, BBH −14.3, TQA −0.5 | −23.4 / −15.2 / −14.9 / −14.3 / −0.5 (r all match) | n=49 |
| CS best-adapt table (all 7 rows) | 81.6/25.6/0.394 … | all reproduce exactly | s42 |
| Robustness (safe LRs /7) | LoRA+wd 6, others 5, DoRA 4, SC-LoRA 1 | exact | s42 |
| LR-artifact table (all 5 methods) | SC-LoRA +26.0, LoRA-Null +19.5, DoRA +5.9, MiLoRA +0.8, CLoRA −14.2 | all reproduce | s42 |
| Fingerprint magnitudes | MiLoRA e_bot 0.115/e_top 0.067, LoRA-Null 0.126, SC-LoRA ein_top 0.41, CorDA ein_bot 0.49 | 0.115/0.067, 0.126, 0.401, 0.494 | 320-adapter battery |
| SC-LoRA erosion | ein_top 0.70→0.21, r=−0.96 | 0.703→0.211, r=−0.962 | 7 LR cells |
| Qwen LoRA-only (subset) | r=−0.88, n=7 | −0.883, n=7 | s42 |
| CE numbers | LoRA 3.57, PiSSA 6.31, MiLoRA 3.66 | 3.570, 6.307, 3.659 | frm recipe |
| Efficiency DoRA | 2.1× slower | 2.14× (train wall-clock) | registry |

### PARTIAL / STALE (data supports MORE than the paper claims)

| Claim | Paper (stale) | Live data | Action |
|---|---|---|---|
| Qwen CS | LoRA-only n=7, "not quantitative" | full n=49, r=−0.86 | B1 — upgrade |
| Math law | 3 adapters, n=14 | 5 adapters, n=35 | N1 — broaden |
| Math headline | GSM8K 50.6 (lrswm) | 67.3 3-seed (frm) | B2 — replace recipe |
| Seeds | "not collected" | s43 ×7, s44 ×6 landed | B5/W1/N2 — incorporate |
| B4 calibration | "not run, embargoed" | 4 b4_sclora cells, SC-LoRA → curve | B5 — resolve |

### DISCREPANCY (stated with precision the data doesn't support)

| Claim | Issue | Severity |
|---|---|---|
| Subspace ΔR²=0.0002, p=0.53 | metric-cherry-picked (only ein_top); others significant | B3 |
| Cross-lit r=−0.98 / slope −14.7 / "ten rows" | not reproducible; excerpt gives −16.2; filter-dependent | B4 |
| Geometry sign-flip "+0.3 to +0.4 across all metrics" | only 2 of 3 flip; e_top stays −0.15; magnitudes +0.27/+0.29 | B3 (same section) |
| Rank partial −0.56 (+ "not in registry") | is in registry; recomputes −0.69/−0.74, sign-flips | W4 |
| CLoRA overhead ~10% | measured 14-17% | W3 |
| BoolQ 69.97 | actual run 71.13 / 3-seed 70.65 | N3 |

### AWAITING QUEUED CELLS (honest, genuinely-pending limitations)

- **CorDA** — withheld; all `lrsw_corda` cells are the contaminated wikitext re-eval (ret 0-23, one F_Δ=515 explosion). A clean nq_open CS re-run is **queued nowhere** in the active `master_dispatch.txt` (only `frc_cordapp` math). Honest limitation; the queue gap should be closed if CorDA is to re-enter.
- **Base ceilings (C5)** for MMLU/ARC/TruthfulQA (no-FT) — not present in the registry; broad-retention stays uncalibrated (correctly disclosed). No explicit no-FT eval cells found queued.
- **Qwen math high-LR** — only 5 low-LR `qwswm_` cells landed; completion sits in `frepro4_qwen.txt`, not the active master dispatch.
- **Qwen seeds 43/44** — Qwen is single-seed (s42); not queued. Acceptable for a "replication of the law," but disclose single-seed.

---

## QUEUE-COVERAGE CHECK (task item 5)

Genuinely pending in `master_dispatch.txt` (58 cells not yet in registry): `frc` 45 (CS wd-ablation c256 + faithful-math structured methods), `frm` 7 (math seeds 43/44), `b4` 5 (calibration expansion), `lrsw` 1 (**DoRA s44** — the one missing CS seed sibling). `frc_reservoir_B.txt` = 40 cells (CorDA++/LoRA-Null/MiLoRA/SC-LoRA faithful-math + α=1r variants).

**Gaps needed but queued NOWHERE:**
1. **Param-matched LoRA+wd @ r16** (strict B5a) — all queued wd cells are `lora_r 32`/`64`. Needed for W2.
2. **CorDA clean nq_open CS re-run** — only CorDA++ (math) is queued.
3. **Base-ceiling no-FT eval** (MMLU/ARC/TruthfulQA) for C5 — not queued.
4. **Qwen seeds 43/44** and **Qwen math high-LR** — not in the active master dispatch.

Covered by queue: DoRA s44 (✓), math seeds 43/44 (✓), structured-math methods (✓ frc_reservoir_B), matched-**rank** (r32) wd ablation (✓ frc_lorawd, but not the r16 control).

---

## FREEZE-READINESS VERDICT

**Not freeze-ready as written, but the fixes are edits, not experiments.** The empirical spine (the Llama-2 commonsense magnitude law and its full statistical battery) is verified to the decimal and is publication-solid. The blockers are (a) the paper being ~9 days behind its own data and the supervisor artifact — Qwen, math, seeds, and B4 all landed and the paper still says "pending/partial/not run" (B1, B2, B5); and (b) three *secondary* numbers stated with false precision that a reviewer will recompute and flip (geometry ΔR²=0.0002 B3, cross-lit slope B4, rank partial W4). Resolve B1-B5 (all text/number edits against existing data) and pre-empt W1-W5; then freeze.

### The 3 changes that most increase credibility
1. **Upgrade Qwen to the full 49-cell replication and adopt the `frm_` 3-seed math headline** (B1+B2). Turns two hedged weaknesses into the paper's second-strongest evidence and removes the paper↔artifact contradiction.
2. **Fix the geometry section's headline numbers** (B3+B4+W4): report the ΔR² *range* across alignment metrics (not the cherry-picked minimum), correct/retag the cross-lit slope, and fix the rank-partial value and its false "not in registry" footnote. This is where an adversarial reviewer will dig.
3. **Retire the stale "not run"/"not collected" TODOs and resolve SC-LoRA** (B5+W1+N2): use the landed seeds and B4 to reclassify SC-LoRA's deviation as a seed/calibration artifact (strengthening "geometry adds nothing"), report LoRA+wd with tight 3-seed bars, and caveat MiLoRA's lucky-seed 79.9.
