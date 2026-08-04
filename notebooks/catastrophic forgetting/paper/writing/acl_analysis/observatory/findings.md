# METRIC OBSERVATORY — per-metric tables & graphs across all configurations, methods, models

**Date:** 2026-07-18. **Pool:** the frozen n=1035 on-pool runs (key_numbers.md §18.1), rebuilt
from `results/*/summary.json` with the canonical loader convention of
`analysis_final/ladder_2026-07-17.py` (families lrsw/lrswm/qwsw/qwswm/frc/frm; finite F_Δ>0 and
finite retention_core; 7 post-freeze stragglers excluded; lora_null split by run name; CorDA/
CorDA++ **withheld** — kept in `master_runs.csv` flagged, never assessed). Preflight in
`00_build_master.py` hard-asserts §18.1: n=1035, pooled r(log F_Δ, ret)=−0.847, all six family
(n, r) cells to 3 decimals — **reproduced exactly**. The 32 quarantined-but-finite runs stay in
the pool (freeze convention, 01_law_final.md §1.1) and are flagged `quarantined` in every CSV.
Geometry join: 1034/1035; CE join: 911/1035 (Qwen sweeps ~61%, seed-block deletion → **no
per-seed Qwen CE anywhere below**). All numbers cite the saved script that produced them.

---

## Ranked top findings

1. **Magnitude first, in every single view.** Pooled r(log F_Δ, retention_core) = −0.847
   (n=1035), per family −0.830…−0.929 (`00_build_master.py` preflight). No other axis in the
   observatory comes close in any family: learning rate −0.47…−0.72, log fro_total −0.71…−0.91,
   log spec_max/dw_sv_max −0.76…−0.87 (`10_m1_tradeoff.py`, `20_m2_magnitude.py`
   → `m2_corr_structure.csv`). F_Δ is the tightest axis in all 6 families without exception.

2. **At matched F_Δ, methods are nearly interchangeable — until the knee.** In the lowest
   matched log-F_Δ bins the max−min spread of per-method mean retention is only **0.6–1.8 pp**
   (lrswm −1.0: 0.63; lrsw −1.0: 0.99; qwswm −1.5: 0.91; qwsw −1.0: 1.78), rising to 4–10 pp in
   at/above-knee bins where runs are few and collapse-contaminated (`50_extra_checks.py` →
   `m1_matched_spread.csv`, from `m1_matched_fdelta.csv`). Method identity matters little where
   training is stable; apparent method gaps concentrate in the unstable high-magnitude regime.

3. **LoRA+wd owns the best corner of the trade-off plane, now multi-seed.** Best-adaptation
   operating points (`10_m1_tradeoff.py` → `m1_op_points.csv`; mean±sd over seeds within the
   one best cell): lrsw LoRA+wd(0.3)@5e-4 = **81.75±0.17 CS-8 / 25.86±0.37 ret** at the
   smallest op-point F_Δ of the family (0.399±0.012) — the frozen multi-seed confirmation of the
   old single-seed §3 point (81.6/25.6). On frm it has the best adaptation (68.48±0.91) at
   **8–21× lower CE drift** than the other methods' op points (KL 0.216 vs up to 4.455;
   `40_m4_cedrift.py` → `m4_op_points.csv`, range check in `50_extra_checks.py`).

4. **The same LR buys a different update size per method, in all 6 families.** SC-LoRA turns a
   given LR into the largest F_Δ everywhere (+0.2–0.5 dex above LoRA+wd at lr 1e-4/3e-4);
   LoRA+wd/CLoRA sit lowest; DoRA's F_Δ explodes (10²–10³) at high LR on the Llama sweeps
   (`20_m2_magnitude.py` transmission block + `m2_fig_fdelta_vs_lr`). LR is a proxy with a
   method-dependent gain — consistent with §18.5.

5. **spec_max is magnitude, not geometry — confirmed observatory-wide.** r(log F_Δ, log
   spec_max) = **+0.931 pooled**, +0.906…+0.954 per family (`20_m2_magnitude.py` →
   `m2_corr_structure.csv`, matches the verification memo's ≈+0.93). Bonus pipeline check:
   log spec_max (geo_drift pipeline) vs log dw_sv_max (summary headline) correlate at
   **r = 1.0000** (n=1034) — they are the same measurement computed twice
   (`50_extra_checks.py`).

6. **A clean update-shape fingerprint exists at matched magnitude — and it is second-order for
   retention.** Pooled median stable_rank(ΔW): DoRA 4.5 < LoRA 5.0 < MiLoRA 8.1 ≈ CLoRA 8.3 ≈
   LoRA-Null 8.7 ≈ LoRA+wd 8.9 « SC-LoRA 14.5 < PiSSA 18.1 (`30_m3_geometry.py`). Inside
   matched F_Δ bins SC-LoRA's stable rank is 2–4× LoRA/DoRA's in every family
   (`m3_matched_fdelta_geom.csv`, `m3_fig_box_stablerank`). Its retention leverage after
   controlling log F_Δ is bounded: partial r(stable_rank, ret | log F_Δ) = −0.32…−0.67 on the
   four Llama families but **≈0 on both Qwen families** (−0.004 qwsw, +0.073 qwswm)
   (`30_m3_geometry.py` → `m3_residual_corr.csv`) — magnitude first, geometry second, and the
   second-order part is model-dependent.

7. **CE drift is the same axis read on the base corpus.** r(log KL, log F_Δ) = +0.84…+0.96 per
   family; r(log KL, retention) = −0.78…−0.92; the identity KL = CE − base-entropy holds to
   1.8e-15, and base entropy is family-constant (sd ≤ 0.021 nats; Llama ≈1.82–1.83, Qwen
   ≈1.93) so KL is cleanly comparable within family (`40_m4_cedrift.py`).

8. **Seed noise is small for retention, large for adaptation, and model-asymmetric.**
   Within-cell sd of retention_core: 0.43–0.94 pp on Llama families vs 1.92–2.89 pp on Qwen
   (means over cells with ≥2 seeds; reproduces §18.1's within-cell SDs). Within-cell sd of
   adaptation is much larger on the CS families (mean 8.7 pp lrsw, 8.2 pp frc; medians ~6 pp),
   driven by format-collapse cells (`10_m1_tradeoff.py` → `m1_seed_noise.csv`). Single-seed
   adaptation rankings on CS are therefore unreliable; retention rankings much less so.

---

## M1 — retention / adaptation trade-off

**Files:** `m1_master.csv`; tables `m1_dist_all`, `m1_op_points`, `m1_matched_fdelta`,
`m1_matched_spread`, `m1_seed_noise` (.csv+.md); figures `m1_fig_scatter_tradeoff`,
`m1_fig_retention_vs_lr`, `m1_fig_adapt_vs_lr`, `m1_fig_box_retention` (.png+.pdf).
Script: `10_m1_tradeoff.py` (+`50_extra_checks.py` for the bin-spread table).

**What varies by method.** At best-adapt operating points the method ordering is
family-dependent but LoRA+wd is on or near the adaptation-retention frontier in every family
(`m1_op_points.csv`): lrsw 81.75/25.86 (best on both axes among ≥3-seed cells), lrswm second
in adaptation (50.67±1.33) with the best retention (24.21±0.32), qwsw best adaptation (87.77,
1-seed E3 cell) — with CLoRA close (87.02±0.19 / 39.52±1.15, 4 seeds). Full-pool per-method
retention distributions (`m1_dist_all.csv`) mostly reflect how much magnitude each method's
LR range reached, not intrinsic robustness: e.g. lrsw means LoRA 24.1, LoRA-Null 24.2,
MiLoRA 24.3 vs SC-LoRA 18.1 — but the matched-F_Δ table (`m1_matched_fdelta.csv`) compresses
those gaps to ≤1 pp in the low bins.

**What varies by LR.** Retention vs LR is monotone-down everywhere
(r(log LR, ret) −0.47…−0.72) but much looser than F_Δ; adaptation vs LR is inverted-U with
per-method optima (`m1_fig_adapt_vs_lr`).

**What is seed noise.** See finding 8; additionally the trade-off correlation itself flips
sign across families: run-level r(adapt, ret) = +0.16 (lrsw), −0.22 (lrswm), +0.49 (qwsw),
+0.71 (qwswm), +0.24 (frc), **+0.86 (frm)** (`10_m1.log`). Positive values mean the pools are
dominated by the "both fail together" (above-knee collapse) axis, not by a frontier trade-off
— the trade-off is only visible near each method's frontier, which is why op-point tables
(not pooled scatter) are the right cross-method comparison.

## M2 — magnitude family (F_Δ primary; fro_total, spec_max, dw_sv_max)

**Files:** `m2_master.csv`; tables `m2_dist_all`, `m2_op_points`, `m2_corr_structure`,
`m2_matched_fdelta_specmax` (.csv+.md); figures `m2_fig_fdelta_vs_lr`, `m2_fig_box_fdelta`,
`m2_fig_specmax_vs_fdelta`. Script: `20_m2_magnitude.py`.

**Interpretation.** The four magnitude metrics form one tight log-log block (r 0.85–0.95 with
log F_Δ in every family; `m2_corr_structure.csv`), and F_Δ is uniformly the most
retention-predictive member (finding 1) — consistent with §18.6 (F_Δ beats ‖ΔW‖_F and
dw_sv_max). Method differences within M2 are transmission differences: at fixed LR the
method ordering of F_Δ is stable (SC-LoRA highest, LoRA+wd/CLoRA lowest, spread ≈0.3–0.5 dex;
transmission block in `20_m2.log`). At matched F_Δ, DoRA carries its magnitude in the most
spiky form (highest log spec_max per bin) while SC-LoRA/CLoRA are flattest
(`m2_matched_fdelta_specmax.csv`) — i.e. what looks like a "geometry" difference in spectral
norms is mostly allocation of the same budget. Seed noise in log F_Δ is small within cells
except in the near-knee Qwen cells flagged in §18.1.

## M3 — geometry / major-minor structure (stable_rank, eff_rank, e_top/e_bot, amp_top)

**Files:** `m3_master.csv`; tables `m3_dist_all`, `m3_op_points`, `m3_matched_fdelta_geom`,
`m3_residual_corr` (.csv+.md); figures `m3_fig_stablerank_vs_lr`, `m3_fig_box_stablerank`,
`m3_fig_box_etop`, `m3_fig_etop_vs_ebot`. Script: `30_m3_geometry.py`.

**Interpretation.** Geometry is where method identity actually lives: the stable-rank boxes
separate methods far more cleanly than any retention/adaptation metric does
(`m3_fig_box_stablerank`; finding 6), and the ordering persists inside matched-F_Δ bins, so
it is a genuine shape fingerprint, not a magnitude echo. The high-spread cluster
{SC-LoRA, PiSSA} vs concentrated {DoRA, LoRA} brackets the §19.3 284B recurrence clusters.
For retention, geometry stays second-order (guardrail; ladder §19.1: ΔR² +0.017 after
magnitude): the per-family partials here (`m3_residual_corr.csv`) put stable_rank/eff_rank at
−0.3…−0.67 on Llama, ≈0 on Qwen, e_top −0.12…−0.40 — real, bounded, model-dependent. Note the
partials are run-level and descriptive (seeds within cells correlated). LR mostly moves
stable rank up (more directions recruited at larger updates) — `m3_fig_stablerank_vs_lr`;
within-cell seed scatter of stable rank is visibly smaller than its method gaps (echoes
§19.2's seed-stability of e_top/stable_rank).

## M4 — CE drift (forgetting_ce, forgetting_kl, base_entropy)

**Files:** `m4_master.csv`; tables `m4_dist_all`, `m4_op_points`, `m4_matched_fdelta_kl`
(.csv+.md); figures `m4_fig_kl_vs_lr`, `m4_fig_box_kl`, `m4_fig_kl_vs_retention`.
Script: `40_m4_cedrift.py`.

**Interpretation.** KL-to-base behaves as a continuous, benchmark-free proxy of the same
magnitude axis (finding 7). Per-method KL distributions mirror the F_Δ transmission ordering
(SC-LoRA/DoRA high, LoRA+wd/CLoRA low; `m4_dist_all.csv`), and at matched F_Δ the per-method
KL means converge (`m4_matched_fdelta_kl.csv`). The op-point KL table is the sharpest
"cheap-retention" exhibit: at each method's own best-adaptation point, KL spans 5× (lrsw),
15× (qwswm, frc) and 21× (frm) across methods, with LoRA+wd lowest in 4/6 families
(`m4_op_points.csv`, `50_extra_checks.py`). Coverage caveat: qwsw 93/151, qwswm 99/164 CE
rows with a seed-block deletion pattern — all Qwen CE numbers here are pooled over available
seeds, and no per-seed Qwen CE comparison is made (verification-memo bar).

---

## Surprises / possibly-new observations

- **The "trade-off" is not a trade-off over most of the pool.** Run-level r(adapt, retention)
  is *positive* in 5/6 families and reaches +0.86 on frm (`10_m1.log`): past the knee, runs
  lose adaptation and retention together. The Pareto framing only exists on the frontier —
  worth a sentence in the paper to preempt a reviewer scatter-reading objection.
- **LoRA+wd sits ~+2 pp above the other methods inside the lrsw −0.5 matched bin** (25.84 vs
  21.85–24.36; `m1_matched_fdelta.csv`). Bin-composition caveat: its runs sit at the low edge
  of the bin. Consistent with its known small above-curve residual, not a new law violation.
- **SC-LoRA's best-adapt point on Qwen math is the family's best**: 77.23±0.79 GSM8K with
  retention 43.14±0.71 at the *smallest* op-point F_Δ (0.107) (`m1_op_points.csv`). Post-E4
  (calibration artifact resolved), SC-LoRA's high magnitude-per-LR is an asset at low LR:
  more adaptation per unit forgetting there. Candidate one-liner for the method-offsets
  discussion.
- **MiLoRA+wd (E6) shows up cleanly in the observatory tables**: 80.22 CS-8 / 26.66 ret
  (n=1 cell, 2 runs on-pool; `m1_op_points.csv`, `m1_dist_all.csv`) — wd transfers; tiny n,
  appendix-only.
- **qwswm's −0.5 matched bin has a 30.6 pp method spread** (`m1_matched_spread.csv`) — this is
  the seed-unstable near-knee Qwen territory of §18.1 (includes quarantined runs); flagged so
  nobody quotes a method ranking from it.
- **Magnitude *suppresses* the geometry signal in lrsw**: partial r(stable_rank, ret | log F_Δ)
  = −0.60 vs raw −0.42 (`m3_residual_corr.csv`) — the geometry association gets *stronger*
  after controlling magnitude on Llama-CS, while it vanishes on Qwen. The model-dependence of
  the second-order axis may be worth one sentence in Limitations.
- **dw_sv_max ≡ spec_max at r=1.0000** across 1034 runs (`50_extra_checks.py`) — two pipelines,
  one measurement; the paper should cite only one of them to avoid pseudo-replication.

## Candidate paper exhibits

| Exhibit | File(s) | Verdict |
|---|---|---|
| Best-adapt operating points, all 6 families, multi-seed | `m1_op_points.csv/.md` | **MAIN** (table; the cross-method comparison done right) |
| Adaptation-vs-retention plane, 6 families, op points starred | `m1_fig_scatter_tradeoff` | **MAIN** (or 2-family cut; full grid → appendix) |
| F_Δ vs LR transmission per method | `m2_fig_fdelta_vs_lr` | **MAIN** candidate (replaces/generalizes old fig7-A at n=1035) |
| Stable-rank fingerprint boxes | `m3_fig_box_stablerank` | **MAIN** candidate (the "geometry is identity, not retention" visual) |
| Matched-F_Δ retention bins + spread | `m1_matched_fdelta.*`, `m1_matched_spread.*` | APPENDIX (supports the fairness claim numerically) |
| Magnitude-family correlation structure | `m2_corr_structure.*` | APPENDIX (axis-choice justification; §18.6 companion) |
| Op-point CE drift (KL) table | `m4_op_points.*` | APPENDIX, with the frm 21× line quotable in text |
| KL vs retention, 6 families | `m4_fig_kl_vs_retention` | APPENDIX (CE corroboration visual) |
| Geometry partials per family | `m3_residual_corr.*` | APPENDIX (second-order + model-dependence) |
| Seed-noise table | `m1_seed_noise.*` | APPENDIX (methodology/limitations support) |
| Retention/adapt/KL vs LR grids, distribution boxes | `m1_fig_retention_vs_lr`, `m1_fig_adapt_vs_lr`, `m1_fig_box_retention`, `m2_fig_box_fdelta`, `m4_fig_kl_vs_lr`, `m4_fig_box_kl`, `m3_fig_*` others | APPENDIX / supplementary |

**Reproduce:** from this directory with the repo venv
(`/home/guyb/UIOrthoLoRA/.venv/bin/python`): `00_build_master.py`, then `10_m1_tradeoff.py`,
`20_m2_magnitude.py`, `30_m3_geometry.py`, `40_m4_cedrift.py`, `50_extra_checks.py`
(logs: `00_build_master.log`, `10_m1.log`, `20_m2.log`, `30_m3.log`, `40_m4.log`,
`50_extra.log`).
