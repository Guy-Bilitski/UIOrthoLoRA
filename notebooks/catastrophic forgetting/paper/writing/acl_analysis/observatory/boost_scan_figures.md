# Boost scan: finished-but-unused figures (audit 2026-08-06)

Scope: rendered figures not `\includegraphics`-ed by `paper/.overleaf-git/main.tex`.
Shipped set (11): fig0_hero, fig_family_scatter (excluded from proposals by instruction),
fig_pareto, fig9_lr_artifact, fig_dose_response, fig_ce_proxy, fig_lr_band,
fig_lrsweep_{lrsw,lrswm,qwsw,qwswm}. Every PNG below was opened and inspected.

## Top 8, best first

### 1. fig_cross_literature.png
- Path: `paper/.overleaf-git/figures/fig_cross_literature.png` (already uploaded to Overleaf; also `paper/writing/figures/`)
- Shows: our Llama-2 CS sweep (n=49) and CLoRA's published Table 4 points overlaid on one retention-vs-F_delta axis; parallel fitted slopes (-14.7 vs -14.3 pp/decade), both base-BBH lines annotated.
- Quality: publication grade. Clear labels, stats in the panel, colorblind-safe two-series palette.
- Generator: `paper/writing/fig_cross_literature.py`
- Supports: Appendix `app:clora` ("CLoRA published cross-check"), which states exactly these numbers (r=-0.98, pooled slope -14.65, base 34.91 vs 33.10) with tables only and no figure. This is the single strongest text-claim-with-no-visual match in the paper. Minor: figure prints slope -14.7 vs text -14.65 (rounding); verify against generator before shipping.
- Equivalent in main.tex: none (Table `tab:clora` only).

### 2. fig_geometry_4panel.png
- Path: `paper/.overleaf-git/figures/fig_geometry_4panel.png` (also `paper/writing/figures/`)
- Shows: 4 panels — (A) retention vs F_delta pooled scatter; (B) partial correlations of geometry metrics given log F_delta collapsing except stable rank; (C) per-method geometry fingerprint heatmap (z-scored e_top/e_bot/e_in/stable rank); (D) SC-LoRA input-side top-subspace energy by layer vs LoRA/MiLoRA baselines with LR-erosion inset.
- Quality: publication grade, dense but legible; annotated.
- Generator: `paper/writing/fig_geometry_4panel.py`
- Supports: §5.2 RQ2 geometry paragraph — the nested-model claim ("geometry adds ~23x less explained variance") and the fingerprint claim are currently carried only by `tab:fingerprint` / `tab:geometry-battery`; panel D also supports the §5.1 SC-LoRA mechanism sentence and `app:geoproj`. Caution: panel A says n=280, pooled r=-0.82 — an older pool than the paper's n=1035 / r=-0.847; would need regeneration or caption care before use.
- Equivalent in main.tex: tables only; no geometry figure anywhere.

### 3. m3_fig_box_stablerank.png
- Path: `paper/writing/acl_analysis/observatory/m3_fig_box_stablerank.png`
- Shows: stable-rank distribution of Delta-W per method, boxes + points, all six run families (SC-LoRA/PiSSA high, DoRA/LoRA low; CorDA marked withheld).
- Quality: publication grade; consistent method palette, per-family panels, full n.
- Generator: `paper/writing/acl_analysis/observatory/30_m3_geometry.py`
- Supports: §5.2 "At matched F_delta the update's stable rank separates the methods" — findings.md marks this the MAIN candidate for "geometry is identity, not retention". Built on the current n=1035 pool (unlike the legacy set below).
- Equivalent in main.tex: `tab:fingerprint` numerically; no visual.

### 4. m2_fig_fdelta_vs_lr.png
- Path: `paper/writing/acl_analysis/observatory/m2_fig_fdelta_vs_lr.png`
- Shows: F_delta vs learning rate per method, six family panels — same LR produces systematically different magnitudes per method (SC-LoRA highest, LoRA+wd lowest).
- Quality: publication grade; log axes, full pool, faint per-seed points behind medians.
- Generator: `paper/writing/acl_analysis/observatory/20_m2_magnitude.py`
- Supports: the RQ2 mechanism sentence "the learning rate matters only through the magnitude it sets" and the fixed-LR-bias argument behind `fig9_lr_artifact` / `tab:lr_artifact`. The transmission step itself (LR -> F_delta) is shown nowhere in main.tex. findings.md: MAIN candidate, "replaces/generalizes old fig7 panel A at n=1035".
- Equivalent in main.tex: downstream consequences shown (fig9, fig_lr_band); the LR->magnitude link itself is not.

### 5. fig5_per_benchmark.png
- Path: `paper/.overleaf-git/figures/fig5_per_benchmark.png` (uploaded, never included)
- Shows: per-benchmark retention vs F_delta (BBH, MMLU, MMLU-Pro, ARC-C, TruthfulQA) with per-benchmark slopes, plus a "which knowledge dies fastest" slope bar chart.
- Quality: good; minor footer/axis-label text overlap at bottom; single family only (Llama-2 CS, n=49, s42) and includes ARC-Challenge which §5.4 excludes as contaminated.
- Generator: `paper/writing/make_figs_split_lora_null.py` (also `paper_figs_v2.py` at repo root, older duplicate)
- Supports: §5.4 RQ4 "Retention benchmarks degrade in a fixed order" — the only RQ subsection with no figure at all. Would want a re-render on the current pool (or an observatory equivalent) rather than shipping as-is.
- Equivalent in main.tex: `tab:fragility` (slopes table); no visual.

### 6. m4_fig_kl_vs_retention.png
- Path: `paper/writing/acl_analysis/observatory/m4_fig_kl_vs_retention.png`
- Shows: retention vs KL(FT||base), six families, per-family r = -0.78 to -0.92 in-panel, all methods.
- Quality: publication grade.
- Generator: `paper/writing/acl_analysis/observatory/40_m4_cedrift.py`
- Supports: §5.3 RQ3 first claim "it correlates with retention at -0.63 to -0.92 per run family" (figure panels show the raw relation the shipped `fig_ce_proxy` calibration is built on). findings.md: APPENDIX candidate. Note the in-panel r range (-0.78..-0.92) is narrower than the text's -0.63..-0.92; reconcile pool/definition before use.
- Equivalent in main.tex: `fig_ce_proxy` shows the calibrated mapping, not the underlying scatter; complementary rather than duplicate.

### 7. fig_ce_vs_magnitude.png
- Path: `paper/writing/figures/fig_ce_vs_magnitude.png` (+ .pdf; not in Overleaf dir)
- Shows: CE-to-base (WikiText-103, MiLoRA Table 8 metric) rising with F_delta (Spearman 0.94, n=6), MiLoRA ~= LoRA at matched magnitude; side panel reproduces MiLoRA's published Table 8 ordering.
- Quality: publication grade; small-n by design (external cross-check).
- Generator: `paper/writing/fig_ce_vs_magnitude.py`
- Supports: a second published-numbers corroboration in the spirit of `app:clora`, tying MiLoRA's own loss-drift metric (cited in §5.3) to the magnitude axis. No section currently shows this; would slot into the appendix next to the CLoRA cross-check.
- Equivalent in main.tex: none.

### 8. fig_efficiency.png
- Path: `paper/writing/figures/fig_efficiency.png` (+ .pdf; not in Overleaf dir)
- Shows: train wall-clock relative to LoRA (DoRA 2.14x, everything else ~1.0x) with one-time init taxes annotated, plus CLoRA additional resident memory vs projector size k (0.42 -> 6.69 GB).
- Quality: publication grade.
- Generator: `paper/writing/fig_efficiency.py`
- Supports: §5.2 closing cost claim and `tab:cost` (table_cost_of_geometry.tex) — "in train time, memory, or an initialization pass". A table equivalent ships, so this is an appendix-visual upgrade, not a gap.
- Equivalent in main.tex: `tab:cost` (numeric equivalent exists).

## Other unused assets (not recommended, with reasons)

Legacy numbered set, uploaded to `paper/.overleaf-git/figures/` but superseded by the
current exhibits; all from the single-seed s42 Llama-CS n=49 era with "provisional /
pending seeds 43/44" caveats baked into the pixels (generator:
`paper/writing/make_figs_split_lora_null.py`; older duplicate `paper_figs_v2.py` at repo root):
- `fig1_magnitude_law.png` — axis-choice (F_delta vs spectral norms). Superseded at full n by `m2_fig_specmax_vs_fdelta.png` (observatory; generator `20_m2_magnitude.py`), which is the better appendix pick for the §A.6 axis-choice justification if wanted.
- `fig2_fairness_residuals.png` — per-method residuals from the pooled law; superseded by fig_family_scatter + observatory matched-F tables; caveat text collides with footer.
- `fig3_pareto.png` — best-op-point plane; equivalent shipped (`fig_pareto`).
- `fig4_lr_sensitivity.png` — adaptation/retention vs LR (generator `paper_figs_v2.py`; not in make_figs_split); equivalent shipped (`fig9_lr_artifact` + `fig_lr_band`).
- `fig6_supporting_structure.png` — adaptation-needs-magnitude + sigma_max unfairness + fixed-budget efficiency; pieces superseded by fig_dose_response, m2_fig_specmax_vs_fdelta, fig_efficiency.
- `fig7_lr_is_the_proxy.png` — LR->F_delta transmission, R2 0.32 -> 0.74; superseded by m2_fig_fdelta_vs_lr (item 4) at n=1035.
- `fig8_magnitude_budget.png` — two-panel adaptation/retention vs F_delta with sweet-spot band; overlapping footer text (axis label collides with note); superseded by fig0_hero + fig_dose_response.
- `op_points.png` (writing/figures) — a table rendered as PNG with overlapping footer text; the paper ships real LaTeX tables for this (tab:grand / tab:lr_artifact).

Observatory, rendered, lower priority (all have generators `10_m1/20_m2/30_m3/40_m4`):
- `m1_fig_scatter_tradeoff.png` — all-runs adaptation-vs-retention plane, 6 families, op points starred. findings.md calls it MAIN, but the shipped `fig_pareto` + `fig_family_scatter` now cover the same story; appendix-grid option only.
- `m2_fig_specmax_vs_fdelta.png` — spec_max rides the magnitude axis (r=+0.91..0.95); clean appendix support for the axis-choice paragraph.
- `m1_fig_retention_vs_lr / m1_fig_adapt_vs_lr / m1_fig_box_retention / m2_fig_box_fdelta / m3_fig_box_etop / m3_fig_etop_vs_ebot / m3_fig_stablerank_vs_lr / m4_fig_kl_vs_lr / m4_fig_box_kl` — supplementary grids/boxes; fine quality, no distinct unshipped claim.

Frozen backup (`paper/writing/figures_frozen_backup/`): every file has a same-name current
equivalent or is a dated pre-revision (`fig0_hero_pre_20260719.png`); nothing unique to flag.

Generator-only outputs never rendered anywhere (dead pipeline, no action proposed):
- `paper_assets.py` (repo root) would write `fig1_pareto_<domain>.png`, `fig2_magnitude.png`, `fig3_leakage.png` — none exist in any figure directory; predates the v2 pipeline.

MISSING-GENERATOR: none among the candidates — every rendered figure above maps to a
tracked generator (house rule satisfied). The only orphan direction is the reverse one
(paper_assets.py outputs with no renders).
